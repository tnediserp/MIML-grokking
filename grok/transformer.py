#!/usr/bin/env python
from argparse import ArgumentParser, Namespace
from typing import Any, Tuple, List, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import cos, sin, sqrt
from torch import tensor, Tensor
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import time
import os

import grok.metrics as metrics
from argparse import ArgumentParser

from grok.data import (
    DEFAULT_DATA_DIR,
    EOS_TOKEN,
    VALID_OPERATORS,
    ArithmeticDataset,
    ArithmeticIterator,
)
from grok.optimizer import CustomAdamW

DEFAULT_LOG_DIR = "logs"


class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise")
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.weight_noise > 0 and self.training:
            bias = self.bias if self.bias is None else self.bias + torch.randn_like(self.bias) * self.weight_noise
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
            # weight = self.weight * torch.exp(torch.randn_like(self.weight) * self.weight_noise)
        else:
            bias = self.bias
            weight = self.weight
            
        return F.linear(
            input,
            weight,
            bias,
        )

class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise")
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.weight_noise > 0 and self.training:
            bias = self.bias if self.bias is None else self.bias + torch.randn_like(self.bias) * self.weight_noise
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
            # weight = self.weight * torch.exp(torch.randn_like(self.weight) * self.weight_noise)
        else:
            bias = self.bias
            weight = self.weight
        return F.layer_norm(
            input,
            self.normalized_shape,
            weight,
            bias,
            self.eps,
        )


class Embedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise")
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.weight_noise > 0 and self.training:
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
            # weight = self.weight * torch.exp(torch.randn_like(self.weight) * self.weight_noise)
        else:
            weight = self.weight
        return F.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_key: int, weight_noise: float) -> None:

        super().__init__()

        self.d_key = d_key

        # head projections
        self.Wq = Linear(d_model, d_key, bias=False, weight_noise=weight_noise)
        self.Wk = Linear(d_model, d_key, bias=False, weight_noise=weight_noise)
        self.Wv = Linear(d_model, d_key, bias=False, weight_noise=weight_noise)

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Union[Tensor, None] = None,
        save_activations: bool = False,
    ) -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:

        # project queries, keys, values
        queries = self.Wq(queries)
        keys = self.Wk(keys)
        values = self.Wv(values)

        # calculate compatibility function
        attn = torch.matmul(queries, torch.transpose(keys, -2, -1))
        attn = attn / sqrt(self.d_key)

        # Filter out attention to future positions
        if mask is not None:
            attn.masked_fill_(mask == 0, float("-inf"))

        # softmax
        attn = self.softmax(attn)

        # sum the weighted value vectors
        result: Tensor = torch.matmul(attn, values)  # shape = (max_context_len, d_key)
        if save_activations:
            leaf_attn = attn.clone().detach()  # type: ignore
            leaf_values = values.clone().detach()  # type: ignore
        else:
            leaf_attn = None  # type: ignore
            leaf_values = None  # type: ignore

        return result, leaf_attn, leaf_values


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, weight_noise: float = 0.0) -> None:
        super().__init__()
        d_key = int(d_model / heads)

        attn_heads = [
            AttentionHead(d_model, d_key, weight_noise=weight_noise)
            for _ in range(heads)
        ]
        self.attn_heads = nn.ModuleList(attn_heads)
        self.Wo = Linear(d_model, d_model, bias=False, weight_noise=weight_noise)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Tensor = None,
        save_activations=False,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:

        head_outputs = [
            h(
                queries=queries,
                keys=keys,
                values=values,
                mask=mask,
                save_activations=save_activations,
            )
            for h in self.attn_heads
        ]
        head_results = [output[0] for output in head_outputs]

        if save_activations:
            layer_attns = list([output[1] for output in head_outputs])
            layer_values = list([output[2] for output in head_outputs])
        else:
            layer_attns = []
            layer_values = []

        multihead_result = torch.cat(head_results, dim=-1)
        multihead_result = self.Wo(multihead_result)
        return multihead_result, layer_attns, layer_values


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        multiplier: int = 4,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()

        d_ff = int(multiplier * d_model)

        non_linearities = {"relu": nn.ReLU, "gelu": nn.GELU}

        self.ffn = nn.Sequential(
            Linear(d_model, d_ff, bias=False, weight_noise=weight_noise),
            non_linearities[non_linearity](),
            Linear(d_ff, d_model, bias=False, weight_noise=weight_noise),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        dropout: float,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, heads, weight_noise=weight_noise)
        # self.self_attn_drop = nn.Dropout(p=dropout)
        self.self_attn_norm = LayerNorm(d_model, weight_noise=weight_noise)

        self.ffn = FFN(d_model, non_linearity=non_linearity, weight_noise=weight_noise)
        self.ffn_drop = nn.Dropout(p=dropout)
        self.ffn_norm = LayerNorm(d_model, weight_noise=weight_noise)

    def forward(
        self,
        x: Tensor,
        self_attn_mask: Tensor = None,
        save_activations: bool = False,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        a1, layer_attns, layer_values = self.self_attn(
            x, x, x, self_attn_mask, save_activations
        )
        # a1 = self.self_attn_drop(a1)
        a1 = self.self_attn_norm(x + a1)

        a2 = self.ffn(a1)
        a2 = self.ffn_drop(a2)
        a2 = self.ffn_norm(a1 + a2)

        return a2, layer_attns, layer_values


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        num_blocks: int,
        dropout: float,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model, heads, dropout, non_linearity, weight_noise=weight_noise
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        x: Tensor,
        self_attn_mask: Tensor = None,
        save_activations=False,
    ) -> Tuple[Tensor, List[List[Tensor]], List[List[Tensor]]]:

        a = x
        attentions = []
        values = []
        for block in self.blocks:
            a, layer_attentions, layer_values = block(
                a, self_attn_mask, save_activations=save_activations
            )
            if save_activations:
                attentions.append(layer_attentions)
                values.append(layer_values)
        return a, attentions, values


class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        n_heads: int = 4,
        d_model: int = 256,
        dropout: float = 0.1,
        max_context_len: int = 1024,
        vocab_len: int = 2000,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.max_context_len = max_context_len
        self.non_linearity = non_linearity

        self.vocab_len = vocab_len

        self.embedding = Embedding(vocab_len, d_model, weight_noise=weight_noise)  # type: ignore
        self.register_buffer(
            "position_encoding", self._position_encoding(max_context_len, d_model)
        )
        self.register_buffer("self_attn_mask", self.make_mask(max_context_len))

        self.decoder = Decoder(
            d_model,
            n_heads,
            n_layers,
            dropout,
            self.non_linearity,
            weight_noise=weight_noise,
        )

        self.linear = Linear(d_model, vocab_len, bias=False, weight_noise=weight_noise)

    @staticmethod
    def make_mask(context_len: int) -> Tensor:
        return torch.ones([context_len, context_len]).tril()

    @classmethod
    def _position_encoding(cls, context_len: int, d_model: int) -> Tensor:
        rows = [
            tensor(
                [
                    sin(pos / (10000 ** (i / d_model)))
                    if i % 2 == 0
                    else cos(pos / (10000 ** ((i - 1) / d_model)))
                    for i in range(d_model)
                ]
            )
            for pos in range(context_len)
        ]
        stack = torch.stack(rows, dim=1)

        return stack.T  # type: ignore

    def embed(self, indices: Tensor) -> Tensor:
        context_len = indices.shape[-1]
        pe = self.position_encoding[:context_len, :]  # type: ignore

        embedded = self.embedding(indices)

        return pe + embedded

    def forward(
        self,
        x: Tensor,
        pos: int = None,
        save_activations: bool = False,
    ) -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        """parameters:
        x:  (rank-1 tensor) vocab indices of decoder input token
                     sequence"""

        # Make sure sampling inputs are on the correct device
        x = x.to(self.embedding.weight.device)

        # make_attention mask
        this_max_context_len = x.shape[-1]
        self_attn_mask = self.self_attn_mask[  # type: ignore
            :this_max_context_len, :this_max_context_len
        ]

        # Decode
        x = self.embed(x) # shape = (batchsize, 2, d_model)
        decoded, attentions, values = self.decoder(
            x, self_attn_mask, save_activations=save_activations
        )
        
        pooled = decoded.mean(dim=-2)  # Pooling across sequence dimension. shape = (batchsize, d_model)

        # Return predictions for specific token
        if pos is not None:
            decoded = decoded[:, pos, :]

        y_hat = self.linear(pooled)
        return y_hat, attentions, values


class TrainableTransformer(LightningModule):
    """
    Adds training methods to train a generic transformer on arithmetic equations
    """

    def __init__(self, hparams: Namespace) -> None:
        """
        :param hparams: An argparse.Namespace with parameters defined in
                        self.add_model_specific_args().
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self.prepare_data()

        self.transformer = Transformer(
            hparams.n_layers,
            hparams.n_heads,
            hparams.d_model,
            hparams.dropout,
            hparams.max_context_len,
            len(self.train_dataset.tokenizer), # vocab_len
            hparams.non_linearity,
            weight_noise=self.hparams.weight_noise,
        )

        self.margin = torch.Tensor([0])
        self.next_epoch_to_eval = -1
        self.next_train_epoch_to_log = 0
        
        self.training_step_outputs = []
        self.validation_step_outputs = [] # store the validation outputs here

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        """
        Defines the hyperparameter arguments needed by instances of this
        class. This is intended to be called when parsing command line
        arguments.

        :param parser: an argparse.ArgumentParser created by the caller
        :returns: the argument parser with the command line arguments added
                  for this class.
        """
        # print("enter func: add model specific args")
        
        parser.add_argument(
            "--batchsize",
            type=float,
            # default=0.25,
            default=0,
            help="-1 -> entire dataset, 0 -> auto-calculate, 0<N<1 -> fraction of dataset, N>1 -> N",
        )

        parser.add_argument("--max_epochs", type=int, default=None)
        parser.add_argument("--max_steps", type=int, default=100000)
        parser.add_argument("--n_layers", type=int, default=2)
        parser.add_argument("--n_heads", type=int, default=4)
        parser.add_argument("--d_model", type=int, default=128)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--weight_noise", type=float, default=0.0)
        parser.add_argument("--non_linearity", type=str, default="relu")
        parser.add_argument("--max_context_len", type=int, default=50)

        parser.add_argument("--math_operator", type=str, default="+")
        parser.add_argument(
            "--operand_length",
            type=int,
            help="for list operations, the length of the lists",
        )

        parser.add_argument("--train_data_pct", type=float, default=50) # The training set size is 50% by default
        parser.add_argument("--warmup_steps", type=int, default=10)
        parser.add_argument("--anneal_lr_steps", type=int, default=100000)
        parser.add_argument("--anneal_lr", dest="anneal_lr", action="store_true")
        parser.set_defaults(anneal_lr=False)

        parser.add_argument("--max_lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=0.1)
        parser.add_argument("--weight_decay_kind", type=str, default="to_zero")
        parser.add_argument("--noise_factor", type=float, default=0)

        parser.add_argument(
            "--save_activations", dest="save_activations", action="store_true"
        )
        parser.set_defaults(save_activations=False)
        parser.add_argument("--save_outputs", dest="save_outputs", action="store_true")
        parser.set_defaults(save_outputs=False)

        parser.add_argument(
            "--logdir",
            type=str,
            default=DEFAULT_LOG_DIR,
        )
        parser.add_argument(
            "--datadir",
            type=str,
            default=DEFAULT_DATA_DIR,
        )

        return parser

    def prepare_data(self) -> None:
        """
        Used by pytorch_lighting

        Loads training data to self.train_dataset
        Loads validation data to self.val_dataset
        """
        print("preparing data")
        (self.train_dataset, self.val_dataset,) = ArithmeticDataset.splits(
            train_pct=self.hparams.train_data_pct,  # type: ignore
            operator=self.hparams.math_operator,  # type: ignore
            operand_length=self.hparams.operand_length,  # type: ignore
            data_dir=self.hparams.datadir,  # type: ignore
        )
        
        print(f"training = {len(self.train_dataset)}, val = {len(self.val_dataset)}")

    def train_dataloader(self) -> ArithmeticIterator:  # type: ignore
        """
        Used by pytorch_lighting

        :returns: an iterator for self.train_dataset
        """
        print(f"epoch {self.current_epoch} enter func: train_dataloader")
        
        device = self.transformer.embedding.weight.device
        iterator = ArithmeticIterator(
            self.train_dataset,
            device,
            batchsize_hint=self.hparams.batchsize,  # type: ignore
        )
        self.train_batchsize = iterator.batchsize
        self.batches_per_epoch = len(iterator)
        
        print(f"num of batches = {self.batches_per_epoch}, batch size = {self.train_batchsize}")
    

        return iterator

    def val_dataloader(self) -> ArithmeticIterator:  # type: ignore
        """
        Used by pytorch_lighting

        :returns: an iterator for self.train_dataset
        """
        # print("enter func: val_dataloader")
        
        device = self.transformer.embedding.weight.device
        iterator = ArithmeticIterator(
            self.val_dataset,
            device,
            batchsize_hint=-1,  # no need to batch validation data
        )
        return iterator

    def test_dataloader(self) -> ArithmeticIterator:  # type: ignore
        """
        Used by pytorch_lighting

        :returns: an iterator for self.train_dataset
        """
        # print("enter func: test_dataloader")
        
        device = self.transformer.embedding.weight.device
        iterator = ArithmeticIterator(
            self.val_dataset, device, batchsize_hint=-1  # type: ignore
        )
        return iterator

    def _scheduler_lr(self, step: int) -> float:
        """
        Used by pytorch_lighting

        :returns: the learning_rate for this training step
        """
        # print("enter func: _scheduler_lr")
        
        max_lr = self.hparams.max_lr  # type: ignore
        min_lr = self.hparams.max_lr / 10  # type: ignore
        warmup_steps = self.hparams.warmup_steps  # type: ignore
        if not self.hparams.anneal_lr:
            if step <= warmup_steps:
                lr = (float(step) / max(warmup_steps, 1)) * max_lr
            else:
                lr = max_lr
        else:
            if step <= warmup_steps:
                lr = (float(step) / max(warmup_steps, 1)) * max_lr
            elif step <= self.hparams.anneal_lr_steps + warmup_steps:
                effective_step = step - warmup_steps
                t = effective_step / self.hparams.anneal_lr_steps
                cos = (1 + np.cos(np.pi * t)) / 2
                lr = min_lr + (max_lr - min_lr) * cos
                # lr = max_lr - ((effective_step / max_effective_step) * (max_lr - min_lr))
            else:
                lr = min_lr
        return lr

    def configure_optimizers(self) -> Tuple[List[Any], List[Dict]]:
        """
        Used by pytorch_lighting

        :returns: optimizers and schedulers.
        """
        # print("enter func: configure_optimizers")
        
        optimizer = CustomAdamW(
            self.parameters(),
            betas=(0.9, 0.98),
            eps=1e-8,
            lr=1,
            weight_decay=self.hparams.weight_decay,
            noise_factor=self.hparams.noise_factor,
            weight_decay_form=self.hparams.weight_decay_kind,
        )
        # optimizer = SAM(
        #     self.parameters(),
        #     base_optimizer=CustomAdamW,
        #     rho=0.05,
        #     betas=(0.9, 0.98),
        #     eps=1e-8,
        #     lr=1,
        #     weight_decay=self.hparams.weight_decay,
        #     noise_factor=self.hparams.noise_factor,
        # )
        schedulers = [
            {
                "scheduler": LambdaLR(optimizer, lr_lambda=self._scheduler_lr),
                "interval": "step",
                "frequency": 1,
            }
        ]
        return [optimizer], schedulers

    def _accuracy(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """
        Takes the most likely solution predicted for each equation and
        calculates the frac of equations in the batch for which these
        answers were correct

        :param y_hat: The softmax tensor output of the transformer
        :param y: A tensor of the token ids for the correct answers to each
                  equation in the batch
        :returns: the fraction of equations correctly answered
        """
        # print("enter func: _accuracy")
        
        # find max prediction from output
        y_hat = torch.max(y_hat, dim=-1).indices  # shape: batchsize
        # row_accuracy = torch.min((y_hat == y), dim=-1).values  # shape: batchsize
        accuracy = (y_hat == y).float() * 100  # shape: batchsize
        return accuracy
    
    def index_extractor(self, equation: Tensor) -> Tensor:
        """
        Given an equation "a + b + ... + z = result", extract [index(a), index(b), ..., index(z)]
        Note: implemented for k-wise addition where k >= 2
        """
        add_token_index = self.train_dataset.tokenizer.stoi["+"]
        
        # Find positions of all "+" tokens in the equation
        add_positions = torch.nonzero(equation[0, :] == add_token_index, as_tuple=False).squeeze()
        # Ensure add_positions is at least 1-D tensor
        if add_positions.dim() == 0:
            add_positions = add_positions.unsqueeze(0)
        if len(add_positions) < 1:
            raise ValueError("Equation must contain at least one '+' token.")
        
        # Initialize a list to hold the indices of all operands
        operand_indices = []

        # Add the first operand before the first "+"
        num_first = equation[..., add_positions[0] - 1]
        operand_indices.append(num_first)

        # Add operands between each pair of "+" tokens
        for i in range(len(add_positions) - 1):
            num_between = equation[..., add_positions[i] + 1]
            operand_indices.append(num_between)
        
        # Add the last operand after the last "+"
        num_last = equation[..., add_positions[-1] + 1]
        operand_indices.append(num_last)

        # Stack the operand indices into a tensor with shape (batch_size, k)
        indices = torch.stack(operand_indices, dim=1)
        
        return indices

    def _step(
        self,
        batch: Dict,
        batch_idx: int,
        train: bool = True,
        reduction: str = "mean",
        grads: bool = False,
    ) -> Tuple[Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor]:
        """
        Performs one forward pass on a training or validation batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :param train: True is this is a training batch, false otherwise
        :returns: The loss from the predicted solutions to the equation,
                  The accuracy of the predicted solutions
                  The fraction of this dataset contained in this batch
                  The portion of the input equations left of the equal sign
                  The softmax probilities for the solutions to the equations
                  A list lists of attention matrices by layer and head
                  A list lists of value matrices by layer and head
                  Margin for this batch
        """
        # print("enter func: _step")
        
        x = batch["text"]  # shape = batchsize * context_len
        y = batch["target"]  # shape = batchsize * context_len
        
        indices_x = self.index_extractor(x) # batchsize * 2
        
        y_hat, attentions, values = self(
            x=indices_x, save_activations=self.hparams.save_activations  # type: ignore
        )  # shape = batchsize * vocab_size
        
        # y_hat = y_hat.transpose(-2, -1)  # shape = batchsize * vocab_size

        # Note: each sample must have exactly one '=' and all of them must
        # have it in the same position.
        eq_token_index = self.train_dataset.tokenizer.stoi["="]
        eq_position_t = torch.nonzero(y[0, :] == eq_token_index, as_tuple=False)
        eq_position = int(eq_position_t.squeeze())

        # only calculate loss/accuracy on right hand side of the equation
        y_rhs = y[..., eq_position + 1]
        y_hat_rhs = y_hat
        x_lhs = x[..., : eq_position + 1]
        

        if train:
            coeff = float(batch["target"].shape[0]) / len(self.train_dataset)
        else:
            coeff = float(batch["target"].shape[0]) / len(self.val_dataset)
        loss = F.cross_entropy(y_hat_rhs, y_rhs, reduction=reduction)

        with torch.no_grad():
            acc = self._accuracy(y_hat_rhs, y_rhs)
            if reduction == "mean":
                acc = acc.mean()

        """
        device = self.transformer.embedding.weight.device
        self.margin = self.margin.to(device)

        output = y_hat_rhs.clone()  # batchsize, vocabsize, rhs tokens
        output_m = output.clone()  # batchsize, vocabsize, rhs tokens
        target = y_rhs.clone()  # batchsize, rhs tokens

        for i in range(output.size(0)):  # batch
            for j in range(output.size(2)):  # rhs tokens
                output_m[i, target[i, j], j] = output_m[i, :, j].min()

        for i in range(output.size(2)):  # rhs tokens
            output_compressed = output[:, target[:, i], i].squeeze().diag()
            output_m_compressed = (
                output_m[:, output_m.max(dim=1).indices[:, i], i].squeeze().diag()
            )
            self.margin = torch.cat(
                (
                    self.margin,
                    (output_compressed - output_m_compressed),
                ),
                0,
            )
        """
        grad_vec = None
        if grads:
            loss.backward()
            for p in self.parameters():
                p.grad.data.div_(batch["text"].shape[0])
                if grad_vec is None:
                    grad_vec = p.grad.data.view(-1)
                else:
                    grad_vec = torch.cat((grad_vec, p.grad.data.view(-1)))
            return loss, grad_vec
        return loss, acc, coeff, x_lhs, y_hat_rhs, attentions, values


    def _save_inputs(self, outputs: Dict, ds: str) -> None:
        """
        Saves the input equations to disk for analysis later

        :param outputs: a list of tuples from self.training_step()
        :param ds: a string ('train' or 'val') naming which dataset
                   these inputs are from.
        :param train: True is this is a training batch, false otherwise
        """
        # print("enter func: save_inputs")
        
        logdir = self.hparams.logdir + "/inputs/" + ds  # type: ignore
        os.makedirs(logdir, exist_ok=True)
        pickle_file = logdir + f"/{ds}.pt"

        x_lhs = torch.cat([x["x_lhs"] for x in outputs])
        with open(pickle_file, "wb") as fh:
            torch.save(x_lhs, fh)

    def _merge_batch_activations(
        self, partial_activations: List[List[Tensor]]
    ) -> List[List[Tensor]]:
        """
        Merges the head_attentions / head_values from all batches in
        this epoch.

        :param partial_activations: A list of
                                   (lists of lists of activations by layer and head)
        :returns: A lists of lists of activations by layer and head
        """
        # print("enter func: _merge_batch_activations")
        
        # num_batches = len(partial_activations)
        num_layers = len(partial_activations[0])
        num_heads = len(partial_activations[0][0])
        activations: List = []
        for _ in range(num_layers):
            activations.append([])
            for _ in range(num_heads):
                activations[-1].append([])

        for minibatch_activations in partial_activations:
            for l, layer_activations in enumerate(minibatch_activations):
                for h, head_attn in enumerate(layer_activations):
                    # # print(f"head_attn = {head_attn}")
                    activations[l][h].append(head_attn)

        for l in range(num_layers):
            for h in range(num_heads):
                activations[l][h] = torch.cat(activations[l][h])

        return activations

    def _save_activations(self, outputs: Dict, ds: str) -> None:
        """
        Saves activations out to disk for analysis later

        :param outputs: a list of tuples from self.training_step()
        """
        # print("enter func: _save_activations")

        output: Dict[str, Any] = {}
        if self.hparams.save_outputs:  # type: ignore
            y_hat_rhs = torch.cat([x["y_hat_rhs"] for x in outputs])
            output["y_hat_rhs"] = y_hat_rhs
        if self.hparams.save_activations:  # type: ignore
            partial_attentions = list([o["partial_attentions"] for o in outputs])
            attentions = self._merge_batch_activations(partial_attentions)
            partial_values = list([o["partial_values"] for o in outputs])
            values = self._merge_batch_activations(partial_values)
            output["attentions"] = attentions
            output["values"] = values
        if self.hparams.save_outputs or self.hparams.save_activations:  # type: ignore
            logdir = self.hparams.logdir + "/outputs/" + ds  # type: ignore
            os.makedirs(logdir, exist_ok=True)
            pickle_file = logdir + f"/epoch_{self.current_epoch:010}.pt"
            with open(pickle_file, "wb") as fh:
                torch.save(output, fh)

    def training_step(self, batch, batch_idx):
        """
        Used by pytorch_lightning
        Runs one forward training pass on one batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with loss, accuracy, lr, probabilities of solutions,
                  attentions, and values
        """
        # print(f"train epoch {self.current_epoch}, batch {batch_idx}")
        
        if batch_idx == 0:
            self.training_epoch_start_time = time.time()
            self.fwd_time_in_epoch = 0

        start = time.time()
        loss, accuracy, coeff, x_lhs, y_hat_rhs, attentions, values = self._step(
            batch=batch, batch_idx=batch_idx, train=True
        )
        self.fwd_time_in_epoch += time.time() - start

        # schedulers = self.trainer.lr_schedulers[0]
        """
        if self.current_epoch != self.next_train_epoch_to_log:
            print(f"current = {self.current_epoch}, next = {self.next_train_epoch_to_log}")
            
            output = {"loss": loss}
        """
            # self.training_step_outputs.append(output)
            # return output
        # lr = schedulers["scheduler"].optimizer.param_groups[0]["lr"]
        
        # obtain the last lr
        
        optimizer = self.optimizers()
        if optimizer is not None:
            lr = optimizer.param_groups[0]["lr"]
        else: print("No optimizer found.")
        
        output = {
            "loss": loss,
            "partial_train_loss": coeff * loss,
            "partial_train_accuracy": coeff * accuracy,
            "learning_rate": torch.tensor([lr]),
            "y_hat_rhs": y_hat_rhs,
            "partial_attentions": attentions,
            "partial_values": values,
        }
        if self.current_epoch == 0:
            output["x_lhs"] = x_lhs
            
        self.training_step_outputs.append(output)
        # print(f"train epoch {self.current_epoch}: appending")

        return output


    # def training_epoch_end(self, outputs):
    #     """
    #     Used by pytorch_lightning
    #     Accumulates results of all forward training passes in this epoch

    #     :param outputs: a list of dicts from self.training_step()
    #     :param batch_idx: which batch this is in the epoch.
    #     :returns: a dict with loss, accuracy, lr, probabilities of solutions,
    #               attentions, and values
    #     """
    #     epoch_is_to_be_logged = self.current_epoch == self.next_train_epoch_to_log
    #     if epoch_is_to_be_logged:
    #         self.next_train_epoch_to_log = max(
    #             int(1.01 * self.next_train_epoch_to_log),
    #             self.next_train_epoch_to_log + 1,
    #         )
    #         with torch.no_grad():
    #             try:
    #                 loss = torch.stack([x["partial_train_loss"] for x in outputs]).sum()
    #             except Exception as e:
    #                 print("!" * 80)
    #                 print(outputs)
    #                 raise e
    #             perplexity = torch.exp(loss)
    #             accuracy = torch.stack(
    #                 [x["partial_train_accuracy"] for x in outputs]
    #             ).sum()
    #         # avg_lr = torch.stack([x["learning_rate"] for x in outputs]).mean()
    #         # max_lr = torch.stack([x["learning_rate"] for x in outputs]).max()
    #         # last_lr = outputs[-1]["learning_rate"]
    #         first_lr = outputs[0]["learning_rate"]

    #         if self.hparams.save_activations or self.hparams.save_outputs:
    #             if self.current_epoch == 0:
    #                 self._save_inputs(outputs, ds="train")
    #             self._save_activations(outputs, ds="train")

    #         logs = {
    #             "train_loss": loss,
    #             "train_accuracy": accuracy,
    #             "train_perplexity": perplexity,
    #             "learning_rate": first_lr,
    #             "len_train_ds": len(self.train_dataset),
    #             "len_val_ds": len(self.val_dataset),
    #             "batches_per_epoch": self.batches_per_epoch,
    #             "time_per_epoch": time.time() - self.training_epoch_start_time,
    #             "fwd_time_in_epoch": self.fwd_time_in_epoch,
    #         }
    #         for k, v in logs.items():
    #             self.log(k, v)


    # replaced with on_train_epoch_end()
    def on_train_epoch_end(self):
        """
        This method is called at the end of each training epoch.
        Replaces `training_epoch_end` in PyTorch Lightning v2.0.

        :param outputs: A list of dictionaries from `training_step()`.
        """
        # print(f"train epoch {self.current_epoch} is about to end, next to log = {self.next_train_epoch_to_log}")
        
        outputs = self.training_step_outputs
        
        epoch_is_to_be_logged = self.current_epoch >= self.next_train_epoch_to_log
        
        
        if len(outputs) == 0:
            1
            # print(f"Epoch {self.current_epoch}: Not a real training step")
        
        elif epoch_is_to_be_logged:
            # print(f"Epoch {self.current_epoch}: logging")
            # Update the next epoch to log
            self.next_train_epoch_to_log = max(
                int(1.01 * self.current_epoch),
                self.current_epoch + 1,
            )
            
            # Collect metrics
            with torch.no_grad():
                try:
                    loss = torch.stack([x["partial_train_loss"] for x in outputs]).sum()
                except Exception as e:
                    print("!" * 80)
                    print(outputs)
                    raise e
                
                perplexity = torch.exp(loss)
                accuracy = torch.stack([x["partial_train_accuracy"] for x in outputs]).sum()

            # Learning rate
            first_lr = outputs[0]["learning_rate"]
            
            # Saving activations/outputs if needed
            if self.hparams.save_activations or self.hparams.save_outputs:
                if self.current_epoch == 0:
                    self._save_inputs(outputs, ds="train")
                self._save_activations(outputs, ds="train")

            # Log the results
            logs = {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_perplexity": perplexity,
                "learning_rate": first_lr,
                "len_train_ds": len(self.train_dataset),
                "len_val_ds": len(self.val_dataset),
                "batches_per_epoch": self.batches_per_epoch,
                "time_per_epoch": time.time() - self.training_epoch_start_time,
                "fwd_time_in_epoch": self.fwd_time_in_epoch,
            }

            for k, v in logs.items():
                self.log(k, v)
                
        self.training_step_outputs.clear()
        
        # print(f"train epoch {self.current_epoch} ends, clearing")

    def validation_step(self, batch, batch_idx):
        """
        Used by pytorch_lightning
        Runs one forward validation pass on one batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with val_loss, val_accuracy, probabilities of solutions,
                  attentions, and values
        """
        # print(f"validate epoch {self.current_epoch}, batch {batch_idx}")
        
        if self.next_epoch_to_eval < self.current_epoch:
            self.next_epoch_to_eval = self.current_epoch
        if self.current_epoch != self.next_epoch_to_eval:
            return {}
        with torch.no_grad():
            loss, accuracy, coeff, x_lhs, y_hat_rhs, attentions, values = self._step(
                batch=batch, batch_idx=batch_idx, train=False
            )
        output = {
            "partial_val_loss": coeff * loss,
            "partial_val_accuracy": coeff * accuracy,
            "y_hat_rhs": y_hat_rhs,
            "partial_attentions": attentions,
            "partial_values": values,
        }
        if self.current_epoch == 0:
            output["x_lhs"] = x_lhs
            
        self.validation_step_outputs.append(output)

        return output

    # def validation_epoch_end(self, outputs):
    #     """
    #     Used by pytorch_lightning
    #     Accumulates results of all forward validation passes in this epoch

    #     :param outputs: a list of dicts from self.validation_step()
    #     :param batch_idx: which batch this is in the epoch.
    #     :returns: a dict with val_loss, val_accuracy
    #     """
    #     validation_is_real = len(outputs[0]) != 0

    #     if validation_is_real:
    #         self.next_epoch_to_eval = max(
    #             int(1.02 * self.next_epoch_to_eval), self.next_epoch_to_eval + 1
    #         )

    #         loss = torch.stack([x["partial_val_loss"] for x in outputs]).sum()
    #         perplexity = torch.exp(loss)
    #         accuracy = torch.stack([x["partial_val_accuracy"] for x in outputs]).sum()

    #         if self.hparams.save_activations or self.hparams.save_outputs:
    #             if self.current_epoch == 0:
    #                 self._save_inputs(outputs, ds="val")
    #             self._save_activations(outputs, ds="val")

    #         logs = {
    #             "val_loss": loss,
    #             "val_accuracy": accuracy,
    #             "val_perplexity": perplexity,
    #         }
    #         for name, param in self.named_parameters():
    #             # n parameters
    #             n_params = param.numel()
    #             # get the l2 norm of the parameter
    #             logs["paramnorm_" + name] = torch.norm(
    #                 param, 2
    #             ).detach().cpu().numpy() / np.sqrt(n_params)

    #         # train accuracy
    #         device = self.transformer.embedding.weight.device
    #         train_data = self.train_dataset.data.to(device)
    #         training_data = {"text": train_data[:, :-1], "target": train_data[:, 1:]}
    #         with torch.no_grad():
    #             tr_loss, tr_acc, *_ = self._step(training_data, 0)
    #             logs["full_train_loss"] = tr_loss
    #             logs["full_train_acc"] = tr_acc

    #         for k, v in logs.items():
    #             self.log(k, v)
    #     # save a checkpoint if the epoch is a power of 2
    #     if (
    #         self.current_epoch > 0
    #         and int(2 ** (int(np.log(self.current_epoch) / np.log(2))))
    #         == self.current_epoch
    #     ):
    #         self.trainer.save_checkpoint(
    #             os.path.join(
    #                 self.hparams.checkpoint_path,
    #                 "epoch_" + str(self.current_epoch) + ".ckpt",
    #             )
    #         )
    #     if validation_is_real:
    #         return logs
    
    def on_validation_epoch_end(self):
        """
        Used by pytorch_lightning
        Accumulates results of all forward validation passes in this epoch

        :param outputs: a list of dicts from self.validation_step()
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with val_loss, val_accuracy
        """
        # print(f"validation at epoch {self.current_epoch} is about to end")
        
        outputs = self.validation_step_outputs
        
        validation_is_real = len(outputs) != 0

        if validation_is_real:
            self.next_epoch_to_eval = max(
                int(1.02 * self.next_epoch_to_eval), self.next_epoch_to_eval + 1
            )

            loss = torch.stack([x["partial_val_loss"] for x in outputs]).sum()
            perplexity = torch.exp(loss)
            accuracy = torch.stack([x["partial_val_accuracy"] for x in outputs]).sum()

            if self.hparams.save_activations or self.hparams.save_outputs:
                if self.current_epoch == 0:
                    self._save_inputs(outputs, ds="val")
                self._save_activations(outputs, ds="val")

            logs = {
                "val_loss": loss,
                "val_accuracy": accuracy,
                "val_perplexity": perplexity,
            }
            for name, param in self.named_parameters():
                # n parameters
                n_params = param.numel()
                # get the l2 norm of the parameter
                logs["paramnorm_" + name] = torch.norm(
                    param, 2
                ).detach().cpu().numpy() / np.sqrt(n_params)

            # train accuracy
            device = self.transformer.embedding.weight.device
            train_data = self.train_dataset.data.to(device)
            training_data = {"text": train_data[:, :-1], "target": train_data[:, 1:]}
            with torch.no_grad():
                tr_loss, tr_acc, *_ = self._step(training_data, 0)
                logs["full_train_loss"] = tr_loss
                logs["full_train_acc"] = tr_acc

            for k, v in logs.items():
                self.log(k, v)
            
            # print(f"Epoch {self.current_epoch}: Is a real validation step")
        
        # else: print(f"Epoch {self.current_epoch}: Not a real validation step")
        
        # save a checkpoint if the epoch is a power of 2
        if (
            self.current_epoch > 0
            and (
            int(2 ** (int(np.log(self.current_epoch) / np.log(2))))
            == self.current_epoch
            or int(2 ** (int(np.log(self.current_epoch + 1) / np.log(2))))
            == self.current_epoch + 1
            )
        ):
            self.trainer.save_checkpoint(
                os.path.join(
                    self.hparams.checkpoint_path,
                    "epoch_" + str(self.current_epoch) + ".ckpt",
                )
            )
            
        self.validation_step_outputs.clear()
        
        # print(f"validation epoch {self.current_epoch} ends")
        
        if validation_is_real:
            return logs

    def test_step(self, batch, batch_idx):
        """
        Used by pytorch_lightning
        Runs one forward validation pass on one batch

        :param batch: The batch of equations to process
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with val_loss, val_accuracy, probabilities of solutions,
                  attentions, and values
        """

        loss, accuracy, coeff, x_lhs, y_hat_rhs, attentions, values = self._step(
            batch=batch, batch_idx=batch_idx, train=False, reduction="none"
        )
        output = {
            "partial_test_loss": coeff * loss,
            "partial_test_accuracy": coeff * accuracy,
            "y_hat_rhs": y_hat_rhs,
            "partial_attentions": attentions,
            "partial_values": values,
        }
        if self.current_epoch == 0:
            output["x_lhs"] = x_lhs

        return output

    def test_epoch_end(self, outputs):
        """
        Used by pytorch_lightning
        Accumulates results of all forward validation passes in this epoch

        :param outputs: a list of dicts from self.validation_step()
        :param batch_idx: which batch this is in the epoch.
        :returns: a dict with val_loss, val_accuracy
        """
        loss = torch.cat([x["partial_test_loss"] for x in outputs], dim=0)  # .sum()
        # loss = list([x["partial_test_loss"] for x in outputs])  # .sum()
        perplexity = torch.exp(loss)
        accuracy = torch.cat([x["partial_test_accuracy"] for x in outputs], dim=0)

        logs = {
            "test_loss": loss,
            "test_accuracy": accuracy,
            "test_perplexity": perplexity,
        }

        return {"test_loss": loss, "log": logs}

    def forward(self, *args, **kwargs) -> Any:
        """Passes all arguments directly to Tranformer.forward()"""
        return self.transformer(*args, **kwargs)
