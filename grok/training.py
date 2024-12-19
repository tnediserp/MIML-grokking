#!/usr/bin/env python

import argparse
import copy
import json
import logging
import math
import os
import sys
import pickle
from argparse import ArgumentParser, Namespace
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

import grok.metrics as metrics
from grok.data import (
    DEFAULT_DATA_DIR,
    EOS_TOKEN,
    VALID_OPERATORS,
    ArithmeticDataset,
    ArithmeticIterator,
)
from grok.transformer import TrainableTransformer
from grok.lstm import TrainableLSTM
from grok.mlp import TrainableMLP
from grok.measure import get_sharpness

DEFAULT_LOG_DIR = "logs"




def train(hparams: Namespace) -> None:
    """
    This is the main trainer_method. This sets up and runs experiment with
    the defined hyperparameters

    :param hparams: An argparse.Namespace with all of the relevant hyperparameters
    """

    # Process the args
    if hparams.logdir is None:
        hparams.logdir = os.environ.get("LOGDIR", ".")
    hparams.logdir = os.path.abspath(hparams.logdir)

    # Make sure d_model, heads, and d_key are compatible
    assert (
        hparams.d_model % hparams.n_heads == 0
    ), "n_heads=%s does not evenly divide d_model=%s" % (
        hparams.n_heads,
        hparams.d_model,
    )
    hparams.d_key = hparams.d_model / hparams.n_heads

    # Set up the RNGs for repeatability
    if hparams.random_seed != -1:
        torch.manual_seed(hparams.random_seed)
        torch.cuda.manual_seed(hparams.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    checkpoint_path = hparams.logdir + "/checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    hparams.checkpoint_path = checkpoint_path
    
    
    # if hparams.load_ckpt == True:
    #     ckpt_file = checkpoint_path + f"/epoch_{hparams.ckpt_epoch}.ckpt"
        
    #     if os.path.exists(ckpt_file) == False:
    #         print(f"checkpoint {ckpt_file}: No such file.")
    #     else:
    #         print(f"Loading checkpoint {hparams.ckpt_epoch}")
    #         checkpoint = torch.load(ckpt_file)
    #         model.load_state_dict(checkpoint["state_dict"])


    logger = CSVLogger(hparams.logdir)

    # checkpointer = ModelCheckpoint(
    #     filepath=checkpoint_path,
    #     monitor="save_ckpt",
    #     mode="max",
    #     save_top_k=len(hparams.ckpt_epochs),
    #     verbose=False,
    # )

    trainer_args = {
        "max_steps": hparams.max_steps,
        "min_steps": hparams.max_steps,
        "max_epochs": int(1e8),
        "val_check_interval": 1,
        "profiler": False,
        # "checkpoint_callback": checkpointer,
        "logger": logger,
        "log_every_n_steps": 1,
        # "flush_logs_every_n_steps": 1000, ### "flush_logs_every_n_steps" is outdated
    }
    
    if torch.cuda.is_available() and hparams.gpu >= 0:
        trainer_args["devices"] = [hparams.gpu]  # List of GPU indices (e.g., [0])
        trainer_args["accelerator"] = "gpu"     # Use GPU as accelerator

    trainer = Trainer(**trainer_args)

    ckpt_file = os.path.join(hparams.checkpoint_path, "epoch_" + str(hparams.ckpt_epoch) + ".ckpt")

    
    if hparams.ckpt_epoch == "0":
        print("No checkpoint to load, starting from the beginning.")
        ckpt_file = None
        
    elif os.path.exists(ckpt_file) == False:
        print(f"checkpoint {ckpt_file}: No such file. Starting from the beginning.")
        ckpt_file = None
    
    else: 
        print(f"Loading checkpoint {hparams.ckpt_epoch}")
        # trainer_args["resume_from_checkpoint"] = ckpt_file
        
    # Create the model
    if hparams.model == "Transformer":
        model = TrainableTransformer(hparams).float()
        print("Using model Transformer")
    elif hparams.model == "LSTM":
        model = TrainableLSTM(hparams).float()
        print("Using model LSTM")
    elif hparams.model == "MLP":
        model = TrainableMLP(hparams).float()
    else:
        print(f"Model {hparams.model} not implemented. Please choose from the following: [Transformer] [LSTM] [MLP]")
        assert(False)
        
    torch.save(model, os.path.join(checkpoint_path, "init.pt"))

    trainer.fit(model=model, ckpt_path=ckpt_file)  # type: ignore
    """
    margin = np.percentile(model.margin.detach().cpu().numpy(), 5)
    device = transformer.embedding.weight.device
    measures, bounds = metrics.calculate(
        transformer,
        transformer_init.to(device),
        device,
        dataset_size,
        margin,
        input_dim=hparams.d_model,
    )

    measures_file = os.path.join(logger.log_dir, "measures.json")
    bounds_file = os.path.join(logger.log_dir, "bounds.json")
    with open(measures_file, "w") as fh:
        json.dump(measures, fh)
    with open(bounds_file, "w") as fh:
        json.dump(bounds, fh)
    """
    return hparams.logdir


def compute_sharpness(hparams: Namespace, ckpts) -> None:
    """
    This is the compute_sharpness method. This loads a series of checkpoints in
    the defined hyperparameters

    :param hparams: An argparse.Namespace with all of the relevant hyperparameters
    """

    # Process the args
    if hparams.logdir is None:
        hparams.logdir = os.environ.get("LOGDIR", ".")
    hparams.logdir = os.path.abspath(hparams.logdir)

    # Make sure d_model, heads, and d_key are compatible
    assert (
        hparams.d_model % hparams.n_heads == 0
    ), "n_heads=%s does not evenly divide d_model=%s" % (
        hparams.n_heads,
        hparams.d_model,
    )
    hparams.d_key = hparams.d_model / hparams.n_heads

    # Set up the RNGs for repeatability
    if hparams.random_seed != -1:
        torch.manual_seed(hparams.random_seed)
        torch.cuda.manual_seed(hparams.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    checkpoint_path = hparams.logdir + "/checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    hparams.checkpoint_path = checkpoint_path

    # Create the model
    model = TrainableTransformer(hparams).float()

    torch.save(model, os.path.join(checkpoint_path, "init.pt"))

    logger = CSVLogger(hparams.logdir)


    trainer_args = {
        "max_steps": hparams.max_steps,
        "min_steps": hparams.max_steps,
        "max_epochs": int(1e8),
        "val_check_interval": 1,
        "profiler": False,
        # "checkpoint_callback": checkpointer,
        "logger": logger,
        "log_every_n_steps": 1,
        "flush_logs_every_n_steps": 1000,
    }
    if torch.cuda.is_available() and hparams.gpu >= 0:
        trainer_args["gpus"] = [hparams.gpu]

    trainer = Trainer(**trainer_args)

    for ckpt in ckpts:
        print(f"Loading checkpoint {ckpt}")
        # model = torch.load(ckpt)
        # model.load_state_dict(torch.load(ckpt))

        checkpoint = torch.load(ckpt)
        # print(dir(checkpoint), type(checkpoint), "Ckpt")
        # for k, v in checkpoint.items():
        #     print(k)
        # print(checkpoint["hyper_parameters"])

        hps = checkpoint["hyper_parameters"]
        hps = argparse.Namespace(**hps)
        model = TrainableTransformer(hps).float()
        model.load_state_dict(checkpoint["state_dict"])

        phi = get_sharpness(model.train_dataloader(), model)
        results = {}
        results[ckpt] = phi
        pickle.dump(results, open(f"results/results_SD-{i}.pkl", "wb"))


def add_args(parser=None) -> Namespace:
    """
    Parses the command line arguments

    :returns: an argparse.Namespace with all of the needed arguments
    """
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=-1)
    parser.add_argument("--gpu", type=int, default=0)
    # parser.add_argument("--load_ckpt", dest="load_ckpt", action="store_true")
    # parser.set_defaults(load_ckpt=False)
    parser.add_argument("--ckpt_epoch", type=str, default="0")
    parser.add_argument("--model", type=str, default="Transformer")
    # parser.add_argument("--checkpoint_period", type=int, default=1)
    
    model_name = parser.parse_args().model
    if model_name == "Transformer":
        parser = TrainableTransformer.add_model_specific_args(parser) 
    elif model_name == "LSTM":
        parser = TrainableLSTM.add_model_specific_args(parser) 
    elif model_name == "MLP":
        parser = TrainableMLP.add_model_specific_args(parser) 
    else:
        print(f"Model {model_name} not implemented.")
        assert(False)
    return parser
