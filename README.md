# PKU MIML Course Project: Grokking

<!-- ## Paper

Implementing experiments in the paper [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) by Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra.

### Related research papers
Search via DBLP/scholar
- Towards Understanding Grokking: An Effective Theory of Representation Learning. [[conference](http://papers.nips.cc/paper_files/paper/2022/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html)] [[arxiv](https://doi.org/10.48550/arXiv.2205.10343)]
- Progress measures for grokking via mechanistic interpretability. [[conference](https://openreview.net/forum?id=9XFSbDPmdW)] [[arxiv](https://doi.org/10.48550/arXiv.2301.05217)]
- Why Do You Grok? A Theoretical Analysis on Grokking Modular Addition. [[conference](https://openreview.net/forum?id=ad5I6No9G1)] [[arxiv](https://doi.org/10.48550/arXiv.2407.12332)]
- Grokking as a First Order Phase Transition in Two Layer Networks. [[conference](https://openreview.net/forum?id=3ROGsTX3IR)] [[arxiv](https://doi.org/10.48550/arXiv.2310.03789)] -->

This is the MIML course project by 沙柯岑 (2200010611)，吴于子恒 (2200010878)，岳镝 (2100012961).

## Installation

Our code is based on the official code from OpenAI [https://github.com/openai/grok](https://github.com/openai/grok).
Clone the repository by running
```
git clone git@github.com:tnediserp/MIML-grokking.git
cd ./MIML-grokking
```
Run the following command to install the required libraries.
```bash
pip install -e .
```

## Training
To train the model, run
```bash
python ./scripts/train.py --model [MODEL] --max_steps [MAX_STEPS] --train_data_pct [ALPHA] --dropout [DROPOUT] --optimizer [OPTIM] --max_lr [LR] --weight_decay [WD] --num_operand [K] --A_fraction [LAMBDA] --ckpt_epoch [CKPT]
```

We explain these arguments below:
- `--model`: The model to be trained, options include [Transformer], [LSTM] and [MLP]. Set to [Transformer] by default.
- `--max_step`: The number of steps to train, set to $10^5$ by default.
- `--tran_data_pct`: The training data *percentage*. Should be a float in $(0, 100)$. Set to $50$ by default.
- `--dropout`: The dropout of the model, a value in $(0, 1)$, set to $0.1$ by default.
- `--optimizer`: The optimizer to use, options include [AdamW], [Adam], [SGD] and [RMSProp]. Set to [AdamW] by default. Note: only implemented for transformer.
- `--max_lr`: Learning rate, set to $10^{-3}$ by default.
- `--weight_decay`: Weight decay, set to $0.1$ by default.
- `--num_operand`: $K$-wise addition, set to $K = 2$ by default. Note: only implemented for transformer.
- `--A_fraction`: The non-linearity $\lambda \in (0, 1)$ (refer to Section 4 of our paper), set to $\lambda = 1$ by default.
- `--ckpt_epoch`: Which checkpoint to load. Should be an integer that appears in `./checkpoints`. Set to $0$ by default, which means starting from the beginning.

After training, the logs can be found in `./lighting_logs/version_0/metrics.csv`, and the hyperparameters are stored in `./lighting_logs/version_0/hparams.yaml`. You can also find existing logs in `./lightning_logs` that correspond to our experiments.

## Visualization

To produce the learning curves, please run
```bash
python ./scripts/visualize_metrics.py -i [INPUT_DIR] -o ./output --model [MODEL]
```
- `-i`: The input directory. Specifically, this should be the directory in which your `metrics.csv` is located. For example, if it is in `./lighting_logs/version_0/`, then you should set `INPUT_DIR` to `./lighting_logs/version_0`.
- `--model`: Your model, options include [Transformer], [LSTM] and [MLP]. Set to [Transformer] by default.

After running this command, you can find the accuracy curve in `./output/grokking_curves` and the loss curve in `./output/loss_curves`.

To produce e.g. Figure 2a in our paper, run
```
python ./scripts/draw_alpha_steps.py -i ./lightning_logs/1AdamW_lr0.001_dropout0.1 -o ./output
```
The resulting curve can be found in `./output/Transformer_alpha`.

To produce e.g. Figure 5 in our paper, run 
```
python ./scripts/draw_alpha_acc.py
```
The resulting curves can be found in `./output/different_settings/`.

To produce Figure 11 in our paper, run 
```
python ./scripts/draw_p_and_Afraction.py
```
The resulting curves can be found in `./output/grokking_curves` and  `./output/loss_curves`.

## Files
- `./scripts/train.py`: The main script of training.
- `./scripts/visualize_metrics.py`, `./scripts/draw_alpha_acc.py`, `./scripts/draw_alpha_steps.py`, `./scripts/draw_p_and_Afraction.py`: Scripts for drawing curves.
- `./grok/data.py`: Preparing data.
- `./grok/training.py`: Training.
- `./grok/optimizer.py`: Implementation of customized optimizers.
- `./grok/transformer.py`: Transformer model.
- `./grok/lstm.py`: LSTM model.
- `./grok/mlp.py`: MLP model.