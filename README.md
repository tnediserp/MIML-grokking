# PKU MIML Course Project: Grokking

## Paper

Implementing experiments in the paper [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) by Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra.

## Installation and Training

```bash
pip install -e .
python ./scripts/train.py
```

During the training process, the models at epochs $2^n$ and $2^n-1$ will be saved as checkpoints in directory `checkpoints/`. (Please check the naming convention of the checkpoints.)

To start from a given checkpoint $i$, run
```bash
python ./scripts/train.py  --ckpt_epoch i
```

## Parameters
total size of data: $97^2 = 9409$, training set: $4704$, validation set $4705$, batch size $512$, learning rate $10^{-3}$...

## Experiment
The training loss converges after about $300$ epochs, and the training accuracy remains $\approx 100\%$.

The validation accuracy quickly rises to $\approx 30\%$ and remains. We have to run far more epochs (e.g. $\approx 10^5$) to see the grokking phenomenon.

See `/ligntning_logs/version_[i]/metrics.csv` for the detailed logs.