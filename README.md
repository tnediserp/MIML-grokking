# PKU MIML Course Project: Grokking

## News (updated)
- Grokking is produced for MLP (It converges in $\approx 5 \times 10^4$ steps, i.e., $10^4$ epochs.)
- Please pay attention to the file structure. Each model is now implemented in a separate file. E.g., MLP is implemented in `./grok/mlp.py`.

## Missions (updated)
When you are working on a file or modified some codes, please inform us in the WeChat group and update the codes in time.
- 沙柯岑
  - Work on the data preparing code.
  - Try to generalize it to the $K$-wise addition, (or figure out an appropriate way to generate training/validation data for $K$-wise addition).
  - Tokens: Perhaps `./grok/data.py`, and the `prepare_data()` functions in every implemented model.
- 吴于子恒
  - Do task 3 on the MLP model, i.e., do experiments on optimizers, training data fractions, dropouts, weight decay... (Hopefully this could be fast, because the training of MLP can be done very fast.)
  - Tokens: Perhaps `./grok/mlp.py`, especially some hyper-parameters defined in the `add_model_specific_args()` function.
- 岳镝
  - Try to produce grokking phenomenon for Transformer and LSTM
  - Tokens: `./grok/transformer.py` and `./grok/lstm.py`.

## Paper

Implementing experiments in the paper [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) by Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra.

## Installation and Training

```bash
pip install -e .
python ./scripts/train.py --model [MODEL]
```

During the training process, the models at epochs $2^n$ and $2^n-1$ will be saved as checkpoints in directory `checkpoints/`. (Please check the naming convention of the checkpoints.)

To start from a given checkpoint $i$, run
```bash
python ./scripts/train.py  --ckpt_epoch [i]
```

For example, to train with Transformer starting from epoch $256$, please run
```
python ./scripts/train.py --model Transformer --ckpt_epoch 256
```

## Parameters
total size of data: $97^2 = 9409$, training set: $4704$, validation set $4705$, batch size $512$, learning rate $10^{-3}$...

## Experiment
The training loss converges after about $300$ epochs, and the training accuracy remains $\approx 100\%$.

The validation accuracy quickly rises to $\approx 30\%$ and remains. We have to run far more epochs (e.g. $\approx 10^5$) to see the grokking phenomenon.

See `./ligntning_logs/version_[i]/metrics.csv` for the detailed logs.

## Visualization
```
python ./scripts/visualize_metrics.py -i ./lightning_logs/addition_50%_[MODEL]/ -o ./output --model [MODEL]
```
When you use this script to create the grokking curve, please first make sure that you put all logs in the same directory.
For example, if you have your logs in `./lightning_logs/addition_50%_MLP`, then you should run 
```
python ./scripts/visualize_metrics.py -i ./lightning_logs/addition_50%_MLP -o ./output --model MLP
```
It is OK to have sub-dirs such as version_0, version_1 in that directory, and the important thing is that they should be **consistent** (with the same training fraction, model..., and with no overlap in epochs/steps)

You will find the output curves in directory `./output/grokking_curves`.