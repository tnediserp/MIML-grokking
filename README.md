# PKU MIML Course Project: Grokking

## News (updated)
- Subtasks 1, 2 finished.
- Grokking is produced for Transformer (converges in $5 \times 10^4$ steps). The grokking curve looks nice.
- Grokking is produced for LSTM (converges in $10^5$ steps with default hparams. If you want it to converge faster, increase the weight decay, e.g. to $1.0$.)
- Grokking is produced for MLP (It converges in $\approx 5 \times 10^4$ steps, i.e., $10^4$ epochs.)
- Please pay attention to the file structure. Each model is now implemented in a separate file. E.g., MLP is implemented in `./grok/mlp.py`.

## Missions (updated)
When you are working on a file or modified some codes, please inform us in the WeChat group and update the codes in time.
- 沙柯岑
  - Do subtask 4.
  - Option 1: Choose a small prime $p$. Draw the $\alpha$-(generalization step) curve for small $K$'s (e.g., $K = 2, 3, 4$). Plot them in the same figure.
  - Option 2: Try large $K$'s by using only a (randomly chosen) subset of all $p^K$ equations.
- 吴于子恒
  - Do subtask 3.
- 岳镝
  - Do subtask 3.

## Paper

Implementing experiments in the paper [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) by Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra.

### Related research papers
Search via DBLP/scholar
- Towards Understanding Grokking: An Effective Theory of Representation Learning. [[conference](http://papers.nips.cc/paper_files/paper/2022/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html)] [[arxiv](https://doi.org/10.48550/arXiv.2205.10343)]
- Progress measures for grokking via mechanistic interpretability. [[conference](https://openreview.net/forum?id=9XFSbDPmdW)] [[arxiv](https://doi.org/10.48550/arXiv.2301.05217)]
- Why Do You Grok? A Theoretical Analysis on Grokking Modular Addition. [[conference](https://openreview.net/forum?id=ad5I6No9G1)] [[arxiv](https://doi.org/10.48550/arXiv.2407.12332)]
- Grokking as a First Order Phase Transition in Two Layer Networks. [[conference](https://openreview.net/forum?id=3ROGsTX3IR)] [[arxiv](https://doi.org/10.48550/arXiv.2310.03789)]

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

To examine the influence of the training data percentage $\alpha$, run 
```
python ./scripts/draw_alpha_steps.py -i ./lightning_logs/Transformer_alpha -o ./output
```
The $\alpha$-generalization step curve can be found in `./output/Transformer_alpha`.