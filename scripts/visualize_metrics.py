#!/usr/bin/env python
# coding: utf-8
################################################################################
# When you use this script to create the grokking curve, please first make sure that you put all logs in the same directory.
# For example, if you have your logs in ./lightning_logs/addition_50%_Transformer, then you should run 
# python .\scripts\visualize_metrics.py -i ./lightning_logs/addition_50%_Transformer -o ./output --model Transformer
# It is OK to have sub-dirs such as version_0, version_1 in that directory, and the important thing is that they should be **consistent** (with the same training fraction, model..., and with no overlap in epochs/steps)
################################################################################
import csv
import json
import logging
import os
import subprocess
from argparse import ArgumentParser
from copy import deepcopy
from glob import glob
from pprint import pprint

import blobfile as bf
import grok
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import torch
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

# take args: input_dir output_dir
parser = ArgumentParser()
parser.add_argument(
    "-i",
    "--input_dir",
    type=str,
    required=True,
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    required=True,
)
# parser.add_argument(
#     "--model",
#     type=str,
#     required=True
# )
parser = grok.training.add_args(parser)
args = parser.parse_args()
print(args, flush=True)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def load_expt_metrics(
    expt_dir,
    args,
):
    """load the metrics for one experiment"""
    args = deepcopy(args)

    # load the hparams for this experiment
    with open(f"{expt_dir}/hparams.yaml", "r") as fh:
        hparams_dict = yaml.safe_load(fh)

    for k, v in hparams_dict.items():
        setattr(args, k, v)

    # load the summarized validation and training data for every epoch
    val_data = {
        "step": [],
        "epoch": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    train_data = {
        "step": [],
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "learning_rate": [],
    }

    with open(f"{expt_dir}/metrics.csv", "r") as fh:
        print(f"open {expt_dir}/metrics.csv")
        for row in csv.DictReader(fh):
            if row["train_loss"] != "":
                for k in train_data:
                    if k in ["step", "epoch"]:
                        v = int(row[k])
                    else:
                        v = float(row[k])
                    train_data[k].append(v)
            else:
                for k in val_data:
                    if k in ["step", "epoch"]:
                        v = int(row[k])
                    else:
                        v = float(row[k])
                    val_data[k].append(v)

    return {
        "hparams": hparams_dict,
        "train": train_data,
        "val": val_data,
        # "raw": raw_data,
    }


def load_run_metrics(
    run_dir,
    args=args,
):
    """load all the metrics for a collection of experiments with the same architecture
    across various amounts of training data"""
    metric_data = {}
    from os import walk

    # expt_dirs = [version_0, version_1...]
    _, expt_dirs, _ = next(os.walk(run_dir))
    for expt_dir in tqdm(expt_dirs, unit="expt"):
        try:
            expt_data = load_expt_metrics(f"{run_dir}/{expt_dir}", args)
            train_data_pct = expt_data["hparams"]["train_data_pct"]
            
            if train_data_pct not in metric_data:
                metric_data[train_data_pct] = expt_data
            else: # this version_i might come from a checkpoint. In this case, append data.
                for phase in ["train", "val"]:
                    for key, value in expt_data[phase].items():
                        metric_data[train_data_pct][phase][key].extend(value)
        
        except FileNotFoundError:
            print(f"{run_dir}/{expt_dir}: File not found")
            pass
    return metric_data


def add_metric_graph(
    fig,
    ax,
    arch,
    metric,
    metric_data,
    scales,
    cmap="viridis",
    by="step",  # step or epoch
    max_increment=0,
):
    ax.set_title(metric)
    ax.set_xscale(scales["x"])
    ax.set_yscale(scales["y"])
    ax.set_xlabel(by)

    if "accuracy" in metric:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ymin = 1e-16
        ymax = 101
        ax.axis(ymin=ymin, ymax=ymax)
    if "loss" in metric:
        ymin = 1e-16
        ymax = 15
        ax.axis(ymin=ymin, ymax=ymax)

    total_plots = 0
    logger.debug(f"processing {metric}")
    plots = []
    T = list(sorted(metric_data.keys()))
    T_max = int(T[-1])
    T_min = int(T[0])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=T[0], vmax=T[-1]))
    colors = sm.to_rgba(T)
    for i, t in enumerate(T):
        # print(f"i = {i}")
        if "val" in metric:
            this_data = metric_data[t]["val"]
        else:
            this_data = metric_data[t]["train"]

        X = this_data[by]
        Y = this_data[metric]
        if max_increment > 0:
            X = [x for x in X if x <= max_increment]
            Y = Y[: len(X)]

        if len(X) != len(Y):
            logger.warning(f"Mismatched data: {metric} at t={t}")
            continue
        if not Y:
            logger.warning(f"No data for {metric}i at t={t}")
            continue

        label = arch + f" t={t}"

        if "accuracy" in metric:
            label += " (max = %.2f)" % max(Y)
        elif "loss" in metric:
            label += " (min = %.2f)" % min(Y)
        total_plots += 1
        ax.plot(X, Y, label=label, color=colors[i])
    if T_max - T_min <= 10:
        ax.legend()
    else:
        fig.colorbar(
            sm,
            ax=ax,
            label="% training data",
            ticks=range(T_min, T_max + 1, int((T_max - T_min) / 5)),
        )


def add_max_accuracy_graph(
    ax,
    arch,
    metric,
    metric_data,
    scales,
    by="step",
    max_increment=0,
):
    ax.set_title(f"max {metric}")
    ax.set_xlabel("% of total data trained on")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    xmin = 0
    xmax = 100
    ymin = 1e-16
    ymax = 101
    ax.axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    ax.set_xscale(scales["x"])
    ax.set_yscale(scales["y"])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    T = list(sorted(metric_data.keys()))
    T_max = int(T[-1])
    T_min = int(T[0])
    Y = []
    for i, t in enumerate(T):
        if "val" in metric:
            this_data = metric_data[t]["val"]
        else:
            this_data = metric_data[t]["train"]
        X = this_data[by]
        if max_increment > 0:
            X = [x for x in X if x <= max_increment]
            max_idx = len(X)
        else:
            max_idx = -1
        try:
            Y.append(max(this_data[metric][:max_idx]))
        except ValueError:
            Y.append(np.nan)

    ax.set_xticks(np.arange(0, 100, 5))
    label = f"max {metric} {arch}"
    ax.plot(T, Y, label=label)

def create_grokking_curves(
    metric_data,
    operation,
    train_percent=50,
    model_name="Transformer",
    most_interesting_only=False,
    image_dir=args.output_dir,
    by="step",
    max_increment=0,
    cmap="viridis"
):
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.title(f"Modular {operation} (training on {train_percent}% of data with {model_name})")
    
    plt.xscale('log')
    plt.xlabel(by)
    plt.yscale('linear')
    plt.ylabel('Acc')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ymin = 1e-16
    ymax = 101
    ax.axis(ymin=ymin, ymax=ymax)
    
    T = list(sorted(metric_data.keys()))
    T_max = int(T[-1])
    T_min = int(T[0])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=T[0], vmax=T[-1]))
    
    X_train = metric_data[train_percent]["train"][by]
    X_val = metric_data[train_percent]["val"][by]
    Y_train = metric_data[train_percent]["train"]["train_accuracy"]
    Y_val = metric_data[train_percent]["val"]["val_accuracy"]

    if max_increment > 0:
        X_train = [x for x in X_train if x <= max_increment]
        Y_train = Y_train[: len(X_train)]
        
        X_val = [x for x in X_val if x <= max_increment]
        Y_val = Y_val[: len(X_val)]

    if len(X_train) != len(Y_train) or len(X_val) != len(Y_val):
        logger.warning(f"Mismatched train or val data")

    if not Y_train or not Y_val:
        logger.warning(f"No train or val data")
    
    ax.plot(X_train, Y_train, label="train", color="red")
    ax.plot(X_val, Y_val, label="val", color="blue")
    ax.legend()
    
    img_file = f"{image_dir}/grokking_curves/{operation}_{train_percent}%_{model_name}_{by}"
    if most_interesting_only:
        img_file += "_most_interesting"
    img_file += ".png"
    d = os.path.split(img_file)[0]
    os.makedirs(d, exist_ok=True)
    print(f"Writing {img_file}")
    fig.savefig(img_file)
    plt.close(fig)

def create_loss_curves(
    metric_data,
    arch,
    operation,
    # epochs,
    most_interesting_only=False,
    image_dir=args.output_dir,
    by="step",
    max_increment=0,
    cmap="viridis",
):
    scales = {
        "x": "log",
        "y": "linear",
    }

    ncols = 2
    nrows = 3
    fig_width = ncols * 8
    fig_height = nrows * 5
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))

    add_metric_graph(
        fig,
        axs[0, 0],
        arch,
        "val_loss",
        metric_data,
        scales,
        cmap,
        by,
        max_increment=max_increment,
    )
    add_metric_graph(
        fig,
        axs[0, 1],
        arch,
        "val_accuracy",
        metric_data,
        scales,
        cmap,
        by,
        max_increment=max_increment,
    )
    add_metric_graph(
        fig,
        axs[0, 1],
        arch,
        "train_accuracy",
        metric_data,
        scales,
        cmap,
        by,
        max_increment=max_increment,
    )
    add_metric_graph(
        fig,
        axs[1, 0],
        arch,
        "train_loss",
        metric_data,
        scales,
        cmap,
        by,
        max_increment=max_increment,
    )
    add_metric_graph(
        fig,
        axs[1, 1],
        arch,
        "train_accuracy",
        metric_data,
        scales,
        cmap,
        by,
        max_increment=max_increment,
    )
    add_metric_graph(
        fig,
        axs[1, 1],
        arch,
        "val_accuracy",
        metric_data,
        scales,
        cmap,
        by,
        max_increment=max_increment,
    )
    add_metric_graph(
        fig,
        axs[2, 0],
        arch,
        "learning_rate",
        metric_data,
        scales,
        cmap,
        by,
        max_increment=max_increment,
    )
    fig.suptitle(f"{operation} {arch} {max_increment:06d} {by}s")
    fig.tight_layout()

    img_file = f"{image_dir}/loss_curves/{operation}_loss_curves_{arch}__upto_{max_increment:010d}_{by}"
    if most_interesting_only:
        img_file += "_most_interesting"
    img_file += ".png"
    d = os.path.split(img_file)[0]
    os.makedirs(d, exist_ok=True)
    print(f"Writing {img_file}")
    fig.savefig(img_file)
    plt.close(fig)


def create_max_accuracy_curves(
    metric_data,
    arch,
    operation,
    by="step",
    max_increment=0,
    image_dir=args.output_dir,
):
    scales = {
        "x": "linear",
        "y": "linear",
    }

    ncols = 1
    nrows = 2
    fig_width = ncols * 8
    fig_height = nrows * 5
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))

    add_max_accuracy_graph(
        axs[0],
        arch,
        "val_accuracy",
        metric_data,
        scales,
        by=by,
        max_increment=max_increment,
    )
    axs[0].legend()
    add_max_accuracy_graph(
        axs[1],
        arch,
        "train_accuracy",
        metric_data,
        scales,
        by=by,
        max_increment=max_increment,
    )
    axs[1].legend()
    fig.suptitle(f"{operation} {arch} {max_increment:06d} {by}s")
    fig.tight_layout()

    img_file = f"{image_dir}/max_accuracy/{operation}_max_accuracy_{arch}_upto_{max_increment:010d}_{by}.png"
    d = os.path.split(img_file)[0]
    os.makedirs(d, exist_ok=True)
    print(f"Writing {img_file}")
    fig.savefig(img_file)
    plt.close(fig)


def create_tsne_graphs(
    operation,
    expt,
    run_dir,
    image_dir=args.output_dir,
):

    saved_pt_dir = f"{run_dir}/activations"
    saved_pts = []

    loss_ts = []
    accuracy_ts = []
    epochs_ts = []
    print(f'glob = {saved_pt_dir + "/activations_*.pt"}')
    files = sorted(glob.glob(saved_pt_dir + "/activations_*.pt"))
    print(f"files = {files}")

    for file in files:
        print(f"Loading {file}")
        saved_pt = torch.load(file)
        saved_pts.append(saved_pt)
        loss_ts.append(saved_pt["val_loss"].mean(dim=-1))
        accuracy_ts.append(saved_pt["val_accuracy"])
        epochs_ts.append(saved_pt["epochs"].squeeze())

    loss_t = torch.cat(loss_ts, dim=0).T.detach()
    accuracy_t = torch.cat(accuracy_ts, dim=0).T.detach()
    epochs_t = torch.cat(epochs_ts, dim=0).detach()
    print(loss_t.shape)
    print(accuracy_t.shape)
    print(epochs_t.shape)
    ######
    a = 0
    num_eqs = len(loss_t)
    b = a + num_eqs

    print("Doing T-SNE..")
    loss_tsne = TSNE(n_components=2, init="pca").fit_transform(loss_t)
    print("...done T-SNE.")

    ncols = 1
    nrows = 1
    fig_width = ncols * 8
    fig_height = nrows * 5
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))

    axs.scatter(loss_tsne[:, 0], loss_tsne[:, 1])

    img_file = f"{image_dir}/tsne/{operation}_{expt}.png"
    d = os.path.split(img_file)[0]
    os.makedirs(d, exist_ok=True)
    print(f"Writing {img_file}")
    fig.savefig(img_file)
    plt.close(fig)


def get_arch(metric_data):
    k = list(metric_data.keys())[0]
    hparams = metric_data[k]["hparams"]
    arch = f'L-{hparams["n_layers"]}_H-{hparams["n_heads"]}_D-{hparams["d_model"]}_B-{hparams["batchsize"]}_S-{hparams["random_seed"]}_DR-{hparams["dropout"]}'
    return arch


def get_operation(metric_data):
    k = list(metric_data.keys())[0]
    hparams = metric_data[k]["hparams"]
    operator = hparams["math_operator"]
    operand_length = hparams["operand_length"]
    _, operation = grok.data.ArithmeticDataset.get_file_path(operator, operand_length)
    return operation


def get_max_epochs(metric_data):
    k = list(metric_data.keys())[0]
    hparams = metric_data[k]["hparams"]
    return hparams["max_epochs"]

def get_train_percentage(metric_data):
    k = list(metric_data.keys())[0]
    hparams = metric_data[k]["hparams"]
    return hparams["train_data_pct"]


rundir = args.input_dir
model_name = args.model

try:
    metric_data = load_run_metrics(rundir, args)
    print("load metric data: success")
    
    arch = get_arch(metric_data)
    operation = get_operation(metric_data)
    print(f"operation: {operation}")
    
    max_epochs = get_max_epochs(metric_data)
    print(f"max epoch = {max_epochs}")
    
    train_pct = get_train_percentage(metric_data)
    print(f"training data percentage = {train_pct}%")
    
    if not (model_name == "Transformer" or model_name == "LSTM"):
        print("model name = [Transformer] [LSTM] [MLP]")
        assert(False)
        
    print(f"model name = {model_name}")
    
    create_grokking_curves(metric_data, operation, train_pct, model_name=model_name, by="step")

    # for by in ["step", "epoch"]:
    #     create_loss_curves(metric_data, arch, operation, by=by)

    # by = "epoch"
    # last_i = -1
    # for i in sorted(list(set(2 ** (np.arange(167) / 10)))):
    #     if i > max_epochs:
    #         break
    #     i = int(round(i))
    #     create_max_accuracy_curves(
    #         metric_data,
    #         arch,
    #         operation,
    #         by=by,
    #         max_increment=i,
    #     )

    # # make a video
    # in_files = os.path.join(
    #     args.output_dir,
    #     "max_accuracy",
    #     f"{operation}_max_accuracy_{arch}_upto_%*.png",
    # )
    # out_file = os.path.join(args.output_dir, f"{operation}_{arch}_max_accuracy.mp4")
    # cmd = [
    #     "ffmpeg",
    #     "-y",
    #     "-r",
    #     "16",
    #     "-i",
    #     in_files,
    #     "-vcodec",
    #     "libx264",
    #     "-crf",
    #     "25",
    #     "-pix_fmt",
    #     "yuv420p",
    #     out_file,
    # ]
    # subprocess.check_call(cmd)

except BaseException as e:
    print(f"{rundir} failed: {e}")
