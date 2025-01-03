import csv
import os
from argparse import ArgumentParser
import blobfile as bf
import grok
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import torch
import yaml
from tqdm import tqdm


def load_expt_metrics(
    expt_dir
):
    # load the hparams for this experiment
    with open(f"{expt_dir}/hparams.yaml", "r") as fh:
        hparams_dict = yaml.safe_load(fh)

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


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    

input_p = "./lightning_logs/Different_p"
input_Afraction = "./lightning_logs/Different_A_fraction"

p_list = [23, 97, 151, 199]
p_dirs = ["./lightning_logs/Different_p/SGD_p=23",
            "./lightning_logs/Different_p/SGD_p=97",
            "./lightning_logs/Different_p/SGD_p=151",
            "./lightning_logs/Different_p/SGD_p=199"]

Afraction_dirs = [os.path.join(input_Afraction, folder) for folder in os.listdir(input_Afraction)
                      if os.path.isdir(os.path.join(input_Afraction, folder))]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

try:
    ############################################################################## acc curves for different p
    ############################################################################
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title("Accuracy curves of different $p$")
    
    plt.xscale('linear')
    plt.xlabel('step')
    plt.yscale('linear')
    plt.ylabel('Acc')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ymin = -1
    ymax = 101
    ax.axis(ymin=ymin, ymax=ymax)

    for i, expt_dir in enumerate(tqdm(p_dirs, unit="expt")):
        expt_metrics = load_expt_metrics(expt_dir)
        train_pct = expt_metrics["hparams"]["train_data_pct"]
        
        X_train = expt_metrics["train"]["step"]
        X_val = expt_metrics["val"]["step"]
        Y_train = expt_metrics["train"]["train_accuracy"]
        Y_val = expt_metrics["val"]["val_accuracy"]
        
        color = colors[i % len(colors)]
        
        ax.plot(X_train, Y_train, linestyle="--", color=color, label=f"p = {p_list[i]}, train")
        ax.plot(X_val, Y_val, linestyle="-", color=color, label=f"p = {p_list[i]}, val")
        ax.legend()
    
    image_file = "./output/grokking_curves/different_p_acc.png"

    d = os.path.split(image_file)[0]
    os.makedirs(d, exist_ok=True)
    print(f"Writing {image_file}")
    fig.savefig(image_file)
    plt.close(fig)
    
    
    ############################################################################## Loss curves for different p
    ############################################################################
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title("Loss curves of different $p$")
    
    plt.xscale('linear')
    plt.xlabel('step')
    plt.yscale('linear')
    plt.ylabel('loss')

    for i, expt_dir in enumerate(tqdm(p_dirs, unit="expt")):
        expt_metrics = load_expt_metrics(expt_dir)
        
        X_train = expt_metrics["train"]["step"]
        X_val = expt_metrics["val"]["step"]
        Y_train = expt_metrics["train"]["train_loss"]
        Y_val = expt_metrics["val"]["val_loss"]
        
        color = colors[i % len(colors)]
        
        ax.plot(X_train, Y_train, linestyle="--", color=color, label=f"p = {p_list[i]}, train")
        ax.plot(X_val, Y_val, linestyle="-", color=color, label=f"p = {p_list[i]}, val")
        ax.legend()
    
    image_file = "./output/loss_curves/different_p_loss.png"

    d = os.path.split(image_file)[0]
    os.makedirs(d, exist_ok=True)
    print(f"Writing {image_file}")
    fig.savefig(image_file)
    plt.close(fig)
    
    ############################################################################## Acc curves for different A_fraction
    ############################################################################
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title("Accuracy curves of different $\lambda$")
    
    plt.xscale('linear')
    plt.xlabel('step')
    plt.yscale('linear')
    plt.ylabel('Acc')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ymin = -1
    ymax = 101
    ax.axis(ymin=ymin, ymax=ymax)

    for i, expt_dir in enumerate(tqdm(Afraction_dirs, unit="expt")):
        expt_metrics = load_expt_metrics(expt_dir)
        
        A_fraction = expt_metrics["hparams"]["A_fraction"]
        
        X_train = expt_metrics["train"]["step"]
        X_val = expt_metrics["val"]["step"]
        Y_train = expt_metrics["train"]["train_accuracy"]
        Y_val = expt_metrics["val"]["val_accuracy"]
        
        color = colors[i % len(colors)]
        
        ax.plot(X_train, Y_train, linestyle="--", color=color, label=f"$\lambda$ = {A_fraction}, train")
        ax.plot(X_val, Y_val, linestyle="-", color=color, label=f"$\lambda$ = {A_fraction}, val")
        ax.legend()
    
    image_file = "./output/grokking_curves/different_Afraction_acc.png"

    d = os.path.split(image_file)[0]
    os.makedirs(d, exist_ok=True)
    print(f"Writing {image_file}")
    fig.savefig(image_file)
    plt.close(fig)
    
    
    ############################################################################## Loss curves for different A_fractions
    ############################################################################
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title("Loss curves of different $\lambda$")
    
    plt.xscale('linear')
    plt.xlabel('step')
    plt.yscale('linear')
    plt.ylabel('loss')
    

    for i, expt_dir in enumerate(tqdm(Afraction_dirs, unit="expt")):
        expt_metrics = load_expt_metrics(expt_dir)
        
        A_fraction = expt_metrics["hparams"]["A_fraction"]
        
        X_train = expt_metrics["train"]["step"]
        X_val = expt_metrics["val"]["step"]
        Y_train = expt_metrics["train"]["train_loss"]
        Y_val = expt_metrics["val"]["val_loss"]
        
        color = colors[i % len(colors)]
        
        ax.plot(X_train, Y_train, linestyle="--", color=color, label=f"$\lambda$ = {A_fraction}, train")
        ax.plot(X_val, Y_val, linestyle="-", color=color, label=f"$\lambda$ = {A_fraction}, val")
        ax.legend()
    
    image_file = "./output/loss_curves/different_Afraction_loss.png"

    d = os.path.split(image_file)[0]
    os.makedirs(d, exist_ok=True)
    print(f"Writing {image_file}")
    fig.savefig(image_file)
    plt.close(fig)
    

except BaseException as e:
    print(f"failed: {e}")