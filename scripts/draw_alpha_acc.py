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


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
def load_best_acc(expt_dir):
    """
    read files {expt_dir}/hparams.yaml and {expt_dir}/metrics.csv, return (alpha, a_t, a_v),
    where a_t is the highest training accuracy, 
    and a_v is the highest validation accuracy.
    """
    with open(f"{expt_dir}/hparams.yaml", "r") as fh:
        hparams_dict = yaml.safe_load(fh)
    
    train_data_pct = hparams_dict["train_data_pct"] # train data pct
    
    best_train_acc = 0
    best_val_acc = 0
    
    with open(f"{expt_dir}/metrics.csv", "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["train_accuracy"] != "": # training stage
                best_train_acc = max(best_train_acc, float(row["train_accuracy"]))
        
            if row["val_accuracy"] != "": # val stage
                best_val_acc = max(best_val_acc, float(row["val_accuracy"]))
        
        
    return train_data_pct, best_train_acc, best_val_acc

def load_all_metrics(run_dir):
    _, expt_dirs, _ = next(os.walk(run_dir))
    
    alpha = []
    train_acc_list = []
    val_acc_list = []
    
    for expt_dir in tqdm(expt_dirs, unit="expt"):
        try:
            train_data_pct, best_train_acc, best_val_acc = load_best_acc(f"{run_dir}/{expt_dir}")
            
            alpha.append(train_data_pct)
            train_acc_list.append(best_train_acc)
            val_acc_list.append(best_val_acc)
        
        except FileNotFoundError:
            print(f"{run_dir}/{expt_dir}: File not found")
            pass
    
    return alpha, train_acc_list, val_acc_list

input_dirs = [
    "./lightning_logs/1AdamW_lr0.001_dropout0.1", 
    "./lightning_logs/8AdamW_lr0.0001_dropout0.1",
    "./lightning_logs/9AdamW_lr0.001_dropout0",
    "./lightning_logs/4SGD_lr0.001",
    "./lightning_logs/5SGD_lr0.01",
    "./lightning_logs/6SGD_lr0.001+weight_decay",
    "./lightning_logs/2Adam",
    "./lightning_logs/3Adam+full_batch",
    "./lightning_logs/7RMSprop"
    ]

titles = [
    "AdamW",
    "AdamW,0.1x baseline LR",
    "AdamW, dropout 0",
    "SGD with Nesterov",
    "SGD with Nesterov,10x baseline LR",
    "SGD with Nesterov,weight decay 0.1",
    "Minibatch Adam",
    "Full batch Adam",
    "RMSprop, alpha 0.9"
    ]

output_dir = "./output/different_settings"

try:
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    fig.suptitle("Highest validation accuracy within 100000 training steps", fontsize=16)
    
    fig.text(0.5, 0.04, "training data fraction", ha="center", fontsize=14)
    fig.text(0.04, 0.5, "Highest validation accuracy", va="center", rotation="vertical", fontsize=14)

    # plt.xlabel('training data fraction')
    # plt.ylabel("Highest validation accuracy")

    for i, ax in enumerate(axes.flat):
        if i >= len(titles):
            break
        
        ax.set_title(titles[i])
        ymin = -5
        ymax = 105
        xmin = 15
        xmax = 85
        ax.axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        
        if input_dirs[i] == "":
            continue
        
        alpha, train_acc_list, val_acc_list = load_all_metrics(input_dirs[i])
        
        ax.plot(alpha, val_acc_list, marker="o", markersize=6, linestyle="-")
        
        # ax.legend()

    plt.subplots_adjust(hspace=0.5)
    
    image_file = output_dir + "/different_settings.png"

    d = os.path.split(image_file)[0]
    os.makedirs(d, exist_ok=True)
    print(f"Writing {image_file}")
    fig.savefig(image_file)
    plt.close(fig)

except BaseException as e:
    print(f"failed: {e}")