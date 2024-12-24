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
# parser = grok.training.add_args(parser)
args = parser.parse_args()
print(args, flush=True)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
def load_generalization_steps(expt_dir):
    """
    read files {expt_dir}/hparams.yaml and {expt_dir}/metrics.csv, return (alpha, S_t, S_v),
    where S_t is the minimum steps that training acc reaches 99%, 
    and S_v is the minimum steps that validation acc reaches 99%.
    """
    with open(f"{expt_dir}/hparams.yaml", "r") as fh:
        hparams_dict = yaml.safe_load(fh)
    
    train_data_pct = hparams_dict["train_data_pct"] # train data pct
    
    target_row = None
    with open(f"{expt_dir}/metrics.csv", "r") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        for row in rows:
            if row["train_accuracy"] != "": # training stage
                if (float(row["train_accuracy"]) >= 99 
                and target_row is not None 
                and float(target_row["train_accuracy"]) >= 99):
                    break
                target_row = row
        
        required_training_steps = int(target_row["step"])
        
        target_row = None
        for row in reversed(rows):
            if row["val_accuracy"] != "": # val stage
                if (float(row["val_accuracy"]) < 99
                    # and target_row is not None
                    ):
                    break
                target_row = row
        
        if target_row is None:
            required_val_steps = 99999
        else: 
            required_val_steps = int(target_row["step"])
        
    return train_data_pct, required_training_steps, required_val_steps

def load_all_metrics(run_dir):
    _, expt_dirs, _ = next(os.walk(run_dir))
    
    alpha = []
    train_steps = []
    val_steps = []
    
    for expt_dir in tqdm(expt_dirs, unit="expt"):
        try:
            train_data_pct, train_step, val_step = load_generalization_steps(f"{run_dir}/{expt_dir}")
            
            alpha.append(train_data_pct)
            train_steps.append(train_step)
            val_steps.append(val_step)
        
        except FileNotFoundError:
            print(f"{run_dir}/{expt_dir}: File not found")
            pass
    
    return alpha, train_steps, val_steps

rundir = args.input_dir
image_dir = args.output_dir

try:
    alpha, train_steps, val_steps = load_all_metrics(rundir)
    print("load metric data: success")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.title(f"Steps until generalization for modular addition")
    
    plt.xscale('linear')
    plt.xlabel('Training data percentage')
    plt.yscale('log')
    plt.ylabel('Steps until accuracy > 99%')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ymin = 100
    ymax = 100000
    ax.axis(ymin=ymin, ymax=ymax)
    
    ax.plot(alpha, train_steps, label="train", color="red", marker="o", markersize=6, linestyle="-")
    ax.plot(alpha, val_steps, label="val", color="blue", marker="x", markersize=6, linestyle="-")
    ax.legend()
    
    image_file = f"{image_dir}/Transformer_alpha/Transformer_alpha"
    image_file += ".png"
    
    d = os.path.split(image_file)[0]
    os.makedirs(d, exist_ok=True)
    print(f"Writing {image_file}")
    fig.savefig(image_file)
    plt.close(fig)
    
    
except BaseException as e:
    print(f"{rundir} failed: {e}")
            