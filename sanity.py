import os, json, argparse, copy, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from tqdm import tqdm

from run_xai import (
    SeqgraDataset, CNN1D, train_model, 
    compute_attributions_batch, set_seed
)

def get_layers_top_to_bottom(model: nn.Module):
    fc_layers = []
    for m in reversed(model.fc):
        if isinstance(m, nn.Linear):
            fc_layers.append(m)
            
    conv_layers = []
    for m in reversed(model.net):
        if isinstance(m, nn.Conv1d):
            conv_layers.append(m)
            
    return fc_layers + conv_layers

def reinit_layer(layer):
    if isinstance(layer, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

def fisher_mean(rs, eps=1e-7):
    rs = np.clip(np.asarray(rs, dtype=float), -1+eps, 1-eps)
    zs = np.arctanh(rs)
    z_mean = np.nanmean(zs)
    r_mean = np.tanh(z_mean)
    return float(r_mean)
    
def compute_spearman(attr1, attr2):
    corrs_signed = []
    corrs_abs = []
    
    for i in range(len(attr1)):
        a1 = attr1[i]
        a2 = attr2[i]
        if np.std(a1) < 1e-9 or np.std(a2) < 1e-9:
            corrs_signed.append(0.0)
            corrs_abs.append(0.0)
            continue

        s, _ = spearmanr(a1, a2)
        corrs_signed.append(s)

        s_abs, _ = spearmanr(np.abs(a1), np.abs(a2))
        corrs_abs.append(s_abs)
        
    return fisher_mean(corrs_signed), fisher_mean(corrs_abs)

def run_cascade_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SeqgraDataset(f"{args.data_root}/training.txt", args.label_pos)
    val_ds = SeqgraDataset(f"{args.data_root}/validation.txt", args.label_pos)
    test_ds = SeqgraDataset(f"{args.data_root}/test.txt", args.label_pos)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    records = []

    for seed in args.seeds:
        print(f"--- Processing Seed {seed} ---")
        set_seed(seed)

        model = CNN1D(L=200).to(device)
        model, _ = train_model(model, train_loader, val_loader, device, epochs=5)
        
        originals = {}
        for method in args.methods:
            attrs = []
            for x, y in test_loader:
                x = x.to(device)
                a = compute_attributions_batch(
                    model, x, method, "dinuc", seed, 64, "attr_x_input"
                )
                attrs.append(a.detach().cpu().numpy())
            originals[method] = np.concatenate(attrs, axis=0)
        
        perturb_model = copy.deepcopy(model)
        layers_to_randomize = get_layers_top_to_bottom(perturb_model)
        
        for step, layer in enumerate(tqdm(layers_to_randomize, desc="Cascading")):
            reinit_layer(layer)
            for method in args.methods:
                
                curr_attrs_list = []
                for x, y in test_loader:
                    x = x.to(device)
                    a = compute_attributions_batch(
                        perturb_model, x, method, "zero", seed, 50, "attr_x_input"
                    )
                    curr_attrs_list.append(a.detach().cpu().numpy())
                
                curr_attrs = np.concatenate(curr_attrs_list, axis=0)
                
                rho_signed, rho_abs = compute_spearman(originals[method], curr_attrs)
                
                records.append({
                    "grammar": args.grammar_name,
                    "seed": seed,
                    "method": method,
                    "step": step + 1,
                    "layer_type": layer.__class__.__name__,
                    "rho_signed": rho_signed,
                    "rho_abs": rho_abs
                })
                
    df = pd.DataFrame(records)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(f"{args.out_dir}/cascade_{args.grammar_name}.csv", index=False)
    print(f"Saved cascade results to {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--dataset_type", required=True)
    ap.add_argument("--out_dir", default="results_cascade")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1,2,3,4,5])
    ap.add_argument("--methods", nargs="+", default=["ig", "deeplift", "guidedgradcam", "gradxinput"])
    ap.add_argument("--label_pos", default="c1")
    args = ap.parse_args()
    
    run_cascade_experiment(args)