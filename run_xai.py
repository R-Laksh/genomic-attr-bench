import os, json, time, argparse, random
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET

from sklearn.metrics import roc_auc_score, average_precision_score

from captum.attr import IntegratedGradients, DeepLift, DeepLiftShap, LRP, GuidedGradCam
from captum.attr._utils.lrp_rules import EpsilonRule

from ushuffle import shuffle, Shuffler

# --------------------------
# Utils
# --------------------------
ALPH = {"A":0, "C":1, "G":2, "T":3}
ALPH_DECODE = {0:"A", 1:"C", 2:"G", 3:"T"}
ALPH_CHAR = "ACGT"

def set_seed(seed: int):
    torch.cuda.empty_cache()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def one_hot(seq: str):
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i,ch in enumerate(seq):
        idx = ALPH.get(ch, None)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr

def ohe_to_seq(x: np.ndarray):
    idxs = x.argmax(axis=0)
    return "".join(ALPH_DECODE[int(i)] for i in idxs)

def parse_seqgra_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    general = root.find("general")
    gid = general.attrib.get("id", "")
    name = (general.findtext("name") or "").strip()
    seed = int((general.findtext("seed") or "0").strip())

    minlength = int(root.findtext("background/minlength") or "0")
    maxlength = int(root.findtext("background/maxlength") or "0")

    split_counts = {}
    sets = root.findall("datageneration/sets/set")
    for s in sets:
        split = s.attrib.get("name")
        total = 0
        for ex in s.findall("example"):
            total += int(ex.attrib.get("samples", "0"))
        split_counts[split] = total

    return {
        "grammar_id": gid,
        "grammar_name": name,
        "grammar_seed": seed,
        "minlength": minlength,
        "maxlength": maxlength,
        "split_counts": split_counts,
        "xml_path": str(xml_path),
    }

# --------------------------
# Dataset
# --------------------------
class SeqgraDataset(Dataset):
    def __init__(self, txt_path: Path, label_pos: str = "c1"):
        df = pd.read_csv(txt_path, sep="\t")
        self.seqs = df["x"].tolist()
        self.x = np.stack([one_hot(s) for s in self.seqs])
        yraw = df["y"].astype(str).tolist()
        self.y = np.array([1.0 if y==label_pos else 0.0 for y in yraw], dtype=np.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i])

# --------------------------
# Models
# --------------------------
class CNN1D(nn.Module):
    def __init__(self, L: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4),
        )
        with torch.no_grad():
            dummy = torch.zeros(1,4,L)
            out = self.net(dummy)
            self.flat = out.shape[1] * out.shape[2]
        self.fc = nn.Sequential(
            nn.Linear(self.flat, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        z = self.net(x)
        z = z.reshape(z.size(0), -1)
        return self.fc(z).squeeze(1)

class DeepSTARR(nn.Module):
    """
    DeepSTARR model from de Almeida et al., 2022;
    see <https://www.nature.com/articles/s41588-022-01048-5>
    Taken from: https://colab.research.google.com/drive/1a2fiRPBd1xvoJf0WNiMUgTYiLTs1XETf?usp=sharing#scrollTo=mPSQhbr4v05i
    """
    def __init__(self, output_dim, d=256,
                 conv1_filters=None, learn_conv1_filters=True,
                 conv2_filters=None, learn_conv2_filters=True,
                 conv3_filters=None, learn_conv3_filters=True,
                 conv4_filters=None, learn_conv4_filters=True):
        super().__init__()

        if d != 256:
            print("NB: number of first-layer convolutional filters in original DeepSTARR model is 256; current number of first-layer convolutional filters is not set to 256")

        self.activation = nn.ReLU()
        self.dropout4 = nn.Dropout(0.4)
        self.flatten = nn.Flatten()

        self.init_conv1_filters = conv1_filters
        self.init_conv2_filters = conv2_filters
        self.init_conv3_filters = conv3_filters
        self.init_conv4_filters = conv4_filters

        assert (not (conv1_filters is None and not learn_conv1_filters)), "initial conv1_filters cannot be set to None while learn_conv1_filters is set to False"
        assert (not (conv2_filters is None and not learn_conv2_filters)), "initial conv2_filters cannot be set to None while learn_conv2_filters is set to False"
        assert (not (conv3_filters is None and not learn_conv3_filters)), "initial conv3_filters cannot be set to None while learn_conv3_filters is set to False"
        assert (not (conv4_filters is None and not learn_conv4_filters)), "initial conv4_filters cannot be set to None while learn_conv4_filters is set to False"

        # Layer 1 (convolutional), constituent parts
        if conv1_filters is not None:
            if learn_conv1_filters:
                self.conv1_filters = nn.Parameter( torch.Tensor(conv1_filters) )
            else:
                self.register_buffer("conv1_filters", torch.Tensor(conv1_filters))
        else:
            self.conv1_filters = nn.Parameter(torch.zeros(d, 4, 7))
            nn.init.kaiming_normal_(self.conv1_filters)
        self.batchnorm1 = nn.BatchNorm1d(d)
        self.activation1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(2)

        # Layer 2 (convolutional), constituent parts
        if conv2_filters is not None:
            if learn_conv2_filters:
                self.conv2_filters = nn.Parameter( torch.Tensor(conv2_filters) )
            else:
                self.register_buffer("conv2_filters", torch.Tensor(conv2_filters))
        else:
            self.conv2_filters = nn.Parameter(torch.zeros(60, d, 3))
            nn.init.kaiming_normal_(self.conv2_filters)
        self.batchnorm2 = nn.BatchNorm1d(60)
        self.activation2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(2)

        # Layer 3 (convolutional), constituent parts
        if conv3_filters is not None:
            if learn_conv3_filters:
                self.conv3_filters = nn.Parameter( torch.Tensor(conv3_filters) )
            else:
                self.register_buffer("conv3_filters", torch.Tensor(conv3_filters))
        else:
            self.conv3_filters = nn.Parameter(torch.zeros(60, 60, 5))
            nn.init.kaiming_normal_(self.conv3_filters)
        self.batchnorm3 = nn.BatchNorm1d(60)
        self.activation3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(2)

        # Layer 4 (convolutional), constituent parts
        if conv4_filters is not None:
            if learn_conv4_filters:
                self.conv4_filters = nn.Parameter( torch.Tensor(conv4_filters) )
            else:
                self.register_buffer("conv4_filters", torch.Tensor(conv4_filters))
        else:
            self.conv4_filters = nn.Parameter(torch.zeros(120, 60, 3))
            nn.init.kaiming_normal_(self.conv4_filters)
        self.batchnorm4 = nn.BatchNorm1d(120)
        self.activation4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool1d(2)

        # Layer 5 (fully connected), constituent parts
        self.fc5 = nn.LazyLinear(256, bias=True)
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.activation5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.4)
        
        # Layer 6 (fully connected), constituent parts
        self.fc6 = nn.Linear(256, 256, bias=True)
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.activation6 = nn.ReLU()
        self.dropout6 = nn.Dropout(0.4)
        
        # Output layer (fully connected), constituent parts
        self.fc7 = nn.Linear(256, output_dim)

    def get_which_conv_layers_transferred(self):
        layers = []
        if self.init_conv1_filters is not None:
            layers.append(1)
        if self.init_conv2_filters is not None:
            layers.append(2)
        if self.init_conv3_filters is not None:
            layers.append(3)
        if self.init_conv4_filters is not None:
            layers.append(4)
        return layers

    def forward(self, x):
        # Layer 1
        cnn = torch.conv1d(x, self.conv1_filters, stride=1, padding="same")
        cnn = self.batchnorm1(cnn)
        cnn = self.activation1(cnn)
        cnn = self.maxpool1(cnn)

        # Layer 2
        cnn = torch.conv1d(cnn, self.conv2_filters, stride=1, padding="same")
        cnn = self.batchnorm2(cnn)
        cnn = self.activation2(cnn)
        cnn = self.maxpool2(cnn)

        # Layer 3
        cnn = torch.conv1d(cnn, self.conv3_filters, stride=1, padding="same")
        cnn = self.batchnorm3(cnn)
        cnn = self.activation3(cnn)
        cnn = self.maxpool3(cnn)

        # Layer 4
        cnn = torch.conv1d(cnn, self.conv4_filters, stride=1, padding="same")
        cnn = self.batchnorm4(cnn)
        cnn = self.activation4(cnn)
        cnn = self.maxpool4(cnn)

        # Layer 5
        cnn = self.flatten(cnn)
        cnn = self.fc5(cnn)
        cnn = self.batchnorm5(cnn)
        cnn = self.activation5(cnn)
        cnn = self.dropout5(cnn)

        # Layer 6
        cnn = self.fc6(cnn)
        cnn = self.batchnorm6(cnn)
        cnn = self.activation6(cnn)
        cnn = self.dropout5(cnn)

        # Output layer
        y_pred = self.fc7(cnn)

        return y_pred.squeeze(1)

class ExpActivation(nn.Module):
    """
    Exponential activation function from Koo & Ploenzke, 2021 (PMID: 34322657)
    Taken from : https://github.com/wassermanlab/ExplaiNN/blob/main/explainn/models/networks.py
    """
    def __init__(self):
        super(ExpActivation, self).__init__()

    def forward(self, x):
        return torch.exp(x)


class Unsqueeze(torch.nn.Module):
    """
    Unsqueeze for sequential models
    Taken from : https://github.com/wassermanlab/ExplaiNN/blob/main/explainn/models/networks.py
    """
    def forward(self, x):
        return x.unsqueeze(-1)
    
class ExplaiNN(nn.Module):
    """
    The ExplaiNN model (PMID: 37370113)
    Taken from : https://github.com/wassermanlab/ExplaiNN/blob/main/explainn/models/networks.py
    """
    def __init__(self, num_cnns, input_length, num_classes, filter_size=19, num_fc=2, pool_size=7, pool_stride=7,
                 weight_path=None):
        """
        :param num_cnns: int, number of independent cnn units
        :param input_length: int, input sequence length
        :param num_classes: int, number of outputs
        :param filter_size: int, size of the unit's filter, default=19
        :param num_fc: int, number of FC layers in the unit, default=2
        :param pool_size: int, size of the unit's maxpooling layer, default=7
        :param pool_stride: int, stride of the unit's maxpooling layer, default=7
        :param weight_path: string, path to the file with model weights
        """
        super(ExplaiNN, self).__init__()

        self._options = {
            "num_cnns": num_cnns,
            "input_length": input_length,
            "num_classes": num_classes,
            "filter_size": filter_size,
            "num_fc": num_fc,
            "pool_size": pool_size,
            "pool_stride": pool_stride,
            "weight_path": weight_path
        }

        if num_fc == 0:
            self.linears = nn.Sequential(
                nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size,
                          groups=num_cnns),
                nn.BatchNorm1d(num_cnns),
                ExpActivation(),
                nn.MaxPool1d(input_length - (filter_size-1)),
                nn.Flatten())
        elif num_fc == 1:
            self.linears = nn.Sequential(
                nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size,
                          groups=num_cnns),
                nn.BatchNorm1d(num_cnns),
                ExpActivation(),
                nn.MaxPool1d(pool_size, pool_stride),
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(in_channels=int(((input_length - (filter_size-1)) - (pool_size-1)-1)/pool_stride + 1) * num_cnns,
                          out_channels=1 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(1 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU(),
                nn.Flatten())
        elif num_fc == 2:
            self.linears = nn.Sequential(
                nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size,
                          groups=num_cnns),
                nn.BatchNorm1d(num_cnns),
                ExpActivation(),
                nn.MaxPool1d(pool_size, pool_stride),
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(in_channels=int(((input_length - (filter_size-1)) - (pool_size-1)-1)/pool_stride + 1) * num_cnns,
                          out_channels=100 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(100 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(in_channels=100 * num_cnns,
                          out_channels=1 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(1 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU(),
                nn.Flatten())
        else:
            self.linears = nn.Sequential(
                nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size,
                          groups=num_cnns),
                nn.BatchNorm1d(num_cnns),
                ExpActivation(),
                nn.MaxPool1d(pool_size, pool_stride),
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(in_channels=int(((input_length - (filter_size-1)) - (pool_size-1)-1)/pool_stride + 1) * num_cnns,
                          out_channels=100 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(100 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU())

            self.linears_bg = nn.ModuleList([nn.Sequential(nn.Dropout(0.3),
                                                           nn.Conv1d(in_channels=100 * num_cnns,
                                                                     out_channels=100 * num_cnns, kernel_size=1,
                                                                     groups=num_cnns),
                                                           nn.BatchNorm1d(100 * num_cnns, 1e-05, 0.1, True),
                                                           nn.ReLU()) for i in range(num_fc - 2)])

            self.last_linear = nn.Sequential(nn.Dropout(0.3),
                                             nn.Conv1d(in_channels=100 * num_cnns, out_channels=1 * num_cnns,
                                                       kernel_size=1,
                                                       groups=num_cnns),
                                             nn.BatchNorm1d(1 * num_cnns, 1e-05, 0.1, True),
                                             nn.ReLU(),
                                             nn.Flatten())

        self.final = nn.Linear(num_cnns, num_classes)

        if weight_path:
            self.load_state_dict(torch.load(weight_path))

    def forward(self, x):
        x = x.repeat(1, self._options["num_cnns"], 1)
        if self._options["num_fc"] <= 2:
            outs = self.linears(x)
        else:
            outs = self.linears(x)
            for i in range(len(self.linears_bg)):
                outs = self.linears_bg[i](outs)
            outs = self.last_linear(outs)
        out = self.final(outs)
        return out.squeeze(1)

def build_model(model_name: str, L: int):
    if model_name == "cnn1d": return CNN1D(L)
    elif model_name == "deepstarr": return DeepSTARR(1)
    elif model_name == "explainnn": return ExplainNN(L)
    raise ValueError(f"Unknown model_name={model_name}")

# --------------------------
# Metrics
# --------------------------
@torch.no_grad()
def eval_auc(model, loader, device):
    model.eval()
    y_true, y_score = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        y_true.extend(y.numpy())
        y_score.extend(torch.sigmoid(logits).cpu().numpy())
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }


def dinuc_shuffle_seq(seq: str):
    shuffler = Shuffler(seq.encode("utf-8"), 2)
    out = shuffler.shuffle()
    return out
def make_baseline(x_batch: torch.Tensor, baseline_kind: str, seed: int):
    if baseline_kind == "zero":
        return torch.zeros_like(x_batch)
    if baseline_kind == "uniform":
        return torch.full_like(x_batch, 0.25)

    if baseline_kind == "dinuc":
        xb = x_batch.detach().cpu().numpy()
        baselines = []
        for i in range(xb.shape[0]):
            seq = ohe_to_seq(xb[i])
            shuf = dinuc_shuffle_seq(seq)
            baselines.append(one_hot(shuf))
        b = torch.from_numpy(np.stack(baselines)).to(x_batch.device)
        return b

    raise ValueError(f"Unknown baseline_kind={baseline_kind}")

# --------------------------
# Attribution methods
# --------------------------
def attach_lrp_rules(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            m.rule = EpsilonRule()

def reduce_to_base_scores(attr: torch.Tensor, x: torch.Tensor, mode: str):
    """
    attr, x: (B,4,L)
    mode:
      - "attr_x_input": sum_c attr*c * x*c
      - "sum_channels": sum_c attr
    returns (B,L)
    """
    if mode == "attr_x_input":
        return (attr * x).sum(dim=1)
    if mode == "sum_channels":
        return attr.sum(dim=1)
    raise ValueError(mode)

def compute_attributions_batch(model, x, method: str, baseline_kind: str, seed: int,
                               ig_steps: int, reduce_mode: str):
    model.eval()
    if method == "gradxinput":
        xreq = x.detach().requires_grad_(True)
        out = model(xreq).sum()
        out.backward()
        attr = xreq.grad.detach() * x.detach()
        return reduce_to_base_scores(attr, x, reduce_mode)

    if method == "ig":
        ig = IntegratedGradients(model)
        base = make_baseline(x, baseline_kind, seed)
        attr = ig.attribute(x, baselines=base, n_steps=ig_steps)
        return reduce_to_base_scores(attr, x, reduce_mode)

    if method == "deeplift":
        dl = DeepLift(model)
        base = make_baseline(x, baseline_kind, seed)
        attr = dl.attribute(x, baselines=base)
        return reduce_to_base_scores(attr, x, reduce_mode)

    if method == "deepliftshap":
        dls = DeepLiftShap(model)
        base = make_baseline(x, baseline_kind, seed)
        attr = dls.attribute(x, baselines=base)
        return reduce_to_base_scores(attr, x, reduce_mode)

    if method == "lrp":
        attach_lrp_rules(model)
        lrp = LRP(model)
        attr = lrp.attribute(x)
        return reduce_to_base_scores(attr, x, reduce_mode)

    if method == "guidedgradcam":
        if isinstance(model, DeepSTARR):
            target_layer = model.activation4
        elif isinstance(model, CNN1D): 
            target_layer = model.net[6]
        ggc = GuidedGradCam(model, target_layer)
        attr = ggc.attribute(x)
        return reduce_to_base_scores(attr, x, reduce_mode)

    raise ValueError(f"Unknown method={method}")

# --------------------------
# Runner
# --------------------------
def train_model(model, train_loader, val_loader, device, epochs=5, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()

    curve = []
    best_auc, best_state = -1.0, None

    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        for x,y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = lossf(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        valm = eval_auc(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **valm}
        curve.append(row)

        if valm["roc_auc"] > best_auc:
            best_auc = valm["roc_auc"]
            best_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, pd.DataFrame(curve)

def plot_curve(df: pd.DataFrame, out_png: Path):
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"])
    plt.xlabel("epoch")
    plt.ylabel("train_loss")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--xml", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model", type=str, default="cnn1d")
    ap.add_argument("--label_pos", type=str, default="c1")

    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--methods", type=str, nargs="+",
                    default=["ig", "deepliftshap", "gradxinput", "lrp"])
    ap.add_argument("--baselines", type=str, nargs="+",
                    default=["uniform", "zero", "dinuc"])

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ig_steps", type=int, default=64)
    ap.add_argument("--reduce_mode", type=str, default="attr_x_input",
                    choices=["attr_x_input", "sum_channels"])
    ap.add_argument("--save_attr", action="store_true",
                    help="If set, saves per-method attributions to npz")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_xml = parse_seqgra_xml(Path(args.xml))

    train_ds = SeqgraDataset(data_root/"training.txt", label_pos=args.label_pos)
    val_ds   = SeqgraDataset(data_root/"validation.txt", label_pos=args.label_pos)
    test_ds  = SeqgraDataset(data_root/"test.txt", label_pos=args.label_pos)
    L = train_ds.x.shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in args.seeds:
        set_seed(seed)

        run_id = f"{meta_xml['grammar_id']}__{args.model}__seed{seed}"
        run_dir = out_dir/run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir/"attributions").mkdir(exist_ok=True)

        meta = {
            **meta_xml,
            "run_id": run_id,
            "seed": seed,
            "model": args.model,
            "label_pos": args.label_pos,
            "data_root": str(data_root),
            "L": int(L),
            "methods": args.methods,
            "baselines": args.baselines,
            "reduce_mode": args.reduce_mode,
        }
        (run_dir/"meta.json").write_text(json.dumps(meta, indent=2))

        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

        model = build_model(args.model, L).to(device)
        model, curve = train_model(model, train_loader, val_loader, device,
                                   epochs=args.epochs, lr=1e-3)
        torch.save(model.state_dict(), run_dir/"model_best.pt")
        curve.to_csv(run_dir/"train_curve.csv", index=False)
        plot_curve(curve, run_dir/"train_curve.png")

        testm = eval_auc(model, test_loader, device)

        attr_index = []
        for method in args.methods:
            baseline_list = args.baselines if method in ("ig", "deeplift", "deepliftshap") else ["none"]
            for base in baseline_list:
                torch.cuda.empty_cache()
                all_scores = []
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    scores = compute_attributions_batch(
                        model, xb, method=method, baseline_kind=base,
                        seed=seed, ig_steps=args.ig_steps, reduce_mode=args.reduce_mode
                    )
                    all_scores.append(scores.detach().cpu().numpy())
                    del scores
                    del xb
                scores_np = np.concatenate(all_scores, axis=0)

                key = f"{method}__{base}"
                attr_index.append({"key": key, "method": method, "baseline": base,
                                   "path": f"attributions/{key}.npz"})
                if args.save_attr:
                    np.savez_compressed(run_dir/f"attributions/{key}.npz", scores_np)

        metrics = {
            "test_roc_auc": testm["roc_auc"],
            "test_auprc": testm["auprc"],
            "attr_index": attr_index,
        }
        (run_dir/"metrics.json").write_text(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
