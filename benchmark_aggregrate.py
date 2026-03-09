import os, json, glob

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score
import argparse
    
import warnings
warnings.filterwarnings('ignore')

MOTIF_A = "TATAAA"
MOTIF_B = "CCAAT"
MOTIF_D = "CGCCAT" 

def load_split(root: str, split: str):
    X = pd.read_csv(f"{root}{split}.txt", sep='\t')
    A = pd.read_csv(f"{root}{split}-annotation.txt", sep='\t')
    seqs = X['x'].tolist()
    labels = X['y'].tolist()
    ann = A['annotation'].tolist()
    return seqs, labels, ann

def parse_annotation_to_binary(ann_row):
    return np.array([1 if ch=='G' else 0 for ch in ann_row], dtype=np.int8)


def create_masks(seqs, annotations):
    N = len(seqs)
    L = len(seqs[0])
    
    causal_mask = np.zeros((N, L), dtype=int)
    dummy_mask = np.zeros((N, L), dtype=int)
    
    CAUSAL_KEYWORDS = [MOTIF_A, MOTIF_B] 
    DUMMY_KEYWORDS = [MOTIF_D]

    for i, ann_str in enumerate(annotations):
        seq = seqs[i]
        ann = parse_annotation_to_binary(ann_str)
        for motif in CAUSAL_KEYWORDS:
            start = 0
            while True:
                idx = seq.find(motif, start)
                if idx == -1: break
                if np.all(ann[idx : idx+len(motif)] > 0):
                    causal_mask[i, idx : idx+len(motif)] = 1
                start = idx + 1
                
        for motif in DUMMY_KEYWORDS:
            start = 0
            while True:
                idx = seq.find(motif, start)
                if idx == -1: break
                if np.all(ann[idx : idx+len(motif)] > 0):
                    dummy_mask[i, idx : idx+len(motif)] = 1
                start = idx + 1
                
    return causal_mask, dummy_mask



def extract_motif_stats(seq, attr_row, ann_str=None):
    stats = {"A": [], "B": [], "D": []}
    motifs = [(MOTIF_A, "A"), (MOTIF_B, "B"), (MOTIF_D, "D")]

    ann = None
    if ann_str is not None:
        ann = parse_annotation_to_binary(ann_str)

    for motif, key in motifs:
        mlen = len(motif)
        start = 0
        while True:
            idx = seq.find(motif, start)
            if idx == -1:
                break

            if ann is not None and not np.all(ann[idx:idx+mlen] > 0):
                start = idx + 1
                continue

            region = slice(idx, idx + mlen)
            mean_attr = float(np.mean(attr_row[region]))
            median_attr = float(np.median(attr_row[region]))
            stats[key].append({"start": idx, "end": idx + mlen, "mean": mean_attr, "median": median_attr})
            start = idx + 1
    return stats

def extract_background_stats(seq, attr_row, causal_mask, dummy_mask):
    L = len(seq)
    background_mask = 1 - (causal_mask + dummy_mask)
    median = np.median(attr_row[background_mask == 1])
    return median
    
def candidate_B_regions_after_A(seq_len, a_start, a_end,
                                min_gap=3, max_gap=5):
    regions = []
    mlen_B = len(MOTIF_B)
    for gap in range(min_gap, max_gap + 1):
        start = a_end + gap
        end = start + mlen_B
        if end <= seq_len:
            regions.append((start, end))
    return regions


def candidate_A_regions_before_B(seq_len, b_start, b_end,
                                 min_gap=3, max_gap=5):
    regions = []
    mlen_A = len(MOTIF_A)
    for gap in range(min_gap, max_gap + 1):
        end = b_start - gap
        start = end - mlen_A
        if start >= 0:
            regions.append((start, end))
    return regions
    
def has_AB_spacing_ok(stats_A, stats_B, min_gap=3, max_gap=5):
    for a in stats_A:
        for b in stats_B:
            gap = b["start"] - a["end"]
            if min_gap <= gap <= max_gap:
                return True
    return False
    
class XAIBenchmark:
    def __init__(self, attributions, causal_mask, dummy_mask,ann):
        """
        attributions: (N, L) numpy array
        causal_mask: (N, L) binary
        dummy_mask: (N, L) binary
        """
        self.attr = np.abs(attributions)
        self.raw_attr = attributions
        self.causal = causal_mask
        self.dummy = dummy_mask
        self.ann = ann
        self.N, self.L = self.attr.shape
        
    def eval_localization(self):
        """
        Computes Token-level AUPRC (Average Precision)
        """
        y_true = self.causal.flatten()
        y_scores = self.attr.flatten()
        return average_precision_score(y_true, y_scores)

    def eval_precision_at_k(self):
        """
        Fraction of top-k attributions that hit the causal motif
        """
        precisions = []
        for i in range(self.N):
            k = int(np.sum(self.causal[i]))
            if k == 0: continue

            top_k_indices = np.argsort(self.attr[i])[::-1][:k]

            hits = np.sum(self.causal[i][top_k_indices])
            precisions.append(hits / k)
            
        return np.mean(precisions)
        
    def eval_causal_relevance_score(self):
        """
        Ratio of avg attribution in causal vs all regions
        """
        crs_scores = []
        skipped_count = 0
        for i in range(self.N):
            if np.sum(self.causal[i]) == 0:
                continue
                
            avg_causal = np.mean(self.attr[i][self.causal[i] == 1])
            avg_total = np.mean(self.attr[i])

            if avg_total < 1e-9:
                skipped_count += 1
                crs_scores.append(0.0)
            else:
                crs_scores.append(avg_causal / avg_total)
        if skipped_count > 0.05 * self.N:
            print(f"Warning: Skipped {skipped_count} sequences due to 0 causal attribution")    
                
        return np.mean(crs_scores)
        

    def eval_dummy_relevance_score(self):
        """
        Ratio of avg attribution in dummy vs causal regions
        """
        drs_scores = []
        skipped_count = 0
        for i in range(self.N):
            if np.sum(self.causal[i]) == 0 or np.sum(self.dummy[i]) == 0:
                continue
                
            avg_causal = np.mean(self.attr[i][self.causal[i] == 1])
            avg_dummy = np.mean(self.attr[i][self.dummy[i] == 1])
            # if i % 100 ==0:
            #     print(f'avg_causal: {avg_causal}')
            #     print(f'avg_dummy: {avg_dummy}')

            if avg_causal < 1e-9:
                if avg_dummy > 1e-9:
                    skipped_count += 1
                    continue
                else:
                    drs_scores.append(1.0) 
            else:
                drs_scores.append(avg_dummy / avg_causal)

        if skipped_count > 0.05 * self.N:
            print(f"Warning: Skipped {skipped_count} sequences due to 0 causal attribution")
                
        return np.mean(drs_scores) 


    def eval_grammar_satisfiability(
        self,
        seqs,
        labels,
        dataset_type: str,
        positive_labels,
        window: int = 5,
    ):
        assert len(seqs) == self.N and len(labels) == self.N

        pos_flags = []
        neg_flags = []

        for i in range(self.N):
            seq = seqs[i]
            label = labels[i]
            is_pos = label in positive_labels
            L = len(seq)

            stats = extract_motif_stats(seq, self.raw_attr[i], ann_str=self.ann[i])
            stats_A = stats["A"]
            stats_B = stats["B"]
            stats_D = stats["D"]

            has_A = len(stats_A) > 0
            has_B = len(stats_B) > 0
            has_D = len(stats_D) > 0

            bg_median = extract_background_stats(seq, self.raw_attr[i],self.causal[i], self.dummy[i])
            if dataset_type == "NOT":
                if is_pos:
                    sat = not (has_A)
                else:
                    sat = has_A and all(s["mean"] < 0 for s in stats_A) and (bg_median > 0)

                    
            elif dataset_type == "OR":
                if is_pos:
                    if not (has_A or has_B):
                        sat = False
                    else:
                        sat = all(s["mean"] > 0 for s in (stats_A + stats_B))
                else:
                    sat = not (has_A or has_B)

            elif dataset_type in ("AND_XOR", "AND_NAND", "DUMMY"):
                if is_pos:
                    sat = has_A and has_B and has_AB_spacing_ok(stats_A, stats_B)
                    if sat:
                        sat = all(s["mean"] > 0 for s in (stats_A + stats_B))
                else:
                    if has_A and not has_B:
                        a_mean = np.mean([s["mean"] for s in stats_A])

                        partner_vals = []
                        for sA in stats_A:
                            for start, end in candidate_B_regions_after_A(
                                L, sA["start"], sA["end"], min_gap=3, max_gap=5
                            ):
                                partner_vals.append(
                                    np.mean(self.raw_attr[i][start:end])
                                )
                        b_expected = np.min(partner_vals) if partner_vals else 0.0

                        sat = (a_mean > 0) and (b_expected < 0)

                    elif has_B and not has_A:
                        b_mean = np.mean([s["mean"] for s in stats_B])

                        partner_vals = []
                        for sB in stats_B:
                            for start, end in candidate_A_regions_before_B(
                                L, sB["start"], sB["end"], min_gap=3, max_gap=5
                            ):
                                partner_vals.append(
                                    np.mean(self.raw_attr[i][start:end])
                                )
                        a_expected = np.min(partner_vals) if partner_vals else 0.0

                        sat = (b_mean > 0) and (a_expected < 0)

                    elif (not has_A) and (not has_B):
                        sat = True
                    else:
                        sat = False

                    if dataset_type == "DUMMY" and has_D:
                        dummy_abs = np.mean([abs(s["mean"]) for s in stats_D])
                        if has_A or has_B:
                            causal_abs = np.mean(
                                [abs(s["mean"]) for s in (stats_A + stats_B)]
                            )
                            sat = sat and (dummy_abs < causal_abs)
                        else:
                            sat = sat and (dummy_abs < 1e-3)

            elif dataset_type == "XOR_XNOR":
                if is_pos:
                    if has_A ^ has_B:
                        if has_A:
                            a_mean = np.median([s["median"] for s in stats_A])

                            partner_vals = []
                            for sA in stats_A:
                                for start, end in candidate_B_regions_after_A(
                                    L, sA["start"], sA["end"], min_gap=3, max_gap=5
                                ):
                                    partner_vals.append(
                                        np.mean(self.raw_attr[i][start:end])
                                    )
                            b_expected = np.max(partner_vals) if partner_vals else 0.0
                            sat = (a_mean < 0)

                        else:
                            b_mean = np.median([s["median"] for s in stats_B])

                            partner_vals = [];
                            for sB in stats_B:
                                for start, end in candidate_A_regions_before_B(
                                    L, sB["start"], sB["end"], min_gap=3, max_gap=5
                                ):
                                    partner_vals.append(
                                        np.mean(self.raw_attr[i][start:end])
                                    )
                            a_expected = np.max(partner_vals) if partner_vals else 0.0

                            sat = (b_mean < 0)
                    else:
                        sat = False
                else:
                    if (has_A and has_B) or (not has_A and not has_B):
                        if has_A or has_B:
                            sat = all(s["mean"] < 0 for s in (stats_A + stats_B))
                        else:
                            sat = True
                    else:
                        sat = False

            elif dataset_type == "COUNT_3":
                if is_pos:
                    sat = len(stats_A) >= 3 and all(s["mean"] > 0 for s in stats_A) and (bg_median < 0)
                else:
                    sat = (len(stats_A) < 3) and all(s["mean"] > 0 for s in stats_A) and (bg_median < 0) 

            elif dataset_type == "NIMPLY":
                if is_pos:
                    a_mean = np.mean([s["mean"] for s in stats_A])
                    partner_vals = []
                    for sA in stats_A:
                        for start, end in candidate_B_regions_after_A(
                                L, sA["start"], sA["end"], min_gap=3, max_gap=5
                            ):
                                partner_vals.append(
                                    np.mean(self.raw_attr[i][start:end])
                                )
                    b_expected = np.max(partner_vals) if partner_vals else 0.0
                    sat = (a_mean > 0) and (b_expected > 0) and (not has_B)
                else:
                    partner_vals = []
                    if (not has_A) and (not has_B):
                        sat = True
                    elif (has_A) and (has_B):
                        a_mean = np.mean([s["mean"] for s in stats_A])
                        for sA in stats_A:
                            for start, end in candidate_B_regions_after_A(
                                L, sA["start"], sA["end"], min_gap=3, max_gap=5
                                ):
                                    partner_vals.append(
                                        np.mean(self.raw_attr[i][start:end])
                                    )
                        b_expected = np.max(partner_vals) if partner_vals else 0.0
                        sat = (a_mean > 0) and (b_expected < 0) and all(s["mean"] < 0 for s in stats_B)
                    elif has_B:
                        sat =  all(s["mean"] < 0 for s in stats_B)
                    else:
                        sat = True
            else:
                raise ValueError(f"Unknown dataset_type: {dataset_type}")

            if is_pos:
                pos_flags.append(int(sat))
            else:
                neg_flags.append(int(sat))

        pos_gs = float(np.mean(pos_flags)) if pos_flags else np.nan
        neg_gs = float(np.mean(neg_flags)) if neg_flags else np.nan

        return {
            "pos_gs": pos_gs,
            "neg_gs": neg_gs,
        }

def evaluate_run(run_dir: str, dataset_type: str, positive_labels: str = "c1"):
    run_dir = run_dir.rstrip("/")
    meta_path = os.path.join(run_dir, "meta.json")
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(meta_path) or not os.path.exists(metrics_path):
        raise FileNotFoundError("Expected meta.json and metrics.json in run_dir")

    meta = json.load(open(meta_path))
    metrics = json.load(open(metrics_path))
    data_root = meta["data_root"]
    if not data_root.endswith("/"):
        data_root = data_root + "/"

    seqs, labels, ann = load_split(data_root, "test")
    causal_mask, dummy_mask = create_masks(seqs, ann)

    rows = []
    for item in metrics.get("attr_index", []):
        key = item["key"]
        method = item.get("method", "")
        baseline = item.get("baseline", "none")
        relpath = item["path"]
        fpath = os.path.join(run_dir, relpath)

        arr = np.load(fpath)["arr_0"]

        if arr.ndim == 3:
            arr = np.sum(arr, axis=1)

        bench = XAIBenchmark(arr, causal_mask, dummy_mask, ann)

        row = {
            "run_id": meta.get("run_id", os.path.basename(run_dir)),
            "grammar_id": meta.get("grammar_id", ""),
            "model": meta.get("model", ""),
            "seed": meta.get("seed", None),

            "key": key,
            "method": method,
            "baseline": baseline,

            "AUPRC": bench.eval_localization(),
            "TopK": bench.eval_precision_at_k(),
            "CRS": bench.eval_causal_relevance_score(),
            "DRS": bench.eval_dummy_relevance_score(),
        }
        row.update(
            bench.eval_grammar_satisfiability(
                seqs=seqs,
                labels=labels,
                dataset_type=dataset_type,
                positive_labels=positive_labels,
            )
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(run_dir, "per_method_metrics.csv")
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)
    return df

def bootstrap_mean_ci(x, n_boot=2000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(x, size=len(x), replace=True)
        boots.append(np.mean(samp))
    boots = np.sort(np.asarray(boots))
    lo = boots[int((alpha/2) * len(boots))]
    hi = boots[int((1 - alpha/2) * len(boots)) - 1]
    return float(np.mean(x)), float(lo), float(hi)

def aggregate_runs(runs_dir: str, dataset_type: str, out_csv: str,
                   n_boot=2000, seed=0):
    run_meta_paths = glob.glob(os.path.join(runs_dir, "**", "meta.json"), recursive=True)
    for mp in sorted(run_meta_paths):
        rd = os.path.dirname(mp)
        pm = os.path.join(rd, "per_method_metrics.csv")
        if not os.path.exists(pm):
            try:
                evaluate_run(rd, dataset_type=dataset_type)
            except Exception as e:
                print("Skip (eval failed):", rd, "->", e)
    metric_paths = glob.glob(os.path.join(runs_dir, "**", "per_method_metrics.csv"), recursive=True)
    if not metric_paths:
        raise FileNotFoundError("No per_method_metrics.csv found under runs_dir")
    df = pd.concat([pd.read_csv(p) for p in metric_paths], ignore_index=True)

    group_cols = ["grammar_id", "model", "method", "baseline"]
    metrics = ["AUPRC", "TopK", "CRS", "DRS", "pos_gs", "neg_gs"]

    out_rows = []
    for keys, g in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = int(len(g))
        for m in metrics:
            mean, lo, hi = bootstrap_mean_ci(g[m].values, n_boot=n_boot, seed=seed)
            row[f"{m}_mean"] = mean
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows).sort_values(group_cols).reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)
    return out_df
        
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_run = sub.add_parser("eval-run")
    ap_run.add_argument("--run_dir", required=True)
    ap_run.add_argument("--dataset_type", required=True)
    ap_run.add_argument("--positive_labels", default="c1")

    ap_agg = sub.add_parser("aggregate")
    ap_agg.add_argument("--runs_dir", required=True)
    ap_agg.add_argument("--dataset_type", required=True)
    ap_agg.add_argument("--out_csv", required=True)
    ap_agg.add_argument("--n_boot", type=int, default=2000)
    ap_agg.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if args.cmd == "eval-run":
        evaluate_run(args.run_dir, dataset_type=args.dataset_type, positive_labels=args.positive_labels)
    elif args.cmd == "aggregate":
        aggregate_runs(args.runs_dir, dataset_type=args.dataset_type, out_csv=args.out_csv,
                       n_boot=args.n_boot, seed=args.seed)