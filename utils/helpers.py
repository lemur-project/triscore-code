import os
import json
import warnings
import pandas as pd
import numpy as np
from utils.configs import *

# results utilities ----------------

def make_results_dir(base_path, config, allow_existing=True):
    """
    Create a unique directory by appending _v1, _v2, ... if needed.
    If a train_config.json exists and contains the same config, use that directory.
    Otherwise, add vN until either finding the correct config or creating a new vN.
    Returns the path of the directory.
    """

    print("EXPERIMENT CONFIGURATION ==============")
    [print(f"\t{k}: {v}") for k, v in config.items()]
    print("=======================================")
    
    def config_matches(config_path, config):
        try:
            with open(config_path, "r") as f:
                existing_log = json.load(f)
            return existing_log == config
        except Exception:
            return False

    path = base_path
    suffix = 1
    found_match_or_new = False
    while not found_match_or_new:
        config_path = os.path.join(path, "train_config.json")
        if os.path.exists(path):
            if os.path.exists(config_path) and config_matches(config_path, config) and not allow_existing:
                raise FileExistsError(f"Directory {path} already exists with the same config. To allow using existing directories, set allow_existing=True.")              
            else:
                path = f"{base_path}_v{suffix}"
                suffix += 1
        else:
            os.makedirs(path)
            with open(os.path.join(path, f"train_config.json"), 'w') as f:
                json.dump(config, f, indent=4)
            found_match_or_new = True

    return path


def create_log_file(file_path, exp_config):
    """Create a CSV file with experiment configuration and results headers."""
    headers = list(exp_config.keys()) + [
        "search_space", "embedding_dimension",
        "classifier", "k", 
        "metric", "audio", "image", "score"
    ]
    df = pd.DataFrame(columns=headers)
    df.to_csv(file_path, index=False, sep="\t")
    
    [print(h, end="\t") for h in headers]
    print()
    return df

def log_results(df, exp_config, fit_modality, emb_size, classifier, k, metric, log_metrics):
    """Append results to the CSV log file."""
    row_data = {
        **exp_config,
        "search_space": fit_modality,
        "embedding_dimension": emb_size,
        "classifier": classifier,
        "k": k,
        "metric": metric,
        "audio": log_metrics["audio"],
        "image": log_metrics["image"],
        "score": log_metrics["score"]
    }
    df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
    [print(v, end="\t") for v in row_data.values()]
    print()
    return df


def log_metrics(
    df, exp_config, per_modality_scores, fit_modality, emb_size,
    classifier, k, metrics
):
    """
    Append one CSV row per metric to df.
    Modalities absent from per_modality_scores (or lacking a given metric)
    are filled with "-".
    """
    for metric in metrics:
        if not any(metric in scores for scores in per_modality_scores.values()):
            continue

        log_metrics = {
            mod: (
                str(round(per_modality_scores[mod][metric] * 100, 4))
                if mod in per_modality_scores and metric in per_modality_scores[mod] and per_modality_scores[mod][metric] is not None
                else "-"
            )
            for mod in MODALITIES
        }
        if not all(v == "-" for v in log_metrics.values()):
            df = log_results(
                df=df,
                exp_config=exp_config,
                fit_modality=fit_modality,
                emb_size=emb_size,
                classifier=classifier,
                k=k,
                metric=metric,
                log_metrics=log_metrics,
            )
    return df


def find_best_k(val_df, fit_modality, hidden_dim=None):
    f1_rows = val_df[
        (val_df["metric"] == "f1")
        & (val_df["search_space"] == fit_modality)
    ].copy()
    if hidden_dim is not None:
        f1_rows = f1_rows[f1_rows["embedding_dimension"] == hidden_dim]
        
    avg_f1s = f1_rows.apply(
        lambda row: np.mean([float(row[mod]) for mod in MODALITIES if row[mod] != "-" and pd.notna(row[mod])]),axis=1,
    )
    f1_rows.loc[:, "mean_f1"] = avg_f1s
    max_f1 = avg_f1s.max()
    best_ks = sorted(f1_rows.loc[avg_f1s == max_f1, "k"].astype(int).unique().tolist())
   
    return best_ks, max_f1


def update_log_file(file_path, df):
    """Update the CSV log file with the provided DataFrame."""
    df.to_csv(file_path, index=False, sep="\t")


def set_default_train_args(args):
    if args.metrics is None:
        if args.category != "title":
            args.metrics = ["f1"]
        else:
            args.metrics = ["topn_acc"]

    if args.ds_dir is None:
        args.ds_dir = DATASETS[args.dataset]["default_dir"]
        print(f"Using default dataset directory: {args.ds_dir}, use ds_dir argument to override.")
    try:
        assert sum([args.weighted_loss, args.oversampled_dataset]) <= 1, \
            "Choose only one class imbalance handling method: weighted_loss, oversampled_dataset, or smote_embeddings."
        if args.beta is None and args.weighted_loss:
            args.beta = 0.9
    except AttributeError:
        pass
    
    if args.ks_to_test is None:
        has_standard = any(m in args.metrics for m in ("f1", "acc"))
        has_topn     = "topn_acc" in args.metrics
        if has_standard and has_topn:
            args.ks_to_test = sorted(set(K_VALUES) | set(K_VALUES_TOPN))
        elif has_topn:
            args.ks_to_test = list(K_VALUES_TOPN)
        else:
            args.ks_to_test = list(K_VALUES)

    return args


def get_batch_mod_dict(batch_size, modalities):
    batch_modality_size = batch_size // len(modalities)
    batch_mod_dict = {modality: batch_modality_size for modality in modalities}
    batch_mod_dict[modalities[0]] += batch_size - sum(batch_mod_dict.values())

    return batch_mod_dict

def normalized_windows_path(path):
    absolute_path = os.path.abspath(path)
    absolute_path = absolute_path.replace("/", "\\")
    return "\\\\?\\" + absolute_path
