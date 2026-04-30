import os
import argparse
import json
import torch

from networks.embedding_networks import MLP
from utils.embeddings import EmbeddingsbyModality
from networks.encoders import AudioEncoder, ImageEncoder, SymbolicEncoder
from utils.analysis import run_embeddings_analysis
from utils.evaluation import evaluate_mlp, knn_search_cross_modal
from utils.helpers import *
from utils.configs import *
from utils.seed import seed_everything

seed_everything(42)

def test_mlp(checkpoints_dir, ds_dir, hidden_dim="512", metrics=["f1"], ks_to_test=[1, 11]):
    
    # PATHS AND CONFIGURATION ------------------        

    with open(os.path.join(checkpoints_dir, "train_config.json"), 'r') as f:
        config = json.load(f)
        config["dataset"] = "triscore" if config["dataset"] == "muscuts" else config["dataset"]
        dataset_name = config['dataset']
        category = config['category']
        balanced_dataset = config.get("oversampled_dataset", False)
        normalized_embeddings = config.get("normalized_embeddings", False)
        smote_embeddings = config.get("smote_embeddings", False)
        ds_dir = DATASETS[dataset_name]["default_dir"] if ds_dir is None else ds_dir
        if metrics is None: metrics = ["f1"] if category != "title" else ["topn_acc"]

        assert hidden_dim in config['hidden_dims'], f"MLP to test (hidden_dim={hidden_dim}) not in trained models {config['hidden_dims']}"

        log_df = create_log_file(os.path.join(checkpoints_dir, "test_results.csv"), exp_config=config)
    try:
        with open(os.path.join(checkpoints_dir, f"{hidden_dim}_mlp.pth"), 'rb') as f:
            checkpoint = torch.load(f, weights_only=False)
            trained_mlp_model = MLP(**checkpoint['args'])
            trained_mlp_model.load_state_dict(checkpoint['model_state_dict'])
    except Exception:
        raise ValueError(f"Could not load MLP model from {checkpoints_dir} with hidden_dim={hidden_dim}")
    print(f"MLP Test for category={category}")

    
    # DATA AND EMBEDDING LOADING --------------------
    encoders_per_modality = {
        "image": ImageEncoder(model_name=config["encoder_image"]), 
        "audio": AudioEncoder(model_name=config["encoder_audio"]), 
        "score": SymbolicEncoder(model_name=config["encoder_score"])
    }
    embs_by_mod = EmbeddingsbyModality(
        encoders_per_modality=encoders_per_modality,
        ds_dir=ds_dir,
        ds_name=dataset_name,
        category=category,
        normalized_embeddings=normalized_embeddings,
        oversampled_dataset=balanced_dataset,
        smote_embeddings=smote_embeddings,
        partitions=["train", "test"]
    )
    # {modality: (embeddings, labels, ids)}
    train_arrays = embs_by_mod.get_embeddings_array("train")
    test_embs_by_mod = embs_by_mod.get_embeddings_dict("test", padded=True)

    # EMBEDDING ANALYSIS --------------------

    run_embeddings_analysis(
        train_arrays=train_arrays,
        test_arrays=embs_by_mod.get_embeddings_array("test"),
        trained_mlp_model=trained_mlp_model,
        results_path=checkpoints_dir,
        net_name=f"MLP-{hidden_dim}",
        category=category,
    )

    # EVAL MLP CLASSIFICATION --------------------
    print(f"Evaluating MLP classification on test set for dataset={dataset_name}, category={category}...")
    mlp_scores = evaluate_mlp(test_embs_by_mod, trained_mlp_model)
    log_df = log_metrics(
        log_df, config, mlp_scores,
        fit_modality="all", emb_size=hidden_dim,
        classifier="MLP", k="-", metrics=["f1"]
    )

    # EVAL MLP + KNN -------------------
    for fit_scenario in MODALITIES + ["combined", "all"]:
        for k in ks_to_test:
            cross_modal_scores = knn_search_cross_modal(
                obj_embs_by_mod=embs_by_mod,
                fit_scenario=fit_scenario,
                dict_Z_Q=test_embs_by_mod,
                mlp=trained_mlp_model,
                metrics=metrics,
                k=k
            )
            log_df = log_metrics(
                log_df, config, cross_modal_scores,
                fit_modality=fit_scenario, emb_size=hidden_dim,
                classifier="kNN", k=k, metrics=metrics
            )
    
    update_log_file(os.path.join(checkpoints_dir, "test_results.csv"), log_df)
    print(f"Successfully tested MLP models for dataset={dataset_name}, category={category}. Results saved in {checkpoints_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run pretrained models with Triamese or MLP classification on a given dataset and category.')
    # REQUIRED ARGUMENTS
    parser.add_argument('--checkpoints_dir', type=str, required=True, help='Path to the directory containing model checkpoints.')
    parser.add_argument('--ks_to_test', type=int, nargs='+', default=None, required=True, help='List of k values to test for kNN classification.')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of the MLP model to test.')
    parser.add_argument('--ds_dir', type=str, default=None, help='Path to the dataset directory.')
    parser.add_argument('--metrics', type=str, nargs='+', default=None, choices=['f1', 'acc', 'topn_acc'], help='Metrics to evaluate and log for kNN classification. If not provided, defaults to "f1" for all categories except "title", for which it defaults to "topn_acc".')

    
    args = parser.parse_args()
    test_mlp(
        checkpoints_dir=args.checkpoints_dir,
        hidden_dim=args.hidden_dim,
        ds_dir=args.ds_dir,
        metrics=args.metrics,
        ks_to_test=args.ks_to_test
    )