import os
import argparse

from utils.helpers import *
from utils.embeddings import EmbeddingsbyModality
from networks.encoders import AudioEncoder, ImageEncoder, SymbolicEncoder
from utils.evaluation import knn_search
from utils.configs import *
from utils.seed import seed_everything

seed_everything(42)

def train_knn(
        dataset_name, 
        category,
        metrics, 
        eval_ks,
        ds_dir,
        fit_modalities=MODALITIES,
        encoders_config=1,
        results_dir="results"
    ):  
    
    # SETUP ------------------
    encoders_per_modality = {
        "image": ImageEncoder(model_name=ENC_CONFIGS[encoders_config]["image"]), 
        "audio": AudioEncoder(model_name=ENC_CONFIGS[encoders_config]["audio"]), 
        "score": SymbolicEncoder(model_name=ENC_CONFIGS[encoders_config]["score"])
    }
    config = {
        "network": "kNN",
        "dataset": dataset_name,
        "category": category,
        "encoder_image": encoders_per_modality["image"].name,
        "encoder_audio": encoders_per_modality["audio"].name,
        "encoder_score": encoders_per_modality["score"].name,
    }
    results_path = make_results_dir(
        os.path.join(results_dir, f"{dataset_name}-{category}", "kNN"),
        config=config
    )
    val_log_path = os.path.join(results_path, "val_results.csv")
    test_log_path = os.path.join(results_path, "test_results.csv")
    val_log_df = create_log_file(val_log_path, exp_config=config)
    test_log_df = create_log_file(test_log_path, exp_config=config)

    # DATA AND EMBEDDING LOADING --------------------
    
    embs_by_mod = EmbeddingsbyModality(
        encoders_per_modality=encoders_per_modality,
        ds_dir=ds_dir,
        ds_name=dataset_name,
        category=category,
        normalized_embeddings=False,
        smote_embeddings=False,
        oversampled_dataset=False,
        partitions=["train", "val", "test"]
    )
    # {modality: (embeddings, labels, ids)}
    
    val_embs_by_mod = embs_by_mod.get_embeddings_dict("val", padded=True)
    test_embeddings_by_mod = embs_by_mod.get_embeddings_dict("test", padded=True)
    
    for fit_modality in fit_modalities:
        
        train_arrays = embs_by_mod.get_embeddings_array("train", modalities=[fit_modality])

        # KNN TRAINING AND EVALUATION --------------------
        print(f"Training kNN classifiers with {fit_modality} modality training and evaluating on validation set...")

        for k in eval_ks:            
            per_modality_scores = knn_search(
                arrays_z_T=train_arrays,
                dict_z_Q=val_embs_by_mod,
                metrics=metrics,
                k=k,
            )

            val_log_df = log_metrics(
                val_log_df, config, per_modality_scores,
                fit_modality=fit_modality, emb_size="-",
                classifier="kNN", k=k, metrics=metrics
            )

        best_ks = eval_ks
        if "f1" in metrics:            
            best_ks, _ = find_best_k(val_log_df, fit_modality)
        update_log_file(val_log_path, val_log_df)

        # TEST THE BEST kNN CLASSIFIER --------------------
        print(f"Evaluating kNN with k={best_ks} on test set...")

        for k in best_ks:            

            per_modality_scores = knn_search(
                arrays_z_T=train_arrays,
                dict_z_Q=test_embeddings_by_mod,
                metrics=metrics,
                k=k
            )

            test_log_df = log_metrics(
                test_log_df, config, per_modality_scores,
                fit_modality=fit_modality, emb_size="-",
                classifier="kNN", k=k, metrics=metrics
            )

        update_log_file(test_log_path, test_log_df)


    update_log_file(test_log_path, test_log_df)
    print(f"Successfully trained and evaluated kNN models for dataset={dataset_name}, category={category} with {fit_modality} modality training. Results saved in {results_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run pretrained classification models on a given dataset and category.')
    # REQUIRED ARGUMENTS
    parser.add_argument('--dataset', type=str, required=True, choices=['mtd', 'triscore', 'primus'], help='Dataset name: "mtd", "triscore" or "primus".')
    parser.add_argument('--category', type=str, required=True, help='Category to classify: MTD: "composer", "ensemble", or "instruments" // TriScore: "composer", "title", or "instruments".')
    # OPTIONAL ARGUMENTS
    parser.add_argument('--encoders_config', type=int, default=1, choices=ENC_CONFIGS.keys(), help='Which encoder configuration to use for the experiment. See ENC_CONFIGS in utils/configs.py for details.')
    parser.add_argument('--fit_modalities', type=str, nargs='+', default=MODALITIES, choices=MODALITIES, help='Modalities to train kNN on (default: all).')
    parser.add_argument('--ds_dir', type=str, default=None, help='Path to the dataset directory. If not provided, the default path from DATASET_DIRS will be used.')
    parser.add_argument('--results_dir', type=str, default="results", help='Directory to save results and logs.')
    parser.add_argument('--metrics', type=str, nargs='+', default=None, choices=['f1', 'acc', 'topn_acc'], help='Metrics to evaluate and log for kNN classification. If not provided, defaults to "f1" for all categories except "title", for which it defaults to "topn_acc".')
    parser.add_argument('--ks_to_test', type=int, nargs='+', default=None, help='List of k values to test for kNN classification. If not provided, defaults to [5, 10] for "topn_acc" metric and best k based on validation for other metrics.')

    args = parser.parse_args()
    args = set_default_train_args(args)
    train_knn(
        dataset_name=args.dataset,
        category=args.category,
        ds_dir=args.ds_dir,
        encoders_config=args.encoders_config,   
        metrics=args.metrics,
        eval_ks=args.ks_to_test,
        results_dir=args.results_dir,
        fit_modalities=args.fit_modalities
    )