import os
import argparse
from torch.utils.data import DataLoader

from data_utils.embeddings_dataset import EmbeddingMultiModalDataset
from networks.embedding_networks import MLP
from utils.embeddings import EmbeddingsbyModality
from networks.encoders import AudioEncoder, ImageEncoder, SymbolicEncoder
from utils.evaluation import knn_search, evaluate_mlp
from utils.helpers import *
from utils.configs import *
from utils.seed import seed_everything

seed_everything(42)

def train_mlp(
    dataset_name, 
    category, 
    ds_dir=None, 
    epochs=100, 
    batch_size=16,
    beta=0.99,
    metrics=["f1"],
    eval_ks=K_VALUES,
    hidden_dims=HIDDEN_DIMENSIONS,
    oversampled_dataset=False, 
    weighted_loss=False, 
    normalized_embeddings=False, 
    batch_modality_balanced=False,
    encoders_config=1,
    results_dir="results",
):

    # MODELS AND CONFIGURATION --------------------
    encoders_per_modality = {
        "image": ImageEncoder(model_name=ENC_CONFIGS[encoders_config]["image"]), 
        "audio": AudioEncoder(model_name=ENC_CONFIGS[encoders_config]["audio"]), 
        "score": SymbolicEncoder(model_name=ENC_CONFIGS[encoders_config]["score"])
    }
    config = {
        "network": "MLP",
        "hidden_dims": hidden_dims,
        "dataset": dataset_name,
        "category": category,
        "normalized_embeddings": normalized_embeddings,
        "oversampled_dataset": oversampled_dataset,
        "weighted_loss": weighted_loss,
        "beta": beta,
        "batch_modality_balanced": batch_modality_balanced,
        "batch_size": batch_size,
        "encoder_image": encoders_per_modality["image"].name,
        "encoder_audio": encoders_per_modality["audio"].name,
        "encoder_score": encoders_per_modality["score"].name,
    }
    results_path = make_results_dir(
        os.path.join(results_dir, f"{dataset_name}-{category}", "MLP"),
        config= config
    )
    log_df = create_log_file(os.path.join(results_path, "val_results.csv"), exp_config=config)

    # DATA AND EMBEDDING LOADING --------------------
    
    embs_by_mod = EmbeddingsbyModality(
        encoders_per_modality=encoders_per_modality,
        ds_dir=ds_dir,
        ds_name=dataset_name,
        category=category,
        normalized_embeddings=normalized_embeddings,
        oversampled_dataset=oversampled_dataset,
        partitions=["train", "val"],
    )
    # {modality: (embeddings, labels, ids)}
    train_embs_by_mod = embs_by_mod.get_embeddings_dict("train", padded=True)
    val_embs_by_mod = embs_by_mod.get_embeddings_dict("val", padded=True)
    train_arrays = embs_by_mod.get_embeddings_array("train")

    batch_mod_dict = get_batch_mod_dict(batch_size, MODALITIES) if batch_modality_balanced else None
    train_dataset = EmbeddingMultiModalDataset(train_embs_by_mod, batch_mod_dict)
    class_weights = train_dataset.get_weight_classes(method="effective", beta=beta) if weighted_loss else None
    train_emb_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_emb_loader = DataLoader(
        EmbeddingMultiModalDataset(val_embs_by_mod),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    #  TRAIN MLP CLASSIFIER  AND EVALUATE WITH kNN -------------------

    for hidden_dim in hidden_dims:

        mlp_model = MLP(
            input_dim=embs_by_mod.max_embedding_dim, 
            hidden_dim=hidden_dim,
            num_classes=embs_by_mod.num_classes,
            loss_weights=class_weights
        )
        mlp_model.fit(
            train_emb_loader, 
            val_emb_loader, 
            epochs=epochs, 
            early_stopping=True if oversampled_dataset else False, 
            patience=20
        )
        per_modality_scores = evaluate_mlp(val_embs_by_mod, mlp_model)
        log_df = log_metrics(
            log_df, config, per_modality_scores,
            fit_modality="all", emb_size=hidden_dim,
            classifier="MLP", k="-", metrics=metrics
        )
        
        mlp_model.save_model(os.path.join(results_path, f"{hidden_dim}_mlp.pth"))

        for k in eval_ks:            
            per_modality_scores = knn_search(
                dict_z_Q=val_embs_by_mod,
                arrays_z_T=train_arrays,
                metrics=metrics, 
                k=k,
                mlp=mlp_model
            )
            log_df = log_metrics(
                log_df, config, per_modality_scores,
                fit_modality="all", emb_size=hidden_dim,
                classifier="kNN", k=k, metrics=metrics
            )

        if "f1" in metrics:
            print(f"Best k during validation for MLP with hidden_dim={hidden_dim}: {find_best_k(log_df, 'all', hidden_dim)}")

    update_log_file(os.path.join(results_path, "val_results.csv"), log_df)
    print(f"Successfully trained and evaluated MLP models for dataset={dataset_name}, category={category}. Results saved in {results_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run pretrained models with MLP classification on a given dataset and category.')
    # REQUIRED ARGUMENTS
    parser.add_argument('--dataset', type=str, required=True, choices=['mtd', 'triscore', 'primus'], help='Dataset name: "mtd", "triscore" or "primus".')
    parser.add_argument('--category', type=str, required=True,help='Category to classify: MTD: "composer", "ensemble", or "instruments" // MUSCUTS: "composer", "title", or "instruments".')
    # OPTIONAL ARGUMENTS
    parser.add_argument('--encoders_config', type=int, default=1, choices=ENC_CONFIGS.keys(), help='Which encoder configuration to use for the experiment. See ENC_CONFIGS in utils/configs.py for details.')
    parser.add_argument('--epochs', type=int, default=100,help='Number of epochs to train the MLP.')
    parser.add_argument('--batch_size', type=int, default=16,help='Batch size for training.')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=HIDDEN_DIMENSIONS, help='List of hidden dimensions for MLP models to train and test.')
    parser.add_argument('--oversampled_dataset', action='store_true',help='Use oversampled dataset.')
    parser.add_argument('--batch_modality_balanced', action='store_true',help='Use balanced batch sizes for each modality when training MLP (only if using oversampled dataset).')
    parser.add_argument('--weighted_loss', action='store_true',help='Use weighted loss to handle class imbalance (only if NOT using balanced dataset).')
    parser.add_argument('--normalized_embeddings', action='store_true',help='Normalize embeddings before training.')
    parser.add_argument('--results_dir', type=str, default="results", help='Directory to save results and models.')
    # these need default setters
    parser.add_argument('--ds_dir', type=str, default=None, help='Path to the dataset directory. If not provided, the default directory for the dataset will be used.')
    parser.add_argument('--beta', type=float, default=None, help='Beta value for effective number of samples when using weighted loss.')
    parser.add_argument('--metrics', type=str, nargs='+', default=['f1'], choices=['f1', 'acc', 'topn_acc'], help='Metrics to evaluate and log for kNN classification.')
    parser.add_argument('--ks_to_test', type=int, nargs='+', default=None, help='List of k values to test for kNN classification.')
    args = parser.parse_args()
    args = set_default_train_args(args)

    train_mlp(
        dataset_name=args.dataset,
        category=args.category,
        epochs=args.epochs,
        batch_size=args.batch_size,
        oversampled_dataset=args.oversampled_dataset,
        batch_modality_balanced=args.batch_modality_balanced,
        weighted_loss=args.weighted_loss,
        normalized_embeddings=args.normalized_embeddings,
        ds_dir=args.ds_dir,
        beta=args.beta,
        metrics=args.metrics,
        eval_ks= args.ks_to_test,
        encoders_config=args.encoders_config,
        results_dir=args.results_dir,
    )
