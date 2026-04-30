#!/bin/bash

CATEGORIES=("composer" "title" "instruments")

# Iterate over categories for both KNN and MLP
for category in "${CATEGORIES[@]}"; do
    # Run KNN model
    python train_knn.py --dataset "triscore" --category "$category" --results_dir "results_baseline_knn"
    # BALANCING BATCH EXPERIMENT
    for balanced_flag in "" "--batch_modality_balanced"; do
        python train_mlp.py --dataset "triscore" --category "$category" $balanced_flag --results_dir "results_mlp_balanced"
    done
    
    # NORMALIZATION EXPERIMENT
    for norm_flag in "" "--normalized_embeddings"; do
        python train_mlp.py --dataset "triscore" --category "$category" $norm_flag --batch_modality_balanced --results_dir "results_mlp_normalization"
    done
    
    # IMBALANCED TECHNIQUES EXPERIMENT
    for imb_flag in "" "--oversampled_dataset" "--weighted_loss"; do
        if [ "$imb_flag" = "--weighted_loss" ]; then
            for beta in "0.9" "0.99" "0.999" "0.9999"; do
                python train_mlp.py --dataset "triscore" --category "$category" --normalized_embeddings --batch_modality_balanced $imb_flag --beta $beta --results_dir "results_mlp_imbalanced_techniques"
            done
        else
            python train_mlp.py --dataset "triscore" --category "$category" --normalized_embeddings --batch_modality_balanced $imb_flag --results_dir "results_mlp_imbalanced_techniques"
        fi
    done
done
