import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from utils.configs import K_VALUES_TOPN, MODALITIES, K_VALUES

def predict_topn(y_Q, y_search, knn, top_N):
    # Ensure NumPy array for safe indexing
    y_search = np.array(y_search)

    # Get neighbor indices from kNN
    _, indices = knn.kneighbors(y_Q, n_neighbors=top_N)

    y_pred_topn = []
    for neighbor_idxs in indices:
        neighbor_labels = y_search[neighbor_idxs]

        # Count frequency of each label
        counts = Counter(neighbor_labels)

        sorted_labels = [y_ for y_, _ in sorted(counts.items(),
                                                      key=lambda x: (-x[1], x[0]))]

        # Keep only the top N
        y_pred_topn.append(sorted_labels[:top_N])

    return np.array(y_pred_topn, dtype=object)

def topn_score(y_Q, y_pred_topn):
    return float(np.mean([y_Q[i] in y_pred_topn[i] for i in range(len(y_pred_topn))]))


def compute_scores(y_true, metrics, y_pred=None, y_topn=None):
    scores = {}
    for metric in metrics:
        if metric == "f1" and y_pred is not None:
            scores["f1"] = f1_score(y_true, y_pred, average="weighted") 
        elif metric == "acc" and y_pred is not None:
            scores["acc"] = accuracy_score(y_true, y_pred)
        elif metric == "topn_acc" and y_topn is not None:
            scores["topn_acc"] = topn_score(y_true, y_topn)
        else:
            scores[metric] = None
    return scores

def evaluate_mlp(
    val_embs_by_mod,
    mlp_model
):
    per_mod_scores = {}
    for current_modality in val_embs_by_mod.keys():           
        z_Q, y_Q, _ = val_embs_by_mod[current_modality]
        y_pred = mlp_model.predict(z_Q)    
        per_mod_scores[current_modality] = compute_scores(
            y_Q, ["f1"], y_pred=y_pred
        )

    return per_mod_scores


def knn_search(
    arrays_z_T,
    dict_z_Q,
    metrics,
    k,
    mlp=None,
):
    """
    Evaluate a model across all modalities in query_data.

    Args:
        arrays_T: dict with "embeddings" and "labels" as np.arrays for the search space
        dict_z_Q: dict of {modality: (embeddings, labels, ids)} for the query space
        metrics:       list of metric names, e.g. ["f1", "acc", "topn_acc"]
        train_labels:  required for topn_acc
        k:             required for topn_acc
        mlp:     optional MLP model to transform embeddings before kNN evaluation; if None, use raw embeddings

    Returns:
        {modality: {metric: float}}
    """

    # search space
    if mlp is not None:
        h_T = mlp.get_embeddings(arrays_z_T["embeddings"])
        y_T = arrays_z_T["labels"]
    else:
        # baseline case
        h_T = arrays_z_T["embeddings"]
        y_T = arrays_z_T["labels"]

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(h_T, y_T)
    y_true = y_T

    # metrics
    compute_cls_metrics = any(m in metrics for m in ("f1", "acc")) and (k in K_VALUES)
    compute_id_metrics = ("topn_acc" in metrics) and (k in K_VALUES_TOPN)

    # query each modality and compute metrics
    per_mod_scores = {}
    for modality, (z_Q, y_Q, _) in dict_z_Q.items():
        if mlp is not None:
            h_Q = mlp.get_embeddings(z_Q)
        else:
            h_Q = z_Q   # baseline: kNN on raw embeddings

        y_pred  = knn.predict(h_Q) if compute_cls_metrics else None
        y_topn  = predict_topn(h_Q, y_true, knn, k) if compute_id_metrics else None

        per_mod_scores[modality] = compute_scores(
            y_true=y_Q, 
            y_pred=y_pred,
            y_topn=y_topn,
            metrics=metrics
        )

    return per_mod_scores


def knn_search_cross_modal(
    obj_embs_by_mod,
    fit_scenario,
    dict_Z_Q,
    mlp,
    metrics,
    k
):
    """
    Evaluate a kNN classifier trained on one modality/scenario on all other modalities.

    Args:
    - obj_embs_by_mod: the EmbeddingsbyModality object, used to query for the appropriate train/test embeddings for each scenario
    - fit_scenario: one of "image", "audio", "score", "combined", "all"
    - dict_Z_Q: dict of {modality: (embeddings, labels, ids)} for the query space
    - mlp: the MLP model trained on the fit_scenario modality(ies), used to extract embeddings for kNN evaluation
    - metrics: list of metric names to compute, e.g. ["f1", "acc", "topn_acc"]
    - k: the k value to use for kNN evaluation
    """

    needs_pred = any(m in metrics for m in ("f1", "acc"))
    needs_topn = ("topn_acc" in metrics) and (k in K_VALUES_TOPN)

    per_mod_scores = {m: {} for m in MODALITIES}
    for query_modality in MODALITIES:
        if query_modality != fit_scenario:
            z_Q, y_Q, _ = dict_Z_Q[query_modality]
            h_Q = mlp.get_embeddings(z_Q)

            # the search space for each query modality is 
            # - combinations excluding the query modality
            # - all modalities
            if fit_scenario == "combined":
                fit_mods = [m for m in MODALITIES if m != query_modality]
            elif fit_scenario == "all":
                fit_mods = MODALITIES
            else:
                fit_mods = [fit_scenario]

            search_arrays = obj_embs_by_mod.get_embeddings_array("train", modalities=fit_mods)
            h_T_local = mlp.get_embeddings(search_arrays["embeddings"])

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(h_T_local, search_arrays["labels"])

            y_pred = knn.predict(h_Q) if needs_pred else None
            y_topn = predict_topn(h_Q, search_arrays["labels"], knn, k) if needs_topn else None

            per_mod_scores[query_modality] = compute_scores(y_Q, metrics, y_pred=y_pred, y_topn=y_topn)
        else:
            per_mod_scores[query_modality] = {metric: None for metric in metrics}

    return per_mod_scores


