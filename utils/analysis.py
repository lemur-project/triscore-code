import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict, Counter
import os
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualize_embeddings(
        embs, lbs, mods,
        method='tsne', perplexity=30, n_components=2, random_state=42,
        figsize=(12, 8), alpha=0.7, s=50, title=None,
        iterative=False, fig=None, ax=None
    ):

    # Convert inputs to numpy arrays if needed
    embs = np.array(embs)
    lbs = np.array(lbs)

    mods_dict = {mod: idx for idx, mod in enumerate(set(mods))}
    mods = np.array([mods_dict[mod] for mod in mods])

    if method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity,
                       random_state=random_state, max_iter=1000)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")

    embs_2d = reducer.fit_transform(embs)

    # Figure & axes management for iterative mode
    if iterative:
        # Create once if missing
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.cla()   # clear instead of creating a new figure
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # Plotting
    mod_names = {idx: mod for mod, idx in mods_dict.items()}
    mod_markers = {0: "o", 1: "s", 2: "^"}
    mod_mark_names = {0: 'Circle', 1: 'Square', 2: 'Triangle'}

    unique_lbs = np.unique(lbs)
    unique_mods = np.unique(mods)

    # Color map
    if len(unique_lbs) <= 10:
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_lbs)))
    elif len(unique_lbs) <= 20:
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_lbs)))
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, len(unique_lbs)))

    label_color_map = dict(zip(unique_lbs, colors))

    # Scatter plots
    for label in unique_lbs:
        for mod in unique_mods:
            mask = (lbs == label) & (mods == mod)
            if np.any(mask):
                ax.scatter(
                    embs_2d[mask, 0],
                    embs_2d[mask, 1],
                    c=[label_color_map[label]],
                    marker=mod_markers[mod],
                    s=s,
                    alpha=alpha,
                    edgecolors='black',
                    linewidth=0.5
                )

    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')

    if title is None:
        modality_names_str = ', '.join(
            [f'{mod_names[m]}-{mod_mark_names[m]}' for m in unique_mods]
        )
        title = f'Embedding Visualization ({method.upper()})\nColors = Labels, Markers = Modalities ({modality_names_str})'
    ax.set_title(title)
    plt.tight_layout()

    return fig, ax, embs_2d


def top_n_fragment_level(q_embs, q_mods, q_ids, category, k_values=[1, 5, 10, 15, 20]):
    """
    Compute top-N cross-modal percentage accuracy for fragment-level retrieval
    """

    colors_per_mods = {
        "audio": "#f2b6bc",
        "image": "#b2dfbc",
        "score": "#abdee5"
    }

    results = {}
    for k in k_values:
        scores = defaultdict(list)  # modality -> list of hits (1/0)

        for modality in sorted(set(q_mods)):
            # Build cross-modal gallery (exclude same modality)
            search_indices = [i for i, m in enumerate(q_mods) if m != modality]
            search_embs = np.vstack([q_embs[i] for i in search_indices])

            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(search_embs)

            # Query only this modality
            query_indices = [i for i, m in enumerate(q_mods) if m == modality]
            query_embs = np.vstack([q_embs[i] for i in query_indices])

            _, neighbors = knn.kneighbors(query_embs)

            for qi, q_idx in enumerate(query_indices):
                for ni in neighbors[qi]:
                    true_gallery_idx = search_indices[ni]
                    hit = int(q_ids[true_gallery_idx] == q_ids[q_idx])
                    scores[modality].append(hit)

        # Compute percentage accuracy for this k
        results[k] = {mod: (np.mean(hits) * 100) if hits else 0.0 for mod, hits in scores.items()}

    # --- Bar plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    modalities = sorted(results[k_values[0]].keys())
    bar_width = 0.15
    x = np.arange(len(k_values))

    for i, modality in enumerate(modalities):
        accs = [results[k][modality] for k in k_values]
        color = colors_per_mods.get(modality, "#777777")  # fallback color if modality not found
        ax.bar(x + i * bar_width, accs, width=bar_width, label=modality,
               color=color, edgecolor='black', linewidth=1.2)
        ax.set_xticks(x + bar_width * (len(modalities) - 1) / 2)
        ax.set_xticklabels([str(k) for k in k_values], fontsize=17)
        ax.set_xlabel('Number of Neighbors (k)', fontsize=17)
        ax.set_ylabel('Accuracy (%)', fontsize=17)
        ax.set_title(f"Top-N Accuracy for Cross-Modal Fragment Retrieval: {category}", fontsize=20, fontweight='bold')
        ax.legend(title='Modality', fontsize=14, title_fontsize=15)
        ax.tick_params(axis='y', labelsize=17)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        visualization = fig
        plt.close(fig)

    return results, visualization


def modality_neighbors(search_embs, search_mods, q_embs, q_mods, ks=[5, 10, 15, 20]):
    results = {}
    unique_mods = list(set(list(search_mods) + list(q_mods)))
    
    # For each k value, analyze neighbor modalities
    for k in ks:
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(search_embs)
        
        # Find neighbors for test samples
        distances, indices = knn.kneighbors(q_embs)
        
        k_results = {
            'modality_distributions': defaultdict(lambda: defaultdict(list)),
            'cross_modality_percentages': defaultdict(dict),
            'same_modality_percentages': defaultdict(float),
            'neighbor_distances': defaultdict(list)
        }
        
        # Analyze each test sample
        for mi, q_mod in enumerate(q_mods):
            neighbor_mods = [search_mods[idx] for idx in indices[mi]]
            neighbor_distances_i = distances[mi]
            
            # Count modality distribution
            mod_counts = Counter(neighbor_mods)
            total_neighs = len(neighbor_mods)
            
            # Store raw counts and percentages
            for modality in unique_mods:
                count = mod_counts.get(modality, 0)
                percentage = (count / total_neighs) * 100
                k_results['modality_distributions'][q_mod][modality].append(percentage)
            
            # Calculate same modality percentage
            same_mod_count = mod_counts.get(q_mod, 0)
            same_mod_pct = (same_mod_count / total_neighs) * 100
            k_results['same_modality_percentages'][q_mod] = same_mod_pct
            
            # Store distances for analysis
            k_results['neighbor_distances'][q_mod].extend(neighbor_distances_i)
        
        # Calculate average percentages across all queries of each modality
        for query_mod in unique_mods:
            if query_mod in k_results['modality_distributions']:
                for neighbor_modality in unique_mods:
                    percentages = k_results['modality_distributions'][query_mod][neighbor_modality]
                    if percentages:
                        k_results['cross_modality_percentages'][query_mod][neighbor_modality] = np.mean(percentages)
                    else:
                        k_results['cross_modality_percentages'][query_mod][neighbor_modality] = 0.0
        
        results[f'k_{k}'] = k_results
    
    return results

def modality_neighbors_crossmodal(search_embs, search_mods, q_embs, q_mods, k_values=[5, 10, 15, 20]):

    results = {}
    unique_mods = list(set(list(search_mods) + list(q_mods)))

    for k in k_values:
        k_results = {
            'modality_distributions': defaultdict(lambda: defaultdict(list)),
            'cross_modality_percentages': defaultdict(dict),
            'same_modality_percentages': defaultdict(float),
            'neighbor_distances': defaultdict(list)
        }

        # For each query separately, exclude same-modality from training
        for i, q_mod in enumerate(q_mods):
            # Select training data excluding this modality
            mask = [m != q_mod for m in search_mods]
            h_search_sub = search_embs[mask]
            m_sub = [m for m, keep in zip(search_mods, mask) if keep]

            # Fit KNN on the reduced set
            knn = NearestNeighbors(n_neighbors=min(k, len(h_search_sub)))
            knn.fit(h_search_sub)

            distances, indices = knn.kneighbors(q_embs[i].reshape(1, -1))
            indices = indices[0]
            distances = distances[0]

            neighbor_mods = [m_sub[idx] for idx in indices]

            # Count modality distribution
            modality_counts = Counter(neighbor_mods)
            total_neighbors = len(neighbor_mods)

            for mod in unique_mods:
                count = modality_counts.get(mod, 0)
                percent = (count / total_neighbors) * 100 if total_neighbors > 0 else 0
                k_results['modality_distributions'][q_mod][mod].append(percent)

            # Same modality is always 0 (since excluded)
            k_results['same_modality_percentages'][q_mod] = 0.0

            # Store distances
            k_results['neighbor_distances'][q_mod].extend(distances)

        # Average percentages across queries of each modality
        for query_mod in unique_mods:
            if query_mod in k_results['modality_distributions']:
                for neighbor_mod in unique_mods:
                    percentages = k_results['modality_distributions'][query_mod][neighbor_mod]
                    k_results['cross_modality_percentages'][query_mod][neighbor_mod] = (
                        np.mean(percentages) if percentages else 0.0
                    )

        results[f'k_{k}'] = k_results

    return results


def plot_neighbor_modality_analysis(results, category):
    k_values = [int(k.split('_')[1]) for k in results.keys()]
    unique_mods = list(results[f'k_{k_values[0]}']['cross_modality_percentages'].keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle(f'Modality Neighbor Analysis: Percentage of Neighbors by Modality for {category}', fontsize=35, fontweight='bold')
    
    cmap = plt.cm.Blues
    norm = colors.Normalize(vmin=0, vmax=100)

    for idx, k in enumerate(k_values):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        # Create matrix for heatmap
        matrix = []
        for query_mod in unique_mods:
            row_data = []
            for neighbor_mod in unique_mods:
                pct = results[f'k_{k}']['cross_modality_percentages'][query_mod][neighbor_mod]
                row_data.append(pct)
            matrix.append(row_data)
        
        matrix = np.array(matrix)
        
        # Create heatmap
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)
        
        # Add text annotations
        for i in range(len(unique_mods)):
            for j in range(len(unique_mods)):
                value = matrix[i, j]
                rgba = cmap(norm(value))
                text_color = 'black' if (0.2 * rgba[0] + 0.7 * rgba[1] + 0.1 * rgba[2]) > 0.5 else 'white'
                ax.text(j, i, f'{value:.1f}%', 
                        ha='center', va='center', 
                        color=text_color,
                        fontsize=30,
                        fontweight='bold'
                        )   
        
        ax.set_xticks(range(len(unique_mods)))
        ax.set_yticks(range(len(unique_mods)))
        ax.set_xticklabels(unique_mods, rotation=45, fontsize=30)
        ax.set_yticklabels(unique_mods, fontsize=30)
        ax.set_xlabel('Neighbor Modality', fontsize=30, fontweight='bold')
        ax.set_ylabel('Query Modality', fontsize=30, fontweight='bold')
        ax.set_title(f'k = {k}', fontsize=35, fontweight='bold')

    fig.subplots_adjust(hspace=0.45, wspace=0.55, top=0.92, bottom=0.1, left=0.1, right=0.9)
    
    return fig

def compute_modality_gap(search_embs, search_mods):
    # get the unique modalities
    unique_mods = list(set(search_mods))
    mod_gaps = {}
    # iterate to compute all the combinations of modalities
    for i, mod_a in enumerate(unique_mods):
        for j, mod_b in enumerate(unique_mods):
            if i >= j:
                continue  # Avoid repeats and self-pairs
            emb_a = search_embs[np.array(search_mods) == mod_a]
            emb_b = search_embs[np.array(search_mods) == mod_b]
            gap = np.linalg.norm(np.mean(emb_a, axis=0) - np.mean(emb_b, axis=0))
            mod_gaps[f"{mod_a}-{mod_b}"] = gap

    # get a single modality gap by averaging all the gaps
    avg_gap =  np.mean(np.array(list(mod_gaps.values())))

    return {k: round(float(v), 2) for k, v in mod_gaps.items()}, round(float(avg_gap), 2)


def run_embeddings_analysis(train_arrays, test_arrays, trained_mlp_model, results_path, net_name, category):

    emb_analysis_dir = os.path.join(results_path, "embedding_analysis")
    os.makedirs(emb_analysis_dir, exist_ok=True)

    print(f"Running embeddings analysis for {net_name}...")
    
    z_T = train_arrays["embeddings"]
    y_T = train_arrays["labels"]
    m_T = train_arrays["modalities"]
   
    z_Q = test_arrays["embeddings"]
    m_Q = test_arrays["modalities"]
    m_V = test_arrays["identifiers"]

    # To avoid repeating the same visualization for multiple model parameters
    pre_fig_path = os.path.join(emb_analysis_dir, f"embeddings_pre_mlp.pdf")
    if not os.path.exists(pre_fig_path):
        fig_pre, _, _ = visualize_embeddings(embs=z_T, lbs=y_T, mods=m_T)
        fig_pre.savefig(pre_fig_path, format='pdf')
        plt.close(fig_pre)
        mod_gaps, avg_gap = compute_modality_gap(z_T, m_T)
        with open(os.path.join(emb_analysis_dir, f"pre_modality_gaps.json"), "w") as f:
            json.dump({
                "modality_gaps": mod_gaps,
                "average_gap": avg_gap
            }, f, indent=4)

    h_T = trained_mlp_model.get_embeddings(z_T)
    h_Q = trained_mlp_model.get_embeddings(z_Q)
    fig_post, _, _ = visualize_embeddings(embs=h_T,lbs=y_T,mods=m_T)
    fig_post.savefig(os.path.join(emb_analysis_dir, f"{net_name}_embeddings_post.pdf"), format='pdf')
    plt.close(fig_post)

    # Compute and log modality gap
    mod_gaps, avg_gap = compute_modality_gap(h_T, m_T)
    with open(os.path.join(emb_analysis_dir, f"{net_name}_modality_gaps.json"), "w") as f:
        json.dump({
            "modality_gaps": mod_gaps,
            "average_gap": avg_gap
        }, f, indent=4)

    # Run cross-modality analysis
    results = modality_neighbors(
        h_T, m_T,   
        h_Q, m_Q,
        ks=[5, 10, 15, 20]
    )
    heatmap_fig = plot_neighbor_modality_analysis(results, category=category)
    heatmap_fig.savefig(os.path.join(emb_analysis_dir, f"{net_name}_neighbor_modality_analysis.pdf"), format='pdf')
    plt.close(heatmap_fig)

    # Run exclusive cross-modality analysis
    results_excl = modality_neighbors_crossmodal(
        h_T, m_T,   
        h_Q, m_Q,
        k_values=[5, 10, 15, 20]
    )
    heatmap_fig_excl = plot_neighbor_modality_analysis(results_excl, category=category)
    heatmap_fig_excl.savefig(os.path.join(emb_analysis_dir, f"{net_name}_neighbor_modality_analysis_q_modal_exclusive.pdf"), format='pdf')
    plt.close(heatmap_fig_excl)

    # Run top-N neighbors analysis
    results, visualization = top_n_fragment_level(
        h_Q, m_Q, m_V,
        k_values=[1, 5, 10, 15, 20],
        category=category
    )
    visualization.savefig(os.path.join(emb_analysis_dir, f"{net_name}_top_n_neighbors_analysis.pdf"), format='pdf')
    plt.close(visualization)
    
    print(f"Embedding analysis completed for {net_name}. Results saved to {emb_analysis_dir}")

    return emb_analysis_dir
