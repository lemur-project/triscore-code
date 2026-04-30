
from typing import Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

class EmbeddingMultiModalDataset(Dataset):
    def __init__(self, embeddings_by_modality: Dict, modality_balance: Optional[Dict[str, int]] | None = None):
        
        self.modality_names = list(embeddings_by_modality.keys())
        self.dict_embeddings_by_modality = {mod: embeddings_by_modality[mod][0] for mod in self.modality_names}
        self.dict_labels_by_modality = {mod: embeddings_by_modality[mod][1] for mod in self.modality_names}
        self.dict_ids_by_modality   = {mod: embeddings_by_modality[mod][2] for mod in self.modality_names}

        if modality_balance is None:
            self.batch_modality_balance = "batch_unbalanced"
            self._flatten_embeddings()
        else:
            self.batch_modality_balance = "batch_balanced"
            self.modality_balance = modality_balance

            self._calculate_modality_balance(modality_balance)

    def _flatten_embeddings(self):
        self.all_embeddings = []
        self.all_labels = []
        for mod in self.modality_names:
            self.all_embeddings.extend(self.dict_embeddings_by_modality[mod])
            self.all_labels.extend(self.dict_labels_by_modality[mod])

    def _calculate_modality_balance(self, modality_balance):
        # assume same num-samples across modalities
        counts = [len(self.dict_embeddings_by_modality[m]) for m in self.modality_names]
        if not all(x == counts[0] for x in counts):
            raise ValueError("All modalities must have the same number of samples.")

        num_samples = counts[0]
        self.samples_per_batch   = sum(modality_balance.values())
        self.num_complete_batches = num_samples // max(modality_balance.values())

        # indices of balanced region
        self.modality_indices = {}
        self.leftover_pairs = []

        for m in self.modality_names:
            usable = self.num_complete_batches * modality_balance[m]
            self.modality_indices[m] = np.arange(usable)
            # save the leftovers
            for i in range(usable, len(self.dict_embeddings_by_modality[m])):
                self.leftover_pairs.append((m, i))

        self.total_len = self.num_complete_batches * self.samples_per_batch + len(self.leftover_pairs)

    def _get_class_frequencies(self):
        if self.batch_modality_balance == "batch_unbalanced":
            labels = self.all_labels
        else:
            labels = sum((self.dict_labels_by_modality[mod] for mod in self.modality_names), [])
        unique, counts = np.unique(labels, return_counts=True)
        max_label = max(unique)
        frequencies = [0] * (max_label + 1)
        for u, c in zip(unique, counts):
            frequencies[u] = c
        return frequencies
    
    def get_weight_classes(self, method="effective", beta=0.99):
        class_frequencies = self._get_class_frequencies()
        if method == "effective":
            effective_num = 1.0 - np.power(beta, class_frequencies)
            class_weights = (1.0 - beta) / np.array(effective_num)
        elif method == "inverse_freq":
            class_weights = 1.0 / np.array(class_frequencies)
        elif method == "inverse_sqrt_freq":
            class_weights = 1.0 / np.sqrt(np.array(class_frequencies))
        else:
            raise ValueError(f"Unknown method: {method}, choose from ['effective', 'inverse_freq', 'inverse_sqrt_freq']")
        
        class_weights = class_weights / np.sum(class_weights) * len(class_frequencies)
        
        return torch.tensor(class_weights, dtype=torch.float32)


    def __len__(self):
        if self.batch_modality_balance == "batch_unbalanced":
            return len(self.all_embeddings)
        return self.total_len

    def __getitem_batch_unbalanced__(self, idx):
        return {
            "embedding": torch.FloatTensor(self.all_embeddings[idx]),
            "label":    self.all_labels[idx],
        }

    def __getitem_batch_balanced__(self, idx):
        balanced_count = self.num_complete_batches * self.samples_per_batch

        if idx < balanced_count:
            batch_idx = idx // self.samples_per_batch
            pos = idx  % self.samples_per_batch

            cumulative = 0
            picked_mod = None
            offset = None
            for mod in self.modality_names:
                if pos < cumulative + self.modality_balance[mod]:
                    picked_mod = mod
                    offset     = pos - cumulative
                    break
                cumulative += self.modality_balance[mod]

            sample_idx = batch_idx * self.modality_balance[picked_mod] + offset
            real_i = self.modality_indices[picked_mod][sample_idx]
            return {
                "embedding": torch.FloatTensor(self.dict_embeddings_by_modality[picked_mod][real_i]),
                "label":    self.dict_labels_by_modality[picked_mod][real_i],
            }
        else:
            leftover_i = idx - balanced_count
            mod, i     = self.leftover_pairs[leftover_i]
            return {
                "embedding": torch.FloatTensor(self.dict_embeddings_by_modality[mod][i]),
                "label":    self.dict_labels_by_modality[mod][i],
            }

    def __getitem__(self, idx):
        return getattr(self, f"__getitem_{self.batch_modality_balance}__")(idx)


