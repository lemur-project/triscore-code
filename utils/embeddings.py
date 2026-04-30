import torch
from tqdm import tqdm
import os
import pickle
import numpy as np
from utils.configs import *
from imblearn.over_sampling import SMOTE
from data_utils.datamodule import MultimodalDataModule
import datasets
from collections import Counter

# Set to True to compute and save embeddings, you will need the dataset prepared, see README for details
COMPUTE_TRISCORE_EMBEDDINGS = False  

class EmbeddingsbyModality:
    def __init__(
            self,
            ds_dir: str,
            ds_name: str,
            category: str,
            encoders_per_modality: dict,
            normalized_embeddings: bool = False,
            oversampled_dataset: bool = False,
            smote_embeddings: bool = False,
            partitions=["train", "val", "test"],
            num_workers: int = 4,
        ):
        self.ds_dir = ds_dir
        self.dataset_name = ds_name
        self.category = category
        self.normalized_embeddings = normalized_embeddings
        self.oversampled_dataset = oversampled_dataset
        self.smote_embeddings = smote_embeddings
        if self.smote_embeddings:
            self.smote = SMOTE(random_state=42)

        self.modalities = MODALITIES
        self.partitions = partitions

        # for embedding extraction
        is_config_1 = set(ENC_CONFIGS[1].values()) == set([e.name for e in encoders_per_modality.values()])
        if self.dataset_name == "triscore" and not COMPUTE_TRISCORE_EMBEDDINGS and is_config_1:
            self.retrieve_from_hf = True
            self.hf_ds_hub = f"ahidalgocenteno/triscore-{category}"
            print("Retrieving embeddings from Hugging Face Hub... ")
        else:
            self.retrieve_from_hf = False
        
        self.num_workers = num_workers
        self.models_dict = encoders_per_modality
        self.max_embedding_dim = max([model.hidden_size for model in self.models_dict.values()])

        # embedding extraction and saving
        self.per_partition_embeddings = self.retrieve_embeddings_per_partition()
        self.num_classes = self.get_number_of_classes()


    # Helper methods to initialize dataloaders and models
    def _get_modality_datamodule(self, modality):
        dm = MultimodalDataModule(
            root_dir=self.ds_dir,
                category=self.category,
                input_modality=modality,
                batch_size=1,
                num_workers=self.num_workers,
                ds_name=self.dataset_name,
                processor=self.models_dict[modality].get_processor(),
                encoder_sr=self.models_dict["audio"].sr,
                symbolic_encoding=self.models_dict["score"].encoding_input,
                return_identifier=True
            )
        dm.setup()
        return dm

    def _extract_embeddings(self, model, dataloader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        model.eval()

        all_embeddings = []
        all_labels = []
        all_identifiers = []

        for batch in tqdm(dataloader, desc=f"Extracting {dataloader.dataset.input_modality} {dataloader.dataset.partition} embeddings"):
            samples, labels, identifiers = batch
            for inputs, lbl, id in zip(samples, labels, identifiers):
                inputs = inputs.to(device)
                with torch.no_grad():
                    emb = model(inputs).squeeze().cpu().numpy()
                all_embeddings.append(emb)
                all_labels.append(lbl)
                all_identifiers.append(id)
        
        return all_embeddings, all_labels, all_identifiers
    

    def _pad_embeddings(self, embs):
        return np.array([
            np.pad(emb, (0, self.max_embedding_dim - emb.shape[0]), mode='constant', constant_values=0)
            for emb in embs
        ])

    def _save_embeddings(self, embeddings, labels, identifiers, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump((embeddings, labels, identifiers), f)
        print(f"Embeddings saved to {save_path}")

    def get_number_of_classes(self):
        partition = self.partitions[0]
        modality = self.modalities[0]
        _, labels, _ = self.get_embeddings_dict(partition, modalities=[modality])[modality]
        return len(set(labels))

    # Methods for retrieving and processing embeddings

    def retrieve_embeddings_per_partition(self):
        embeddings_per_partition = {}

        for partition in self.partitions:
            per_modality_embs = {}
            # load hf partition
            if self.retrieve_from_hf:
                hf_ds = datasets.load_dataset(self.hf_ds_hub, split=partition)
            
                for mod in self.modalities:
                    embs_column_key = f"{mod}-embeddings-{self.models_dict[mod].name}"
                    per_modality_embs[mod] = (
                        list(hf_ds.with_format("numpy")[embs_column_key]), 
                        list(hf_ds["label"]), 
                        list(hf_ds["identifier"])
                    )    
            else:
                for mod in self.modalities:
                    save_path = os.path.join(
                        self.ds_dir, "embeddings", "pretrained",
                        f"{mod}-{self.models_dict[mod].name}",
                        self.category,
                        f"{partition}_embeddings.pkl"
                    )
                    if os.path.exists(save_path):
                        with open(save_path, 'rb') as f:
                            embeddings, labels, identifiers = pickle.load(f)
                    else:
                        dm = self._get_modality_datamodule(mod)
                        dataloader = getattr(dm, f"{partition}_dataloader")()
                        embeddings, labels, identifiers = self._extract_embeddings(
                            model= self.models_dict[mod],
                            dataloader=dataloader
                        )
                    self._save_embeddings(embeddings, labels, identifiers, save_path)
                    per_modality_embs[mod] = (embeddings, labels, identifiers)

            embeddings_per_partition[partition] = per_modality_embs
        print("Embeddings retrieved!")
        return embeddings_per_partition


    def get_embeddings_dict(self, partition: str, padded: bool=True, modalities: list=None):
        """
        partition: "train", "val", "test"
        padded: whether to pad embeddings to max_embedding_dim
        modalities: list of modalities to get embeddings for (default: the ones declared in EmbeddingsbyModality initialization)
        Returns:
            dict with {modality: (embeddings, labels, identifiers)}
        """
        if modalities is None:
            modalities = self.modalities
        
        per_mod_embs_dict = self.per_partition_embeddings.get(partition)
        if per_mod_embs_dict is None:
            raise ValueError(f"Embeddings for partition '{partition}' not found.")

        embeddings_by_modality = {}
        for modality in modalities:
            if modality not in per_mod_embs_dict:
                raise ValueError(f"Embeddings for modality '{modality}' not found in partition '{partition}'.")

            embeddings, labels, identifiers = per_mod_embs_dict[modality]

            if self.normalized_embeddings:
                embeddings = np.array(embeddings)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Prevent division by zero
                embeddings = (embeddings / norms)

            if  self.smote_embeddings and partition == "train":
                # Apply SMOTE to the embeddings
                embeddings, labels_arr  = self.smote.fit_resample(np.array(embeddings), np.array(labels))
                labels = list(labels_arr)
                identifiers = len(labels) * [None]  # SMOTE generates synthetic samples without identifiers

            if self.oversampled_dataset and partition == "train":
                counts = Counter(labels)
                max_count = max(counts.values())
                repeated_indices = []
                for i, label in enumerate(labels):
                    if counts[label] < max_count:
                        repeated_indices.extend([i] * (max_count - counts[label]))
                embeddings = np.array(embeddings)
                embeddings = np.vstack([embeddings] + [embeddings[repeated_indices]] * (max_count - 1))
                labels = [label] * len(embeddings)
                identifiers = len(embeddings) * [None]  # Oversampling generates synthetic samples without identifiers

            if padded: embeddings = self._pad_embeddings(embeddings)
            
            embeddings_by_modality[modality] = (embeddings, labels, identifiers)

        return embeddings_by_modality


    def get_embeddings_array(self, partition: str, modalities: list=None, check_modality_aligment: bool=False):
        """
        embedding_dict: dict with {mod: (embeddings, labels, identifiers)}
        Returns:
            embeddings: np.array of shape (num_samples, embedding_dim)
            labels: np.array of shape (num_samples,)
            identifiers: list of identifiers
        """

        all_embeddings, all_labels, all_identifiers, all_modalities = [], [], [], []
        embedding_dict = self.get_embeddings_dict(partition, modalities=modalities)

        for mod, (embeddings, labels, identifiers) in embedding_dict.items():
            all_embeddings.append(self._pad_embeddings(embeddings))
            all_labels.append(labels)
            all_identifiers.extend(identifiers)
            all_modalities.extend([mod] * len(labels))

        if check_modality_aligment:
            # check that for each modality, the identifiers are the same
            for mod, (embeddings, labels, identifiers) in embedding_dict.items():
                if len(set(identifiers)) != len(identifiers):
                    raise ValueError(f"Identifiers for modality {mod} are not unique.")
                if set(identifiers) != set(all_identifiers):
                    raise ValueError(f"Identifiers for modality {mod} do not match the overall identifiers.")
            
        return {
            "embeddings": np.concatenate(all_embeddings), 
            "labels": np.concatenate(all_labels), 
            "modalities": np.array(all_modalities), 
            "identifiers": np.array(all_identifiers)
        }