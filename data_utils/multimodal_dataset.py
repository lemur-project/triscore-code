import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import math
import os
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

from utils.configs import *

class MultimodalDataset(Dataset, ABC):
    def __init__(
        self,
        ds_name: str,
        input_modality: str,
        category: str,
        partition: str,
        return_id: bool
    ):        
        self.ds_name = ds_name
        self.input_modality = self.validate_modality(input_modality)
        self.category = self.validate_label(category, ds_name)
        self.partition = self.validate_partition(partition)
        
        self.getter_suffix = "with_id" if return_id else "modality" 

        # defined in subclasses
        self.metadata = None
        self.root_dir = None
        self.max_num_classes = None
        self.l2i, self.i2l = None, None
        
    # VALIDATION -----------------------------------------------------------

    def validate_label(self, category, ds_name):
        if category not in DATASETS[ds_name]["categories"]:
            raise ValueError(f"Invalid category '{category}' for dataset {ds_name}. Valid options are: {DATASETS[ds_name]['categories']}.")
        return DATASETS[ds_name]["map_category_names"][category]

    def validate_modality(self, input_modality):
        if input_modality not in MODALITIES:
            raise ValueError(f"Invalid input_modality '{input_modality}'. Valid options are {MODALITIES}.")
        return input_modality
    
    def validate_partition(self, partition):
        if partition not in ["train", "val", "test"]:
            raise ValueError(f"Invalid partition '{partition}'. Valid options are: train, val, test.")
        return partition

    # UTILS ---------------------------------------------------------------

    def _are_partitions_created(self, num_classes=None):
        categories = DATASETS[self.ds_name]["map_category_names"].values()
        partitions_folder = f"partitions{f'_{num_classes}' if num_classes is not None else ''}"
        for partition in ["train", "val", "test"]:  
            for category in categories:
                partition_file = os.path.join(self.root_dir, partitions_folder, category, f"{partition}.lst")
                if not os.path.exists(partition_file):
                    return False
        return True

    def create_partitions(self, test_and_val_size, max_num_classes=None):

        categories = DATASETS[self.ds_name]["map_category_names"].values()
        df = self.load_metadata()

        partitions_dir = os.path.join(
            self.root_dir,
            f"partitions{f'_{max_num_classes}' if max_num_classes is not None else ''}"
        )
        os.makedirs(partitions_dir, exist_ok=True)

        def filter_min_samples(df, label, min_samples):
            valid = df[label].value_counts()
            valid = valid[valid >= min_samples].index
            return df[df[label].isin(valid)]

        # Each class needs enough samples so stratified splitting yields ≥1 per class per side
        min_samples_first_split = math.ceil(1 / test_and_val_size)       # e.g. ceil(1/0.2) = 5
        min_samples_second_split = math.ceil(1 / 0.5)                    # = 2

        for category in categories:
            category_dir = os.path.join(partitions_dir, category)

            if os.path.exists(category_dir) and os.listdir(category_dir) == ["train.lst", "val.lst", "test.lst"]:
                continue

            print(f"Creating partition files for category '{category}'...")

            df_cat = df.copy()

            # filter out multilabel samples
            if df_cat[category].apply(lambda x: isinstance(x, list)).any(): 
                # keep only the samples that are lists of length 1
                mask = df_cat[category].apply(lambda x: isinstance(x, list) and len(x) == 1) 
                df_cat = df_cat[mask].copy()
                df_cat[category] = df_cat[category].apply(lambda x: x[0])

            # discard any label with the value "Anonymous" or "Unknown"
            df_cat = df_cat[~df_cat[category].isin(["Anonymus", "Unknown", "Anonymous"])]

            df_cat = filter_min_samples(df_cat, category, min_samples=min_samples_first_split)

            train, test_and_val = train_test_split(
                df_cat, test_size=test_and_val_size, stratify=df_cat[category], random_state=42
            )

            test_and_val = filter_min_samples(test_and_val, category, min_samples=min_samples_second_split)

            val, test = train_test_split(
                test_and_val, test_size=0.5, stratify=test_and_val[category], random_state=42
            )

            # Drop classes absent from any split
            value_counts = pd.concat([
                train[category].value_counts().rename("count_train"),
                val[category].value_counts().rename("count_val"),
                test[category].value_counts().rename("count_test"),
            ], axis=1)
            drop_classes = value_counts[value_counts.isnull().any(axis=1)].index.tolist()
            train = train[~train[category].isin(drop_classes)]
            val   = val[~val[category].isin(drop_classes)]
            test  = test[~test[category].isin(drop_classes)]

            # Apply max_num_classes on finalized splits
            if max_num_classes is not None:
                value_counts = pd.concat([
                    train[category].value_counts().rename("count_train"),
                    val[category].value_counts().rename("count_val"),
                    test[category].value_counts().rename("count_test"),
                ], axis=1)
                value_counts["total"] = value_counts.sum(axis=1)
                top_classes = value_counts["total"].nlargest(max_num_classes).index
                train = train[train[category].isin(top_classes)]
                val   = val[val[category].isin(top_classes)]
                test  = test[test[category].isin(top_classes)]

            # Final value counts for the CSV
            value_counts = pd.concat([
                train[category].value_counts().rename("count_train"),
                val[category].value_counts().rename("count_val"),
                test[category].value_counts().rename("count_test"),
            ], axis=1).fillna(0).astype(int)
            value_counts["total_count"] = value_counts.sum(axis=1)
            value_counts = value_counts.sort_values("total_count", ascending=False)

            os.makedirs(category_dir, exist_ok=True)

            value_counts.reset_index(names="label").to_csv(
                os.path.join(category_dir, f"{category}_label_distribution.csv"), index=False
            )

            for partition_name, partition_df in [("train", train), ("val", val), ("test", test)]:
                with open(os.path.join(category_dir, f"{partition_name}.lst"), "w", encoding="utf-8") as f:
                    f.write("\n".join(partition_df["id"]) + "\n")

        print("Partitions saved!")
        return partitions_dir

    def balance_dataset(self, category, oversample, partition):
        metadata = self.metadata
        
        if partition == "train":
            if oversample:
                max_count = metadata[category].value_counts().max()
                sampled_list = []
                for cls, group in metadata.groupby(category):
                    group_size = len(group)
                    if group_size < max_count:
                        # Calculate how many extra samples needed
                        n_extra = max_count - group_size
                        extra_samples = group.sample(n_extra, replace=True)
                        # Combine original + extra samples
                        balanced_group = pd.concat([group, extra_samples])
                    else:
                        # If class already has max_count, keep all
                        balanced_group = group
                    sampled_list.append(balanced_group)
                metadata_sampled = pd.concat(sampled_list).reset_index(drop=True)
            else:
                # Undersample: make all classes have the same min count
                min_count = metadata[category].value_counts().min()
                metadata_sampled = metadata.groupby(category).head(min_count).reset_index(drop=True)
            
            return metadata_sampled
        else:
            return metadata
     
    def print_dataset_info(self):
        print(f"Input modality: {self.input_modality}")
        print(f"Label type: {self.category}")
        print(f"Number of samples: {len(self.metadata)}")
        print(f"Number of unique {self.category}s: {len(self.metadata[self.category].unique())}")
        print(f"Unique {self.category}s: {self.metadata[self.category].unique()}")
        print(f"Number of samples in each class: {self.metadata[self.category].value_counts()}")

    def _get_label_dicts(self, unique_labels):
        l2i = {label: idx for idx, label in enumerate(unique_labels)}
        i2l = {idx: label for label, idx in l2i.items()}
        return l2i, i2l

    def get_classes(self):
        return self.metadata[self.category].unique()

    def __len__(self):
        return len(self.metadata)

    # DATA LOADING --------------------------------------------------------
    
    def __getitem__(self, idx):
        sample_metadata = self.metadata.iloc[idx]          
        return getattr(self, f"__getitem_{self.getter_suffix}__")(sample_metadata)

    def __getitem_with_id__(self, sample_metadata):
        return getattr(self, f"__getitem_{self.input_modality}__")(sample_metadata), self.l2i[sample_metadata[self.category]], sample_metadata["id"]
    
    def __getitem_modality__(self, sample_metadata):
        return getattr(self, f"__getitem_{self.input_modality}__")(sample_metadata), self.l2i[sample_metadata[self.category]]


    def __getitem_score__(self, sample_metadata):
        score_path = self._get_score_path_from_metadata(sample_metadata)
        # open the score file
        with open(score_path, "r", encoding="utf-8") as f:
            score = f.read()
        # remove line starting with !!! (Comments with the metadata)
        score = "\n".join(line for line in score.splitlines() if not line.startswith("!!!"))
        return score

    def __getitem_audio__(self, sample_metadata):
        audio_path = self._get_audio_path_from_metadata(sample_metadata)
        waveform, sr = torchaudio.load(audio_path)
        # Convert waveform to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform, sr
        

    def __getitem_image__(self, sample_metadata):
        png_path = self._get_image_path_from_metadata(sample_metadata)
        image = Image.open(png_path).convert("RGB")  
        return image
    
    # ABSTRACT METHODS -------------------------------------------------

    @abstractmethod
    def load_metadata(self, partition):
        pass

    @abstractmethod
    def _get_audio_path_from_metadata(self, metadata_row):
        pass

    @abstractmethod
    def _get_score_path_from_metadata(self, metadata_row):
        pass
    
    @abstractmethod
    def _get_image_path_from_metadata(self, metadata_row):
        pass

    