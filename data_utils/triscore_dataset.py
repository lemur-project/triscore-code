import os
import pandas as pd
import json
from utils.helpers import normalized_windows_path
from data_utils.multimodal_dataset import MultimodalDataset # base class

class TriScoreDataset(MultimodalDataset):
    def __init__(
        self,
        root_dir: str,
        input_modality: str,
        category: str,
        partition: str,
        symbolic_encoding=None, # only applies when input_modality = "score"
        balanced_dataset=True,
        oversample=True,
        return_id=False 
    ):
        super().__init__(
            ds_name="triscore",
            input_modality=input_modality,
            category=category,
            partition=partition,
            return_id=return_id
        )

        self.root_dir = root_dir 
        self.balanced_dataset = balanced_dataset
        if not self._are_partitions_created():
            self.create_partitions(test_and_val_size=0.2, max_num_classes=None)
        self.metadata = self.load_metadata(partition)
        if balanced_dataset:
            self.metadata = self.balance_dataset(self.category, oversample, partition)

        # Create label mappings
        self.unique_labels = sorted(self.metadata[self.category].unique())
        self.l2i, self.i2l = self._get_label_dicts(self.unique_labels)

        self.symbolic_format = self._get_symbolic_format(symbolic_encoding)

    def get_classes(self):
        return self.metadata[self.category].unique()
    
    def _get_symbolic_format(self, symbolic_encoding):
        if symbolic_encoding == "musicxml":
            return "musicxml.xml"
        elif symbolic_encoding == "abc":
            return "abc"
        elif symbolic_encoding == "kern":
            return "krn"
        else:
            raise ValueError(f"Unsupported symbolic encoding: {symbolic_encoding}")
    
    def load_metadata(self, partition=None):
        # load the partition paths
        if partition is not None:
            partition_file = os.path.join(self.root_dir, "partitions", self.category, f"{partition}.lst")
            if not os.path.exists(partition_file):
                raise ValueError(f"Partition file {partition_file} not found, please prepare the dataset first.")
            with open(partition_file, "r", encoding="utf-8") as f:
                paths = [line.strip() for line in f if line.strip()]
        else:
            files_data_path = os.path.join(self.root_dir, "files.lst")
            if not os.path.exists(files_data_path):
                raise ValueError(f"Files list {files_data_path} not found, please prepare the dataset first.")
            with open(files_data_path, "r", encoding="utf-8") as f:
                paths = [line.strip() for line in f]

        # create a dataframe with the metadata
        metadata = []
        for path in paths:
            # split the path with "_-_" to remove the cut suffix
            root_file_name = path.split("_-_")[0]
            # get the metadata file
            metadata_file = os.path.join(self.root_dir, f"{root_file_name}.json")
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # if the partition has been created
                if isinstance(data[self.category], list) and partition is not None:
                    data[self.category] = data[self.category][0]
                data["id"] = path
                metadata.append(data)   

        return pd.DataFrame(metadata)

    
    def _get_audio_path_from_metadata(self, row):
        return os.path.join(self.root_dir, f"{row['id']}.mp3")
    
    def _get_score_path_from_metadata(self, row):
        return os.path.join(self.root_dir, f"{row['id']}.{self.symbolic_format}")
    
    def _get_image_path_from_metadata(self, row):
        return os.path.join(self.root_dir, f"{row['id']}.png")
    


