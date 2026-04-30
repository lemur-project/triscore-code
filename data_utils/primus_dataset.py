import os
import pandas as pd
import json
import urllib.parse

from data_utils.multimodal_dataset import MultimodalDataset # base class
from utils.configs import DATASETS

class PriMusDataset(MultimodalDataset):
    def __init__(
        self,
        root_dir: str,
        category: str, 
        input_modality: str, 
        partition: str,
        symbolic_encoding=None, # only applies when input_modality = "score"
        max_num_classes=100,
        balanced_dataset=True,
        oversample=True,
        return_id=False
    ):
        super().__init__(
            ds_name="primus",
            input_modality=input_modality,
            category=category,
            partition=partition,
            return_id=return_id
        )

        self.root_dir = root_dir            
        self.balanced_dataset = balanced_dataset
        if not self._are_partitions_created(num_classes=max_num_classes):
            self.create_partitions(test_and_val_size=0.2, max_num_classes=max_num_classes)
        self.metadata = self.load_metadata(partition, max_num_classes)
        if balanced_dataset:
            self.metadata = self.balance_dataset(self.category, oversample, partition)
    
        # Create label mappings
        self.unique_labels = sorted(self.metadata[self.category].unique())
        self.l2i, self.i2l = self._get_label_dicts(self.unique_labels)

        self.symbolic_fmt = self._get_symbolic_format(symbolic_encoding)


    def _get_symbolic_format(self, symbolic_encoding):
        if symbolic_encoding is None:
            return None
        elif symbolic_encoding == "xml":
            return "xml"
        elif symbolic_encoding == "abc":
            return "abc"
        elif symbolic_encoding == "kern":
            return "krn"
        else:
            raise ValueError(f"Unsupported symbolic encoding: {symbolic_encoding}")

    def load_metadata(self, partition=None, max_num_classes=None):
        categories = DATASETS["primus"]["map_category_names"].values()
    
        if partition is not None:
            partition_file = os.path.join(
                self.root_dir,
                f"partitions{f'_{max_num_classes}' if max_num_classes is not None else ''}",
                self.category,
                f"{partition}.lst",
            )
            if not os.path.exists(partition_file):
                raise ValueError(f"Partition file {partition_file} not found.")
            with open(partition_file, "r") as f:
                ids = [line.strip() for line in f]
        else:
            ids_file = os.path.join(self.root_dir, "files.lst")
            if not os.path.exists(ids_file):
                raise ValueError(f"IDs file {ids_file} not found.")
            with open(ids_file, "r") as f:
                ids = [line.strip() for line in f]
    
        metadata = []
        for id in ids:
            json_file = os.path.join(self.root_dir, id, f"{id}.json")
            with open(json_file, "r") as fin:
                data = json.load(fin)
    
            # Partition mode → ensure single label
            if isinstance(data[self.category], list) and partition is not None:
                data[self.category] = data[self.category][0]                    
            # Assign ID
            row = {"id": id}
            # Normalize and copy ALL category fields (but append ONCE)
            for cat in categories:
                lbl = data.get(cat)
                if isinstance(lbl, str):
                    lbl = urllib.parse.unquote(lbl).replace("+", " ")
                elif isinstance(lbl, list):
                    lbl = [urllib.parse.unquote(x).replace("+", " ") for x in lbl]
                row[cat] = lbl
    
            metadata.append(row)
    
        return pd.DataFrame(metadata)
        
    # DATA LOADING --------------------------------------------------------

    def _get_audio_path_from_metadata(self, row):
        return os.path.join(self.root_dir, row['id'], f"{row['id']}.wav")
    
    def _get_score_path_from_metadata(self, row):
        return os.path.join(self.root_dir, row['id'], f"{row['id']}.{self.symbolic_fmt}")

    def _get_image_path_from_metadata(self, row):
        return  os.path.join(self.root_dir, row['id'],  f"{row['id']}.png")
