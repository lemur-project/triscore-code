import os
import json
import glob
import pandas as pd
from data_utils.multimodal_dataset import MultimodalDataset # base class

class MTDataset(MultimodalDataset):
    def __init__(
        self,
        root_dir: str,
        category: str, 
        input_modality: str, 
        partition: str,
        symbolic_encoding=None, # only applies when input_modality = "score"
        balanced_dataset=True,
        oversample=True,
        return_id=False
    ):
        super().__init__(
            ds_name="mtd",
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

        # symbolic format and folder
        self.symbolic_format, self.symbolic_folder = self._get_symbolic_format_and_folder(symbolic_encoding)
        
    # INTERNAL --------------------------------------------------------
    
    def _get_symbolic_format_and_folder(self, symbolic_encoding):
        if symbolic_encoding is None:
            return None, None
        elif symbolic_encoding == "musicxml":
            return "xml", "data_SCORE_XML"
        elif symbolic_encoding == "abc":
            return "abc", "data_SCORE_ABC"
        elif symbolic_encoding == "kern":
            return "krn", "data_SCORE_KERN"
        else:
            raise ValueError(f"Unsupported symbolic encoding: {symbolic_encoding}")
        
    def _get_mtd_list_field(self, key, data, partition=None):
        labels = data.get(key, [])
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",")]
        if partition is not None:
            return labels[0]
        return labels

    def load_metadata(self, partition=None):
        meta_dir = os.path.join(self.root_dir, "data_META")
        meta_files = glob.glob(os.path.join(meta_dir, "*.json"))

        metadata = []
        for meta_file in meta_files:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                metadata.append({
                    "id": data["MTDID"],
                    "WorkID": data["WorkID"],
                    "WorkTitle": data.get("WorkTitle", "Unknown"),
                    "ComposerID": data["ComposerID"],
                    "ThemeInstruments": self._get_mtd_list_field("ThemeInstruments", data, partition),
                    "Ensemble": self._get_mtd_list_field("Ensemble", data, partition),
                    "Polyphony": data.get("Polyphony", "Unknown"),
                })
        
        metadata = pd.DataFrame(metadata)

        # get the metadata from the corresponding partition
        if partition is not None:
            partition_file = os.path.join(self.root_dir, "partitions", self.category, f"{partition}.lst")
            if os.path.exists(partition_file):
                with open(partition_file, "r") as f:
                    partition_ids = [line.strip() for line in f]
                metadata = metadata[metadata["id"].isin(partition_ids)]
            else:
                raise ValueError(f"Partition file {partition_file} not found, please prepare the dataset first.")
        
        return metadata

    # GETTERS FOR THE DATASET ------------------------------------------------

    def _get_audio_path_from_metadata(self, row):
        return os.path.join(self.root_dir, "data_AUDIO", f"MTD{row['id']}_{row['ComposerID']}_{row['WorkID']}.wav")
    
    def _get_score_path_from_metadata(self, row):
        return os.path.join(self.root_dir, self.symbolic_folder, f"MTD{row['id']}_{row['ComposerID']}_{row['WorkID']}.{self.symbolic_format}")
    
    def _get_image_path_from_metadata(self, row):
        return os.path.join(self.root_dir, "data_SCORE_IMG_PNG", f"MTD{row['id']}_{row['ComposerID']}_{row['WorkID']}.png")
