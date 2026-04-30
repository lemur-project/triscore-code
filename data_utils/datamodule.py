from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from data_utils.mtd_dataset import MTDataset
from data_utils.triscore_dataset import TriScoreDataset
from data_utils.primus_dataset import PriMusDataset
from data_utils.data_preprocessing import BatchModalityCollator

class MultimodalDataModule(LightningDataModule):

    def __init__(
        self,
        root_dir: str,
        category: str,
        input_modality: str,
        ds_name: str,
        processor,
        encoder_sr,
        symbolic_encoding,
        batch_size: int = 16,
        num_workers: int = 0,
        balanced_dataset=False,
        oversample=True,
        return_identifier=False,
        augmentations=None,
    ):
        super(MultimodalDataModule, self).__init__()
        self.root_dir = root_dir
        self.category = category
        self.input_modality = input_modality
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balanced_dataset = balanced_dataset
        self.oversample = oversample
        self.ds_name = ds_name
        self.augmentations = augmentations
        self.processor = processor
        self.encoder_sr = encoder_sr
        self.return_identifier = return_identifier
        self.symbolic_encoding = symbolic_encoding

    def setup(self, stage=None):
        # initialize the collator object
        self.collator = BatchModalityCollator(processor=self.processor, modality=self.input_modality, encoder_sr=self.encoder_sr, symbolic_encoding=self.symbolic_encoding)
        ds_args = {
            "root_dir": self.root_dir,
            "category": self.category,
            "input_modality": self.input_modality,
            "oversample": self.oversample,
            "return_id": self.return_identifier,
            "symbolic_encoding": self.symbolic_encoding,
        }
        if self.ds_name == "mtd":
            if stage == "fit" or stage is None:
                self.train_ds = MTDataset(**ds_args,partition="train",balanced_dataset=self.balanced_dataset)
                self.val_ds = MTDataset(**ds_args,partition="val",balanced_dataset=False)
                self.num_classes = len(self.train_ds.unique_labels)
            if stage == "test" or stage is None:
                self.test_ds = MTDataset(**ds_args,partition="test",balanced_dataset=False)
        elif self.ds_name == "triscore":
            if stage == "fit" or stage is None:
                self.train_ds = TriScoreDataset(**ds_args,partition="train",balanced_dataset=self.balanced_dataset)
                self.val_ds = TriScoreDataset(**ds_args,partition="val",balanced_dataset=False)
                self.num_classes = len(self.train_ds.unique_labels)
            if stage == "test" or stage is None:
                self.test_ds = TriScoreDataset(**ds_args,partition="test",balanced_dataset=False)
        elif self.ds_name == "primus":
            if stage == "fit" or stage is None:
                self.train_ds = PriMusDataset(**ds_args, partition="train", balanced_dataset=self.balanced_dataset)
                self.val_ds = PriMusDataset(**ds_args, partition="val", balanced_dataset=False)
                self.num_classes = len(self.train_ds.unique_labels)
            if stage == "test" or stage is None:
                self.test_ds = PriMusDataset(**ds_args, partition="test", balanced_dataset=False)
        else:
            raise ValueError(f"Dataset {self.ds_name} is not supported.")
        
        self.num_classes = len(self.train_ds.unique_labels) if hasattr(self, 'train_ds') else len(self.test_ds.unique_labels)

    def get_num_classes(self):
        return self.num_classes
    
    def get_i2l(self):
        return self.train_ds.i2l if hasattr(self, 'train_ds') else self.test_ds.i2l

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
