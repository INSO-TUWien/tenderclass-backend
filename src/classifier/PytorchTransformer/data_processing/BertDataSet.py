import torch
from torch.utils.data import Dataset

from src.classifier.PytorchTransformer.data_processing.BertPreprocessor import BertPreprocessor
from src.classifier.PytorchTransformer.config.TransformerModelConfig import PytorchTransformerConfig


class BertDataSet(Dataset):

    def __init__(self, df, config: PytorchTransformerConfig):
        self.titles = df["title"].values
        self.descriptions = df["description"].values
        self.labels = torch.tensor(df["label"].values)
        self.processor = BertPreprocessor

        self.config = config

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        title = self.titles[idx]
        desc = self.descriptions[idx]
        label = self.labels[idx]

        return BertPreprocessor.get_sample(self.config, title, desc, label)
