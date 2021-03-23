import torch
from torch.utils.data import Dataset

from src.classifier.FullTextTransformerModel.config.TransformerModelConfig import \
    PytorchTransformerConfig
from src.classifier.FullTextTransformerModel.PytorchTransformer.data_processing.BertPreprocessor import BertPreprocessor


class BertDataSet(Dataset):

    def __init__(self, df, config: PytorchTransformerConfig):
        self.titles = None
        self.descriptions = None
        self.labels = None

        if "title" in df.columns:
            self.titles = df["title"].values

        if "description" in df.columns:
            self.descriptions = df["description"].values

        if "label" in df.columns:
            self.labels = torch.tensor(df["label"].values)

        self.processor = BertPreprocessor
        self.config = config

    def __len__(self):
        if self.titles is not None:
            return len(self.titles)
        elif self.descriptions is not None:
            return len(self.descriptions)
        else:
            return len(self.labels)

    def __getitem__(self, idx):

        title = "" if self.titles is None else self.titles[idx]
        desc = "" if self.descriptions is None else self.descriptions[idx]
        label = -1 if self.labels is None else self.labels[idx]

        return BertPreprocessor.get_sample(self.config, title, desc, label)
