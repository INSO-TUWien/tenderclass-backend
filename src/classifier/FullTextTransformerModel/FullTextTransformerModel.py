import gc
from typing import List

import pandas as pd
import torch
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from src.classifier.FullTextTransformerModel.PytorchTransformer.PyTorchLighningTransformer import \
    PyTorchTransformerLightning
from src.classifier.FullTextTransformerModel.config.TransformerModelConfig import \
    PytorchTransformerConfig
from src.classifier.FullTextTransformerModel.PytorchTransformer.data_processing.BertDataSet import BertDataSet
from src.classifier.TenderClassClassifier import TenderClassClassifier
from src.entity.ValidationResult import ValidationResult
from src.entity.LabeledTenderCollection import LabelledTenderCollection
from src.entity.Tender import Tender


class FullTextTransformerModel(TenderClassClassifier):

    def __init__(self, config=PytorchTransformerConfig.bert_german_full()):
        self.config = None
        self.model = None
        self.create_new_model(config)

    def predict(self, df):
        data = BertDataSet(df, self.config)
        dataloader = DataLoader(data, batch_size=1)
        predictions = []

        for batch_ndx, sample in enumerate(dataloader):
            title_input_ids = sample["title_input_ids"]
            title_attention_masks = sample["title_attention_mask"]
            description_input_ids = sample["description_input_ids"]
            description_attention_masks = sample["description_attention_mask"]
            logits = self.model(title_input_ids, title_attention_masks, description_input_ids,
                                description_attention_masks)
            _, predicted = torch.max(logits, 1)
            predictions.append(predicted.data[0].item())

        return predictions

    def classify(self, tenders: List[Tender]):
        titles = list(map(lambda x: x.get_title("DE"), tenders))
        descriptions = list(map(lambda x: x.get_description("DE"), tenders))
        labels = list(map(lambda x: -1, tenders))

        df = pd.DataFrame({"title": titles, "description": descriptions, "label": labels})
        df = df.dropna()

        predictions = self.predict(df)

        tuples = zip(tenders, predictions)
        selected_tenders = [t for t, p in tuples if p == 1]
        return selected_tenders

    def prepare_data(self, labelled_tenders):
        labelled_tenders_collection = LabelledTenderCollection(labelled_tenders)

        training_df = pd.DataFrame({"title": labelled_tenders_collection.get_titles(),
                                    "description": labelled_tenders_collection.get_original_language_entity_description(),
                                    "label": labelled_tenders_collection.get_labels()})
        return training_df.dropna()

    def train(self, labelled_tenders):
        training_df = self.prepare_data(labelled_tenders)

        train_df, val_df = train_test_split(training_df, test_size=0.1)

        # Create the DataLoader for our training set
        train_data = BertDataSet(train_df, self.config)
        train_dataloader = DataLoader(train_data, batch_size=self.config.batch_size, num_workers=6)

        # Create the DataLoader for our validation set
        val_data = BertDataSet(val_df, self.config)
        val_dataloader = DataLoader(val_data, batch_size=self.config.batch_size, num_workers=6)

        self.model.set_total_training_steps(self.config.batch_size * len(train_dataloader))
        wandb_logger = WandbLogger()
        trainer = Trainer(gradient_clip_val=1.0, gpus=1, max_epochs=self.config.epochs, logger=wandb_logger)
        trainer.fit(self.model, train_dataloader, val_dataloader)

    def validate(self, labelled_tenders):
        df = self.prepare_data(labelled_tenders)
        X = df[['title', 'description']]
        ylabels = df['label']
        y_pred = self.predict(X)
        return ValidationResult(ylabels, y_pred)

    def save_model(self):
        torch.save(self.model.state_dict(), "./data/" + self.config.name)

    def create_new_model(self, *args):
        if len(args) is 1:
            self.config = args[0]

        del self.model
        torch.cuda.empty_cache()
        gc.collect()

        self.model = PyTorchTransformerLightning(self.config)
