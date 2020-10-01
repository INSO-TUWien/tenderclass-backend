import pandas as pd
import torch
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from src.classifier.Lighning.BertDataSet import BertDataSet
from src.classifier.Lighning.PyTorchTransformerLighning import PyTorchTransformerLightning
from src.classifier.interfaces.MLModelInterface import MlModelInterface
from src.entity.LabeledTenderCollection import LabelledTenderCollection
from pytorch_lightning.loggers import TensorBoardLogger

class FullTextBertModelLightning(MlModelInterface):

    def __init__(self):
        self.tokenizerLong = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
        self.modelLong = AutoModel.from_pretrained("bert-base-multilingual-uncased")
        self.tokenizerShort = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.modelShort = AutoModel.from_pretrained("distilbert-base-uncased")
        self.max_len = 16
        self.batch_size = 4
        self.evaluation = True
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def classify(self, tenders):
        pass

    def train(self, labelled_tenders):
        labelled_tenders_collection = LabelledTenderCollection(labelled_tenders)
        logger = TensorBoardLogger('tb_logs', name='FullTextBertModelLightning')

        training_df = pd.DataFrame({"title": labelled_tenders_collection.get_titles(), "description": labelled_tenders_collection.get_original_language_entity_description(), "label": labelled_tenders_collection.get_labels()})
        training_df = training_df.dropna()

        train_df, val_df = train_test_split(training_df, test_size=0.1)

        # Create the DataLoader for our training set
        train_data = BertDataSet(train_df, self.max_len, self.tokenizerShort, self.tokenizerLong)
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size)

        # Create the DataLoader for our validation set
        val_data = BertDataSet(val_df, self.max_len, self.tokenizerShort, self.tokenizerLong)
        val_dataloader = DataLoader(val_data, batch_size=self.batch_size)

        model = PyTorchTransformerLightning(self.modelShort, self.modelLong, False, total_steps=self.batch_size*len(train_dataloader))
        trainer = Trainer(gradient_clip_val=1.0, gpus=1, max_epochs=4, logger=logger)
        trainer.fit(model, train_dataloader, val_dataloader)

    def create_new_model(self):
        pass
