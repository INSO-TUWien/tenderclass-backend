import pandas as pd
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from src.classifier.PytorchTransformer.BertDataSet import BertDataSet
from src.classifier.PytorchTransformer.PyTorchLighningTransformer import PyTorchTransformerLightning
from src.classifier.PytorchTransformer.config.PytorchTransformerConfig import PytorchTransformerConfig
from src.classifier.interfaces.MLModelInterface import MlModelInterface
from src.entity.LabeledTenderCollection import LabelledTenderCollection


class FullTextTransformerModel(MlModelInterface):

    def __init__(self):
        self.config: PytorchTransformerConfig = PytorchTransformerConfig.description_only()

    def classify(self, tenders):
        pass

    def train(self, labelled_tenders):
        labelled_tenders_collection = LabelledTenderCollection(labelled_tenders)

        training_df = pd.DataFrame({"title": labelled_tenders_collection.get_titles(),
                                    "description": labelled_tenders_collection.get_original_language_entity_description(),
                                    "label": labelled_tenders_collection.get_labels()})
        training_df = training_df.dropna()

        train_df, val_df = train_test_split(training_df, test_size=0.1)

        # Create the DataLoader for our training set
        train_data = BertDataSet(train_df, self.config)
        train_dataloader = DataLoader(train_data, batch_size=self.config.batch_size, num_workers=6)

        # Create the DataLoader for our validation set
        val_data = BertDataSet(val_df, self.config)
        val_dataloader = DataLoader(val_data, batch_size=self.config.batch_size, num_workers=6)

        model = PyTorchTransformerLightning(self.config, total_steps=self.config.batch_size * len(train_dataloader))
        wandb_logger = WandbLogger()
        trainer = Trainer(gradient_clip_val=1.0, gpus=1, max_epochs=self.config.epochs, logger=wandb_logger)
        trainer.fit(model, train_dataloader, val_dataloader)

    def create_new_model(self):
        pass
