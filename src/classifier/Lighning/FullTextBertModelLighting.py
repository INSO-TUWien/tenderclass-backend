import pandas as pd
import torch
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from src.classifier.Lighning.BertDataSet import BertDataSet
from src.classifier.Lighning.PyTorchTransformerLighning import PyTorchTransformerLightning
from src.classifier.interfaces.MLModelInterface import MlModelInterface
from src.entity.LabeledTenderCollection import LabelledTenderCollection

class FullTextBertModelLightning(MlModelInterface):

    def __init__(self):
        self.tokenizerLong = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.modelLong = AutoModel.from_pretrained("bert-base-multilingual-cased")
        self.tokenizerShort = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.modelShort = AutoModel.from_pretrained("distilbert-base-uncased")
        self.max_len = 128
        self.batch_size = 16
        self.evaluation = True
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def classify(self, tenders):
        pass

    def train(self, labelled_tenders):
        labelled_tenders_collection = LabelledTenderCollection(labelled_tenders)

        training_df = pd.DataFrame({"title": labelled_tenders_collection.get_titles(), "description": labelled_tenders_collection.get_original_language_entity_description(), "label": labelled_tenders_collection.get_labels()})
        training_df = training_df.dropna()

        train_df, val_df = train_test_split(training_df, test_size=0.1)

        # Create the DataLoader for our training set
        train_data = BertDataSet(train_df, self.max_len, self.tokenizerShort, self.tokenizerLong)
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size)

        # Create the DataLoader for our validation set
        val_data = BertDataSet(val_df, self.max_len, self.tokenizerShort, self.tokenizerLong)
        val_dataloader = DataLoader(val_data, batch_size=self.batch_size)

        model = PyTorchTransformerLightning(self.modelLong, self.modelShort, False)
        trainer = Trainer(gradient_clip_val=1.0, gpus=1, max_epochs=4)
        trainer.fit(model, train_dataloader, val_dataloader)

    def create_new_model(self):
        pass

    # Create a function to tokenize a set of texts
    def preprocessing_for_bert(self, data):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                      tokens should be attended to by the model.
        """
        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []

        # For every sentence...
        for text in data:
            # `encode_plus` will:
            #    (1) Tokenize the sentence
            #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
            #    (3) Truncate/Pad sentence to max length
            #    (4) Map tokens to their IDs
            #    (5) Create attention mask
            #    (6) Return a dictionary of outputs
            encoded_sent = self.tokenizerShort.encode_plus(
                truncation=True,
                text=text,  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=self.max_len,  # Max length to truncate/pad
                pad_to_max_length=True,  # Pad sentence to max length
                # return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True  # Return attention mask
            )

            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks
