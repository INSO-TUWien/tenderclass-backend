import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from transformers import AdamW


class PyTorchTransformerLightning(LightningModule):
    """Bert Model for Classification Tasks.
    """

    def __init__(self, modelLong, modelShort, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(PyTorchTransformerLightning, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Instantiate BERT model
        self.bertTitles = modelShort
        self.bertDescriptions = modelShort

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in * 2, H),
            nn.ReLU(),
            ##nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bertTitles.parameters():
                param.requires_grad = False

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          lr=5e-5,  # Default learning rate
                          eps=1e-8  # Default epsilon value
                          )

        return optimizer

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputsTitles = self.bertTitles(input_ids=input_ids,
                                        attention_mask=attention_mask)

        outputsDescriptions = self.bertTitles(input_ids=input_ids,
                                              attention_mask=attention_mask)

        concat_output = torch.cat((outputsTitles[0][:, 0, :], outputsDescriptions[0][:, 0, :]), dim=1)

        # Feed input to classifier to compute logits
        logits = self.classifier(concat_output)

        return logits

    def training_step(self, batch, batch_idx):
        title_input_ids = batch["title_input_ids"]
        title_attention_masks = batch["title_attention_mask"]
        description_input_ids = batch["description_input_ids"]
        description_attention_masks = batch["description_attention_mask"]
        labels = batch["label"]
        # Perform a forward pass. This will return logits.
        logits = self.forward(title_input_ids, title_attention_masks)

        # Compute loss and accumulate the loss values
        loss = self.loss_fn(logits, labels)
        logs = {'train_loss': loss}
        return {'loss': loss, 'logs': logs}

    def validation_step(self, batch, batch_idx):
        title_input_ids = batch["title_input_ids"]
        title_attention_masks = batch["title_attention_mask"]
        description_input_ids = batch["description_input_ids"]
        description_attention_masks = batch["description_attention_mask"]
        labels = batch["label"]

        # Perform a forward pass. This will return logits.
        logits = self.forward(title_input_ids, title_attention_masks)

        # Compute loss and accumulate the loss values
        loss = self.loss_fn(logits, labels)

        #Compute the accuracy
        preds = torch.argmax(logits, dim=1).flatten()
        accuracy = torch.tensor((preds == labels).cpu().numpy().mean() * 100)

        return {'loss': loss, 'acc': accuracy}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}

        print(f'Accuracy: {avg_val_acc}')

        return {'avg_val_loss': avg_loss, 'logs': tensorboard_logs}
