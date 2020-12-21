import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from transformers import AdamW, get_linear_schedule_with_warmup

from classifier.FullTextTransformerModel.config import TransformerModelConfig


class PyTorchTransformerLightning(LightningModule):

    def __init__(self, config: TransformerModelConfig, total_steps=0):
        super(PyTorchTransformerLightning, self).__init__()

        self.config = config

        # Specify number of input feature, size of hidden state and number of output features
        D_in, H, D_out = 768, 50, 2

        # Instantiate pretrained model
        self.model_title = config.model_title if config.model_title is not None else self.empty_model
        self.model_desc = config.model_desc if config.model_desc is not None else self.empty_model

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in * self.config.num_models, H),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(H, D_out),
            nn.Sigmoid()
        )

        # Freeze the pretrained model
        if config.freeze_pretrained_layers:
            if config.model_title is not None:
                for param in self.model_title.parameters():
                    param.requires_grad = False
            if config.model_desc is not None:
                for param in self.model_desc.parameters():
                    param.requires_grad = False

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.total_steps = total_steps

    def empty_model(self, input_ids, attention_mask):
        return None

    def set_total_training_steps(self, training_steps):
        self.total_steps = training_steps

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          lr=5e-5,  # Default learning rate
                          eps=1e-8  # Default epsilon value
                          )

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value
                                                    num_training_steps=self.total_steps)

        return [optimizer], [scheduler]

    def forward(self, title_input_ids, title_attention_masks, description_input_ids, description_attention_masks):
        # Feed input to BERT
        outputs_titles = self.model_title(input_ids=title_input_ids, attention_mask=title_attention_masks)
        outputs_descs = self.model_desc(input_ids=description_input_ids, attention_mask=description_attention_masks)

        if self.config.use_title is False:
            output = outputs_descs[0][:, 0, :]
        elif self.config.use_desc is False:
            output = outputs_titles[0][:, 0, :]
        else:
            output = torch.cat((outputs_titles[0][:, 0, :], outputs_descs[0][:, 0, :]), dim=1)

        logits = self.classifier(output)

        return logits

    def training_step(self, batch, batch_idx):
        title_input_ids = batch["title_input_ids"]
        title_attention_masks = batch["title_attention_mask"]
        description_input_ids = batch["description_input_ids"]
        description_attention_masks = batch["description_attention_mask"]
        labels = batch["label"]

        logits = self.forward(title_input_ids, title_attention_masks, description_input_ids,
                              description_attention_masks)

        correct = logits.argmax(dim=1).eq(labels).sum().item()

        total = len(labels)

        loss = self.loss_fn(logits, labels)
        logs = {'train_loss': loss}

        self.logger.log_metrics({"Test Accuracy": correct / total, "Test Loss": loss})

        return {'loss': loss,
                'logs': logs,
                "correct": correct,
                "total": total
                }

    def validation_step(self, batch, batch_idx):
        title_input_ids = batch["title_input_ids"]
        title_attention_masks = batch["title_attention_mask"]
        description_input_ids = batch["description_input_ids"]
        description_attention_masks = batch["description_attention_mask"]
        labels = batch["label"]

        logits = self.forward(title_input_ids, title_attention_masks, description_input_ids,
                              description_attention_masks)

        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1).flatten()
        accuracy = torch.tensor((preds == labels).cpu().numpy().mean() * 100)

        self.logger.log_metrics({"Val Accuracy": accuracy, "Val Loss": loss})

        return {'loss': loss, 'acc': accuracy}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}

        print(f'Accuracy: {avg_val_acc}')
        self.logger.log_metrics({"Epoch Val Accuracy": avg_val_acc, "Epoch Val Loss": avg_loss})

        return {'avg_val_loss': avg_loss, 'logs': tensorboard_logs}
