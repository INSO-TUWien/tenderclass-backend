import torch
import torch.nn as nn


# Create the BertClassfier class
class PyTorchTransformer(nn.Module):
    """Bert Model for Classification Tasks.
    """

    def __init__(self, modelLong, modelShort, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(PyTorchTransformer, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
        self.bertTitles = modelShort
        self.bertDescriptions = modelShort

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in*2, H),
            nn.ReLU(),
            ##nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bertTitles.parameters():
                param.requires_grad = False

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

        outputsDescriptions= self.bertTitles(input_ids=input_ids,
                                        attention_mask=attention_mask)

        concat_output = torch.cat((outputsTitles[0][:, 0, :], outputsDescriptions[0][:, 0, :]), dim=1)

        # Feed input to classifier to compute logits
        logits = self.classifier(concat_output)

        return logits