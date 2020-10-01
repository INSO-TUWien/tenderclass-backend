import torch
from torch.utils.data import Dataset


class BertDataSet(Dataset):

    def __init__(self, df, max_len, tokenizer_short, tokenizer_long):
        self.titles = df["title"].values
        self.descriptions = df["description"].values
        self.labels = torch.tensor(df["label"].values)

        self.max_len = max_len
        self.tokenizer_short = tokenizer_short
        self.tokenizer_long = tokenizer_long

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        title_ids, title_mask = self.preprocessing_for_bert(self.titles[idx], self.tokenizer_short)
        desc_ids, desc_mask = self.preprocessing_for_bert(self.titles[idx], self.tokenizer_long)
        sample = {'title_input_ids': title_ids, 'title_attention_mask': title_mask, 'description_input_ids': desc_ids, 'description_attention_mask': desc_mask, "label": self.labels[idx]}

        return sample

    def preprocessing_for_bert(self, data, tokenizer):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                      tokens should be attended to by the model.
        """
        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []

        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            truncation=True,
            text=data,  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=self.max_len,  # Max length to truncate/pad
            pad_to_max_length=True,  # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids = encoded_sent.get('input_ids')
        attention_masks = encoded_sent.get('attention_mask')

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks
