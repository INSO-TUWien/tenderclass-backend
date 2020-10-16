import torch
from torch.utils.data import Dataset

from src.classifier.PytorchTransformer.config.PytorchTransformerConfig import PytorchTransformerConfig


class BertDataSet(Dataset):

    def __init__(self, df, config: PytorchTransformerConfig):
        self.titles = df["title"].values
        self.descriptions = df["description"].values
        self.labels = torch.tensor(df["label"].values)

        self.config = config

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        title = self.titles[idx]
        desc = self.descriptions[idx]

        if self.config.truncate:
            title = self.truncate(title, self.config.max_len_title)
            desc = self.truncate(desc, self.config.max_len_desc)

        title_ids, title_mask, desc_ids, desc_mask = ([], [], [], [])

        if self.config.use_title:
            title_ids, title_mask = self.preprocessing_for_bert(title, self.config.tokenizer_title, self.config.max_len_title)

        if self.config.use_desc:
            desc_ids, desc_mask = self.preprocessing_for_bert(desc, self.config.tokenizer_desc, self.config.max_len_desc)

        sample = {'title_input_ids': title_ids, 'title_attention_mask': title_mask, 'description_input_ids': desc_ids, 'description_attention_mask': desc_mask, "label": self.labels[idx]}

        return sample

    def preprocessing_for_bert(self, data, tokenizer, max_len):
        encoded_sent = tokenizer.encode_plus(
            truncation=True,
            text=data,  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=max_len,  # Max length to truncate/pad
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

    def truncate(self, text, max_len):
        splitted = text.split()

        if len(splitted) <= max_len:
            return text

        first_part = splitted[:129]
        second_part = splitted[-383:]
        retstr = "".join(first_part) + " " + " ".join(second_part)
        return retstr
