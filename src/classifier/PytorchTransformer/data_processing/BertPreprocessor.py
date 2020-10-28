import string

import torch

from src.classifier.PytorchTransformer.config.PytorchTransformerConfig import PytorchTransformerConfig
from src.entity.Tender import Tender


class BertPreprocessor:

    @staticmethod
    def get_sample_from_tender(config: PytorchTransformerConfig, tender: Tender):
        return BertPreprocessor.get_sample(config, tender.get_title("EN"), tender.original_lang_entity.description)

    @staticmethod
    def get_sample(config: PytorchTransformerConfig, title: string, desc: string, label=-1):
        if config.truncate:
            title = BertPreprocessor.truncate(title, config.max_len_title)
            desc = BertPreprocessor.truncate(desc, config.max_len_desc)

        title_ids, title_mask, desc_ids, desc_mask = ([], [], [], [])

        if config.use_title:
            title_ids, title_mask = BertPreprocessor.preprocessing_for_bert(title, config.tokenizer_title,
                                                                          config.max_len_title)

        if config.use_desc:
            desc_ids, desc_mask = BertPreprocessor.preprocessing_for_bert(desc, config.tokenizer_desc,
                                                                        config.max_len_desc)

        sample = {'title_input_ids': title_ids, 'title_attention_mask': title_mask, 'description_input_ids': desc_ids,
                  'description_attention_mask': desc_mask, "label": label}

        return sample

    @staticmethod
    def preprocessing_for_bert(data, tokenizer, max_len):
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

    @staticmethod
    def truncate(text, max_len):
        splitted = text.split()

        if len(splitted) <= max_len:
            return text

        first_part = splitted[:129]
        second_part = splitted[-383:]
        retstr = "".join(first_part) + " " + " ".join(second_part)
        return retstr