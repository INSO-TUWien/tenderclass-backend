from transformers import AutoTokenizer, AutoModel


class PytorchTransformerConfig:

    def __init__(self, tokenizer_title, tokenizer_desc, model_title, model_desc, max_len_title=512, max_len_desc=512,
                 batch_size=32, epochs=10, truncate=False, use_title=True, use_description=True, freeze_bert=True, name="Transformer-Model"):

        self.name = name
        self.tokenizer_title = tokenizer_title
        self.tokenizer_desc = tokenizer_desc
        self.model_title = model_title
        self.model_desc = model_desc
        self.max_len_title = max_len_title
        self.max_len_desc = max_len_desc
        self.batch_size = batch_size
        self.epochs = epochs
        self.truncate = truncate
        self.use_title = use_title
        self.use_desc = use_description
        self.freeze_bert = freeze_bert

        if self.use_desc is False:
            self.model_desc = None
            self.tokenizer_desc = None

        if self.use_title is False:
            self.model_title = None
            self.tokenizer_title = None

        self.num_models = 2 if (self.use_title and self.use_desc) else 1

    @classmethod
    def standard_model(cls):
        tokenizer_desc = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
        model_desc = AutoModel.from_pretrained("bert-base-multilingual-uncased")
        tokenizer_title = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model_title = AutoModel.from_pretrained("distilbert-base-uncased")
        max_len_title = 75
        max_len_desc = 300
        batch_size = 32
        epochs = 2

        return PytorchTransformerConfig(tokenizer_title, tokenizer_desc, model_title, model_desc, max_len_title,
                                        max_len_desc, batch_size, epochs, name="StandardModel")

    @classmethod
    def train_bert_layers(cls):
        tokenizer_desc = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
        model_desc = AutoModel.from_pretrained("bert-base-multilingual-uncased")
        tokenizer_title = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model_title = AutoModel.from_pretrained("distilbert-base-uncased")
        max_len_title = 75
        max_len_desc = 300
        batch_size = 32
        epochs = 4

        return PytorchTransformerConfig(tokenizer_title, tokenizer_desc, model_title, model_desc, max_len_title,
                                        max_len_desc, batch_size, epochs, name="TrainBertLayers", freeze_bert=False)

    @classmethod
    def xlm_model(cls):
        tokenizer_desc = AutoTokenizer.from_pretrained("xlm-roberta-base")
        model_desc = AutoModel.from_pretrained("xlm-roberta-base")
        tokenizer_title = AutoTokenizer.from_pretrained("distilbert-base-cased")
        model_title = AutoModel.from_pretrained("distilbert-base-cased")
        max_len_title = 75
        max_len_desc = 300
        batch_size = 32
        epochs = 20

        return PytorchTransformerConfig(tokenizer_title, tokenizer_desc, model_title, model_desc, max_len_title,
                                        max_len_desc, batch_size, epochs, name="XLMModel")

    @classmethod
    def description_only(cls):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
        model = AutoModel.from_pretrained("bert-base-multilingual-uncased")

        return PytorchTransformerConfig(None, tokenizer, None, model, use_title=False, max_len_desc=300, batch_size=128, name="description_only")

    @classmethod
    def title_only(cls):
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
        model = AutoModel.from_pretrained("distilbert-base-cased")

        return PytorchTransformerConfig(tokenizer, None, model, None, use_description=False, max_len_title=75, batch_size=128,
                                        name="description_only")