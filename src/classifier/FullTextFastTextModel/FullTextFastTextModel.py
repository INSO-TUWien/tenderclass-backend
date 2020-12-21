import csv
import logging
import string
from typing import List

import pandas as pd
import fasttext.util
from sklearn import svm
from spacy.lang.de.stop_words import STOP_WORDS

from spacy.lang.de import German

from classifier.TenderClassClassifier import TenderClassClassifier
from entity.ValidationResult import ValidationResult
from src.entity.LabeledTenderCollection import LabelledTenderCollection
from src.entity.Tender import Tender

punctuations = string.punctuation
logger = logging.getLogger(__name__)


class FullTextFastTextModel(TenderClassClassifier):

    def __init__(self):
        self.domain_stopwords = ["Ausschreibung", "Bekanntmachung"]
        self.parser = German()
        self.stopwords = list(STOP_WORDS)
        self.stopwords.extend(self.domain_stopwords)
        self.fast_text_model = None
        self.svm_average_model = None
        self.create_new_model()

    def predict(self, df):
        val_svm = pd.DataFrame({
            "title_pos_prob": [self.extract_pos_probability(self.fast_text_model.predict(x)) for x in
                               df["title"].values.tolist()],
            "desc_pos_prob": [self.extract_pos_probability(self.fast_text_model.predict(x)) for x in
                              df["description"].values.tolist()]
        })

        return self.svm_average_model.predict(val_svm[["title_pos_prob", "desc_pos_prob"]])

    def classify(self, tenders: List[Tender]):
        pass

    def spacy_tokenizer(self, sentence):
        sentence_tokens = self.parser(sentence)
        sentence_tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in
                           sentence_tokens]
        sentence_tokens = [word for word in sentence_tokens if word not in self.stopwords and word not in punctuations]
        return sentence_tokens

    def prepare_data(self, labelled_tenders):
        labelled_tenders_collection = LabelledTenderCollection(labelled_tenders)

        complete_df = pd.DataFrame({"title": labelled_tenders_collection.get_titles("DE"),
                                    "description": labelled_tenders_collection.get_descriptions("DE"),
                                    "label": labelled_tenders_collection.get_labels()})
        complete_df = complete_df.dropna()

        # tokenize
        logger.info("Tokenize")
        complete_df.iloc[:, 0] = complete_df.iloc[:, 0].apply(lambda x: " ".join(self.spacy_tokenizer(x)))
        complete_df.iloc[:, 1] = complete_df.iloc[:, 1].apply(lambda x: " ".join(self.spacy_tokenizer(x)))

        # prefix label to FastText format
        logger.info("Prefixing")
        complete_df.iloc[:, 2] = complete_df.iloc[:, 2].apply(lambda x: "__label__" + str(x))

        return complete_df

    def save_dataset(self, dataset, name):
        dataset.to_csv(name,
                       index=False,
                       sep=" ",
                       header=None,
                       quoting=csv.QUOTE_NONE,
                       quotechar="",
                       escapechar=" ")

    def train(self, labelled_tenders):
        complete_df = self.prepare_data(labelled_tenders)

        title_df = complete_df[["title", "label"]]
        title_df.columns = ['value', 'label']

        desc_df = complete_df[["description", "label"]]
        desc_df.columns = ['value', 'label']

        fasttext_training_df = pd.concat([title_df, desc_df])

        self.save_dataset(fasttext_training_df, "fasttext_train.csv")

        logger.info("Training model")
        self.fast_text_model = fasttext.train_supervised("fasttext_train.csv", wordNgrams=2)

        # train the linear classifier
        svm_train = pd.DataFrame({
            "title_pos_prob": [self.extract_pos_probability(self.fast_text_model.predict(x)) for x in
                               complete_df["title"].values.tolist()],
            "desc_pos_prob": [self.extract_pos_probability(self.fast_text_model.predict(x)) for x in
                              complete_df["description"].values.tolist()],
            "label": [0 if x == "__label__0" else 1 for x in complete_df["label"].values.tolist()]
        })

        X, y = (svm_train[["title_pos_prob", "desc_pos_prob"]], svm_train["label"])

        self.svm_average_model = svm.SVC(kernel='linear')
        self.svm_average_model.fit(X, y)

    def validate(self, labelled_tenders):
        complete_df = self.prepare_data(labelled_tenders)
        y_pred = self.predict(complete_df)
        y_labels = [0 if x == "__label__0" else 1 for x in complete_df["label"].values.tolist()]

        return ValidationResult(y_labels, y_pred)

    def extract_pos_probability(self, fast_text_prediction):
        (label, probability) = fast_text_prediction
        label = 0 if label[0] == "__label__0" else 1
        probability = probability[0]

        return probability if label == 1 else (1 - probability)

    def save_model(self):
        pass

    def create_new_model(self):
        self.fast_text_model = None
        self.svm_average_model = None
