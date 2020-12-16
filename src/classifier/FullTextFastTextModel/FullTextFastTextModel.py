import csv
import logging
import string
from typing import List

import pandas as pd
import fasttext.util
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from spacy.lang.de.stop_words import STOP_WORDS
import spacy

from sklearn.model_selection import train_test_split
from spacy.lang.de import German

from classifier.TenderClassClassifier import TenderClassClassifier
from src.entity.LabeledTenderCollection import LabelledTenderCollection
from src.entity.Tender import Tender

punctuations = string.punctuation
logger = logging.getLogger(__name__)


class FullTextFastTextModel(TenderClassClassifier):

    def __init__(self):
        self.nlp = spacy.load('de_core_news_sm')
        self.domain_stopwords = ["Ausschreibung", "Bekanntmachung"]
        self.parser = German()
        self.stopwords = list(STOP_WORDS)
        self.stopwords.extend(self.domain_stopwords)
        # fasttext.util.download_model('de', if_exists='ignore')
        # self.create_new_model()

    def classify(self, tenders: List[Tender]):
        pass

    def spacy_tokenizer(self, sentence):
        sentence_tokens = self.parser(sentence)
        sentence_tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in
                           sentence_tokens]
        sentence_tokens = [word for word in sentence_tokens if word not in self.stopwords and word not in punctuations]
        return sentence_tokens

    def save_dataset(self, dataset, name):
        dataset.to_csv(name,
                       index=False,
                       sep=" ",
                       header=None,
                       quoting=csv.QUOTE_NONE,
                       quotechar="",
                       escapechar=" ")

    def train(self, labelled_tenders):
        labelled_tenders_collection = LabelledTenderCollection(labelled_tenders)

        complete_df = pd.DataFrame({"title": labelled_tenders_collection.get_titles(),
                                    "description": labelled_tenders_collection.get_original_language_entity_description(),
                                    "label": labelled_tenders_collection.get_labels()})
        complete_df = complete_df.dropna()

        # tokenize
        logger.info("Tokenize")
        complete_df.iloc[:, 0] = complete_df.iloc[:, 0].apply(lambda x: " ".join(self.spacy_tokenizer(x)))
        complete_df.iloc[:, 1] = complete_df.iloc[:, 1].apply(lambda x: " ".join(self.spacy_tokenizer(x)))

        # prefix label to FastText format
        logger.info("Prefixing")
        complete_df.iloc[:, 2] = complete_df.iloc[:, 2].apply(lambda x: "__label__" + str(x))

        # train / test split
        train_df, val_df = train_test_split(complete_df, test_size=0.1)

        self.save_dataset(train_df[["title", "label"]], "title_train.csv")
        self.save_dataset(val_df[["title", "label"]], "title_val.csv")

        self.save_dataset(train_df[["description", "label"]], "desc_train.csv")
        self.save_dataset(val_df[["description", "label"]], "desc_val.csv")

        logger.info("Training model")
        model_title = fasttext.train_supervised("title_train.csv", wordNgrams=2)
        model_desc = fasttext.train_supervised("desc_train.csv", wordNgrams=2)
        logger.info("Title acc.: " + str(model_title.test("title_val.csv")))
        logger.info("Description acc.: " + str(model_desc.test("desc_val.csv")))

        # train the linear classifier
        svm_train = pd.DataFrame({
            "title_pos_prob": [self.extract_pos_probability(model_title.predict(x)) for x in train_df["title"].values.tolist()],
            "desc_pos_prob": [self.extract_pos_probability(model_desc.predict(x)) for x in train_df["description"].values.tolist()],
            "label": [0 if x == "__label__0" else 1 for x in train_df["label"].values.tolist()]
        })

        X_train, y_train = (svm_train[["title_pos_prob", "desc_pos_prob"]], svm_train["label"])

        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)

        # validation
        # train the linear classifier
        val_svm = pd.DataFrame({
            "title_pos_prob": [self.extract_pos_probability(model_title.predict(x)) for x in val_df["title"].values.tolist()],
            "desc_pos_prob": [self.extract_pos_probability(model_desc.predict(x)) for x in val_df["description"].values.tolist()],
            "label": [0 if x == "__label__0" else 1 for x in val_df["label"].values.tolist()]
        })

        y_pred = clf.predict(val_svm[["title_pos_prob", "desc_pos_prob"]])
        logger.info("Accuracy:", metrics.accuracy_score(val_svm["label"], y_pred))

        tn, fp, fn, tp = confusion_matrix(val_svm["label"], y_pred).ravel()
        logger.info(f"tn: {tn} fp: {fp}")
        logger.info(f"fn: {fn} tp:{tp}")

    def extract_pos_probability(self, fast_text_prediction):
        (label, probability) = fast_text_prediction
        label = 0 if label[0] == "__label__0" else 1
        probability = probability[0]

        return probability if label == 1 else (1 - probability)

    def save_model(self):
        pass

    def create_new_model(self):
        # self.model = fasttext.load_model('cc.en.300.bin')
        pass
