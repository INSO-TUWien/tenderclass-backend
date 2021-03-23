import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import spacy
import string
import logging

from src.classifier.TenderClassClassifier import TenderClassClassifier
from src.entity.LabeledTenderCollection import LabelledTenderCollection
from src.entity.ValidationResult import ValidationResult

logger = logging.getLogger(__name__)

LANGUAGE = "DE"
MODEL_NAME = "scikit_model"
punctuations = string.punctuation


class SpacyScikitModel(TenderClassClassifier):

    def __init__(self):
        if LANGUAGE == "DE":
            from spacy.lang.de.stop_words import STOP_WORDS
            self.domain_stopwords = ["Ausschreibung", "Bekanntmachung"]
            from spacy.lang.de import German
            self.parser = German()
        elif LANGUAGE == "EN":
            from spacy.lang.en.stop_words import STOP_WORDS
            self.domain_stopwords = ["contract", "system", "service", "tender", "company", "notice", "procurement",
                                     "work", "include", "support", "approximately", "management", "agreement",
                                     "office", "solution", "manage", "product", "design", "program", "project",
                                     "supply", "trust", "equipment"]
            from spacy.lang.en import English
            self.parser = English()
        else:
            raise Exception("unknown language")

        self.stopwords = list(STOP_WORDS)
        self.stopwords.extend(self.domain_stopwords)
        self.pipe = None
        self.create_new_model()

    def spacy_tokenizer(self, sentence):
        sentence_tokens = self.parser(sentence)
        sentence_tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in sentence_tokens]
        sentence_tokens = [word for word in sentence_tokens if word not in self.stopwords and word not in punctuations]
        return sentence_tokens

    class Predictors(TransformerMixin):

        def __clean_text(self, text):
            if text is None:
                return ""
            return str(text).strip().lower()

        def transform(self, X, **transform_params):
            return [self.__clean_text(text) for text in X]

        def fit(self, X, y=None, **fit_params):
            return self

        def get_params(self, deep=True):
            return {}

    def __convert_to_input(self, tenders):
        titles = list(map(lambda x: x.get_title(LANGUAGE), tenders))
        return titles

    def classify(self, tenders):
        titles = self.__convert_to_input(tenders)
        predictions = self.pipe.predict(titles)
        tuples = zip(tenders, predictions)
        selected_tenders = [t for t, p in tuples if p == 1]
        return selected_tenders

    def load(self, name):
        self.pipe = joblib.load("./data/" + name)

    def save(self, name):
        joblib.dump(self.pipe, "./data/" + name)

    def prepare_data(self, labelled_tenders):
        tenders = [i for i, j in labelled_tenders]
        labels = [j for i, j in labelled_tenders]
        titles = self.__convert_to_input(tenders)

        training_df = pd.DataFrame({"title": titles, "label": labels})
        T = training_df["title"]
        y = training_df["label"]

        return T, y

    def train(self, labelled_tenders):
        X, ylabels = self.prepare_data(labelled_tenders)
        self.pipe.fit(X, ylabels)

    def validate(self, labelled_tenders):
        X, ylabels = self.prepare_data(labelled_tenders)
        y_pred = self.pipe.predict(X)
        return ValidationResult(ylabels, y_pred)

    def create_new_model(self):
        vectorizer = CountVectorizer(tokenizer=self.spacy_tokenizer, ngram_range=(1, 2))

        classifier = LinearSVC()
        predictor = self.Predictors()

        self.pipe = Pipeline([("cleaner", predictor),
                              ('vectorizer', vectorizer),
                              ('classifier', classifier)])
