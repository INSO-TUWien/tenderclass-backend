import logging

import joblib
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from spacy.lang.de import German
from spacy.lang.de.stop_words import STOP_WORDS
import string

from src.classifier.TenderClassClassifier import TenderClassClassifier
from src.entity.ValidationResult import ValidationResult
from src.entity.LabeledTenderCollection import LabelledTenderCollection

punctuations = string.punctuation

logger = logging.getLogger(__name__)


class FullTextSvmModel(TenderClassClassifier):

    def __init__(self):
        self.stopwords = list(STOP_WORDS)
        self.domain_stopwords = ["Ausschreibung", "Bekanntmachung"]
        self.stopwords.extend(self.domain_stopwords)
        self.parser = German()
        self.punctuations = string.punctuation
        self.domain_stopwords = ["contract", "system", "service", "tender", "company", "notice", "procurement",
                                 "work", "include", "support", "approximately", "management", "agreement",
                                 "office", "solution", "manage", "product", "design", "program", "project",
                                 "supply", "trust", "equipment"]

        self.stopwords = list(STOP_WORDS)
        self.stopwords.extend(self.domain_stopwords)
        self.model = None
        self.create_new_model()

    def tokenize(self, sentence):
        sentence_tokens = self.parser(sentence)
        sentence_tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in
                           sentence_tokens]
        sentence_tokens = [word for word in sentence_tokens if word not in self.stopwords and word not in punctuations]
        return sentence_tokens

    def classify(self, tenders):
        titles = list(map(lambda x: x.get_title("DE"), tenders))
        predictions = self.model.predict(titles)
        tuples = zip(tenders, predictions)
        selected_tenders = [t for t, p in tuples if p == 1]
        return selected_tenders

    def prepare_data(self, labelled_tenders):
        labelled_tenders_collection = LabelledTenderCollection(labelled_tenders)

        # create the pandas df
        training_df = pd.DataFrame({"titles": labelled_tenders_collection.get_titles("DE"),
                                    "descriptions": labelled_tenders_collection.get_descriptions("DE"),
                                    "label": labelled_tenders_collection.get_labels()})

        # remove null values (description is not alway set)
        training_df = training_df
        training_df.loc[training_df["descriptions"].isnull(), 'descriptions'] = training_df["titles"]
        X = training_df[['titles', 'descriptions']]
        ylabels = training_df['label']

        return X, ylabels

    def train(self, labelled_tenders):
        X, ylabels = self.prepare_data(labelled_tenders)
        self.model.fit(X, ylabels)

    def validate(self, labelled_tenders):
        X, ylabels = self.prepare_data(labelled_tenders)
        y_pred = self.model.predict(X)
        return ValidationResult(ylabels, y_pred)

    def load(self, name):
        self.model = joblib.load("./data/" + name)

    def save(self, name):
        joblib.dump(self.model, "./data/" + name)

    class Extractor(BaseEstimator, TransformerMixin):
        def __init__(self, column):
            self.column = column
            pass

        def transform(self, df, y=None):
            return df[self.column]

        def fit(self, df, y=None):
            return self

    def create_new_model(self):
        pipeline = Pipeline([
            ('union', FeatureUnion(
                transformer_list=[
                    ('titles', Pipeline([
                        ('selector', self.Extractor(column="titles")),
                        ('vect', CountVectorizer(max_features=1000, tokenizer=self.tokenize, ngram_range=(1, 2))),
                        ('tfidf', TfidfTransformer())
                    ])),
                    ('descriptions', Pipeline([
                        ('selector', self.Extractor(column="descriptions")),
                        ('vect', CountVectorizer(max_features=1000, tokenizer=self.tokenize, ngram_range=(1, 2))),
                        ('tfidf', TfidfTransformer())
                    ])),
                ],
            )),
            ('svc', SVC(kernel="linear", random_state=0)),
        ])

        self.model = None
