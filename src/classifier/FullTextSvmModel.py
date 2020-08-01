import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import string

from src.classifier.interfaces.MLModelInterface import MlModelInterface
from src.entity.LabeledTenderCollection import LabelledTenderCollection


class FullTexSvmtModel(MlModelInterface):

    def __init__(self):
        self.stopwords = list(STOP_WORDS)
        self.nlp = spacy.load("en")
        self.parser = English()
        self.punctuations = string.punctuation

    def classify(self, tenders):
        pass

    def my_tokenizer(self, sentence):
        mytokens = self.parser(sentence)
        mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
        mytokens = [word for word in mytokens if word not in self.stopwords and word not in self.punctuations]
        return mytokens

    class Predictors(TransformerMixin):
        def clean_text(self, text):
            return str(text.strip().lower())

        def transform(self, X, **transform_params):
            return [self.clean_text(text) for text in X]

        def fit(self, X, y, **fit_params):
            return self

        def get_params(self, deep=True):
            return {}

    def train(self, labelled_tenders):
        labelled_tenders_collection = LabelledTenderCollection(labelled_tenders)

        training_df = pd.DataFrame({"descriptions": labelled_tenders_collection.get_descriptions(), "label": labelled_tenders_collection.get_labels()})
        # remove null vlaues (description is not alway set)
        training_df = training_df.dropna()
        X = training_df['descriptions']
        ylabels = training_df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)

        # Vectorization
        vectorizer = CountVectorizer(tokenizer=self.my_tokenizer, ngram_range=(1, 1))
        classifier = LinearSVC()

        tfvectorizer = TfidfVectorizer(tokenizer=self.my_tokenizer)

        # Create the  pipeline to clean, tokenize, vectorize, and classify using"Count Vectorizor"
        pipe_countvect = Pipeline([("cleaner", self.Predictors()),
                                   ('vectorizer', vectorizer),
                                   ('classifier', classifier)])
        # Fit our data
        pipe_countvect.fit(X_train, y_train)
        # Predicting with a test dataset
        sample_prediction = pipe_countvect.predict(X_test)

        # Prediction Results
        # 1 = Positive review
        # 0 = Negative review
        for (sample, pred) in zip(X_test, sample_prediction):
            print(sample, "Prediction=>", pred)

        # Accuracy
        print("Accuracy: ", pipe_countvect.score(X_test, y_test))
        print("Accuracy: ", pipe_countvect.score(X_test, sample_prediction))
        # Accuracy
        print("Accuracy: ", pipe_countvect.score(X_train, y_train))

    def create_new_model(self):
        pass
