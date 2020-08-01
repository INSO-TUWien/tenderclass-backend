import pandas as pd
from sklearn.model_selection import train_test_split

from src.classifier.interfaces.MLModelInterface import MlModelInterface
from src.entity.LabeledTenderCollection import LabelledTenderCollection


class FullTexSvmtModel(MlModelInterface):

    def __init__(self):
        pass

    def classify(self, tenders):
        pass

    def train(self, labelled_tenders):
        labelled_tenders_collection = LabelledTenderCollection(labelled_tenders)

        training_df = pd.DataFrame({"descriptions": labelled_tenders_collection.get_descriptions(), "label": labelled_tenders_collection.get_labels()})
        # remove null vlaues (description is not alway set)
        training_df = training_df.dropna()
        X = training_df['descriptions']
        ylabels = training_df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)

    def create_new_model(self):
        pass
