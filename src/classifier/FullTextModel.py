import string
import logging

from src.classifier.interfaces.MLModelInterface import MlModelInterface
from src.entity.LabeledTenderCollection import LabelledTenderCollection

logger = logging.getLogger(__name__)

LANGUAGE = "DE"
MODEL_NAME = "scikit_model"
punctuations = string.punctuation


class FullTextModel(MlModelInterface):

    def __init__(self):
        pass

    def classify(self, tenders):
        pass

    def train(self, labelled_tenders):
        labelled_tenders_collection = LabelledTenderCollection(labelled_tenders)
        tenders = labelled_tenders_collection.get_tenders()
        labels = labelled_tenders_collection.get_labels()
        descriptions = labelled_tenders_collection.get_descriptions()
        titles = labelled_tenders_collection.get_titles()
        pass

    def create_new_model(self):
        pass
