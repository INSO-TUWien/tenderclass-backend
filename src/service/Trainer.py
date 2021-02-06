from src.Models.FromDatasetsModelModel import FromDatasetsModel
import random
import logging

from src.Models.ModelNameModel import ModelNameModel
from src.entity.ValidationResult import ValidationResult
from src.persistence.Persistence import Persistence
from src.service.fetcher.Fetcher import Fetcher

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Trainer:
    """
    This class cooordinates training and creation of the machine learning model as well as preparation of data.
    """

    def __init__(self, tender_model):
        self.tender_fetcher = Fetcher()
        self.tender_model = tender_model
        self.persistence = Persistence()

    def create_new(self):
        self.tender_model.create_new_model()

    def save(self, model: ModelNameModel):
        self.tender_model.save(model.name)

    def load(self, model: ModelNameModel):
        self.tender_model.load(model.name)

    def train(self, tender_ids, labels):
        search_arg = " OR ".join(tender_ids)
        search_criteria = f" AND ND=[{search_arg}]"
        tenders = self.tender_fetcher.get(0, search_criteria=search_criteria)

        labelled_tenders = list(map(lambda x: (x, labels[tender_ids.index(x.id)], tenders)))

        self.tender_model.train(labelled_tenders)

    def validate(self, model: FromDatasetsModel) -> ValidationResult:
        pos_tenders = self.persistence.load(model.pos_filename)
        neg_tenders = self.persistence.load(model.neg_filename)

        pos_labels = [1] * len(pos_tenders)
        neg_labels = [0] * len(neg_tenders)

        labelled_tenders = list(zip(pos_tenders, pos_labels)) + list(zip(neg_tenders, neg_labels))

        random.shuffle(labelled_tenders)

        return self.tender_model.validate(labelled_tenders)

    def load_and_train(self, model: FromDatasetsModel):
        pos_tenders = self.persistence.load(model.pos_filename)
        neg_tenders = self.persistence.load(model.neg_filename)

        return self.train_from_entities(neg_tenders, pos_tenders)

    def train_from_entities(self, neg_tenders, pos_tenders):
        pos_labels = [1] * len(pos_tenders)
        neg_labels = [0] * len(neg_tenders)

        labelled_tenders = list(zip(pos_tenders, pos_labels)) + list(zip(neg_tenders, neg_labels))

        random.shuffle(labelled_tenders)

        self.tender_model.train(labelled_tenders)
        logger.info("tenders successfully laoded and labelled")
