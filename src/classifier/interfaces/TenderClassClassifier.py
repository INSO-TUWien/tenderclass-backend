from abc import ABC, abstractmethod


class TenderClassClassifier(ABC):

    @abstractmethod
    def create_new_model(self):
        pass

    @abstractmethod
    def train(self, labelled_tenders):
        pass

    @abstractmethod
    def classify(self, tenders):
        pass
