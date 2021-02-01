from abc import ABC, abstractmethod


class TenderClassClassifier(ABC):

    @abstractmethod
    def create_new_model(self, *args):
        pass

    @abstractmethod
    def train(self, labelled_tenders):
        pass

    @abstractmethod
    def validate(self, labelled_tenders):
        pass

    @abstractmethod
    def classify(self, tenders):
        pass

    @abstractmethod
    def save(self, name):
        pass

    @abstractmethod
    def load(self, name):
        pass
