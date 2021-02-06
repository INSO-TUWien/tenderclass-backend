from src.service.fetcher.Fetcher import Fetcher


class Recommender:
    """
    This class gets all tenders from today, classifies them and returns only the positive tenders.
    """

    def __init__(self, tender_model):
        self.tender_fetcher = Fetcher()
        self.tender_model = tender_model
        self.cached_search_criteria = ""

    def get_recommendations(self, count, search_criteria = ""):
        tenders = self.tender_fetcher.get(count, search_criteria=search_criteria)
        return self.tender_model.classify(tenders)

    def get_all(self, count, search_criteria=""):
        tenders = self.tender_fetcher.get(count, search_criteria=search_criteria)
        return tenders

