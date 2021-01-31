from typing import List

from src.Models.TedSaveModel import TedSaveModel
from src.entity.Tender import Tender
from src.persistence.Persistence import Persistence
from src.service.fetcher.ted.TedFetcher import TedFetcher


class Fetcher:
    """
    This class fetches tenders from provides databases.
    Currently, only TED serves as database.
    """

    def __init__(self):
        self.ted_fetcher = TedFetcher()
        self.persistence = Persistence()

    def get(self, count: int, load_documents: bool = False, search_criteria: str = "", languages: List[str] = ["DE", "EN"], page_offset: int = 0) -> List[Tender]:
        return self.ted_fetcher.get(count, load_documents, search_criteria, languages, languages, page_offset)

    def fetch_and_save(self, model: TedSaveModel):
        tenders = self.ted_fetcher.from_ted_save_model(model)
        self.persistence.save(tenders, model.dataset_name + ".json")
        return

