class TedSaveModel:
    def __init__(self, amount, search_criteria, dataset_name, original_languages=None, languages=None):
        self.amount: int = amount
        self.search_criteria: str = search_criteria
        self.original_languages: list[str] = original_languages
        self.languages: list[str] = languages
        self.dataset_name: str = dataset_name
