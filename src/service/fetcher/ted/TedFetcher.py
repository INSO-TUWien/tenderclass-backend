from typing import List

from src.Models.TedSaveModel import TedSaveModel
from src.entity.Tender import Tender
from src.service.fetcher.ted.TedDownloader import TedDownloader
from src.service.fetcher.ted.TedExtractor import TedExtractor
import sys


class TedFetcher:
    """
    This class fetches the tenders from the TED downlaoder, parses them and returns them as list of tender entities.
    """

    # maximum number of tenders which should be loaded per API call to the TED database
    MAX_PAGE_COUNT = 1000

    def __init__(self):
        self.ted_downloader = TedDownloader()
        self.ted_extractor = TedExtractor()

    def from_ted_save_model(self, model: TedSaveModel):
        return self.get(count=model.amount, search_criteria=model.search_criteria,
                        original_languages=model.original_languages, languages=model.languages)

    def get(self, count: int, load_documents: bool = False, search_criteria: str = "", original_languages=[],
            languages: List[str] = ["DE", "EN"], page_offset: int = 0):

        if original_languages is None:
            original_languages = []

        if languages is None:
            languages = []

        # each original language must occur in the ifnal tender entity
        for language in original_languages:
            if language not in languages:
                languages.append(language)

        if count <= 0:
            count = sys.maxsize

        ted_docs = []
        page = 1
        last_docs_count = -1

        while last_docs_count < len(ted_docs):
            last_docs_count = len(ted_docs)

            xml_docs = self.ted_downloader.get_xml_contracts(page, self.MAX_PAGE_COUNT, search_criteria, page_offset)

            for xml_doc in xml_docs:
                if xml_doc is not None:

                    doc: Tender = self.ted_extractor.extract(xml_doc, languages)

                    if doc is not None:
                        documents = []
                        if load_documents:
                            pass
                            # doc_links = self.ted_extractor.extract_doc_links(xml_doc)
                            # logger.info("found doc links: " + str(doc_links))
                            # for doc_link in doc_links:
                            #    documents.append(doc_parse.get_doc_content(doc_link))
                            # TODO add document links to tender

                        # if filter by orignal language then just use tender in those languages
                        if len(original_languages) > 0:
                            if doc.original_lang in original_languages:
                                ted_docs.append(doc)
                        else:
                            ted_docs.append(doc)

                        if len(ted_docs) >= count:
                            return ted_docs
            page += 1

        return ted_docs
