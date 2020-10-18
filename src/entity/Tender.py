from typing import List

from src.entity.TenderLanguageEntity import TenderLanguageEntity


class Tender:
    """
    This class serves as entity for one tender. It holds the id, the CPV codes and a number of language entities,
    where each of which is a collection of title and description in a certain language.
    """

    @classmethod
    def from_json_dict(cls, serialized_dict):
        id = serialized_dict["id"]
        original_lang = serialized_dict["original_lang"]
        cpvs = serialized_dict["cpvs"]
        lang_entities = {}

        serialized_original_lang_entity = serialized_dict["original_lang_entity"]
        original_lang_entity = TenderLanguageEntity.from_json_dict(serialized_original_lang_entity)

        for e in serialized_dict["languageentities"]:
            lang_entry = TenderLanguageEntity(e["title"], e["description"], e["link"])
            lang_entities[e["language"]] = lang_entry
        return cls(id, cpvs, lang_entities, original_lang, original_lang_entity)

    def __init__(self, id: str, cpvs: List[str], lang_entities=None, original_lang=None, original_lang_entity=None):
        self.original_lang = original_lang
        self.original_lang_entity: TenderLanguageEntity = original_lang_entity
        self.id = id
        self.cpvs = cpvs
        if lang_entities is None:
            lang_entities = {}
        self.lang_entities = lang_entities


    def add_language_entity(self, language_key, title, description="", link=""):
        entity = TenderLanguageEntity(title, description, link)
        self.lang_entities[language_key] = entity

    def set_original_language_entity(self, language_key, title, description="", link=""):
        entity = TenderLanguageEntity(title, description, link)
        self.original_lang_entity = entity
        self.original_lang = language_key

    def get_title(self, language):
        return self.lang_entities[language].title

    def get_description(self, language):
        return self.lang_entities[language].description

    def get_dict(self):
        contract = {"id": self.id, "cpvs": list(self.cpvs), "original_lang": self.original_lang}
        lang_list = []

        original_lang_entity = self.original_lang_entity.get_dict()
        contract["original_lang_entity"] = original_lang_entity

        for k, v in self.lang_entities.items():
            lang_entry = {"language": k, "title": v.title, "description": v.description, "link": v.link}
            lang_list.append(lang_entry)
        contract["languageentities"] = lang_list

        return contract
