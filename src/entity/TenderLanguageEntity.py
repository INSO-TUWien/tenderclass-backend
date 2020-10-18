class TenderLanguageEntity:
    """
    This class holds the title and description of one tender for a certain language.
    """

    def __init__(self, title: str, description: str, link: str):
        self.title: str = title
        self.description: str = description
        self.link: str = link

    @classmethod
    def from_json_dict(cls, serialized_dict):
        title = serialized_dict["title"]
        description = serialized_dict["description"]
        link = serialized_dict["link"]

        return cls(title, description, link)

    def get_dict(self):
        lang_entity = {"title": self.title, "description": self.description, "link": self.link}
        return lang_entity
