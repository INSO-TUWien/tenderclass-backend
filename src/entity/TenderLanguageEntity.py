class TenderLanguageEntity:
    """
    This class holds the title and description of one tender for a certain language.
    """

    def __init__(self, title: str, description: str, link: str):
        self.title: str = title
        self.description: str = description
        self.link: str = link
