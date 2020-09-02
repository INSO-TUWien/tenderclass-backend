class LabelledTenderCollection:

    def __init__(self, labelled_tenders):
        self.labelled_tenders = labelled_tenders

    def get_titles(self):
        return list(map(lambda x: x.get_title("EN"), self.get_tenders()))

    def get_descriptions(self):
        return list(map(lambda x: x.get_description("EN"), self.get_tenders()))

    def get_tenders(self):
        return [i for i, j in self.labelled_tenders]

    def get_labels(self):
        return [int(j) for i, j in self.labelled_tenders]