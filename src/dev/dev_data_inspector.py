import pickle
import pandas as pd
import matplotlib.pyplot as plt

from src.entity.LabeledTenderCollection import LabelledTenderCollection


class DevDATAInspector:
    def main(self):
        labelled_tenders = self.load_obj()
        labelled_tenders_collection = LabelledTenderCollection(labelled_tenders)
        df = pd.DataFrame({"title": labelled_tenders_collection.get_titles(), "description": labelled_tenders_collection.get_descriptions(), "label": labelled_tenders_collection.get_labels()})
        df["title_len"] = df["title"].str.split().str.len()
        df["description_len"] = df["description"].str.split().str.len()

        title_len_fig = plt.figure()
        df.boxplot(column=['title_len'])

        desc_len_fig = plt.figure()
        df.boxplot(column=['description_len'])

        title_len_fig.savefig("title_len.png", format="png")
        desc_len_fig.savefig("desc_len.png", format="png")

    def load_obj(self):
        with open('labelled_tenders.pkl', 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    DevDATAInspector().main()