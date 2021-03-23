import json
from sklearn.model_selection import train_test_split

from src.entity.Tender import Tender


class TrainTestSplitter:
    def main(self):
        self.split("dev_pos_tenders")

    def split(self, path):

        tenders = self.load(path + ".json")
        train, test = train_test_split(tenders, test_size=0.1, random_state=42)
        self.save(train, path + "_train.json")
        self.save(test, path + "_test.json")

    def load(self, path):
        with open("../data/" + path, 'r', encoding='utf8') as json_file:
            tender_dicts = json.load(json_file)
        tenders = list(map(lambda x: Tender.from_json_dict(x), tender_dicts))
        return tenders

    def save(self, tenders, path):
        with open("../data/" + path, 'w', encoding='utf8') as json_file:
            json.dump(list(map(lambda x: x.get_dict(), tenders)), json_file, ensure_ascii=False)
            json_file.flush()
            json_file.close()

if __name__ == '__main__':
    TrainTestSplitter().main()