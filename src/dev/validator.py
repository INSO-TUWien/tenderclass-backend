import json
import os
import random
import csv
import sys
import time

# Use src path so that the python interpreter can access all modules
sys.path.append(os.getcwd()[:os.getcwd().index('src')])

from sklearn.model_selection import KFold

from src.classifier.FullTextFastTextModel.FullTextFastTextModel import FullTextFastTextModel
from src.classifier.FullTextFastTextModel.validation.FullTextFastTextModelDescOnly import FullTextFastTextModelDescOnly
from src.classifier.FullTextFastTextModel.validation.FullTextFastTextModelTitleOnlyl import FullTextFastTextModelTitleOnly
from src.classifier.FullTextSvmModel.FullTextSvmModel import FullTextSvmModel
from src.classifier.FullTextSvmModel.validation.FullTextSvmModelDescOnly import FullTextSvmModelDescOnly
from src.classifier.FullTextSvmModel.validation.FullTextSvmModelTitleOnly import FullTextSvmModelTitleOnly
from src.classifier.FullTextTransformerModel.FullTextTransformerModel import FullTextTransformerModel
from src.classifier.FullTextTransformerModel.config.TransformerModelConfig import PytorchTransformerConfig
from src.entity.Tender import Tender

models = {
    "FullTextFastTextModel": FullTextFastTextModel(),
    "FullTextFastTextModelTitleOnly": FullTextFastTextModelTitleOnly(),
    "FullTextFastTextModelDescOnly": FullTextFastTextModelDescOnly(),
    "FullTextTransformerModel": FullTextTransformerModel(PytorchTransformerConfig.bert_german_full()),
    "FullTextTransformerModelTitleOnly": FullTextTransformerModel(PytorchTransformerConfig.bert_german_title_only()),
    "FullTextTransformerModelDescOnly": FullTextTransformerModel(PytorchTransformerConfig.bert_german_description_only()),
    "FullTextSvmModel": FullTextSvmModel(),
    "FullTextSvmModelTitleOnly": FullTextSvmModelTitleOnly(),
    "FullTextSvmModelDescOnly": FullTextSvmModelDescOnly(),
}


def load(path):
    with open(path, 'r', encoding='utf8') as json_file:
        tender_dicts = json.load(json_file)
    tenders = list(map(lambda x: Tender.from_json_dict(x), tender_dicts))
    return tenders


class Validator:
    def __init__(self):
        self.start = time.time()
        labelled_tenders = self.get_tenders()
        #labelled_tenders = labelled_tenders[:1000]
        kfold = KFold(5, True, 1)

        # enumerate splits
        iteration = 0
        for train, test in kfold.split(labelled_tenders):
            iteration = iteration + 1
            print('K %s of 5' % iteration)
            print('train: %s, test: %s' % (len(train), len(test)))
            for name, impl in models.items():
                impl.create_new_model()
                impl.train([labelled_tenders[i] for i in train])
                self.write_result(name, iteration, impl.validate([labelled_tenders[i] for i in test]))
                pass

            #break

    def write_result(self, name, iteration, val_result):
        with open('test_result_' + str(self.start) + '.csv', mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            writer.writerow([name, iteration, val_result.tn, val_result.fp, val_result.fn, val_result.tp,
                                      val_result.accuracy, val_result.precision, val_result.recall, val_result.f1])

    def get_tenders(self):
        pos_path = "dev_pos_tenders.json"
        neg_path = "dev_neg_tenders.json"

        pos_tenders = load(pos_path)
        neg_tenders = load(neg_path)

        pos_labels = [1] * len(pos_tenders)
        neg_labels = [0] * len(neg_tenders)

        labelled_tenders = list(zip(pos_tenders, pos_labels)) + list(zip(neg_tenders, neg_labels))

        random.shuffle(labelled_tenders)
        return labelled_tenders


if __name__ == '__main__':
    Validator()
