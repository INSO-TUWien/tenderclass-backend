import json
from transformers import AutoTokenizer

import pandas as pd
import matplotlib.pyplot as plt

from entity.Tender import Tender


class DevDATAInspector:
    def main(self):
        tenders = self.load("dev_neg_tenders.json") + self.load("dev_pos_tenders.json")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")

        words = []
        tokens_per_word = []

        for tender in tenders:
            langEnt = tender.get_original_language_entity()

            for word in langEnt.title.split():
                words.append(word)

            for word in langEnt.description.split():
                words.append(word)

        for word in words:
            tokens_per_word.append(len(self.tokens_of_sequence(tokenizer, word)))

        print(sum(tokens_per_word) / len(tokens_per_word))

    def tokens_of_sequence(self, tokenizer, data):
        encoded_sent = tokenizer.encode_plus(
            text=data,
            add_special_tokens=False
        )

        input_ids = encoded_sent.get('input_ids')
        return input_ids


    def load(self, path):
        with open(path, 'r', encoding='utf8') as json_file:
            tender_dicts = json.load(json_file)
        tenders = list(map(lambda x: Tender.from_json_dict(x), tender_dicts))
        return tenders

    def sequence_length(self):
        tenders = self.load("dev_neg_tenders.json") + self.load("dev_pos_tenders.json")
        df = pd.DataFrame()
        df["title_len"] = [len((x.get_title("DE")).split()) for x in tenders]
        df["description_len"] = [len((x.get_description("DE")).split()) for x in tenders]

        title_len_fig = plt.figure()
        title_len_fig.suptitle('')
        df.boxplot(column=['title_len'], showfliers=False)

        desc_len_fig = plt.figure()
        desc_len_fig.suptitle('')
        df.boxplot(column=['description_len'], showfliers=False)

        title_len_fig.savefig("title_len.png", format="png")
        desc_len_fig.savefig("desc_len.png", format="png")

if __name__ == '__main__':
    DevDATAInspector().main()