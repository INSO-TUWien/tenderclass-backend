# Use src path so that the python interpreter can access all modules
import os
import sys

from src.classifier.FullTextTransformerModel import FullTextTransformerModel
from src.persistence.Persistence import Persistence
from src.service.Recommender import Recommender
from src.service.Trainer import Trainer

sys.path.append(os.getcwd()[:os.getcwd().index('src')])


# TODO: select the Machine Learning model
tender_model = FullTextTransformerModel()
# tender_model = TransformerModel()

tender_recommender = Recommender(tender_model)
tender_trainer = Trainer(tender_model)
tender_persistence = Persistence()
develop = True
