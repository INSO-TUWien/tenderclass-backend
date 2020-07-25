# Use src path so that the python interpreter can access all modules
import os
import sys

from src.classifier.FullTextModel import FullTextModel

sys.path.append(os.getcwd()[:os.getcwd().index('src')])

from src.classifier.SpacyScikitModel import SpacyScikitModel

from src.persistence.Persistence import Persistence
from src.service.Recommender import Recommender
from src.service.Trainer import Trainer

# TODO: select the Machine Learning model
tender_model = FullTextModel()
# tender_model = TransformerModel()

tender_recommender = Recommender(tender_model)
tender_trainer = Trainer(tender_model)
tender_persistence = Persistence()
