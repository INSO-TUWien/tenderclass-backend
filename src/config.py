from src.classifier.FullTextFastTextModel.FullTextFastTextModel import FullTextFastTextModel
from src.classifier.FullTextSvmModel.FullTextSvmModel import FullTextSvmModel
from src.classifier.FullTextTransformerModel.FullTextTransformerModel import FullTextTransformerModel
from src.classifier.SpacyScikitModel.SpacyScikitModel import SpacyScikitModel
from src.persistence.Persistence import Persistence
from src.service.Recommender import Recommender
from src.service.Trainer import Trainer


# TODO: select the Machine Learning model
#tender_model = FullTextTransformerModel()
tender_model = FullTextSvmModel()
# tender_model = TransformerModel()

tender_recommender = Recommender(tender_model)
tender_trainer = Trainer(tender_model)
tender_persistence = Persistence()
develop = True
