from flask import Blueprint, request

from src.config import tender_recommender, tender_persistence, tender_trainer

persistence_blueprint = Blueprint('persistence_blueprint', __name__)


### Additional endpoints for saving tenders and training tenders from the file system.
### NOT documented yet because it is not scope of this bachelor thesis

@persistence_blueprint.route("/save", methods=['POST'])
def post_save():
    path = request.json["path"]
    search_criteria = request.json["search_criteria"]
    count = int(request.args.get('count'))
    tenders = tender_recommender.get_all(count, search_criteria=search_criteria)
    tender_persistence.save(tenders, path)

    return "ok"


@persistence_blueprint.route("/train", methods=['POST'])
def post_train_from_persistence():
    neg_path = request.json["neg_path"]
    pos_path = request.json["pos_path"]
    neg_tenders = tender_persistence.load(neg_path)
    pos_tenders = tender_persistence.load(pos_path)
    tender_trainer.train_from_entities(neg_tenders, pos_tenders)

    return "ok"
