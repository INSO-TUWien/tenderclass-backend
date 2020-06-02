from flask import Blueprint, request
from src.config import tender_trainer

model_blueprint = Blueprint('model_blueprint', __name__)


@model_blueprint.route("/new", methods=['POST'])
def post_create_new():
    body = request.json
    pos_number = body["pos_number"]
    neg_number = body["neg_number"]
    pos_search_criteria = body["pos_search_criteria"]
    neg_search_criteria = body["neg_search_criteria"]

    tender_trainer.create_and_init(pos_number, pos_search_criteria, neg_number, neg_search_criteria)

    return "ok"
