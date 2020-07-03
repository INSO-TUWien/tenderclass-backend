from http.client import BAD_REQUEST

from flask import Blueprint, request, abort
from src.config import tender_trainer
from src.validation.create_new_validation import CreateNewValidation
from marshmallow import ValidationError

model_blueprint = Blueprint('model_blueprint', __name__)
create_new_validation = CreateNewValidation()


@model_blueprint.route("/new", methods=['POST'])
def post_create_new():
    try:
        model = create_new_validation.load(request.json)

        tender_trainer.create_and_init(model.pos_number, model.pos_search_criteria, model.neg_number, model.neg_search_criteria)

        return "ok"
    except ValidationError as err:
        abort(BAD_REQUEST, str(err.messages))
