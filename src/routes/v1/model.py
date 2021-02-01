from http.client import BAD_REQUEST

from flask import Blueprint, request, abort
from src.config import tender_trainer
from src.validation.create_from_datasets_validation import CreateFromDatasetsValidation
from src.validation.create_new_validation import CreateNewValidation
from marshmallow import ValidationError

from src.validation.model_name_validation import ModelNameValidation

model_blueprint = Blueprint('model_blueprint', __name__)
create_new_validation = CreateNewValidation()
create_from_datasets_validation = CreateFromDatasetsValidation()
model_name_validation = ModelNameValidation()


@model_blueprint.route("/new", methods=['POST'])
def post_create_new():
    try:
        tender_trainer.create_new()

        return "ok"
    except ValidationError as err:
        abort(BAD_REQUEST, str(err.messages))


@model_blueprint.route("/train-from-datasets", methods=['POST'])
def post_train_from_datasets():
    try:
        model = create_from_datasets_validation.load(request.json)

        tender_trainer.load_and_train(model)

        return "ok"
    except ValidationError as err:
        abort(BAD_REQUEST, str(err.messages))


@model_blueprint.route("/save", methods=['POST'])
def save_model():
    try:
        model = model_name_validation.load(request.json)

        tender_trainer.save(model)

        return "ok"
    except ValidationError as err:
        abort(BAD_REQUEST, str(err.messages))


@model_blueprint.route("/load", methods=['POST'])
def load_model():
    try:
        model = model_name_validation.load(request.json)

        tender_trainer.load(model)

        return "ok"
    except ValidationError as err:
        abort(BAD_REQUEST, str(err.messages))
