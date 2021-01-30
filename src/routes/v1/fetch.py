from http.client import BAD_REQUEST

from flask import Blueprint, request, abort
from marshmallow import ValidationError

from src.config import tender_fetcher
from src.validation.ted_save_validation import TedSaveValidation

fetch_blueprint = Blueprint('download_blueprint', __name__)
ted_save_validation = TedSaveValidation()


@fetch_blueprint.route("/ted", methods=['POST'])
def post_create_new():
    try:
        model = ted_save_validation.load(request.json)
        tender_fetcher.fetch_and_save(model)

        return "ok"
    except ValidationError as err:
        abort(BAD_REQUEST, str(err.messages))
