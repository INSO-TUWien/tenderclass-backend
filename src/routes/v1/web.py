from flask import Blueprint, request, jsonify
from datetime import date
from datetime import datetime
from src.config import tender_recommender, tender_trainer

web_blueprint = Blueprint('web_blueprint', __name__)


@web_blueprint.route("/recommendations", methods=['GET'])
def get_recommendations():
    # use query parameters to overwrite default count and date
    count = int(request.args.get('count'))
    if count is None:
        # download all tenders (indicated by count=0)
        count = 0
    today = request.args.get('date')
    if today is None:
        # DEFAULT: get tenders of today
        today = datetime.strftime(date.today(), "%Y%m%d")
    search_criteria = " AND PD=[" + today + "]"
    tenders = tender_recommender.get_recommendations(count, search_criteria)
    return jsonify(list(map(lambda x: x.get_dict(), tenders)))


@web_blueprint.route("/train", methods=['POST'])
def post_train_from_web():
    body = request.json
    train_tender_ids = body["ids"]
    train_tender_labels = body["labels"]
    tender_trainer.train(train_tender_ids, train_tender_labels)

    return "ok"


### Additional endpoints for saving tenders and training tenders from the file system.
### NOT documented yet because it is not scope of this bachelor thesis

@web_blueprint.route("/", methods=['GET'])
def get_all():
    count = int(request.args.get('count'))
    date_filter = request.args.get('date')
    search_criteria = ""
    if date_filter:
        search_criteria = " AND PD=[" + date_filter + "]"
    tenders = tender_recommender.get_all(count, search_criteria=search_criteria)
    return jsonify(list(map(lambda x: x.get_dict(), tenders)))
