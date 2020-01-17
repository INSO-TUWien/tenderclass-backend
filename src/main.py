import os
import sys
sys.path.append(os.getcwd()[:os.getcwd().index('src')])

from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint

from src.fetcher.TenderFetcher import TenderFetcher
from src.service.TenderRecommender import TenderRecommender
from src.service.TenderTrainer import TenderTrainer
from datetime import date


app = Flask(__name__)

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'tenderclass-backend': "API specification for the Machine Learning classification solution for public tenders"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)

tender_recommender = TenderRecommender()
tender_trainer = TenderTrainer()


@app.route("/api/v1/tenders", methods=['GET'])
def get_all():
    count = int(request.args.get('count'))
    tenders = tender_recommender.get_all(count)
    return jsonify(list(map(lambda x: x.get_dict(), tenders)))


@app.route("/api/v1/model/recommendations", methods=['GET'])
def get_recommendations():
    count = int(request.args.get('count'))
    today = date.today()
    tenders = tender_recommender.get_recommendations(count, today)
    return jsonify(list(map(lambda x: x.get_dict(), tenders)))


@app.route("/api/v1/model/train", methods=['POST'])
def post_train_tenders():
    body = request.json
    train_tender_ids = body["ids"]
    train_tender_labels = body["labels"]
    tender_trainer.train(train_tender_ids, train_tender_labels)

    return "ok"


@app.route("/api/v1/model/new", methods=['POST'])
def post_create_new():
    body = request.json
    pos_number = body["pos_number"]
    neg_number = body["neg_number"]
    pos_search_criteria = body["pos_search_criteria"]
    neg_search_criteria = body["neg_search_criteria"]

    tender_trainer.create_and_init(pos_number, pos_search_criteria, neg_number, neg_search_criteria)

    return "ok"


if __name__ == "__main__":
    app.run()
