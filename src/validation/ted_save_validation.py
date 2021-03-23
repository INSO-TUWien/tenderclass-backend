from marshmallow import Schema, fields, post_load

from src.Models.TedSaveModel import TedSaveModel


class TedSaveValidation(Schema):
    amount = fields.Int(required=True)
    search_criteria = fields.String(required=True)
    original_languages = fields.List(fields.String(), required=False)
    languages = fields.List(fields.String(), required=False)
    dataset_name = fields.String(required=True)

    @post_load
    def make_model(self, data, **kwargs):
        return TedSaveModel(**data)
