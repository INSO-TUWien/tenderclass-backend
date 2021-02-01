from marshmallow import Schema, fields, post_load

from src.Models.ModelNameModel import ModelNameModel


class ModelNameValidation(Schema):
    name = fields.String(required=True)

    @post_load
    def make_model(self, data, **kwargs):
        return ModelNameModel(**data)