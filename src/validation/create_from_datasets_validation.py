from marshmallow import Schema, fields, post_load

from src.Models.FromDatasetsModelModel import FromDatasetsModel


class CreateFromDatasetsValidation(Schema):
    pos_filename = fields.String(required=True)
    neg_filename = fields.String(required=True)

    @post_load
    def make_model(self, data, **kwargs):
        return FromDatasetsModel(**data)