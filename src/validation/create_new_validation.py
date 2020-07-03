from marshmallow import Schema, fields, post_load

from src.Models.NewModelModel import NewModelModel


class CreateNewValidation(Schema):
    pos_number = fields.Int(required=True)
    neg_number = fields.Int(required=True)
    pos_search_criteria = fields.String(required=True)
    neg_search_criteria = fields.String(required=True)

    @post_load
    def make_user(self, data, **kwargs):
        return NewModelModel(**data)