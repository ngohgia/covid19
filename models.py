from app import db
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.types import ARRAY


class Result(db.Model):
    __tablename__ = 'results'

    id = db.Column(db.Integer, primary_key=True)
    orig_urls = db.Column(ARRAY(db.String()))
    pred_urls = db.Column(ARRAY(db.String()))
    ref_urls  = db.Column(ARRAY(db.String()))

    def __init__(self, orig_urls, pred_urls, ref_urls):
        self.orig_urls = orig_urls
        self.pred_urls = pred_urls
        self.ref_urls = ref_urls

    def __repr__(self):
        return '<id {}>'.format(self.id)
