import os
from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_sqlalchemy import SQLAlchemy
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

from models import *

@app.route('/', methods=['GET'])
def index():
    result = db.session.query(Result).get(1) # validation results
    data = {
      'origUrls': result.orig_urls,
      'predUrls': result.pred_urls,
      'refUrls': result.ref_urls,
    }
    return render_template('index.html', flaskData=data)

@app.route('/eval/<path:filename>')
def eval(filename):
    return send_from_directory('eval', filename, as_attachment=True)

if __name__ == '__main__':
    app.debug = True
    app.run()
