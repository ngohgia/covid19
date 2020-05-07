import os
from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_sqlalchemy import SQLAlchemy
from config import Config
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

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

@app.route('/validation/<path:filename>')
def eval(filename):
    return send_from_directory('validation', filename, as_attachment=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
  if request.method == 'POST':
    if 'file' not in request.files:
        raise("No file part in request")
    file = request.files['file']
    if file.filename == '':
        raise("No selected file")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_DIR'], filename))
    else:
        raise("Invalid file type")
  result = db.session.query(Result).get(1) # validation results
  data = {
    'origUrls': result.orig_urls,
    'predUrls': result.pred_urls,
    'refUrls': result.ref_urls,
  }
  return render_template('index.html', flaskData=data)

if __name__ == '__main__':
    app.debug = True
    app.run()
