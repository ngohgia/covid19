import os
from flask import Flask, render_template, request, send_from_directory, jsonify, flash, redirect
from flask_sqlalchemy import SQLAlchemy
from config import Config
from werkzeug.utils import secure_filename

from rq import Queue
from rq.job import Job
from worker import conn
import random
import string
    
from pipeline import predict_tools as pt

ALLOWED_EXTENSIONS = {'nii', 'nii.gz'}

app = Flask(__name__)
app.config.from_object(Config)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

q = Queue(connection=conn)

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

def allowed_file(filename):
    extension = ".".join(filename.split('.')[1:])
    return '.' in filename and extension in ALLOWED_EXTENSIONS

def call_predict(input_file_path, output_dir):
    errors = []
    result = pt.predict(input_file_path, output_dir)

    try:
        result = Result(
            orig_urls=result['origUrls'],
            pred_urls=result['predUrls'],
            ref_urls=result['refUrls']
        )
        db.session.add(result)
        db.session.commit()
        return result.id
    except:
        errors.append("Unable to add item to database.")
        return {"error": errors}

@app.route('/upload', methods=['POST'])
def upload_file():
  if request.method == 'POST':
    from app import call_predict

    if 'file' not in request.files:
        raise("No file part in request")
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        raise("No selected file")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_file_path = os.path.join(app.config['UPLOAD_DIR'], filename)
        file.save(input_file_path)
        flash('File successfully uploaded')

        rand_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) 
        output_dir = os.path.join(Config.OUTPUT_DIR, rand_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        job = q.enqueue_call(
            func=call_predict, args=(input_file_path, rand_id), result_ttl=5000
        )
        print(job.id)
        return redirect("/")
    else:
        raise("Invalid file type")
        return redirect(request.url)

@app.route("/results/<job_key>", methods=['GET'])
def get_results(job_key):
    job = Job.fetch(job_key, connection=conn)

    if job.is_finished:
        result = db.session.query(Result).get(job.result)
        data = {
          'origUrls': result.orig_urls,
          'predUrls': result.pred_urls,
          'refUrls': result.ref_urls,
        }
        return render_template('index.html', flaskData=data)
    else:
        return "Crunching!", 202

@app.route('/validation/<path:filename>')
def validation(filename):
    return send_from_directory('validation', filename, as_attachment=True)

@app.route('/results/<path:filename>')
def outputs(filename):
    return send_from_directory('results', filename, as_attachment=True)

if __name__ == '__main__':
    app.debug = True
    app.run()
