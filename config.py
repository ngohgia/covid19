import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql:///covid19_segmentation_dev')
    UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    SECRET_KEY = os.getenv('COVID19_SECRET', "ag23gawegae23g")
