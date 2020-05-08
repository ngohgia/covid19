# covid19
Website for evaluating Covid-19 ML models

## Requirements
- PostgreSQL
- Redis

## Setup
```
python -m venv env 
source env/bin/activate
pip install -r requirements.txt
bash setup.sh
```

Initialize DB
```
python manage.py db init
python manage.py db migrate
python manage.py db upgrade
python manage.py seed
```

If there is problem installing psycopg2, try the following command:
```
env LDFLAGS='-L/usr/local/lib -L/usr/local/opt/openssl/lib
-L/usr/local/opt/readline/lib' pip install psycopg2
```

Trained models and validation results need to be copied into `trained_models` and `training_images` (TBD)
