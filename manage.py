import os
from config import Config
from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand

from app import app, db
from models import Result

app.config.from_object(Config)

migrate = Migrate(app, db)
manager = Manager(app)

manager.add_command('db', MigrateCommand)

@manager.command
def seed():
    "Add seed data to results table."
    NUM_VAL_SAMPLES = 20
    db.session.add(Result(
      [("/eval/orig/%d.jpg" % i) for i in range(NUM_VAL_SAMPLES)],
      [("/eval/pred/%d.png" % i) for i in range(NUM_VAL_SAMPLES)],
      [("/eval/ref/%d.png" % i) for i in range(NUM_VAL_SAMPLES)],
    ))
    db.session.commit()

if __name__ == '__main__':
    manager.run()
