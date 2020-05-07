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
    print("Add seed data to results table.")
    NUM_VAL_SAMPLES = 20
    db.session.add(Result(
      [("/validation/unet_lr0.0001_seed23_losstype0_augTrue_ver1/orig/%d.jpg" % i) for i in range(NUM_VAL_SAMPLES)],
      [("/validation/unet_lr0.0001_seed23_losstype0_augTrue_ver1/pred/%d.png" % i) for i in range(NUM_VAL_SAMPLES)],
      [("/validation/unet_lr0.0001_seed23_losstype0_augTrue_ver1/ref/%d.png" % i) for i in range(NUM_VAL_SAMPLES)],
    ))
    db.session.commit()

if __name__ == '__main__':
    manager.run()
