import os
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/eval/<path:filename>')
def eval(filename):
    return send_from_directory('eval', filename, as_attachment=True)

if __name__ == '__main__':
    app.debug = True
    app.run()
