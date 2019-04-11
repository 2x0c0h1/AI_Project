import os
import numpy as np
import cv2
from keras.backend import clear_session
from keras.models import model_from_yaml

from flask import Flask, flash, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.debug = True
app.testing = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"

classes = ['Asus ROG Zephyrus', 'Dell XPS', 'Huawei MateBook', 'Lenovo Ideapad', 'Macbook', 'Razer Blade Stealth']

def classify(file_path):
    # load YAML and create model
    yaml_file = open('../model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("../model.h5")
    print("Loaded model")

    img = cv2.imread(file_path)
    img = cv2.resize(img,(128,128))
    img = np.reshape(img,[1,128,128,3])

    print("Running model")
    predictions = loaded_model.predict(img)
    clear_session()
    return predictions


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    if request.args.get('prediction') is None:
        return render_template('index.html')
    prediction = request.args['prediction']
    return render_template('index.html', prediction=prediction)

@app.route('/', methods=['POST'])
def upload_file():
    print("Uploading")
    if request.method == 'POST':
        if 'Image' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['Image']
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = app.config['UPLOAD_FOLDER']
            file.save(file_path)
            predictions = classify(file_path)
            print("-----------------------------------")
            print(predictions[0])
            print(np.argmax(predictions[0]))
            print(classes[np.argmax(predictions[0])])
            print("-----------------------------------")
            return redirect(url_for('.index', prediction=classes[np.argmax(predictions[0])]))

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080)
