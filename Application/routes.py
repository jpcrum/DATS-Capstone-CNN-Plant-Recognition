from flask import Flask, render_template, flash, redirect, url_for, request, jsonify
from forms import UploadForm
import CNN
from CNN import load_images, preprocess_images, sort, torch_predict_images
import os
from keras.models import load_model
import os



app = Flask(__name__)
app.config['SECRET_KEY'] = '7074880a859c01bf7af98676ef9fbcb1'



@app.route('/results', methods=['POST', 'GET'])
# Classify a single image given the path from upload.html
def results():
    form = UploadForm()
    path = form.folder.data
    images = load_images(form.folder.data)
    preprocessed = preprocess_images(images)
    preds, predict = torch_predict_images(preprocessed)

    classes = {'0':'Black-Grass', '1':'Charlock', '2':'Cleavers', '3':'Common Chickweed', '4':'Common wheat', '5':'Fat Hen', '6':'Loose Silky-bent',
               '7':'Maize', '8':'Scentless Mayweed', '9':'Shepherds Purse', '10':'Small-flowered Cranesbill', '11':'Sugar beet', '12':'Undecided'}
    for species in classes.values():
        if not os.path.exists(path + '/' + species):
            os.makedirs(path + '/' + species)

    sort(preds, images)

    result = {'Black-Grass': int(predict[0]), 'Charlock': int(predict[1]), 'Cleavers': int(predict[2]), 'Common Chickweed': int(predict[3]),
              'Common Wheat': int(predict[4]), 'Fat Hen': int(predict[5]), 'Loose Silky-Bent': int(predict[6]), 'Maize': int(predict[7]),
              'Scentless Mayweed': int(predict[8]), 'Shepherds Purse': int(predict[9]), 'Small-Flowered Cranesbill': int(predict[10]),
              'Sugar Beet': int(predict[11]), 'Undecided': int(predict[12])}

    return render_template("results.html", results = result)



@app.route('/upload', methods=['POST', 'GET'])
# Render the user interface for classifyAJAX()
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        return redirect(url_for('result'))
    return render_template('upload.html', form=form)



@app.route('/')
@app.route('/index')
# Render the front page of the application
def index():
    return render_template('index.html')
