from flask import Flask, render_template, url_for, send_from_directory, send_file
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['CLASS_FOLDER'] = 'classes'


def all_class_images():
    csv_df = pd.read_csv("birds.csv")
    dataset = pd.DataFrame(csv_df).to_numpy()
    filepaths = dataset[:, 1]
    class_ids = dataset[:, 0]
    train_ids = class_ids[:81950]
    train_filepaths = filepaths[:81950]
    id_to_filepaths = dict(zip(train_ids, train_filepaths))
    return id_to_filepaths


CLASS_IMAGES = all_class_images()


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload and Classify Image")


@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/classes/<url>')
def get_class_image(url):
    return send_from_directory(app.config['CLASS_FOLDER'], url)
    # return send_from_directory(app.config['CLASS_FOLDER'], filename)


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        filepath = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            app.config['UPLOAD_FOLDER'],
            secure_filename(file.filename)
        )
        file.save(filepath)
        predicted_id, predicted_class, predicted_prob, top_ids, top_probs, top_classes = classifyImage(filepath)
        image_url = "/uploads/" + file.filename

        class_images = []
        for id in top_ids:
            class_images.append("/classes/" + str(int(id)) + ".jpg")

        class_image_url = "/classes/" + str(int(predicted_id)) + ".jpg"
        print("class image url")
        print(class_image_url)

        return render_template(
            'index.html',
            form=form,
            image_url=image_url,
            class_image_url=class_image_url,
            predicted_class=predicted_class,
            predicted_prob=predicted_prob,
            top_ids=top_ids,
            top_probs=top_probs,
            top_classes=top_classes,
            class_images=class_images
        )

    return render_template('index.html', form=form)


def classifyImage(filepath):
    one_epoch_model = tf.keras.models.load_model("batch_normalized_5_epochs")
    one_epoch_model.summary()

    test_image = Image.open(filepath).resize((224, 224))
    image_array = np.expand_dims(np.asarray(test_image), axis=0)
    predict_x = one_epoch_model.predict(image_array)
    classes_x = np.argmax(predict_x, axis=1)
    print(classes_x)
    predictions = predict_x[0]
    top_indices = [i for i in np.argsort(predictions)[-5:]]
    top_indices.reverse()
    print("top indices")
    print(top_indices)
    top_probs = [predictions[i] for i in top_indices]
    print(top_probs)

    # Generate class labels
    csv_df = pd.read_csv("birds.csv")
    dataset = pd.DataFrame(csv_df).to_numpy()
    labels = dataset[:, 2]
    class_ids = dataset[:, 0]
    id_to_names = dict(zip(class_ids, labels))

    predicted_id = classes_x[0]
    predicted_class = id_to_names[classes_x[0]]
    predicted_prob = top_probs[0]
    top_classes = [id_to_names[i] for i in top_indices]
    print(top_classes)

    return predicted_id, predicted_class, predicted_prob, top_indices, top_probs, top_classes

if __name__ == '__main__':
    app.run(debug=True)