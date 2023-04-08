from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload and Classify Image")

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
        predicted_class = classifyImage(filepath)
        return render_template('index.html', form=form, predicted_class=predicted_class)

    return render_template('index.html', form=form)

def classifyImage(filepath):
    one_epoch_model = tf.keras.models.load_model("batch_normalized_1_epoch")
    one_epoch_model.summary()

    test_image = Image.open(filepath).resize((224, 224))
    image_array = np.expand_dims(np.asarray(test_image), axis=0)
    predict_x = one_epoch_model.predict(image_array)
    classes_x = np.argmax(predict_x, axis=1)
    print(classes_x)

    # Generate class labels
    csv_df = pd.read_csv("birds.csv")
    dataset = pd.DataFrame(csv_df).to_numpy()
    labels = dataset[:, 2]
    class_ids = dataset[:, 0]
    id_to_name = dict(zip(class_ids, labels))

    predicted_class = id_to_name[classes_x[0]]

    return predicted_class

if __name__ == '__main__':
    app.run(debug=True)