from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import os
import numpy as np

# Create the Flask application instance
app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
BATCH_SIZE = 32
IMAGE_SIZE = 255
CHANNEL = 3
EPOCHS = 10

def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method=='POST':
        if "file" not in request.files:
            return render_template('index.html', messages="no file found")
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            file.save(filepath)

            # Read the image
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))

            # Predict using the loaded model
            predicted_class, confidence = predict(img)

            # Render the template with the uploaded image, actual and predicted labels, and confidence
            return render_template('index.html', image_path=filepath, actual_label=predicted_class,
                                   predicted_label=predicted_class, confidence=confidence)



    return render_template('index.html', message='Upload an image')
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == "__main__":
    app.run(debug=True)