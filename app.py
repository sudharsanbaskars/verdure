import numpy as np
from flask import Flask, render_template, request, Markup, redirect, url_for
from PIL import Image
import torch
import io
from utils.model import ResNet9

from torchvision import transforms

from utils.disease import disease_dic

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']


IMAGE_SIZE = 256
# apple_model = tf.keras.models.load_model("models/apple_model.h5",compile=False)
# corn_model = tf.keras.models.load_model("models/corn_model.h5",compile=False)
# grape_model = tf.keras.models.load_model("models/grape_model.h5",compile=False)
# potato_model = tf.keras.models.load_model("models/potatoes.h5",compile=False)
# tomato_model = tf.keras.models.load_model("models/tomato_model.h5",compile=False)
# peach_model = tf.keras.models.load_model("models/peach_model.h5",compile=False)
# pepper_model = tf.keras.models.load_model("models/pepper_model.h5",compile=False)
# cherry_model = tf.keras.models.load_model("models/cherry_model.h5",compile=False)

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# def predict_image1(img_path, model):
#     img = image.load_img(img_path, grayscale=False, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = np.array(x, 'float32')
#     x /= 255
#     preds = model.predict(x)
#     ind=np.argmax(preds[0])
#     return ind

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/apple-disease-predict', methods=['GET', 'POST'])
def apple_disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('apple-disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            plant_name, crop_name = prediction.split("___")

            if plant_name != "Apple":
                return render_template('invalid.html', plant_name="Apple")

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('apple-disease-result.html', prediction=prediction, title=title)
        except:
            #print("somthing wrong")
            pass
    return render_template('apple-disease.html', title=title)

@app.route('/corn-disease-predict', methods=['GET', 'POST'])
def corn_disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)
            plant_name, crop_name = prediction.split("___")

            if plant_name != "Corn_(maize)":
                return render_template('invalid.html', plant_name="Apple")

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)

@app.route('/grape-disease-predict', methods=['GET', 'POST'])
def grape_disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('grape-disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)
            plant_name, crop_name = prediction.split("___")

            if plant_name != "Grape":
                return render_template('invalid.html', plant_name="Grape")

            prediction = Markup(str(disease_dic[prediction]))

            return render_template('grape-disease-result.html', prediction=prediction, title=title)
        except Exception as e:
            print(e)
            pass
    return render_template('grape-disease.html', title=title)


@app.route('/potato-disease-predict', methods=['GET', 'POST'])
def potato_disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            plant_name, crop_name = prediction.split("___")

            if plant_name != "Potato":
                return render_template('invalid.html', plant_name="Potato")

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('potato-disease.html', title=title)


@app.route('/tomato-disease-predict', methods=['GET', 'POST'])
def tomato_disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('tomato-disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)
            plant_name, crop_name = prediction.split("___")

            if plant_name != "Tomato":
                return render_template('invalid.html', plant_name="Tomato")

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('tomato-disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('tomato-disease.html', title=title)


# @app.route('/peach-disease-predict', methods=['GET', 'POST'])
# def peach_disease_prediction():
#     title = 'Harvestify - Disease Detection'
#
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files.get('file')
#         if not file:
#             return render_template('disease.html', title=title)
#         try:
#             img = file.read()
#
#             prediction = predict_image(img)
#
#             prediction = Markup(str(disease_dic[prediction]))
#             return render_template('disease-result.html', prediction=prediction, title=title)
#         except:
#             pass
#     return render_template('disease.html', title=title)

@app.route('/pepper-disease-predict', methods=['GET', 'POST'])
def pepper_disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('pepper-disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)
            plant_name, crop_name = prediction.split(",")

            if plant_name != "Pepper":
                return render_template('invalid.html', plant_name="Pepper")

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('pepper-disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('pepper-disease.html', title=title)


@app.route('/cherry-disease-predict', methods=['GET', 'POST'])
def cherry_disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)

@app.route('/stawberry-disease-predict', methods=['GET', 'POST'])
def stawberry_disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('strawberry-disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)
            plant_name, crop_name = prediction.split("___")

            if plant_name != "Strawberry":
                return render_template('invalid.html', plant_name="Strawberry")

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('strawberry-disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('strawberry-disease.html', title=title)


if __name__ == "__main__":
    app.run(debug=True)