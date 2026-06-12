from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_bcrypt import Bcrypt
from flask_pymongo import PyMongo
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
import pickle
import numpy as np
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['MONGO_URI'] = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/crop_recommendation')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fallback_secret_key')

mongo = PyMongo(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

with open('RandomForest.pkl', 'rb') as model_file:
    models = pickle.load(model_file)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.environ.get('MODEL_PATH', 'VGG_PV.h5')
_vgg_model = None


def get_vgg_model():
    global _vgg_model
    if _vgg_model is None:
        model_url = os.environ.get('MODEL_URL')
        if model_url and not os.path.exists(MODEL_PATH):
            import urllib.request
            urllib.request.urlretrieve(model_url, MODEL_PATH)

        from tensorflow.keras.applications import VGG16
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, Flatten, Dropout

        vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in vgg_base.layers:
            layer.trainable = False
        x = Flatten()(vgg_base.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(38, activation='softmax')(x)
        _vgg_model = Model(inputs=vgg_base.input, outputs=predictions)
        _vgg_model.load_weights(MODEL_PATH)
    return _vgg_model


class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id


@login_manager.user_loader
def load_user(user_id):
    user = mongo.db.users.find_one({'_id': user_id})
    if user:
        return User(user_id=user['_id'])
    return None


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        usermailid = request.form["usermaild"]
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        existing_user = mongo.db.users.find_one({'_id': username})
        if existing_user:
            return render_template('register.html', error='Username already exists. Please choose a different one.')

        mongo.db.users.insert_one({'_id': username, 'password': hashed_password, "usermaild": usermailid})
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = mongo.db.users.find_one({'_id': username})

        if user and bcrypt.check_password_hash(user['password'], password):
            login_user(User(username))
            return redirect(url_for('mainpage'))
        return render_template('login.html', error='Invalid username or password.')
    return render_template('login.html')


@app.route("/mainpage")
def mainpage():
    return render_template("mainpage.html")


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/crop-recommendation', methods=['GET'])
@login_required
def home():
    try:
        import serial
        ser = serial.Serial('COM3', 9600)
        datas = []
        while len(datas) <= 3:
            if ser.in_waiting > 0:
                data = ser.readline().decode('utf-8').strip().split(",")
                temp = float(data[0].split("Temperature: ")[1].split(" ")[0])
                humidity = float(data[1].split("Humidity: ")[1].split(" ")[0])
                ph = float(data[2].split("pH Value: ")[1])
                datas.append(data)
                return render_template('index1.html', temperature=temp, humidity=humidity, ph=ph)
    except Exception:
        pass
    return render_template('index1.html', iot_unavailable=True)


@app.route('/recommend_crop', methods=['POST'])
@login_required
def recommend_crop():
    try:
        N = request.form.get('N', type=float)
        P = request.form.get('P', type=float)
        K = request.form.get('K', type=float)
        temperature = request.form.get('temperature', type=float)
        humidity = request.form.get('humidity', type=float)
        ph = request.form.get('ph', type=float)
        rainfall = request.form.get('rainfall', type=float)

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = models.predict(features)

        return render_template('result.html', recommended_crop=prediction[0])
    except Exception as e:
        return jsonify({'error': str(e)})


def model_predict(img_path):
    from tensorflow.keras.preprocessing import image as keras_image
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = img_batch / 255.0
    return get_vgg_model().predict(img_preprocessed)


@app.route('/leaf-disease', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('leafresult.html', error="No file uploaded.")

    file = request.files['file']
    if file.filename == '':
        return render_template('leafresult.html', error="No selected file.")

    fname = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, fname)
    file.save(file_path)

    preds = model_predict(file_path)
    pred_class = preds.argmax(axis=-1)
    plant_disease_list = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
        "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
        "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
        "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
        "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
        "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]
    result = plant_disease_list[pred_class[0]]
    return render_template('leafresult.html', result=result, image_path=fname)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)