from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_bcrypt import Bcrypt
from flask_pymongo import PyMongo
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import pickle
import numpy as np
import serial
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image



app = Flask(__name__)
app.config['MONGO_URI'] = 'mongodb://localhost:27017/crop_recommendation'
app.config['SECRET_KEY'] = 'your_secret_key'

mongo = PyMongo(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

with open('RandomForest.pkl', 'rb') as model_file:
    models = pickle.load(model_file)

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
        usermailid =  request.form["usermaild"]
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        existing_user = mongo.db.users.find_one({'username': username})
        if existing_user:
            return 'User already exists!'
        
        mongo.db.users.insert_one({'_id': username, 'password': hashed_password,"usermaild":usermailid})
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
        return 'Invalid credentials!'
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
    print("Request is coming")
    datas = []
    ser = serial.Serial('COM3', 9600)
    ['Temperature: 32.00 °C, Humidity: 60.80 %, pH Value: 8.18']
    while len(datas) <= 3:
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8').strip()
            data=data.split(",")
            print(data)
            temp=data[0].split("Temperature: ")[1].split(" ")[0]
            temp=float(temp)
            humidity=data[1].split("Humidity: ")[1].split(" ")[0]
            humidity=float(humidity)
            ph = data[2].split("pH Value: ")[1]
            ph= float(ph)
            datas.append(data)
            return render_template('index1.html', temperature=temp, humidity=humidity, ph=ph)
    return "123"
    
    # humidity = float(datas[2].split('%\t%')[0])
    # temperature = float(datas[2].split('%\t%')[1].split(': ')[1].replace('°C', ''))
    # ser.close()
    humidity =  60
    temperature =  70
    return render_template('index1.html', temperature=temperature, humidity=humidity)

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
        print("here is the",prediction)
        
        return render_template('result.html', recommended_crop=prediction[0])
    except Exception as e:
        return jsonify({'error': str(e)})


def create_model(input_shape, num_classes):
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in vgg_base.layers:
        layer.trainable = False

    x = Flatten()(vgg_base.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=vgg_base.input, outputs=predictions)
    return model

model_path = 'VGG_PV.h5'
input_shape = (224, 224, 3)
num_classes = 38

# Recreate the exact same model, including its weights and optimizer
model = create_model(input_shape, num_classes)
model.load_weights(model_path)  # Load weights

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = img_batch / 255.0

    prediction = model.predict(img_preprocessed)
    return prediction

@app.route('/leaf-disease', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('result.html', error="No file uploaded.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', error="No selected file.")
    
    if file:
        basepath = os.path.dirname(__file__)
        fname=secure_filename(file.filename)
        file_path = os.path.join(basepath, 'static/uploads', fname)
        file.save(file_path)
        preds = model_predict(file_path, model)
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
    app.run(debug=True)
