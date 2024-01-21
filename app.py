from flask import Flask,render_template, request, redirect
import os
import torch
import torchvision.transforms as transforms
import io,cv2,base64
from werkzeug.utils import secure_filename
from PIL import ImageChops, ImageEnhance
from flask.globals import session
from tensorflow.keras.models import model_from_json
import numpy as np
from PIL import Image
# from your_model_module import Model, predict_video_forgery


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'


# image
def convert_to_ela_image(image, quality):
    temp_filename = 'temp_file.jpg'
    ela_filename = 'temp_ela_file.png'
    
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image


import numpy as np
q = [4.0, 12.0, 2.0]
filter1 = [[0, 0, 0, 0, 0],
           [0, -1, 2, -1, 0],
           [0, 2, -4, 2, 0],
           [0, -1, 2, -1, 0],
           [0, 0, 0, 0, 0]]
filter2 = [[-1, 2, -2, 2, -1],
           [2, -6, 8, -6, 2],
           [-2, 8, -12, 8, -2],
           [2, -6, 8, -6, 2],
           [-1, 2, -2, 2, -1]]
filter3 = [[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 1, -2, 1, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]]


filter1 = np.asarray(filter1, dtype=float) / q[0]
filter2 = np.asarray(filter2, dtype=float) / q[1]
filter3 = np.asarray(filter3, dtype=float) / q[2]
    
filters = filter1+filter2+filter3



image_size = (128, 128)

def prepare_image(image_path):
    ela_image = convert_to_ela_image(image_path, 85)
    resized_ela_image = ela_image.resize(image_size)
    normalized_image = np.array(resized_ela_image).flatten() / 255.0
    print("Shape of normalized_image:", normalized_image.shape)
    return normalized_image



json_file = open('v1model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights("v1model.h5")


json_file2 = open('dunetm.json', 'r')
loaded_model_json = json_file2.read()
json_file2.close()
#load weights 
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("dunet.h5")

def predict(image, model):
    try:
        # Convert image content to PIL Image
        im = Image.open(io.BytesIO(image.read()))
        
        ela_img = prepare_image(im)
        print("Shape of ela_img before reshaping:", ela_img.shape)
        ela_img = ela_img.reshape(1, 128, 128, 3)
        print("Shape of ela_img after reshaping:", ela_img.shape)
        prediction = model.predict(ela_img)

        return ela_img, prediction

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

def predict_region(img, model):
    img = np.array(Image.open(img))
    temp_img_arr = cv2.resize(img, (512, 512))
    temp_preprocess_img = cv2.filter2D(temp_img_arr, -1, filters)
    temp_preprocess_img = cv2.resize(temp_preprocess_img, (512, 512))
    temp_img_arr = temp_img_arr.reshape(1, 512, 512, 3)
    temp_preprocess_img = temp_preprocess_img.reshape(1, 512, 512, 3)
    print("Shape of temp_img_arr:", temp_img_arr.shape)
    print("Shape of temp_preprocess_img:", temp_preprocess_img.shape)
    model_temp = model.predict([temp_img_arr, temp_preprocess_img])
    print("Shape of model_temp:", model_temp.shape)
    model_temp = model_temp[0].reshape(512, 512)
    for i in range(model_temp.shape[0]):
        for j in range(model_temp.shape[1]):
            if model_temp[i][j] > 0.75:
                model_temp[i][j] = 1.0
            else:
                model_temp[i][j] = 0.0
    
    return model_temp


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'gif', 'webm', 'avi', '3gp', 'wmv', 'flv', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/video')
def video():
    return render_template('video.html')


@app.route('/predict', methods = ['POST','GET'])
def predict_page():
    return "it is fake video"
    # if request.method == 'POST':
    #     if 'file' not in request.files:
    #         return 'No Video File provided'
    # file = request.files['file']

    # if file.filename == '':
    #     return 'No selected video file'
    
    # if file and allowed_file(file.filename):
    #     filename = secure_filename(file.filename)
    #     video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #     file.save(video_path)


    #     result, confidence = predict(model, video_path)
    #     return render_template('result.html', result=result, confidence=confidence, video_path=video_path)
    # return render_template('video.html')

@app.route('/imgupload', methods=['POST'])
def imgupload():
    result = "This is a fake image"  # Default value
    prediction_text = None
    ela_img_base64 = None
    predi_base64 = None

    if 'file' not in request.files:
        return render_template('resultimg.html', result="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('resultimg.html', result="No selected file")

    try:
       
       
        ela_img, pred = predict(file, model)

        if ela_img is not None and pred is not None:
            prediction_text = f"Probability of input image to be real is {pred[0]}, Probability of input image to be fake is {1-pred[0]}"

            if pred[0] >= 0.5:
                result = "This is a pristine image"
            else:
                predi = predict_region(file, loaded_model)

                # Encode images as base64 strings
                ela_img_base64 = base64.b64encode(Image.fromarray((ela_img[0] * 255).astype(np.uint8))).decode('utf-8')
                predi_base64 = base64.b64encode(Image.fromarray((predi * 255).astype(np.uint8))).decode('utf-8')
               
    except Exception as e:
        print(f"Error during file processing: {e}")

    # print("uploaded_img_base64:", uploaded_img_base64)
    return render_template('resultimg.html', result=result, prediction_text=prediction_text, ela_img=ela_img_base64, predi=predi_base64)
    # return render_template('resultimg.html', result="Error during prediction")

if __name__ == "__main__":
    app.run(debug=True)