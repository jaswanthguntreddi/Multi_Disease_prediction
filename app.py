from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import io
import base64
import requests
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import markdown
from flask import url_for

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Model loading utility
def load_model(model_path, model_type='pickle'):
    try:
        working_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(working_dir, model_path)
        if model_type == 'pickle':
            with open(full_path, 'rb') as f:
                return pickle.load(f)
        elif model_type == 'tensorflow':
            return tf.keras.models.load_model(full_path)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        class DummyModel:
            def predict(self, x): return [1]
        return DummyModel()

# Load models
models = {
    'diabetes': load_model('saved_models/diabetes.pkl'),
    'heart': load_model('saved_models/heart.pkl'),
    'kidney': load_model('saved_models/kidney.pkl'),
    'pneumonia': load_model('saved_models/trained.h5', 'tensorflow'),
    'brain': load_model('saved_models/model.h5', 'tensorflow'),
    'parkinson': load_model('saved_models/parkinsons_model.sav'),
}

# Load medicine data
try:
    medicine_dict = pickle.load(open('saved_models/medicine_dict.pkl', 'rb'))
    medicines = pd.DataFrame(medicine_dict)
    similarity = pickle.load(open('saved_models/similarity.pkl', 'rb'))
except Exception as e:
    print(f"Error loading medicine data: {e}")
    medicines = pd.DataFrame({'Drug_Name': ['Medicine1', 'Medicine2']})
    similarity = np.array([[1, 0], [0, 1]])

# Utility functions
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def process_image(file, target_size):
    image = Image.open(io.BytesIO(file.read()))
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[..., :3]
    elif img_array.shape[-1] == 1:  # Single channel
        img_array = np.repeat(img_array, 3, axis=-1)
    
    return img_array, image

def recommend_medicines(medicine):
    try:
        medicine_series = medicines['Drug_Name'].astype(str)
        index = medicine_series[medicine_series == str(medicine)].index[0]
        distances = similarity[index]
        recommended_indices = sorted(range(len(distances)), key=lambda i: distances[i], reverse=True)[1:6]
        return medicines.iloc[recommended_indices]['Drug_Name'].tolist()
    except Exception as e:
        print(f"Error in recommendation: {e}")
        return []

# Routes
@app.route('/')
def index():
    medicine_list = medicines['Drug_Name'].tolist()
    return render_template('index.html', medicines=medicine_list)

@app.route('/predict', methods=['POST'])
def predict():
    prediction_type = request.form.get('prediction_type')
    
    if prediction_type == 'diabetes':
        try:
            NewBMI_Overweight = NewBMI_Underweight = NewBMI_Obesity_1 = 0
            NewBMI_Obesity_2 = NewBMI_Obesity_3 = NewInsulinScore_Normal = 0
            NewGlucose_Low = NewGlucose_Normal = NewGlucose_Overweight = NewGlucose_Secret = 0

            form_data = request.form
            BMI = form_data.get('BMI')
            Insulin = form_data.get('Insulin')
            Glucose = form_data.get('Glucose')

            if float(BMI) <= 18.5:
                NewBMI_Underweight = 1
            elif 18.5 < float(BMI) <= 24.9:
                pass
            elif 24.9 < float(BMI) <= 29.9:
                NewBMI_Overweight = 1
            elif 29.9 < float(BMI) <= 34.9:
                NewBMI_Obesity_1 = 1
            elif 34.9 < float(BMI) <= 39.9:
                NewBMI_Obesity_2 = 1
            elif float(BMI) > 39.9:
                NewBMI_Obesity_3 = 1

            if 16 <= float(Insulin) <= 166:
                NewInsulinScore_Normal = 1

            if float(Glucose) <= 70:
                NewGlucose_Low = 1
            elif 70 < float(Glucose) <= 99:
                NewGlucose_Normal = 1
            elif 99 < float(Glucose) <= 126:
                NewGlucose_Overweight = 1
            elif float(Glucose) > 126:
                NewGlucose_Secret = 1

            user_input = [
                form_data.get('Pregnancies'), Glucose, form_data.get('BloodPressure'),
                form_data.get('SkinThickness'), Insulin, BMI,
                form_data.get('DiabetesPedigreeFunction'), form_data.get('Age'),
                NewBMI_Underweight, NewBMI_Overweight, NewBMI_Obesity_1,
                NewBMI_Obesity_2, NewBMI_Obesity_3, NewInsulinScore_Normal,
                NewGlucose_Low, NewGlucose_Normal, NewGlucose_Overweight,
                NewGlucose_Secret
            ]
            
            user_input = [float(x) for x in user_input]
            prediction = models['diabetes'].predict([user_input])
            result = "The person has diabetes" if prediction[0] == 1 else "The person does not have diabetes"
            
            return jsonify({
                'status': 'success',
                'prediction_type': 'diabetes',
                'result': result,
                'form_data': form_data.to_dict()
            })
        
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})

    elif prediction_type == 'heart':
        try:
            user_input = [
                request.form.get('age'), request.form.get('sex'), request.form.get('cp'),
                request.form.get('trestbps'), request.form.get('chol'), request.form.get('fbs'),
                request.form.get('restecg'), request.form.get('thalach'), request.form.get('exang'),
                request.form.get('oldpeak'), request.form.get('slope'), request.form.get('ca'),
                request.form.get('thal')
            ]
            
            user_input = [float(x) for x in user_input]
            prediction = models['heart'].predict([user_input])
            result = "This person has heart disease" if prediction[0] == 1 else "This person does not have heart disease"
            
            return jsonify({
                'status': 'success',
                'prediction_type': 'heart',
                'result': result,
                'form_data': request.form.to_dict()
            })
        
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})

    elif prediction_type == 'kidney':
        try:
            fields = [
                'Age', 'Blood Pressure', 'Specific Gravity', 'Albumin', 'Sugar',
                'Red Blood Cell', 'Pus Cell', 'Pus Cell Clumps', 'Bacteria',
                'Blood Glucose Random', 'Blood Urea', 'Serum Creatinine', 'Sodium',
                'Potassium', 'Haemoglobin', 'Packet Cell Volume', 'White Blood Cell Count',
                'Red Blood Cell Count', 'Hypertension', 'Diabetes Mellitus',
                'Coronary Artery Disease', 'Appetitte', 'Peda Edema', 'Aanemia'
            ]
            
            inputs = [request.form.get(field) for field in fields]
            inputs = [float(x) for x in inputs]
            prediction = models['kidney'].predict([inputs])
            result = "The person has kidney disease" if prediction[0] == 1 else "The person does not have kidney disease"
            
            return jsonify({
                'status': 'success',
                'prediction_type': 'kidney',
                'result': result,
                'form_data': request.form.to_dict()
            })
        
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})

    elif prediction_type == 'pneumonia':
        try:
            if 'file' not in request.files:
                return jsonify({'status': 'error', 'message': 'No file uploaded'})
            
            file = request.files['file']
            if not file:
                return jsonify({'status': 'error', 'message': 'No file selected'})
            
            img_array, image = process_image(file, (300, 300))
            if img_array.shape != (300, 300, 3):
                return jsonify({'status': 'error', 'message': f'Unexpected image shape: {img_array.shape}'})
            
            prediction = models['pneumonia'].predict(img_array.reshape(1, 300, 300, 3))[0][0]
            result = "Pneumonia detected in the X-ray." if prediction >= 0.5 else "No Pneumonia detected in the X-ray."
            
            return jsonify({
                'status': 'success',
                'prediction_type': 'pneumonia',
                'result': result,
                'confidence': float(prediction),
                'image': image_to_base64(image)
            })
        
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})

    elif prediction_type == 'brain_tumor':
        try:
            class_labels = ['Pituitary Tumor', 'Glioma Tumor', 'No Tumor', 'Meningioma Tumor']
            
            if 'file' not in request.files:
                return jsonify({'status': 'error', 'message': 'No file uploaded'})
            
            file = request.files['file']
            if not file:
                return jsonify({'status': 'error', 'message': 'No file selected'})
            
            img_array, image = process_image(file, (128, 128))
            img_array = np.expand_dims(img_array, axis=0)

            prediction = models['brain'].predict(img_array)
            predicted_index = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            result = class_labels[predicted_index]
            
            return jsonify({
                'status': 'success',
                'prediction_type': 'brain_tumor',
                'result': result,
                'confidence': f"{confidence*100:.2f}%",
                'image': image_to_base64(image)
            })
        
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})

    elif prediction_type == 'parkinson':
        try:
            parkinson_fields = [
                "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
                "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
                "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
                "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
            ]
            
            inputs = [request.form.get(field) for field in parkinson_fields]
            input_values = [float(x) for x in inputs]
            prediction = models['parkinson'].predict([input_values])
            result = "The person has Parkinson's disease" if prediction[0] == 1 else "The person does not have Parkinson's disease"
            
            return jsonify({
                'status': 'success',
                'prediction_type': 'parkinson',
                'result': result,
                'form_data': request.form.to_dict()
            })
        
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})

    elif prediction_type == 'medicine_recommender':
        try:
            selected_medicine = request.form.get('medicine')
            recommendations = recommend_medicines(selected_medicine)
            return jsonify({
                'status': 'success',
                'prediction_type': 'medicine_recommender',
                'selected_medicine': selected_medicine,
                'recommendations': [str(med) for med in recommendations] if recommendations else []
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})

    elif prediction_type == 'medical_chatbot':
        try:
            prompt = request.form.get('prompt')
            image_file = request.files.get('image')

            if not prompt:
                return jsonify({'status': 'error', 'message': 'Please enter your medical question'})

            content = [{"type": "text", "text": prompt}]
            image_data = None
        
            if image_file:
                # Process the image
                image = Image.open(io.BytesIO(image_file.read()))
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                })

            payload = {
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 1000
            }

            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }

            response = requests.post(
                GROQ_API_URL,
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                model_response = result["choices"][0]["message"]["content"]
            
                return jsonify({
                    'status': 'success',
                    'prediction_type': 'medical_chatbot',
                    'response': model_response,
                    'image': image_data if image_data else None
                })
            else:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                return jsonify({'status': 'error', 'message': error_msg})
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': f"An unexpected error occurred: {e}"})  
# Add this new route (place it with your other routes)
@app.route('/medical_chatbot')
def medical_chatbot():
    return render_template('medical_chatbot.html')

# Add this new endpoint for chatbot requests
@app.route("/ask", methods=["POST"])
def ask():
    prompt = request.form["prompt"]
    image_file = request.files.get("image")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    content = [{"type": "text", "text": prompt}]
    if image_file:
        base64_image = encode_image(image_file)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": 1,
        "max_completion_tokens": 1024,
        "top_p": 1,
        "stream": False
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=data)

    try:
        output = response.json()
        answer = output["choices"][0]["message"]["content"]
        html_response = markdown.markdown(answer)  # Converts markdown to HTML
    except Exception as e:
        html_response = f"<strong>Error:</strong> {e}<br><br><code>{response.text}</code>"

    return render_template("medical_chatbot.html", response=html_response)

# Add this helper function
def encode_image(file):
    return base64.b64encode(file.read()).decode('utf-8')  

if __name__ == '__main__':
    app.run(debug=True)
