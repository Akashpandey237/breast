import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Define the path to the model file (including the filename)
model_file_path = 'C:\\Users\\akash pandey\\breast_cancer_detection\\breast_cancer_detector.pickle'

# Check if the model file exists before attempting to load it
if os.path.exists(model_file_path):
    try:
        with open(model_file_path, 'rb') as model_file:
            model = pickle.load(model_file)
    except Exception as e:
        # Handle exceptions related to model loading
        model = None
else:
    # Handle the case when the model file is not found
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text='Model not found. Please check the model file.')

    try:
        input_features = [float(x) for x in request.form.values()]
        features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                         'mean smoothness', 'mean compactness', 'mean concavity',
                         'mean concave points', 'mean symmetry', 'mean fractal dimension',
                         'radius error', 'texture error', 'perimeter error', 'area error',
                         'smoothness error', 'compactness error', 'concavity error',
                         'concave points error', 'symmetry error', 'fractal dimension error',
                         'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                         'worst smoothness', 'worst compactness', 'worst concavity',
                         'worst concave points', 'worst symmetry', 'worst fractal dimension']

        df = pd.DataFrame([input_features], columns=features_name)
        output = model.predict(df)

        if output == 0:
            res_val = "** breast cancer **"
        else:
            res_val = "no breast cancer"

        return render_template('index.html', prediction_text='Patient has {}'.format(res_val))
    except Exception as e:
        # Handle exceptions gracefully, e.g., form input errors
        return render_template('index.html', prediction_text='An error occurred. Please check your input.')

if __name__ == "__main__":
    app.run()

