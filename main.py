# api.py
from flask import Flask, request, jsonify
import pickle
import joblib
import numpy as np 

app = Flask(__name__)

# Load the pre-trained machine learning model
# with open('ModelFinal.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

model = joblib.load('ModelFinal.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        # data = request.get_json()
        # input_array = np.array(["Bakso", "baksi"]).reshape(-1, 1)


        # Make predictions using the loaded model
        prediction = model.predict([["Bakso", "baksi"]])  # Adjust this line based on your model input

        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)
