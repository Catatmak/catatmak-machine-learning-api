from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('./models/ModelFinal.pkl')

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('./models/tfidfvectorizer.pkl')  # Assuming you saved the vectorizer during training

Y_train_columns = joblib.load('./models/Y_train_columns.pkl')  # Assuming you saved the columns during training

@app.route('/categorize', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    nama = data['nama']

    # Transform the input text using the TF-IDF vectorizer
    text_vectorizer = tfidf_vectorizer.transform(nama)

    # Make predictions using the model
    prediction = model.predict(text_vectorizer)

    # Extract the predicted categories
    predicted_categories = [Y_train_columns[i] for i in np.where(prediction == 1)[1]]

    # Return the result as JSON
    result = {'predicted_categories': predicted_categories}
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
