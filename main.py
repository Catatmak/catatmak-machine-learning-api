from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

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

@app.route('/insight', methods=['POST'])
def get_insight():
    json_data = request.get_json(force=True)

    df = pd.DataFrame(json_data)
    # Convert 'price' column to numeric
    df['price'] = pd.to_numeric(df['price'])

    # Convert 'created_at' column to datetime
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Filter data for the last week
    end_date = df['created_at'].max()
    start_date = end_date - timedelta(days=6)
    df_last_week = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)]

    # 1. Total Pengeluaran selama 1 minggu
    total_expenses_last_week = df_last_week[df_last_week['type'] == 'outcome']['price'].sum()
    print('Total Pengeluaran selama 1 minggu:', (total_expenses_last_week))

    # 2. Total Pengeluaran Kategori Terbanyak
    most_expensive_category = df_last_week[df_last_week['type'] == 'outcome']['category'].mode().iloc[0]
    total_expenses_most_category = df_last_week[df_last_week['category'] == most_expensive_category]['price'].sum()
    print('Total Pengeluaran Kategori Terbanyak (', most_expensive_category, '):', (total_expenses_most_category))

    # 3. Selama seminggu pengeluaran terbanyak di hari tanggal apa dan berapa?
    day_and_date_most_expenses = df_last_week[df_last_week['type'] == 'outcome'].groupby(df_last_week['created_at'].dt.strftime('%A, %Y-%m-%d'))['price'].sum().idxmax()
    total_expenses_on_day_most_expenses = df_last_week[df_last_week['created_at'].dt.strftime('%A, %Y-%m-%d') == day_and_date_most_expenses]['price'].sum()
    print('Selama seminggu pengeluaran terbanyak pada', (day_and_date_most_expenses.split(',')[0]), day_and_date_most_expenses.split(',')[1], 'sebesar', (total_expenses_on_day_most_expenses))

    # 4. Kesimpulan naik atau turun dibandingkan minggu kemaren (assuming data is available for the previous week)
    previous_end_date = start_date - timedelta(days=1)
    previous_start_date = previous_end_date - timedelta(days=6)
    df_previous_week = df[(df['created_at'] >= previous_start_date) & (df['created_at'] <= previous_end_date)]

    total_expenses_previous_week = df_previous_week[df_previous_week['type'] == 'outcome']['price'].sum()

    if total_expenses_last_week > total_expenses_previous_week:
        conclusion = 'Naik'
        price_difference = total_expenses_last_week - total_expenses_previous_week
    elif total_expenses_last_week < total_expenses_previous_week:
        conclusion = 'Turun'
        price_difference = total_expenses_previous_week - total_expenses_last_week
    else:
        conclusion = 'Sama'
        price_difference = 0

    print(f'Kesimpulan naik atau turun dibandingkan minggu kemaren: {conclusion}')
    print(f'Total perbedaan harga: {(price_difference)}')

    # 5. Prediksi pengeluaran selama seminggu kedepan itu berapa (using a simple linear regression model)
    df['is_outcome'] = df['type'] == 'outcome'
    X = df[['is_outcome']]
    y = df['price']

    model = LinearRegression()
    model.fit(X, y)

    future_dates = pd.date_range(end_date + timedelta(days=1), end_date + timedelta(days=7))
    future_df = pd.DataFrame({'created_at': future_dates, 'is_outcome': True})

    predicted_expenses = model.predict(future_df[['is_outcome']])
    total_predicted_expenses = predicted_expenses.sum()

    print('Prediksi pengeluaran selama seminggu kedepan:', (total_predicted_expenses))
    responses = {
        'total_expenses_last_week': int(total_expenses_last_week),
        'most_expensive_category': str(most_expensive_category),
        'total_expenses_most_category': int(total_expenses_most_category),
        'date_most_expenses': day_and_date_most_expenses.split(',')[1],
        'total_expenses_on_day_most_expenses': int(total_expenses_on_day_most_expenses),
        'conclusion': str(conclusion),
        'price_difference': int(price_difference),
        'total_predicted_expenses': int(total_predicted_expenses),
    }

    return jsonify(responses)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
