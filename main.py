from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle

app = Flask(__name__)

# Load the model and polynomial features
my_model = pickle.load(open('4.Model Testing and Validation/my_saved_model_v1.sav', 'rb'))
poly = pickle.load(open('4.Model Testing and Validation/poly_features.sav', 'rb'))

# Load the dataset
winrate_df = pd.read_csv("2.Feature Engineering/wrangled_data.csv")
winrate_df['date'] = pd.to_datetime(winrate_df['date'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        deaths = float(request.form['deaths'])
        kpg = float(request.form['kpg'])
        br = float(request.form['br'])
        nation = request.form['nation']

        # Calculate kd
        kd = kpg / deaths

        # Filter data for the past 80 days
        current_date = datetime.now()
        one_month_ago = current_date - timedelta(days=80)
        winrate_fdf = winrate_df[
            (winrate_df['date'] >= one_month_ago) &
            (winrate_df['date'] <= current_date) &
            (winrate_df['rb_lower_br'] == br) &
            (winrate_df['nation'] == nation)
        ]

        # Calculate win_rate
        win_rate = winrate_fdf['rb_win_rate'].mean()
        if np.isnan(win_rate):
            win_rate = 0  # Handle missing win_rate

        # Scaling parameters
        MIN_KPG = 1
        MAX_KPG = 15
        MIN_KD = 1
        MAX_KD = 5

        # Handle zero division in scaling
        kpg_scaled = 0 if MAX_KPG == MIN_KPG else (kpg - MIN_KPG) / (MAX_KPG - MIN_KPG)
        kd_scaled = 0 if MAX_KD == MIN_KD else (kd - MIN_KD) / (MAX_KD - MIN_KD)

        # Calculate input_value
        input_value = (kd_scaled * 0.333 + kpg_scaled * 0.666) + (win_rate - 0.5)
        input_value = np.array([[input_value]])

        # Transform and predict
        input_poly = poly.transform(input_value)
        predicted_value = my_model.predict(input_poly)
        predicted_value100 = predicted_value[0] * 100

        return render_template('index.html', prediction=predicted_value100)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)