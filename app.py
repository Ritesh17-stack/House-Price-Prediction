from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("house_price_model.joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        square_feet = float(request.form['SquareFeet'])
        bedrooms = float(request.form['Bedrooms'])
        bathrooms = float(request.form['Bathrooms'])
        neighborhood = request.form['Neighborhood']
        year_built = float(request.form['YearBuilt'])

        # Create a DataFrame with the input data
        input_data = pd.DataFrame([[square_feet, bedrooms, bathrooms, neighborhood, year_built]],
                                  columns=['SquareFeet', 'Bedrooms', 'Bathrooms', 'Neighborhood', 'YearBuilt'])

        # Make prediction
        prediction = model.predict(input_data)

        # Format the prediction as currency
        formatted_prediction = "${:,.2f}".format(prediction[0])

        return render_template('index.html', prediction_text=f'Estimated House Price: {formatted_prediction}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)