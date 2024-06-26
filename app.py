from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
# This line creates a Flask application instance.
app = Flask(__name__)

# Load the trained model
model = joblib.load("house_price_model.joblib")

# This line defines a route for the home page of the application.
@app.route('/')
# This line defines a function that will be executed when a request is made to the home page.
def home():
    # This line returns the index.html template to be rendered as the response.
    return render_template('index.html')

# This line defines a route for the predict page of the application.    
@app.route('/predict', methods=['POST'])
# This line defines a function that will be executed when a request is made to the prediction page. 
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
        # This code snippet returns the 'index.html' template with a message that includes the estimated house price formatted as currency.
        return render_template('index.html', prediction_text=f'Estimated House Price: {formatted_prediction}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')
# This line checks if the script is being run directly (not imported as a module).
if __name__ == '__main__':
    app.run(debug=True)