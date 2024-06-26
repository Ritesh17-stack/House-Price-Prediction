import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import joblib

# Importing the dataset
dataset = pd.read_csv('housing_price_dataset.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# preprocessing steps
# In OneHotEncoder, drop='first' parameter drops the first category for each feature. This helps avoid multicollinearity in the dataset when using one-hot encoding.
ct = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt']),
        ('cat', OneHotEncoder(drop='first'), ['Neighborhood'])
    ])

# Creating a pipeline
# preprocessor: This step applies column transformations to the input data using the ColumnTransformer class. It scales numerical columns using StandardScaler and one-hot encodes categorical columns using OneHotEncoder.
# regressor: This step trains a linear regression model using the LinearRegression class.This combines preprocessing steps with Linear Regression Model
model = Pipeline([
    ('preprocessor', ct),
    ('regressor', LinearRegression())
])

# Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Fit the model
model.fit(X_train, y_train)

# Save the entire pipeline
joblib.dump(model, "house_price_model.joblib")

#Evaluate the model
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
print(f"R2 score: {r2_score(y_test, y_pred)}")