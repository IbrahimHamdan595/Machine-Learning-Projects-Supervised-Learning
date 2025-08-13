import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


car_sales = pd.read_csv("../data/car-sales-extended-missing-data.csv")
car_sales.dropna(subset=['Price'], inplace=True)
print(car_sales.isna().sum())

categorical_features = ['Make', 'Colour']
categorical_transformer = Pipeline(steps=[
    ("Imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))])                                 
                                   
door_feature = ['Doors']
door_transformer = Pipeline(steps=[
    ("Imputer", SimpleImputer(strategy="constant", fill_value=4))])                                 
     
numeric_features = ['Odometer (KM)']
numeric_transformer = Pipeline(steps=[
    ("Imputer", SimpleImputer(strategy="median"))])                                 
     
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ('door', door_transformer, door_feature),
        ('num', numeric_transformer, numeric_features)
    ]
)

print(car_sales.isna().sum())

regression_models = {
    "Ridge": Ridge(),
    "SVR_linear": SVR(kernel='linear'),
    "SVR_rbf": SVR(kernel='rbf'),
    "RandomForestRegressor": RandomForestRegressor()
}

regression_results = {}

X = car_sales.drop('Price', axis = 1)
y = car_sales['Price']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

for model_name, model in regression_models.items():
    model_pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                     ('model', model)])
    model_pipeline.fit(X_train, y_train)
    regression_results[model_name] = model_pipeline.score(X_test, y_test)
regression_results

print(regression_results)

# The Ridge model is the best pratice for this project

ridge_pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                     ('model', Ridge())])
ridge_pipeline.fit(X_train, y_train)

y_preds = ridge_pipeline.predict(X_test)

# Evaluation

mse = mean_squared_error(y_test, y_preds)
mae = mean_absolute_error(y_test, y_preds)
r2 = r2_score(y_test, y_preds)
