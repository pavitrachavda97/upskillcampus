import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

datafile1 = pd.read_csv('dataset/datafile(1).csv')
datafile2 = pd.read_csv('dataset/datafile(2).csv')
datafile3 = pd.read_csv('dataset/datafile(3).csv')
datafile = pd.read_csv('dataset/datafile.csv')
produce = pd.read_csv('dataset/produce.csv')

# Data Preprocessing
datafile1.fillna(datafile1.mean(), inplace=True)
datafile2.fillna(datafile1.mean(), inplace=True)
datafile3.fillna(datafile1.mean(), inplace=True)
datafile.fillna(datafile1.mean(), inplace=True)
produce.fillna(datafile1.mean(), inplace=True)

# Train-Test Split
X = datafile.drop('target_column', axis=1)
y = datafile['target_column']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Data Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Model Evaluation
y_pred = model.predict(X_test_scaled)

