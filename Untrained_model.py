import datetime

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('dataset.csv')
df = pd.DataFrame(data)
print(df)

X = df[['Year']]
y = df['Petrol_Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)

# Create a linear regression model
a = datetime.datetime.now()
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

b = datetime.datetime.now()

filename = "Completed_model.joblib"
joblib.dump(model, filename)

# Make predictions
y_pred = model.predict(X_test)

c = b - a
print("time to train model", c.total_seconds() * 1000)

# Plot the results
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('Petrol Price Prediction')
plt.xlabel('Year')
plt.ylabel('Petrol Price')
plt.show()

# Predict the price for a future year
future_year = 2025  # Replace with the desired year
future_price = model.predict([[future_year]])

print(f'Predicted petrol price in {future_year}: {future_price[0]:.2f}')

loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, y_test)
print("model score is ", result)
