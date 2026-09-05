# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 26-08-2026



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM :
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Read CSV file
df = pd.read_csv(r"House_Price.csv")

# Given data
print("GIVEN DATA")
print(df.head())

# Create time series using YearBuilt and Price
df = df.groupby("YearBuilt")["Price"].mean().reset_index()


# Select Price column
data = df["Price"].dropna().reset_index(drop=True)

# Augmented Dickey-Fuller Test
result = adfuller(data)

print("\nAUGMENTED DICKEY-FULLER TEST")
print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] < 0.05:
    print("The data is stationary")
else:
    print("The data is not stationary")

# Split data
train_size = int(len(data) * 0.8)

train = data[:train_size]
test = data[train_size:]

# AR model
model = AutoReg(train, lags=5, trend="c")
model_fit = model.fit()

# PACF and ACF
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

plot_pacf(data, lags=15, ax=ax[0])
ax[0].set_title("PACF")

plot_acf(data, lags=15, ax=ax[1])
ax[1].set_title("ACF")

plt.tight_layout()
plt.show()

# Predictions
predictions = model_fit.predict(
    start=len(train),
    end=len(data)-1
)

print("\nPREDICTION")
print(predictions)

# MSE
mse = np.mean((test.values - predictions.values) ** 2)

print("\nFINAL PREDICTION")
print("Mean Squared Error:", mse)

# Plot actual vs predicted
plt.figure(figsize=(12, 5))

plt.plot(test.index, test.values, label="Actual")
plt.plot(test.index, predictions.values, label="Predicted")

plt.title("AR - Actual vs Predicted")
plt.xlabel("Observation")
plt.ylabel("Price")
plt.legend()
plt.show()
```
### OUTPUT:

GIVEN DATA

<img width="770" height="408" alt="image" src="https://github.com/user-attachments/assets/a4c225d3-1c11-4eec-b093-bb51076308ac" />


PACF - ACF

<img width="1199" height="392" alt="image" src="https://github.com/user-attachments/assets/bf1b9f88-6528-4016-be38-a6b051c0c1db" />

<img width="1201" height="395" alt="image" src="https://github.com/user-attachments/assets/dc8f3567-d082-4833-90d5-3131ef6a813b" />


PREDICTION

<img width="614" height="607" alt="image" src="https://github.com/user-attachments/assets/86d67cc3-093e-40b2-a4ef-76cba0c6d9e9" />


FINIAL PREDICTION

<img width="1352" height="673" alt="image" src="https://github.com/user-attachments/assets/8e31f645-24a1-47b1-8800-6572a6a5f84f" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
