## DEVELOPED BY: SHARAN MJ
## REGISTER NO: 212222240097
## DATE:

# Ex.No: 07                                       AUTO REGRESSIVE MODEL

## AIM:
To Implement an Auto Regressive Model using Python
## ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
## PROGRAM
```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

data = pd.read_csv('/content/OnionTimeSeries - Sheet1.csv', parse_dates=['Date'], index_col='Date')

print(data.head())

result = adfuller(data['Min'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

model = AutoReg(train['Min'], lags=13)
model_fit = model.fit()

predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

mse = mean_squared_error(test['Min'], predictions)
print('Mean Squared Error:', mse)

plt.figure(figsize=(10,6))
plt.subplot(211)
plot_pacf(train['Min'], lags=13, ax=plt.gca())
plt.title("PACF - Partial Autocorrelation Function")
plt.subplot(212)
plot_acf(train['Min'], lags=13, ax=plt.gca())
plt.title("ACF - Autocorrelation Function")
plt.tight_layout()
plt.show()

print("PREDICTION:")
print(predictions)

plt.figure(figsize=(10,6))
plt.plot(test.index, test['Min'], label='Actual Price')
plt.plot(test.index, predictions, color='red', label='Predicted Price')
plt.title('Test Data vs Predictions (FINAL PREDICTION)')
plt.xlabel('Date')
plt.ylabel('Minimum Price')
plt.legend()
plt.show()
```
## OUTPUT:

### GIVEN DATA
![Screenshot 2024-11-11 113004](https://github.com/user-attachments/assets/b7654498-e236-48d9-a2d4-66055df33b9d)

### ADF-STATISTIC AND P-VALUE
![Screenshot 2024-11-11 113106](https://github.com/user-attachments/assets/fd0fbbba-17e0-48f1-9602-04699cbe33b5)


### PACF - ACF
![Screenshot 2024-11-11 114401](https://github.com/user-attachments/assets/ea24f263-e9c6-4091-b19f-0f3573295972)
![Screenshot 2024-11-11 114409](https://github.com/user-attachments/assets/8ddba0e4-3d4f-4784-804c-70c23963f93a)

### MSE VALUE

![Screenshot 2024-11-11 113111](https://github.com/user-attachments/assets/4e35a75a-a473-431c-b1fd-d8a162243602)

### PREDICTION
![Screenshot 2024-11-11 114355](https://github.com/user-attachments/assets/a8c8719c-fa32-4f5d-ba42-84a2388367d1)

### FINAL PREDICTION

![Screenshot 2024-11-11 114615](https://github.com/user-attachments/assets/1391647e-31c0-4929-a437-0aaeea4622d6)

### RESULT:
Thus, the program to implement the auto regression function using python is executed successfully.
