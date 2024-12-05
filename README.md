# MSU-Capstone

This repository will include all resources and work completed by Jack D, Tharun B, Shrinidhi K, and Yoensuk C.

# Regression Methods

General Overview
The project uses datasets 371, eba, fb5, and f3d to predict key performance indicators (KPIs) including revenue, invoice number, and customer count. Support Vector Machine (SVM) with an RBF kernel, Long Short-Term Memory (LSTM) network, XGBoost, and Lasso regression were applied for forecasting. These models were evaluated using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Coefficient of Determination (R²), and Relative Absolute Error (RAE).

# SVM code explanation

Overview:

This code implemented a time-series prediction model using Support Vector Regression (SVR) to predict KPIs. The SVR model from the sklearn.svm library was used to forecast the target variable of four distribution companies based on historical data and additional features such as quantity, document_id, and others derived from the past few periods.

Libraries and Modules:

- **Numpy**: Used for numerical operations like calculating performance metrics.
- **Pandas**: Used for data manipulation and reading the dataset.
- **Scikit-Learn**: Used for implementing the SVR model, data scaling, and evaluating the model.
- **Matplotlib**: Used for data visualization (e.g., plotting actual vs predicted revenue).
- **Metrics**: Mean Squared Error (MSE), Mean Absolute Error (MAE), R² Score, and others were used for evaluating the model performance.

Data Preprocessing and Feature Engineering:

-Function: encoder(df, target, window=1, cut=1, drop_timeline=True)
-This function processed the dataset to create the necessary features for the SVR model:
-•	Rolling Window for Target: The function calculated the rolling mean of the target over a specified window to smooth out fluctuations and make the target more suitable for prediction.
-•	Dropping Unnecessary Columns: The year column was dropped if present, as it was not needed for SVR modeling.
-•	Feature Engineering: Additional features, such as the rolling average of revenue, quantity, and document_id, were added to capture recent trends.
Data Scaling
-•	MinMaxScaler was used to scale the features (X) and target (y). This was important for the SVR model to work effectively, as scaling ensured that all features had similar ranges, which is vital for models like SVR.
-•	The target (y) was scaled but then inverse transformed back to its original scale after predictions.

SVR Model Construction:

The SVR model was constructed using the Radial Basis Function (RBF) kernel, which is suitable for non-linear regression problems like this one.
Key Hyperparameters:
-•	C (Regularization parameter): The penalty for misclassification (set to 100), controlling the trade-off between a smooth decision boundary and classifying the training points correctly.
-•	Gamma: The parameter for the kernel, controlling the influence of a single training example (set to 0.1).
-•	Epsilon: Defined a margin of tolerance where no penalty was given for errors (set to 0.1).
-‘svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)’

Model Training:

K-Fold Cross-Validation:
-The model was trained and evaluated using K-fold cross-validation with 2 folds. This helped ensure that the model was generalized and did not overfit to a specific train-test split.
-The SVR model was trained using the fit() method on the training data and was used to predict on the test set:
-‘svr_model.fit(X_train, y_train.ravel())’

Model Evaluation:

Various performance metrics were calculated to evaluate the model's predictive accuracy:
-•	Mean Squared Error (MSE): Measured the average squared difference between predicted and actual values. Lower values indicated better performance.
-•	Root Mean Squared Error (RMSE): The square root of MSE, providing the error in the same units as the target variable.
-•	Mean Absolute Error (MAE): Measured the average absolute difference between predicted and actual values.
-•	Mean Absolute Percentage Error (MAPE): Measured the percentage difference between predicted and actual values.
-•	R² Score: Measured how well the predictions matched the actual data. A score of 1 indicated a perfect fit.
-•	Relative Absolute Error (RAE): Measured the error relative to the mean of the actual values.
-These metrics were calculated for each fold, and their averages were computed at the end.

Visualization:

Visualization was an integral part of understanding the model’s performance. The Matplotlib library was used to plot the actual versus predicted revenue for the last fold. The graph highlighted how well the model captured trends in the target variable, allowing for a visual comparison of its predictive capability.

# SVM code 

## Revenue

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

### Load the data
#file_path = "C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_371.csv"
#file_path = "C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_eba.csv"
file_path = "C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_fb5.csv"
#file_path = "C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_f3d.csv"

data = pd.read_csv(file_path)

### Define the encoder function for feature-target creation
def encoder(df, target, window=1, cut=1, drop_timeline=True):
    # Ensure the DataFrame has a proper index
    df = df.reset_index(drop=True)
   
    # New column 'target' that is the reverse rolling mean of the target column
    df['target'] = df[target].rolling(window=window).mean().shift(-1)
   
    # Drop rows with NaN and drop the last `cut` rows
    df = df.dropna().iloc[:-cut]
   
    if drop_timeline and 'year' in df.columns:
        df = df.drop('year', axis=1)
   
    # Features (X) and target (y)
    X = df.drop(['target'], axis=1)
    y = df['target']
   
    # Add rolling averages for additional features
    X['revenue_recent'] = df['revenue'].rolling(window=3).mean()
    X['quantity_recent'] = df['quantity'].rolling(window=3).mean()
    X['document_id_recent'] = df['document_id'].rolling(window=3).mean()
   
    # Fill missing values with -1
    X = X.fillna(-1)
   
    return X, y

### Prepare the data using the encoder
X, y = encoder(data, 'revenue', window=1, cut=2, drop_timeline=True)

### Scale the features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

### Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

### Function to calculate RAE (Relative Absolute Error)
def relative_absolute_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))

### Initialize KFold for cross-validation
kf = KFold(n_splits=2, shuffle=True, random_state=42)

### Lists to store performance metrics
mse_scores = []
rmse_scores = []
mae_scores = []
mape_scores = []
r2_scores = []
rae_scores = []

### Perform cross-validation
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]

    # Train the SVR model
    svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr_model.fit(X_train, y_train.ravel())

    # Predict on the test set
    y_pred_scaled = svr_model.predict(X_test)
    
    # Inverse transform the predictions and actuals
    y_test_original = scaler_y.inverse_transform(y_test)
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mape = mean_absolute_percentage_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)
    rae = relative_absolute_error(y_test_original, y_pred_original)
    
    # Store the metrics
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    mape_scores.append(mape)
    r2_scores.append(r2)
    rae_scores.append(rae)

### Average performance metrics across folds
avg_mse = np.mean(mse_scores)
avg_rmse = np.mean(rmse_scores)
avg_mae = np.mean(mae_scores)
avg_mape = np.mean(mape_scores)
avg_r2 = np.mean(r2_scores)
avg_rae = np.mean(rae_scores)

### Print performance metrics
print(f"Average Mean Squared Error (MSE): {avg_mse}")
print(f"Average Root Mean Squared Error (RMSE): {avg_rmse}")
print(f"Average Mean Absolute Error (MAE): {avg_mae}")
print(f"Average Mean Absolute Percentage Error (MAPE): {avg_mape}")
print(f"Average R² Score: {avg_r2}")
print(f"Average Relative Absolute Error (RAE): {avg_rae}")

### Visualize results from the last fold
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 3))
plt.plot(y_test_original, label='Actual Revenue', color='blue', linestyle='--', marker='o', alpha=0.7)
plt.plot(y_pred_original, label='Predicted Revenue', color='red', linestyle='-', marker='x', alpha=0.7)
plt.title('Actual vs Predicted Revenue (SVR)')
plt.xlabel('Sample Index')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## Results
For dataset 371

Average Mean Squared Error (MSE): 81930265925.24803 
Average Root Mean Squared Error (RMSE): 284220.65900434833
Average Mean Absolute Error (MAE): 212456.5607986489
Average Mean Absolute Percentage Error (MAPE): 20.23394627551963
Average R² Score: 0.4278386595013954
Average Relative Absolute Error (RAE): 0.6531716440959617

For dataset eba 

Average Mean Squared Error (MSE): 722066159829.0266
Average Root Mean Squared Error (RMSE): 838657.4240307469
Average Mean Absolute Error (MAE): 668771.920062257
Average Mean Absolute Percentage Error (MAPE): 403.1750033726474
Average R² Score: 0.8024632257322452
Average Relative Absolute Error (RAE): 0.3930541930014756

For dataset fb5

Average Mean Squared Error (MSE): 2191957429.5035505
Average Root Mean Squared Error (RMSE): 46768.42884182373
Average Mean Absolute Error (MAE): 35984.80881167662
Average Mean Absolute Percentage Error (MAPE): 8.945571930516156
Average R² Score: 0.7066216526701256
Average Relative Absolute Error (RAE): 0.5463696842689412

For dataset f3d

Average Mean Squared Error (MSE): 290651814898.2412
Average Root Mean Squared Error (RMSE): 536415.5511141435
Average Mean Absolute Error (MAE): 414859.48171069176
Average Mean Absolute Percentage Error (MAPE): 47.64272211351366
Average R² Score: 0.440489469340656
Average Relative Absolute Error (RAE): 0.7129330893076734

## Invoice Number

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

### Load the dataset
file_path = "C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_371.csv"
#file_path = "C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_eba.csv"
#file_path = "C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_fb5.csv"
#file_path = "C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_f3d.csv"
data = pd.read_csv(file_path)

### Define the encoder function for feature-target creation
def encoder(df, target, window=1, cut=1, drop_timeline=True):
    # Ensure the DataFrame has a proper index
    df = df.reset_index(drop=True)
    
    # New column 'target' that is the reverse rolling mean of the target column
    df['target'] = df[target].rolling(window=window).mean().shift(-1)
    
    # Drop rows with NaN and drop the last `cut` rows
    df = df.dropna().iloc[:-cut]
    
    if drop_timeline and 'year' in df.columns:
        df = df.drop('year', axis=1)
    
    # Features (X) and target (y)
    X = df.drop(['target', 'document_id'], axis=1)  # Remove 'document_id' as it is the target
    y = df['target']
    
    # Add rolling averages for additional features (exclude document_id to avoid leakage)
    X['revenue_recent'] = df['revenue'].rolling(window=3).mean()
    X['quantity_recent'] = df['quantity'].rolling(window=3).mean()
    
    # Fill missing values with -1
    X = X.fillna(-1)
    
    return X, y

### Prepare the data using the encoder
X, y = encoder(data, 'document_id', window=1, cut=2, drop_timeline=True)

### Scale the features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

### Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

### Function to calculate RAE (Relative Absolute Error)
def relative_absolute_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))

### Initialize KFold for cross-validation
kf = KFold(n_splits=2, shuffle=True, random_state=42)

### Lists to store performance metrics
mse_scores = []
rmse_scores = []
mae_scores = []
mape_scores = []
r2_scores = []
rae_scores = []

### Perform cross-validation
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]

    # Train the SVR model
    svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr_model.fit(X_train, y_train.ravel())

    # Predict on the test set
    y_pred_scaled = svr_model.predict(X_test)
    
    # Inverse transform the predictions and actuals
    y_test_original = scaler_y.inverse_transform(y_test)
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mape = mean_absolute_percentage_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)
    rae = relative_absolute_error(y_test_original, y_pred_original)
    
    # Store the metrics
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    mape_scores.append(mape)
    r2_scores.append(r2)
    rae_scores.append(rae)

### Average performance metrics across folds
avg_mse = np.mean(mse_scores)
avg_rmse = np.mean(rmse_scores)
avg_mae = np.mean(mae_scores)
avg_mape = np.mean(mape_scores)
avg_r2 = np.mean(r2_scores)
avg_rae = np.mean(rae_scores)

### Print performance metrics
print(f"Average Mean Squared Error (MSE): {avg_mse}")
print(f"Average Root Mean Squared Error (RMSE): {avg_rmse}")
print(f"Average Mean Absolute Error (MAE): {avg_mae}")
print(f"Average Mean Absolute Percentage Error (MAPE): {avg_mape}")
print(f"Average R² Score: {avg_r2}")
print(f"Average Relative Absolute Error (RAE): {avg_rae}")

### Visualize results from the last fold
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 3))
plt.plot(y_test_original, label='Actual Number of Invoices', color='blue', linestyle='--', marker='o', alpha=0.7)
plt.plot(y_pred_original, label='Predicted Number of Invoices', color='red', linestyle='-', marker='x', alpha=0.7)
plt.title('Actual vs Predicted Number of Invoices (SVR)')
plt.xlabel('Sample Index')
plt.ylabel('Number of Invoices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## Results
For dataset 371

Average Mean Squared Error (MSE): 311923.055435903
Average Root Mean Squared Error (RMSE): 554.5197049108219
Average Mean Absolute Error (MAE): 427.11034020178204
Average Mean Absolute Percentage Error (MAPE): 14.527685754017455
Average R² Score: 0.2604774171184646
Average Relative Absolute Error (RAE): 0.8170069657947512

For dataset eba 

Average Mean Squared Error (MSE): 34992595.74469987
Average Root Mean Squared Error (RMSE): 5839.422055836634
Average Mean Absolute Error (MAE): 4657.581252021408
Average Mean Absolute Percentage Error (MAPE): 534.5218221128981
Average R² Score: 0.778937217680552
Average Relative Absolute Error (RAE): 0.4153819685313313

For dataset fb5

Average Mean Squared Error (MSE): 27712.14411202274
Average Root Mean Squared Error (RMSE): 165.71549805565775
Average Mean Absolute Error (MAE): 124.25408725198977
Average Mean Absolute Percentage Error (MAPE): 8.619266969154383
Average R² Score: 0.8985281872073403
Average Relative Absolute Error (RAE): 0.35587750350743097

For dataset f3d

Average Mean Squared Error (MSE): 148551.08505780273
Average Root Mean Squared Error (RMSE): 382.38442979842284
Average Mean Absolute Error (MAE): 299.49359610864644
Average Mean Absolute Percentage Error (MAPE): 26.91027968064954
Average R² Score: 0.24147132559629578
Average Relative Absolute Error (RAE): 0.8638208148211515

## Customer

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

### Load the data
#file_path = "C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_371.csv"
#file_path = "C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_eba.csv"
file_path = "C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_fb5.csv"
#file_path = "C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_f3d.csv"
data = pd.read_csv(file_path)

### Define the encoder function for feature-target creation
### Modify the encoder function for the new target (customer_id)
def encoder_customers(df, target, window=1, cut=1, drop_timeline=True):
    # Ensure the DataFrame has a proper index
    df = df.reset_index(drop=True)
    
    # New column 'target' that is the reverse rolling mean of the target column
    df['target'] = df[target].rolling(window=window).mean().shift(-1)
    
    # Drop rows with NaN and drop the last cut rows
    df = df.dropna().iloc[:-cut]
    
    if drop_timeline and {'year', 'month'}.issubset(df.columns):
        df = df.drop(['year', 'month'], axis=1)
    
    # Features (X) and target (y)
    X = df.drop(['target'], axis=1)
    y = df['target']
    
    # Add rolling averages for additional features
    X['customers_recent_1'] = df[target].shift(1)
    X['customers_recent_2'] = df[target].shift(2)
    X['customers_recent_3'] = df[target].shift(3)
    
    # Fill missing values with -1
    X = X.fillna(-1)
    
    return X, y

### Prepare the data using the modified encoder
X_customers, y_customers = encoder_customers(data, 'customer_id', window=1, cut=3, drop_timeline=True)

### Scale the features and target
scaler_X_customers = MinMaxScaler()
scaler_y_customers = MinMaxScaler()
X_scaled_customers = scaler_X_customers.fit_transform(X_customers)
y_scaled_customers = scaler_y_customers.fit_transform(y_customers.values.reshape(-1, 1))

### Function to calculate performance metrics (MAPE, etc.) remains the same
### Perform KFold cross-validation
kf = KFold(n_splits=2, shuffle=True, random_state=42)

### Lists to store performance metrics
mse_scores_customers = []
rmse_scores_customers = []
mae_scores_customers = []
mape_scores_customers = []
r2_scores_customers = []
rae_scores_customers = []

### Perform cross-validation for customer prediction
for train_index, test_index in kf.split(X_scaled_customers):
    X_train, X_test = X_scaled_customers[train_index], X_scaled_customers[test_index]
    y_train, y_test = y_scaled_customers[train_index], y_scaled_customers[test_index]

    # Train the SVR model
    svr_model_customers = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr_model_customers.fit(X_train, y_train.ravel())

    # Predict on the test set
    y_pred_scaled_customers = svr_model_customers.predict(X_test)
    
    # Inverse transform the predictions and actuals
    y_test_original_customers = scaler_y_customers.inverse_transform(y_test)
    y_pred_original_customers = scaler_y_customers.inverse_transform(y_pred_scaled_customers.reshape(-1, 1))
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test_original_customers, y_pred_original_customers)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original_customers, y_pred_original_customers)
    mape = mean_absolute_percentage_error(y_test_original_customers, y_pred_original_customers)
    r2 = r2_score(y_test_original_customers, y_pred_original_customers)
    rae = relative_absolute_error(y_test_original_customers, y_pred_original_customers)
    
    # Store the metrics
    mse_scores_customers.append(mse)
    rmse_scores_customers.append(rmse)
    mae_scores_customers.append(mae)
    mape_scores_customers.append(mape)
    r2_scores_customers.append(r2)
    rae_scores_customers.append(rae)

### Average performance metrics across folds
avg_mse_customers = np.mean(mse_scores_customers)
avg_rmse_customers = np.mean(rmse_scores_customers)
avg_mae_customers = np.mean(mae_scores_customers)
avg_mape_customers = np.mean(mape_scores_customers)
avg_r2_customers = np.mean(r2_scores_customers)
avg_rae_customers = np.mean(rae_scores_customers)

### Print performance metrics
print(f"Average Mean Squared Error (MSE): {avg_mse_customers}")
print(f"Average Root Mean Squared Error (RMSE): {avg_rmse_customers}")
print(f"Average Mean Absolute Error (MAE): {avg_mae_customers}")
print(f"Average Mean Absolute Percentage Error (MAPE): {avg_mape_customers}")
print(f"Average R² Score: {avg_r2_customers}")
print(f"Average Relative Absolute Error (RAE): {avg_rae_customers}")

### Visualize results from the last fold
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 3))
plt.plot(y_test_original_customers, label='Actual Customers', color='blue', linestyle='--', marker='o', alpha=0.7)
plt.plot(y_pred_original_customers, label='Predicted Customers', color='red', linestyle='-', marker='x', alpha=0.7)
plt.title('Actual vs Predicted Number of Customers (SVR)')
plt.xlabel('Sample Index')
plt.ylabel('Number of Customers')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## Results
For dataset 371

Average Mean Squared Error (MSE): 111009.46059895428
Average Root Mean Squared Error (RMSE): 331.04602850913807
Average Mean Absolute Error (MAE): 269.05923120887667
Average Mean Absolute Percentage Error (MAPE): 12.892843220940069
Average R² Score: 0.25785470729776766
Average Relative Absolute Error (RAE): 0.9414593159270647

For dataset eba 

Average Mean Squared Error (MSE): 114085.304962052
Average Root Mean Squared Error (RMSE): 325.350817505442
Average Mean Absolute Error (MAE): 246.31247861072134
Average Mean Absolute Percentage Error (MAPE): 128.783510735159
Average R² Score: 0.721455284723208
Average Relative Absolute Error (RAE): 0.4421525554929424

For dataset fb5

Average Mean Squared Error (MSE): 33563.40551344717
Average Root Mean Squared Error (RMSE): 177.14663894998995
Average Mean Absolute Error (MAE): 124.90928329738897
Average Mean Absolute Percentage Error (MAPE): 17.66512346899942
Average R² Score: 0.7669025480229001
Average Relative Absolute Error (RAE): 0.40111208996428915

For dataset f3d

Average Mean Squared Error (MSE): 42699.451549423124
Average Root Mean Squared Error (RMSE): 206.21299612906552
Average Mean Absolute Error (MAE): 149.48368050808142
Average Mean Absolute Percentage Error (MAPE): 22.844856829202286
Average R² Score: 0.310315394556936
Average Relative Absolute Error (RAE): 0.7263318936664588

# LSTM code explanation

Overview:

The codes implemented a time-series prediction model using a Long Short-Term Memory (LSTM) network to predict three KPIs. The LSTM model was part of the tensorflow.keras library (built on TensorFlow). The goal was to forecast the target variables of four distribution companies based on historical data and additional features such as quantity, document_id, and others derived from the past few periods.

Libraries and Modules:

•	Numpy: Used for numerical operations like calculating metrics and transforming data.
•	Pandas: Used for data manipulation and reading the dataset.
•	Matplotlib: Used for data visualization (e.g., plotting actual vs predicted revenue).
•	Scikit-Learn: Used for data scaling, splitting datasets, and evaluating models.
•	TensorFlow (Keras): Used to build and train the LSTM model.
•	Metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), and others were used for evaluating the model performance.

Data Preprocessing and Feature Engineering:

Function: encoder(df, target, window=1, cut=1, drop_timeline=True)
This function processed the dataset to create the necessary features for the LSTM model:
•	Rolling Window for Target: The function calculated the rolling mean of the target over a specified window to smooth out fluctuations and make the target more suitable for prediction.
•	Dropping Unnecessary Columns: The year column was dropped if present, as it was not needed for LSTM modeling.
•	Feature Engineering: Additional features, like the rolling average of revenue, quantity, and document_id, were added to capture recent trends.

Data Scaling:

•	MinMaxScaler was used to scale the features (X) and target (y). This was important for neural networks to work effectively, as they perform better when the input data is scaled to a range [0, 1].
•	The target (y) was also scaled but then inverse transformed back to its original scale after predictions.

LSTM Model Construction:

- LSTM model consisted of three layers:
LSTM Layer:
This was the first layer of the model, where the LSTM units (128 units) were used to capture sequential dependencies in the time-series data, and the activation function used is ReLU.
Dropout Layer:
This layer followed the LSTM layer, with a dropout rate of 0.2. It helps prevent overfitting by randomly setting a fraction (20%) of input units to zero during training.
Dense Layer:
This was the final layer of the model, which was a fully connected (dense) layer with a single output unit to predict the target value
- Input Shape: The input shape was (X_train.shape[1], X_train.shape[2]), where X_train.shape[1] was the number of timesteps (1 in this case) and X_train.shape[2] was the number of features.

Model Training:

The model was trained using the fit() method with a batch size of 16 and 20 epochs. The optimizer used was Adam, and the loss function was Mean Squared Error (MSE). 
•	Epochs: The number of passes through the entire dataset during training.
•	Batch Size: Defined the number of samples used in one iteration of training.
*Since the model was trained on a local machine, it was executed on the CPU.

Cross-Validation:

K-Fold Cross-Validation:
The dataset was split into 2 folds using KFold from scikit-learn. The model was trained on the training set and evaluated on the test set in each fold. Metrics were calculated for each fold, and the average performance metrics were displayed at the end.

Model Evaluation:

The model was evaluated using various metrics:
•	Mean Squared Error (MSE): Measured the average squared difference between predicted and actual values. Lower values indicated better model performance.
•	Root Mean Squared Error (RMSE): The square root of MSE, providing error in the same units as the target variable.
•	Mean Absolute Error (MAE): Measured the average absolute difference between predicted and actual values.
•	Mean Absolute Percentage Error (MAPE): Measured the percentage difference between predicted and actual values.
•	R² Score: Measured how well the predictions matched the actual data (1 indicated perfect prediction).
•	Relative Absolute Error (RAE): Measured the error relative to the mean of the actual values.
These metrics were calculated and printed after training for each fold, and the averages were displayed.

Visualization:

The Matplotlib library was used to compare actual and predicted values for the last fold. The visualization illustrated the model’s performance in capturing trends over time, providing insights into its temporal prediction capabilities.

# LSTM code

## Revenue

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

### Load data
#data = pd.read_csv("C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_371.csv")
#data = pd.read_csv("C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_eba.csv")
data = pd.read_csv("C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_fb5.csv")
#data = pd.read_csv("C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_f3d.csv")

### Define MAPE function with zero handling
def mean_absolute_percentage_error(y_true, y_pred):
    non_zero_mask = y_true != 0  # Avoid division by zero
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

### Define RAE function
def relative_absolute_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))

### Define the encoder function for feature-target creation
def encoder(df, target, window=1, cut=1, drop_timeline=True):
    # Ensure the DataFrame has a proper index
    df = df.reset_index(drop=True)
   
    # New column 'target' that is the reverse rolling mean of the target column
    df['target'] = df[target].rolling(window=window).mean().shift(-1)
   
    # Drop rows with NaN
    df = df.dropna().iloc[:-cut]
   
    if drop_timeline and 'year' in df.columns:
        df = df.drop('year', axis=1)
   
    # Features (X) and target (y)
    X = df.drop(['target', 'index'], axis=1)
    y = df['target']
   
    # Add rolling averages for additional features
    X['revenue_recent'] = df['revenue'].rolling(window=3).mean()
    X['quantity_recent'] = df['quantity'].rolling(window=3).mean()
    X['document_id_recent'] = df['document_id'].rolling(window=3).mean()
   
    # Fill missing values with -1
    X = X.fillna(-1)
   
    return X, y

### Prepare the data
X, y = encoder(data, 'revenue', window=1, cut=2, drop_timeline=True)

### Scale the features but not the target
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

### Scale the target (y) but keep track of scaler for inverse transformation
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

### Reshape X for LSTM input (samples, timesteps, features)
X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

### Initialize KFold
kf = KFold(n_splits=2, shuffle=True, random_state=42)

### Lists to store performance metrics
mse_scores = []
rmse_scores = []
mae_scores = []
mape_scores = []
r2_scores = []
rae_scores = []

### K-Fold Cross-Validation
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]  # Use scaled `y` for training
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    
    # Predict
    y_pred_scaled = model.predict(X_test).flatten()
    
    # Inverse transform predictions and actuals to original scale
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate performance metrics on the original scale
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mape = mean_absolute_percentage_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)
    rae = relative_absolute_error(y_test_original, y_pred_original)
    
    # Append metrics to lists
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    mape_scores.append(mape)
    r2_scores.append(r2)
    rae_scores.append(rae)

### Calculate average performance metrics
avg_mse = np.mean(mse_scores)
avg_rmse = np.mean(rmse_scores)
avg_mae = np.mean(mae_scores)
avg_mape = np.mean(mape_scores)
avg_r2 = np.mean(r2_scores)
avg_rae = np.mean(rae_scores)

### Print metrics
print(f"Average Mean Squared Error (MSE): {avg_mse}")
print(f"Average Root Mean Squared Error (RMSE): {avg_rmse}")
print(f"Average Mean Absolute Error (MAE): {avg_mae}")
print(f"Average Mean Absolute Percentage Error (MAPE): {avg_mape}")
print(f"Average R² Score: {avg_r2}")
print(f"Average Relative Absolute Error (RAE): {avg_rae}")

### Visualize results from the last fold
plt.figure(figsize=(6, 3))
plt.plot(y_test_original, label='Actual Revenue', color='blue', linestyle='--', marker='o', alpha=0.7)
plt.plot(y_pred_original, label='Predicted Revenue', color='red', linestyle='-', marker='x', alpha=0.7)
plt.title('Actual vs Predicted Revenue (LSTM)')
plt.xlabel('Sample Index')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## Results
For dataset 371

Average Mean Squared Error (MSE): 58647129180.011536
Average Root Mean Squared Error (RMSE): 240480.28554963035
Average Mean Absolute Error (MAE): 175326.6186617647
Average Mean Absolute Percentage Error (MAPE): 16.802462433758016
Average R² Score: 0.6879598911129752
Average Relative Absolute Error (RAE): 0.4748365585025791

For dataset eba 

Average Mean Squared Error (MSE): 1473796932108.3103
Average Root Mean Squared Error (RMSE): 1212472.2221356742
Average Mean Absolute Error (MAE): 1036550.3004190105
Average Mean Absolute Percentage Error (MAPE): 448.4172761256421
Average R² Score: 0.5989216030220409
Average Relative Absolute Error (RAE): 0.614827863644031

For dataset fb5

Average Mean Squared Error (MSE): 1855159870.2331433
Average Root Mean Squared Error (RMSE): 43030.672736817236
Average Mean Absolute Error (MAE): 33652.10757758621
Average Mean Absolute Percentage Error (MAPE): 8.248253099859728
Average R² Score: 0.7508844755932729
Average Relative Absolute Error (RAE): 0.5046890546272427

For dataset f3d

Average Mean Squared Error (MSE): 387675279702.1819
Average Root Mean Squared Error (RMSE): 618455.5909550928
Average Mean Absolute Error (MAE): 433363.8531721622
Average Mean Absolute Percentage Error (MAPE): 49.53354446302119
Average R² Score: 0.44166187692178227
Average Relative Absolute Error (RAE): 0.7159640166890247

## Invoice number

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

### Load the dataset
data = pd.read_csv("C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_371.csv")
#data = pd.read_csv("C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_eba.csv")
#data = pd.read_csv("C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_fb5.csv")
#data = pd.read_csv("C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_f3d.csv")

### Define MAPE function with zero handling
def mean_absolute_percentage_error(y_true, y_pred):
    non_zero_mask = y_true != 0  # Avoid division by zero
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

### Define RAE function
def relative_absolute_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))

### Define the encoder function for feature-target creation
def encoder(df, target, window=1, cut=1, drop_timeline=True):
    # Ensure the DataFrame has a proper index
    df = df.reset_index(drop=True)
   
    # New column 'target' that is the reverse rolling mean of the target column
    df['target'] = df[target].rolling(window=window).mean().shift(-1)
   
    # Drop rows with NaN
    df = df.dropna().iloc[:-cut]
   
    if drop_timeline and 'year' in df.columns:
        df = df.drop('year', axis=1)
   
   ### Features (X) and target (y)
    X = df.drop(['target', 'document_id'], axis=1)  # Remove 'document_id' as it is the target
    y = df['target']
   
    # Add rolling averages for additional features
    X['revenue_recent'] = df['revenue'].rolling(window=3).mean()
    X['quantity_recent'] = df['quantity'].rolling(window=3).mean()
    X['document_id_recent'] = df['document_id'].rolling(window=3).mean()
   
    # Fill missing values with -1
    X = X.fillna(-1)
   
    return X, y

### Prepare the data for predicting the number of invoices
X, y = encoder(data, 'document_id', window=1, cut=2, drop_timeline=True)

### Scale the features but not the target
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

### Scale the target (y) but keep track of scaler for inverse transformation
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

### Reshape X for LSTM input (samples, timesteps, features)
X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

### Initialize KFold
kf = KFold(n_splits=2, shuffle=True, random_state=42)

### Lists to store performance metrics
mse_scores = []
rmse_scores = []
mae_scores = []
mape_scores = []
r2_scores = []
rae_scores = []

### K-Fold Cross-Validation
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]  # Use scaled `y` for training
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    
    # Predict
    y_pred_scaled = model.predict(X_test).flatten()
    
    # Inverse transform predictions and actuals to original scale
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate performance metrics on the original scale
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mape = mean_absolute_percentage_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)
    rae = relative_absolute_error(y_test_original, y_pred_original)
    
    # Append metrics to lists
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    mape_scores.append(mape)
    r2_scores.append(r2)
    rae_scores.append(rae)

### Calculate average performance metrics
avg_mse = np.mean(mse_scores)
avg_rmse = np.mean(rmse_scores)
avg_mae = np.mean(mae_scores)
avg_mape = np.mean(mape_scores)
avg_r2 = np.mean(r2_scores)
avg_rae = np.mean(rae_scores)

### Print metrics
print(f"Average Mean Squared Error (MSE): {avg_mse}")
print(f"Average Root Mean Squared Error (RMSE): {avg_rmse}")
print(f"Average Mean Absolute Error (MAE): {avg_mae}")
print(f"Average Mean Absolute Percentage Error (MAPE): {avg_mape}")
print(f"Average R² Score: {avg_r2}")
print(f"Average Relative Absolute Error (RAE): {avg_rae}")

### Visualize results from the last fold
plt.figure(figsize=(6, 3))
plt.plot(y_test_original, label='Actual Number of Invoices', color='blue', linestyle='--', marker='o', alpha=0.7)
plt.plot(y_pred_original, label='Predicted Number of Invoices', color='red', linestyle='-', marker='x', alpha=0.7)
plt.title('Actual vs Predicted Number of Invoices (LSTM)')
plt.xlabel('Sample Index')
plt.ylabel('Number of Invoices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


## Results 
For dataset 371

Average Mean Squared Error (MSE): 215811.10815862464
Average Root Mean Squared Error (RMSE): 463.7100205576173
Average Mean Absolute Error (MAE): 363.50296319699754
Average Mean Absolute Percentage Error (MAPE): 12.563581809343642
Average R² Score: 0.5864590801799385
Average Relative Absolute Error (RAE): 0.6358238379366123

For dataset eba 

Average Mean Squared Error (MSE): 71861056.37103133
Average Root Mean Squared Error (RMSE): 8453.86252562438
Average Mean Absolute Error (MAE): 7230.97893778483
Average Mean Absolute Percentage Error (MAPE): 345.35713749337674
Average R² Score: 0.5638244506594419
Average Relative Absolute Error (RAE): 0.6427122607022013

For dataset fb5

Average Mean Squared Error (MSE): 23732.140412156186
Average Root Mean Squared Error (RMSE): 152.74744710944015
Average Mean Absolute Error (MAE): 110.95743113550647
Average Mean Absolute Percentage Error (MAPE): 7.244747971946377
Average R² Score: 0.9104908759671355
Average Relative Absolute Error (RAE): 0.32386980109449676

For dataset f3d

Average Mean Squared Error (MSE): 144072.6764082578
Average Root Mean Squared Error (RMSE): 379.5346948399162
Average Mean Absolute Error (MAE): 292.68298871564434
Average Mean Absolute Percentage Error (MAPE): 26.13112066072066
Average R² Score: 0.2839867889046187
Average Relative Absolute Error (RAE): 0.8279132042834887

## Customer

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

### Load data
#data = pd.read_csv("C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_371.csv")
#data = pd.read_csv("C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_eba.csv")
#data = pd.read_csv("C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_fb5.csv")
data = pd.read_csv("C:\\Users\\Jason\\Downloads\\Capstone Project\\transformed_f3d.csv")

### Define MAPE function with zero handling
def mean_absolute_percentage_error(y_true, y_pred):
    non_zero_mask = y_true != 0  # Avoid division by zero
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

### Define RAE function
def relative_absolute_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))

### Define the encoder function for feature-target creation
def encoder(df, target, window=1, cut=1, drop_timeline=True):
    # Ensure the DataFrame has a proper index
    df = df.reset_index(drop=True)
   
    # New column 'target' that is the reverse rolling mean of the target column
    df['target'] = df[target].rolling(window=window).mean().shift(-1)
   
    # Drop rows with NaN
    df = df.dropna().iloc[:-cut]
   
    if drop_timeline and 'year' in df.columns:
        df = df.drop('year', axis=1)
   
    # Features (X) and target (y)
    X = df.drop(['target', 'index'], axis=1)
    y = df['target']
   
    # Add rolling averages for additional features
    X['customers_recent'] = df['customer_id'].rolling(window=3).mean()
    X['quantity_recent'] = df['quantity'].rolling(window=3).mean()
    X['document_id_recent'] = df['document_id'].rolling(window=3).mean()
   
    # Fill missing values with -1
    X = X.fillna(-1)
   
    return X, y

### Prepare the data
X, y = encoder(data, 'customer_id', window=1, cut=2, drop_timeline=True)

### Scale the features but not the target
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

### Scale the target (y) but keep track of scaler for inverse transformation
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

### Reshape X for LSTM input (samples, timesteps, features)
X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

### Initialize KFold
kf = KFold(n_splits=2, shuffle=True, random_state=42)

### Lists to store performance metrics
mse_scores = []
rmse_scores = []
mae_scores = []
mape_scores = []
r2_scores = []
rae_scores = []

### K-Fold Cross-Validation
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]  # Use scaled `y` for training
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    
    # Predict
    y_pred_scaled = model.predict(X_test).flatten()
    
    # Inverse transform predictions and actuals to original scale
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate performance metrics on the original scale
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mape = mean_absolute_percentage_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)
    rae = relative_absolute_error(y_test_original, y_pred_original)
    
    # Append metrics to lists
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    mape_scores.append(mape)
    r2_scores.append(r2)
    rae_scores.append(rae)

### Calculate average performance metrics
avg_mse = np.mean(mse_scores)
avg_rmse = np.mean(rmse_scores)
avg_mae = np.mean(mae_scores)
avg_mape = np.mean(mape_scores)
avg_r2 = np.mean(r2_scores)
avg_rae = np.mean(rae_scores)

### Print metrics
print(f"Average Mean Squared Error (MSE): {avg_mse}")
print(f"Average Root Mean Squared Error (RMSE): {avg_rmse}")
print(f"Average Mean Absolute Error (MAE): {avg_mae}")
print(f"Average Mean Absolute Percentage Error (MAPE): {avg_mape}")
print(f"Average R² Score: {avg_r2}")
print(f"Average Relative Absolute Error (RAE): {avg_rae}")

### Visualize results from the last fold
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 3))
plt.plot(y_test_original, label='Actual Customers', color='blue', linestyle='--', marker='o', alpha=0.7)
plt.plot(y_pred_original, label='Predicted Customers', color='red', linestyle='-', marker='x', alpha=0.7)
plt.title('Actual vs Predicted Number of Customers (LSTM)')
plt.xlabel('Sample Index')
plt.ylabel('Number of Customers')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## Results
For dataset 371

Average Mean Squared Error (MSE): 89728.92871746744
Average Root Mean Squared Error (RMSE): 297.90692058893103
Average Mean Absolute Error (MAE): 216.71915690104169
Average Mean Absolute Percentage Error (MAPE): 9.865273368373783
Average R² Score: 0.505234887448087
Average Relative Absolute Error (RAE): 0.7310943542822776

For dataset eba 

Average Mean Squared Error (MSE): 140643.57933516643
Average Root Mean Squared Error (RMSE): 374.68258030462164
Average Mean Absolute Error (MAE): 308.19353177812366
Average Mean Absolute Percentage Error (MAPE): 105.23964438849688
Average R² Score: 0.6991145622090849
Average Relative Absolute Error (RAE): 0.5014620164886756

For dataset fb5

Average Mean Squared Error (MSE): 31374.65075727211
Average Root Mean Squared Error (RMSE): 174.4117853527558
Average Mean Absolute Error (MAE): 95.74264920991043
Average Mean Absolute Percentage Error (MAPE): 10.907385156590916
Average R² Score: 0.8438177775429858
Average Relative Absolute Error (RAE): 0.29845526274232936

For dataset f3d

Average Mean Squared Error (MSE): 40241.98929597151
Average Root Mean Squared Error (RMSE): 200.57530626972172
Average Mean Absolute Error (MAE): 154.55176863386822
Average Mean Absolute Percentage Error (MAPE): 23.68892747602844
Average R² Score: 0.3798141104055781
Average Relative Absolute Error (RAE): 0.7465072203634531
