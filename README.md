# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries for data manipulation, model building, evaluation, and visualization.
2. Load the car price dataset and perform preprocessing:
   - Drop irrelevant columns.
   - Convert categorical variables into dummy/indicator variables.
3. Separate the dataset into features (X) and target variable (y).
4. Standardize the features and target using `StandardScaler`.
5. Split the dataset into training and testing sets using `train_test_split`.
6. Define Ridge, Lasso, and ElasticNet regression models with polynomial feature transformation using `Pipeline`.
7. Train each model on the training data.
8. Predict the car prices using the trained models on the testing data.
9. Evaluate the model performance using Mean Squared Error (MSE) and R² score.
10. Visualize the performance metrics for each model using bar plots.


## Program:
```python
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: THARUN V K
RegisterNumber: 212223230231
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("CarPrice_Assignment.csv")

# Data preprocessing
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)

# Splitting the data into features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Standardizing the features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models and pipelines
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    # Create a pipeline with polynomial features and the model
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('regressor', model)
    ])

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    predictions = pipeline.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Store results
    results[name] = {'MSE': mse, 'R² score': r2}

# Print results
for model_name, metrics in results.items():
    print(f"{model_name} - Mean Squared Error: {metrics['MSE']:.2f}, R² score: {metrics['R² score']:.2f}")

# Visualization of the results
# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)

# Set the figure size
plt.figure(figsize=(12, 5))

# Bar plot for MSE
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='MSE', hue='Model', data=results_df, palette='viridis', legend=False)
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)

# Bar plot for R² score
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='R² score', hue='Model', data=results_df, palette='viridis', legend=False)
plt.title('R² Score')
plt.ylabel('R² Score')
plt.xticks(rotation=45)

# Show the plots
plt.tight_layout()
plt.show()

*/
```

## Output:
### PERFORMANCE METRICS:
![Screenshot 2025-04-24 105818](https://github.com/user-attachments/assets/410046e4-ee42-49c1-8c3e-b807e6e51538)


### BAR PLOT FOR MSE:
![Screenshot 2025-04-24 105824](https://github.com/user-attachments/assets/0cb046a8-d5cc-47b8-94ea-d2491f0162d9)

### BAR PLOT FOR R²:
![Screenshot 2025-04-24 105830](https://github.com/user-attachments/assets/4ac105b4-3f50-49e1-a409-a514e92fb863)

## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
