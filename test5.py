import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the Boston Housing dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name='MEDV')

# Introduce some missing values in the 'AGE' column for demonstration
np.random.seed(0)
missing_rows = np.random.randint(0, X.shape[0], 40)
X.loc[missing_rows, 'AGE'] = np.nan

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Separate the data with missing values and the data without missing values
missing_data = X_train[X_train['AGE'].isnull()]
complete_data = X_train[X_train['AGE'].notnull()]

# Fit a linear regression model using the data without missing values
lr = LinearRegression()
lr.fit(complete_data.drop('AGE', axis=1), complete_data['AGE'])

# Predict the missing values using the linear regression model
age_preds = lr.predict(missing_data.drop('AGE', axis=1))

# Add a random error term to the predicted values
st_dev = np.sqrt(mean_squared_error(complete_data['AGE'], lr.predict(complete_data.drop('AGE', axis=1))))
random_error = np.random.normal(0, st_dev, size=age_preds.shape)
age_preds += random_error

# Impute the missing values with the stochastic regression predictions
X_train.loc[missing_data.index, 'AGE'] = age_preds

# Now you can proceed with your analysis using the imputed dataset
