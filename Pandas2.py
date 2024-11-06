# بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ و به نستعين

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import make_pipeline
import seaborn as sns
from scipy.stats import ttest_ind


# Seaborn datasets:
"""
tips: A dataset of 244 observations of tips given to waiters in a restaurant.
titanic: A dataset of 887 observations of passengers on the Titanic.
iris: A dataset of 150 observations of iris flowers with measurements of their sepal length and width, and petal length and width.
flights: A dataset of 144,000 observations of flights departing from New York City in 2013.
exercise: A dataset of 180 observations of physical activity, diet, and other factors for a group of people.
diamonds: A dataset of 53,940 observations of diamond prices with 10 variables, including carat weight, cut, color, clarity, and price.
"""

seperator = "\n" + "#" * 150
np.set_printoptions(linewidth=np.inf, threshold=np.inf)  # Display the numerically processed dataframe's records on the same line
pd.set_option('display.max_columns', None)  # Display all columns in a dataframe
pd.set_option('expand_frame_repr', False)  # Display the dataframe records in the same line

# Load dataframe from Seaborn:
df = sns.load_dataset('titanic')


# Explore the dataset:
"""
NOTE:
- If the "int" datatype columns don't have a large range of values, it's often better to convert them into "categorical", unless you're dealing with trees/forests
- When dealing with trees/forests:
    1.) We do not need to scale (normalize) the numerical features
    2.) It's okay to use ordinal encoding for the categorical variables (even if it's arbitrary ordering)
"""
print(df.info(), seperator)
print(df.describe(), seperator)
print(df, seperator)  # Displays the dataframe's head and tail


# Extract the target feature from the dataset:
target_feature = "survived"
data = df.drop(target_feature, axis=1)
target = df[target_feature].values


# The "object" datatype is like "VARCHAR", it is less efficient than "category" which is used for a limited number of occurrences in a column (classes)
object_cols = data.select_dtypes(include=['object']).columns
data[object_cols] = data[object_cols].astype('category')

discrete_columns = data.select_dtypes(include=['category', int]).columns
for col in discrete_columns:
    unique = np.sort(data[col].dropna().unique())  # Applying ".sort()" method modifies the array in place and does not return anything
    print(f'Unique values in "{col}": {unique}')
    if len(unique) == 2:
        data[col] = df[col].map({unique[0]: False, unique[1]: True})  # Convert binary "category/int" datatypes to binary "bool"
print(seperator)
print(data.head(), seperator)


# Create the dataset:
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.85, random_state=0)


# Instead of preprocessing each feature datatype (numerical and categorical) separately (decoupled), we can combine the preprocessing steps into one pipeline:
numerical_columns = data.select_dtypes(include=[int, float]).columns  # We need to scale them
binary_columns = data.select_dtypes(include=[bool]).columns  # We need to convert them to 1's and 0's
categorical_columns = data.select_dtypes(include=['category']).columns  # We need to transform to one-hot vectors

numerical_preprocessor = StandardScaler()
binary_preprocessor = OrdinalEncoder()
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")  # If a new categorical value is encountered in the test data, it is ignored and not included in the one-hot encoded representation of the data

# The "ColumnTransformer" class takes 3 arguments: preprocess name, the transformer, and the columns of interest
preprocessor = ColumnTransformer([
    ('standard_scaler', numerical_preprocessor, numerical_columns),
    ('binary-encoder', binary_preprocessor, binary_columns),
    ('one-hot-encoder', categorical_preprocessor, categorical_columns)])

processed_columns = preprocessor.fit_transform(data)  # The output of the preprocessor is a numpy matrix
print(processed_columns[:5])
print(processed_columns.shape, seperator)


# Now combine the transformer (preprocessor) with a classifier or regressor model in a pipeline:
# The model "HistGradientBoostingClassifier" can natively handle "NaN" values by default, unlike "LogisticRegression" which requires preprocessing
model = make_pipeline(preprocessor, HistGradientBoostingClassifier(learning_rate=0.1, max_depth=3, random_state=0))
# model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))
print(model, seperator)

model.fit(x_train, y_train)
y_hat = model.predict(x_test)
print(y_hat[:5])
print(y_test[:5])

accuracy = model.score(x_test, y_test)
print(f"Model's accuracy: {accuracy*100:.4}%", seperator)

# Using "cross_validate" will handle the train/test data split and run cross validation over "cv" runs to evaluate a model's generalization and robustness:
cv_results = cross_validate(model, x_train, target, cv=5, scoring='accuracy')  # "data" must remain as a dataframe to use "cross_validate()"
print(cv_results)

accuracies = cv_results['test_score']
print(f"The mean cross-validation accuracy is: {accuracies.mean():.4} ± {accuracies.std():.4}")
