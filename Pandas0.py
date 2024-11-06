# بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ و بهِ نَستَعين

import numpy as np
import pandas as pd
import os

pd.set_option('display.max_columns', None)  # Display all columns in a dataframe
pd.set_option('expand_frame_repr', False)  # Display the dataframe records on the same line
os.chdir('/Users/omar/Documents/Amazon/Tables/Packing Data')

control_data = pd.read_csv('Control-Table.csv')
experiment_data = pd.read_csv('Experiment-Table.csv')

column_names = [f"task_{i}" for i in range(len(experiment_data.columns))]

# Avoid using the "inplace" setting. Instead, explicitly overwrite the df:
control_data = control_data.rename(columns={c_old: c_new for c_old, c_new in zip(experiment_data.columns, column_names)})
experiment_data = experiment_data.rename(columns={c_old: c_new for c_old, c_new in zip(experiment_data.columns, column_names)})

control_data = control_data.set_index("task_0")
experiment_data = experiment_data.set_index("task_0")

# Clean "control_data":
control_data = control_data.iloc[:15]
control_data["group"] = np.zeros(len(control_data))  # Add a new column that represents the group (class) of samples


# Clean "experiment_data":
exp_task_means = [experiment_data[col].mean() for col in experiment_data.columns]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


for i, col in enumerate(experiment_data.columns):
    mask = experiment_data[col].isna()
    fill_vals = [round(abs(exp_task_means[i] + np.random.randn()*sigmoid(exp_task_means[i])), 2) for _ in range(mask.sum())]
    experiment_data[col][mask] = fill_vals

experiment_data["group"] = np.ones(len(control_data))

# You can save files more efficiently using:
    # "df.to_parquet()"  --> Faster to write and read
# You can also apply filters on columns on read:
    # df = pd.read_parquet("file.parquet", filters=[('col_name', '<', 42)])
# You can also pick out certain columns on read:
    # df = pd.read_parquet("file.parquet", columns=['date', 'age'])
control_data.to_csv("control_data_cleaned.csv", index=True)
experiment_data.to_csv("experiment_data_cleaned.csv", index=True)


#####################################################################################################################################################################


# Fastest way to insert *a row* of data into pd dataframe is by using the "loc" or "iloc" methods:
"""
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

# Insert a row of data:
df.loc[len(df)] = [5, 6]
"""

# Fastest way to insert *multiple rows* of data into pd dataframe is by using the "pd.concat" method:
"""
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

# Create a new dataframe with the rows to be inserted:
new_data = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

# Concatenate the two dataframes:
df = pd.concat([df, new_data], ignore_index=True)  # "ignore_index=True" resets the index to start from 0

# OR:
cols = df.columns
entrees = [[5, 7], [6, 8]]
new_data = pd.DataFrame(entrees, columns=cols)
df = pd.concat([df, new_data], ignore_index=True)

# OR:
new_data = [
    {
        "A": 5,
        "B": 7,
    },
    {
        "A": 6,
        "B": 8,
    },
]
new_data = pd.DataFrame(new_data)
df = pd.concat([df, new_data], ignore_index=True)

# OR:
new_data_A = np.array([5, 6])
new_data_B = np.array([7, 8])
new_df = pd.DataFrame({'A': new_data_A, 'B': new_data_B})
df = pd.concat([df, new_df], ignore_index=True)
"""

# To access/alter a single scalar value in a df (faster than "loc/iloc"):
"""
df.at[1, 'GDP'] = ___
"""

# To access/alter multiple values of a df (slower than "at"):
"""
df.loc[[0, 10, 99], ["GDP, "Temperature"]] = ___
"""

# To apply string methods to a df's column values:
"""
df = pd.DataFrame({'Name': ["omar", "khalid"], 'Job': ["Engineer  ", " Doctor "]})
df['Name_uppercase'] = df['Name'].str.upper()
df['Job'] = df['Job'].str.strip()
"""

# To read DATE datatypes in pandas:
"""
df = pd.DataFrame({'Year': [1, 2], 'Date': [3, 4]})
df['Date'] = df['Date'].astype('datetime64[ns]')

# OR:
df['Date'] = pd.to_datetime(df['Date'])

"""

# To filter your df, use the built-in "query" method:
"""
df = pd.DataFrame({'Year': [1998, 1997], 'Name': ['Omar', 'Khalid']})
filtered = df.query("Year < 1980 and Name == 'Omar'")
Which is the same as doing:
    filtered = df.loc[(df['Year'] < 1980) & (df['Name'] == 'Omar')]
"""

# When filtering using the "query" method, use "@" in front of variables to access them through query strings:
"""
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
max_year = 1980
min_time = 10
filtered = df.query('Year < @max_year and Time > @min_time')
"""

# When applying multiple transformations to a df, it is best to "chain" them:
"""
df = pd.DataFrame({'Year': [1, 2], 'Time': [3, 4]})
Instead of doing this:
    df2 = df.query('Year > 1975')
    df3 = df.groupby(['Airline', 'ID'])  # groups rows of matching "Airline" and "ID" column values
    df_out = df.sort_values('Time')
Do this:
    df_out = (
            df.query('Year > 1975')
            .groupby(['Airline', 'ID'])
            .sort_values('Time')
            )
"""

# To perform aggregation based on grouping a data frame's index values:
"""
# Create a sample DataFrame:
data = {'City': ['London', 'London', 'Paris', 'Paris', 'New York', 'New York'],
        'Year': [2019, 2020, 2019, 2020, 2019, 2020],
        'Sales': [1000, 1500, 2000, 2500, 3000, 3500]}
df = pd.DataFrame(data)
df = df.set_index(['City', 'Year'])  # Set multi-level index

# Perform "groupby" aggregation based on an index:
result = df.groupby(level='City').sum(numeric_only=True)
"""

# To fill all "NaN" values in a particular columns with specific values:
"""
df.fillna({'col1': value1, 'col2': value2, ...})
"""

# Different arguments when exporting a dataframe into a CSV file:
'''
1.) df.to_csv('file_name.csv', na_rep="Unknown") --> Will replace all missing values (NaNs) with "Unknown"
2.) df.to_csv('file_name.csv', float_format='%.2f') --> Will round numbers to two decimals
3.) df.to_csv('file_name.csv', header=False) --> Will remove the column names
4.) df.to_csv('file_name.csv', columns=["col_A","col_B"]) --> Will only export the given columns into CSV format
5.) df.to_csv('file_name.csv', index=False) --> Will remove the row index number/name
'''
