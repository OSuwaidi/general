# بسم الله الرحمن الرحيم

import pandas as pd
import seaborn as sb
import numpy as np
from matplotlib import pyplot as plt
import random

line = '-----------------------------------------------------'  # To make preview clearer
country_data = pd.read_csv('country_data.csv')  # Read csv file
print(f"Dataframe: \n{line}\n{country_data}\n\n")

country_data.Industry = np.array([float(val.replace(',', '.')) if type(val) is not float else val for val in country_data.Industry])
print(country_data.Industry.values)
mask = country_data.Industry >= 0.5
print(country_data[mask])

# Display the dataframe indices which are used for indexing:
print(f"Dataframe's indices: {country_data.index}\n\n")

# To change indices names in case the index was: "index='Country'":
'''
country_data = country_data.rename(index=lambda r: r.strip())
print(country_data.loc['Bahrain'])
'''

# Extract the desired rows (based on row indexes) with all the columns from the dataframe:
print(f'Display 3 countries:\n {line}\n{country_data.loc[[0, 50, 101]]} \n\n')

# Extract desired rows (based on row indexes) and desired columns (based on labels) from the dataframe:
print(f'Display 2 countries with 3 columns:\n {line}\n{country_data.loc[[33, 77], ["Country", "Birthrate", "Deathrate"]]} \n\n')

# Note that the above 2 operations can be done using the same logic with "df.iloc[int]", which takes in integer values for row and column indices

# Show all the categories/classes that exist in our dataframe:
print(f"Dataframe Columns (before): \n{line}\n{country_data.columns}\n\n")


# Adjust the names that contain spaces and/or parenthesis (so that we can call them easier):
def fix(text):
    for i in range(len(text)):
        if text[i:i + 2] == ' (':  # Recall that list indexing ends at (last index - 1)
            return text[:i]
    return text


country_data = country_data.rename(columns=lambda text: fix(text))  # Recall that "lambda" function iterates over all elements in whatever object it's called in ("columns")
print(f"Dataframe Columns (after): \n{line}\n{country_data.columns}\n\n")
'''
If you wanted to remove any non-alphanumeric (not alphabets or numbers) characters from your column (or row) strings:
import re
p = re.compile(r'[^\w\s]+')  # Regex pattern (where: "\w" matches an alphanumeric character and "\s" matches a whitespace character)
country_data = country_data.rename(columns=lambda x: p.sub('', x))  # Substitute every "x" from "columns" that belongs in "p" with nothing ('')
'''

# Quick dataframe information:
print(f"First 5 rows: \n{line}\n{country_data.head()} \n\n")  # Display first 5 rows of data (with their corresponding columns)
print(f"Last 5 rows: \n{line}\n{country_data.tail()} \n\n")  # Display last 5 rows of data (with their corresponding columns)
print(f"Statistics: \n{line}\n{country_data.describe()} \n\n")  # Provides statistical information about the numerical variables in the data (columns)
print(f"Data type in each category: \n{line}\n{country_data.dtypes} \n\n")
print(f"Samples under the GDP category: \n{line}\n{country_data['GDP']} \n\n")  # Can add ".values" ad the end to return a numpy array instead
print(f"How many times same id/name showed up in 'Country': \n{line}\n{country_data.Country.value_counts()} \n\n")  # How many times each sample/value under the category "country" was repeated (to show repetitiveness)
print(f"Samples under the category 'Country': \n{line}\n{country_data.Country.values} \n\n")  # Will print all the samples (or values) under the category "country" in a numpy array (without sample index/id because of ".values")
print(f"Number of missing values in each category: \n{line}\n{country_data.isna().sum()} \n\n")  # Prints the total number of missing values in each category (column)

# We found 1 missing value under GDP! Let's remove it:
index = np.where(country_data["GDP"].isna() == True)[0]  # "is True" doesn't work in "np.where()
print(f'Country with no GDP information: {country_data["Country"][index]}')
country_data.drop(index, inplace=True)  # Remove that index's row
print(f"Number of rows: {len(country_data)}\n\n")

# Find countries with GDP equalling 500:
poor_countries = country_data[country_data["GDP"] == 500]
print(f'{poor_countries = } \n')  # This notation will print the variable along its value!!!
print(country_data.groupby(['GDP']).sum(numeric_only=True), "\n\n")  # Groups different samples/id's based on *matching* categorical group (matching GDP value in this case)
# Sanity check: poor_population_total = sum(country_data.Population[i] for i in range(len(country_data)) if i in index_poor)


# Display a heatmap that shows the cross-correlation between each of the variables in your data
plt.style.use('seaborn')
sb.heatmap(country_data.corr(method='pearson', min_periods=1), annot=True)
plt.title('Cross-Correlation Heatmap')
plt.show()

for i in country_data.Country:
    if i == 'Philippines ':
        print(f'Found {i}!')
        break

gdp_list = list(country_data.GDP)


def kmeans(data, k_num, steps):  # Only works for one dimensional data
    centroids = {}
    for _ in range(k_num):
        centroids[random.randint(min(data), max(data))] = []

    for i in range(steps):
        for point in data:
            difference = [(point - cen) ** 2 for cen in centroids]
            index = difference.index(min(difference))
            centroids[[*centroids][index]].append(point)
        for cen in [*centroids]:
            avg = sum(centroids[cen]) / len(centroids[cen])
        if i == (steps - 1):
            continue
        else:
            del centroids[cen]
            centroids[avg] = []
    return centroids


clustered = kmeans(data=gdp_list, k_num=3, steps=10)
colors = ['b', 'g', 'r', 'μ', 'm', 'y']

i = 0
for key in clustered:
    color = colors[i]
    plt.scatter(key, 0, c=color, marker='*', s=700)  # The "y" value is 0 because it's a one dimensional scattering
    for gdp in clustered[key]:
        index = gdp_list.index(gdp)
        plt.scatter(gdp, 0, c=color, marker='.')
        plt.annotate(country_data.Country[index], (gdp, 0), ha='center')
    i += 1

plt.title('GDP ($ per capita)')
plt.show()
