# import libraries
import pandas as pd
import numpy as np
import statistics

# import data
df_main = pd.read_csv('calif_housing_data.csv')

# (1) Python function that takes in a vector and normalizes it
def norm(a):
    data = np.array(a)
    values = (data - data.min())/ (data.max() - data.min())
    return values

#norm(df_main['median_house_value'])


# (2) Python function that takes in a vector and standardizes it

def stand(a):
    data = np.array(a)
    values = (data - np.mean(data)) / (statistics.stdev(data))
    return values

#stand(df_main['median_house_value'])


# (3) Working with a Dataframe

# (a) how many rows does this data set have?
print(len(df_main))
# (a) Answer: 20640 rows

# (a) drop NAs for future calculations
df_main = df_main.dropna()

print(len(df_main))
# (a) Answer: 20433 rows after removing NAs

# (b) Answer: The target vector for the model is the median house value.

# (c) Create a new feature for total bedrooms / number of households
df_main['average_beds_per_house'] = df_main['total_bedrooms'] / df_main['households']

# (c) Answer: This feature represents the average number of bedrooms per house for
# a given neighborhood block.

# (d) Create new data frame with median age, median income, and new feature from part (c)
df_new = df_main[['housing_median_age', 'median_income', 'average_beds_per_house']]
print(df_new)

# (d) print out of the new dataframe
#       housing_median_age  median_income  average_beds_per_house
#0                      41         8.3252                1.023810
#1                      21         8.3014                0.971880
#2                      52         7.2574                1.073446
#3                      52         5.6431                1.073059
#4                      52         3.8462                1.081081
#...                   ...            ...                     ...
#20635                  25         1.5603                1.133333
#20636                  18         2.5568                1.315789
#20637                  17         1.7000                1.120092
#20638                  18         1.8672                1.171920
#20639                  16         2.3886                1.162264


# (e)
print(df_new.apply(stand, axis = 0))

# (e) print out of the standardization of the dataframe
#        housing_median_age  median_income  average_beds_per_house
#0                0.983858       2.345106               -0.153859
#1               -0.607256       2.332575               -0.262930
#2                1.858971       1.782896               -0.049603
#3                1.858971       0.932947               -0.050416
#4                1.858971      -0.013143               -0.033567
#...                   ...            ...                     ...
#20635           -0.289033      -1.216697                0.076183
#20636           -0.845924      -0.692027                0.459410
#20637           -0.925479      -1.143143                0.048372
#20638           -0.845924      -1.055110                0.157229
#20639           -1.005035      -0.780587                0.136949

#[20433 rows x 3 columns]



