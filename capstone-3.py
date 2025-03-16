import pandas as pd

df = pd.read_csv('./dataset_part_1.csv')

# Identify and calculate the percentage of the missing values in each attribute
print(df.isnull().sum() / len(df) * 100)
print('-----------------------------------')

# Identify which columns are numerical and categorical:
print(df.dtypes)
print('-----------------------------------')

# -----------------------------------------------------------------------
# TASK 1: Calculate the number of launches on each site
# -----------------------------------------------------------------------
launch_sites = df['LaunchSite'].value_counts()
print(launch_sites)
print('-----------------------------------')


# -----------------------------------------------------------------------
# TASK 2: Calculate the number and occurrence of each orbit
# -----------------------------------------------------------------------
orbit = df['Orbit'].value_counts()
print(orbit)
print('-----------------------------------')


# -----------------------------------------------------------------------
# TASK 3: Calculate the number and occurrence of mission outcome of the orbits
# -----------------------------------------------------------------------
landing_outcomes = df['Outcome'].value_counts()
print(landing_outcomes)
print('-----------------------------------')

for i, outcome in enumerate(landing_outcomes.keys()):
  print(i, outcome)
print('-----------------------------------')

bad_outcomes = set(landing_outcomes.keys()[[1,3,5,6,7]])
print(bad_outcomes)
print('-----------------------------------')


# -----------------------------------------------------------------------
# TASK 4: Create a landing outcome label from Outcome column
# -----------------------------------------------------------------------
landing_class = []
landing_class = df['Outcome'].isin(bad_outcomes).eq(False).astype(int)
print(landing_class[0:10])
print('-----------------------------------')

df['Class'] = landing_class
df.to_csv("dataset_part_2.csv", index=False)