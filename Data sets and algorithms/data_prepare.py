import pandas as pd

# Column names
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

# read adult.data 
train = pd.read_csv("adult.data", header=None, names=columns, na_values=" ?", skipinitialspace=True)

# read adult.test , skip first row
test = pd.read_csv("adult.test", header=None, names=columns, na_values=" ?", skipinitialspace=True, skiprows=1)

# remove '.' from all income column 
test['income'] = test['income'].str.replace('.', '', regex=False)

# collab education and test sets
df = pd.concat([train, test], ignore_index=True)

# drop absent data
df = df.dropna()

# Save updated data as csv 
df.to_csv("adult_combined.csv", index=False)

print("Data has been prepared and saved as 'adult_combined.csv'.")
print(f"The dataset contains a total of {df.shape[0]} rows and {df.shape[1]} columns.")
