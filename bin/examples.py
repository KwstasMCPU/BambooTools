from bambootools import bambootools
import pandas as pd
import numpy as np

# Set a seed for reproducibility
np.random.seed(0)

# Define the number of records
n_records = 20

# Define the categories for the 'animal' column
animals = ['cat', 'dog', 'lama']

# Generate random data for the 'animal', 'weight', and 'tail length' columns
df = pd.DataFrame({
    'animal': np.random.choice(animals, n_records),
    'weight': np.random.randint(1, 100, n_records),
    'tail length': np.random.randint(1, 100, n_records),
    'color': np.random.choice(['black', 'white', 'brown', 'gray'], n_records),
    'name': [f'name_{i}' for i in range(n_records)]
})

# Insert NULL values in the 'weight', 'tail length' and 'name' columns
for col, n_nulls in zip(['weight', 'tail length', 'name'], [3, 5, 1]):
    null_indices = np.random.choice(df.index, n_nulls, replace=False)
    df.loc[null_indices, col] = np.nan
 
# get dataset's completeness for each column
print(df.bbt.completeness())

# get dataset's completeness per group
print(df.bbt.completeness(by=['animal']))

# find how many values and their percentage which are above a threshold
print(df['weight'].bbt.above(thresh=30))

# find how many values and their percentage which are below a threshold
print(df['weight'].bbt.below(thresh=30))
