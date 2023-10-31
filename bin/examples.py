from bambootools import bambootools
import pandas as pd
import numpy as np
import seaborn as sns

# Set a seed for reproducibility
np.random.seed(0)

# Define the number of records
n_records = 50

# Define the categories for the 'animal' column
animals = ["cat", "dog", "lama"]

# Generate random data
df = pd.DataFrame(
    {
        "animal": np.random.choice(animals, n_records),
        "color": np.random.choice(
            ["black", "white", "brown", "gray"], n_records
        ),
        "weight": np.random.randint(1, 100, n_records),
        "tail length": np.random.randint(1, 50, n_records),
        "height": np.random.randint(10, 500, n_records),
        "date_of_observation": pd.date_range(
            start="2022-01-01", end="2022-02-01", periods=n_records
        ),
    }
)

# Insert NULL values in the 'animal', 'color', 'weight', 'tail length' and
# 'height' columns
for col, n_nulls in zip(df.columns, [2, 15, 20, 48, 17, 0]):
    null_indices = np.random.choice(df.index, n_nulls, replace=False)
    df.loc[null_indices, col] = np.nan

# get dataset's completeness for each column
print(df.bbt.completeness())

# get dataset's completeness per group
print(df.bbt.completeness(by=["animal"]))

# get missing correlation matrix
print(df.bbt.missing_corr_matrix())

# outlier boundaries per group
penguins = sns.load_dataset("penguins")
print(penguins.bbt.outlier_bounds(method="std", by=["sex", "species"]))

# outliers summary
print(penguins.bbt.outlier_summary(method="std"))

# outliers summary per group
print(
    penguins.bbt.outlier_summary(method="iqr", by=["sex", "species"], factor=1)
)

# get duplication summary
print(penguins.bbt.duplication_summary(subset=["sex", "species", "island"]))

# get duplication frequency table
print(
    penguins.bbt.duplication_frequency_table(
        subset=["sex", "species", "island"]
    )
)

# find how many values and their percentage which are above a threshold
print(df["weight"].bbt.above(thresh=30))

# find how many values and their percentage which are below a threshold
print(df["weight"].bbt.below(thresh=30))
