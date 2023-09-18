import pytest
from bambootools import bambootools
import pandas as pd
import numpy as np

@pytest.fixture
def make_dataset():
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
    return df

def test_init(make_dataset):
    assert make_dataset.equals(make_dataset.bbt.pandas_obj), "Expected equal dataframe."
    
def test_completeness(make_dataset):
    result = make_dataset.bbt.completeness()
    assert result.shape == (5, 2), "Wrong table dimensions."
    assert result['count'].max() <= make_dataset.shape[0], "Max value of non missing cannot exceed total number of records."
    assert result['perc'].max() <= 1.0, "Max value of perc cannot exceed 1."
    
def test_completeness_per_group(make_dataset):
    result = make_dataset.bbt.completeness(by=['animal'])
    assert result.shape == (3, 8), "Wrong table dimensions."
    assert result['weight']['perc'].max() <= 1.0, "Max value of perc cannot exceed 1."