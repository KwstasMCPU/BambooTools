import pytest
from bambootools import bambootools
import pandas as pd
import numpy as np
import seaborn as sns


@pytest.fixture
def animals_dataset():
    # Set a seed for reproducibility
    np.random.seed(0)

    # Define the number of records
    n_records = 20

    # Define the categories for the 'animal' column
    animals = ['cat', 'dog', 'lama']

    # Generate random data for the 'animal', 'weight', n 'tail length' columns
    df = pd.DataFrame({
        'animal': np.random.choice(animals, n_records),
        'weight': np.random.randint(1, 100, n_records),
        'tail length': np.random.randint(1, 100, n_records),
        'color': np.random.choice(['black', 'white',
                                   'brown', 'gray'],
                                  n_records),
        'name': [f'name_{i}' for i in range(n_records)]
    })

    # Insert NULL values in the 'weight', 'tail length' and 'name' columns
    for col, n_nulls in zip(['weight', 'tail length', 'name'], [3, 5, 1]):
        null_indices = np.random.choice(df.index, n_nulls, replace=False)
        df.loc[null_indices, col] = np.nan
    return df


@pytest.fixture
def penguins_dataset():
    return sns.load_dataset("penguins")


# tests for BambooToolsDfAccessor
def test_init_dataframe(animals_dataset):
    assert animals_dataset.equals(animals_dataset.bbt.pandas_obj), (
        "Expected equal dataframe.")


def test_completeness(animals_dataset):
    result = animals_dataset.bbt.completeness()
    assert result.shape == (5, 3), "Wrong table dimensions."
    assert result['count'].max() <= animals_dataset.shape[0], (
        "Max value of non missing cannot exceed total number of records.")
    assert result['perc'].max() <= 1.0, (
        "Max value of perc cannot exceed 1.")


def test_completeness_per_group(animals_dataset):
    result = animals_dataset.bbt.completeness(by=['animal'])
    assert result.shape == (3, 12), "Wrong table dimensions."
    assert result['weight']['perc'].max() <= 1.0, (
        "Max value of perc cannot exceed 1.")
    # test if the counts were calculated correctly
    n_cats = animals_dataset['animal'].value_counts()['cat']
    assert result['weight'].loc['cat', 'total'] == n_cats, (
        "Total counts per category must equal the value counts."
    )


def test_missing_corr_matrix(animals_dataset):
    result = animals_dataset.bbt.missing_corr_matrix()
    n_columns = animals_dataset.shape[1]
    assert result.shape == (n_columns, n_columns), (
        "A {}x{} martix was expected".format(n_columns, n_columns)
    )


def test_outlier_summary(penguins_dataset):
    result = penguins_dataset.bbt.outlier_summary(method='iqr',
                                                  by=['sex', 'species']
                                                  )
    assert result.shape == (24, 5), "Shape should be (24, 5). sex*species."
    assert result['n_outliers_lower'].sum() == 8, "N of outliers should be 8."
    # test if the counts were calculated correctly
    n_penguins_grp = penguins_dataset.groupby(['sex', 'species']).size()
    assert n_penguins_grp['Female', 'Adelie'] == result['total_records'].\
        loc[('Female', 'Adelie'), 'bill_depth_mm'], (
            "Total counts per category must equal the original data counts."
        )


# tests for BambooToolsSeriesAccessor
def test_init_series(animals_dataset):
    assert animals_dataset['weight'].equals(
        animals_dataset['weight'].bbt.series_obj), (
        "Expected equal series."
    )


def test_above(animals_dataset):
    result = animals_dataset['weight'].bbt.above(thresh=30)
    assert isinstance(result, tuple), "Did not return a tuple."
    assert result[0] == 10, "Value counts should be 10."


def test_below(animals_dataset):
    result = animals_dataset['weight'].bbt.below(thresh=30)
    assert isinstance(result, tuple), "Did not return a tuple."
    assert result[0] == 6, "Value counts should be 10."
