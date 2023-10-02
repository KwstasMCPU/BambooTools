# BambooTools

BambooTools is a Python library designed to enhance your data analysis workflows. Built as an extension to the widely-used pandas library, BambooTools provides one liner methods for outlier detection and investigation of missing values.

With BambooTools, you can easily identify and handle outliers in your data, enabling more accurate analyses and predictions. The library also offers a completeness summary feature, which provides a quick and efficient way to assess the completeness of your dataset.

## Installation

Install from PiPy

```bash
pip install BambooTools
```

Install from source

```bash
pip install git+https://github.com/KwstasMCPU/BambooTools
```

# Usage

You can find examples in the `bin\examples.py` file. I have illustrated some below as well.

## Completeness summary

`completeness()` retuns a completeness summary table, stating the percentages and counts of complete (not NULL) values for each column:

```python
from bambootools import bambootools
import pandas as pd
import numpy as np

df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
                              'Parrot', 'Parrot',
                              'Lama', 'Falcon'],
                   'Max Speed': [380, 370,
                                 24, 26,
                                 np.nan, np.nan],
                   'Weight': [np.nan, 2,
                              1.5, np.nan,
                              80, 2.2]
                   })
# check the completeness of the dataset per column
print(df.bbt.completeness())
```
|           | perc               | count | total |
|-----------|--------------------|-------|-------|
| Animal    | 1.0                | 6     | 6     |
| Max Speed | 0.6666666666666666 | 4     | 6     |
| Weight    | 0.6666666666666666 | 4     | 6     |

Specifying a list of categorical columns would result the completeness per category:
```python
# check the completeness of the datataset per category
print(df.bbt.completeness(by=['Animal']))
```
|        | Max Speed   |           |           | Weight      |        |        |
|--------|-------------|-----------|-----------|-------------|--------|--------|
| Animal | perc        | count     | total     | perc        | count  | total  |
|        |             |           |           |             |        |        |
| Falcon | 0.666666667 | 2         | 3         | 0.666666667 | 2      | 3      |
| Lama   | 0           | 0         | 1         | 1           | 1      | 1      |
| Parrot | 1           | 2         | 2         | 0.5         | 1      | 2      |

## Missing values correlation matrix
`missing_corr_matrix()` This matrix aims to help to pintpoint relationships between missing values of different columns. Calculates
the conditional probability of a column's value being NaN, given the fact another column value is NaN.

For a dataset with two columns `'A', 'B'` the conditional probability of a value from column `'A'` being NaN is:

$$P(A \text{ is NULL } | B \text{ is NULL}) = \frac{P(A \text{ is NULL } \cap B \text{ is NULL})}{P(B \text{ is NULL})}$$

*Note:* The matrix alone will not tell the whole story. Additional metrics, such dataset's completeness can help if any relationship exists.

```python
# Generate a bigger dataset
# Set a seed for reproducibility
np.random.seed(0)

# Define the number of records
n_records = 50

# Define the categories for the 'animal' column
animals = ['cat', 'dog', 'lama']

# Generate random data
df = pd.DataFrame({
    'animal': np.random.choice(animals, n_records),
    'color': np.random.choice(['black', 'white', 'brown', 'gray'], n_records),
    'weight': np.random.randint(1, 100, n_records),
    'tail length': np.random.randint(1, 50, n_records),
    'height': np.random.randint(10, 500, n_records)
})

# Insert NULL values in the 'animal', 'color', 'weight', 'tail length' and 'height' columns
for col, n_nulls in zip(df.columns, [2, 15, 20, 48, 17]):
    null_indices = np.random.choice(df.index, n_nulls, replace=False)
    df.loc[null_indices, col] = np.nan

# missing values correlations
print(df.bbt.missing_corr_matrix())
```
|             | animal   | color    | weight   | tail length | height   |
|-------------|----------|----------|----------|-------------|----------|
| animal      | NaN      | 0.5      | 0.5      | 1           | 0        |
| color       | 0.066667 | NaN      | 0.333333 | 1           | 0.4      |
| weight      | 0.05     | 0.25     | NaN      | 0.95        | 0.25     |
| tail length | 0.041667 | 0.3125   | 0.395833 | NaN         | 0.354167 |
| height      | 0        | 0.352941 | 0.294118 | 1           | NaN      |

## Outlier summary

`outlier_summary()` retuns a summary of the outliers found in the dataset based on a specific method (eg. IQR).
It returns the number of outliers below and above the boundaries calculated by the specific method.
```python
penguins = sns.load_dataset("penguins")
# identify outliers using the  Inter Quartile Range approach
print(penguins.bbt.outlier_summary('iqr', factor=1))
```
|                   | n_outliers_upper | n_outliers_lower | n_non_outliers | n_total_outliers | total_records |
|-------------------|------------------|------------------|----------------|------------------|---------------|
| bill_depth_mm     | 0                | 0                | 342            | 0                | 342           |
| bill_length_mm    | 2                | 0                | 340            | 2                | 342           |
| body_mass_g       | 4                | 0                | 338            | 4                | 342           |
| flipper_length_mm | 0                | 0                | 342            | 0                | 342           |

You can also get the summary per group:

```python
# outliers per category
print(penguins.bbt.outlier_summary(method='iqr', by=['sex', 'species'], factor=1))
```
|                         |                   | n_non_outliers | n_outliers_lower | n_outliers_upper | n_total_outliers | total_records |
|-------------------------|-------------------|----------------|------------------|------------------|------------------|---------------|
| ('Female', 'Adelie')    | bill_depth_mm     | 71             | 1                | 1                | 2                | 73            |
| ('Female', 'Adelie')    | bill_length_mm    | 71             | 1                | 1                | 2                | 73            |
| ('Female', 'Adelie')    | body_mass_g       | 73             | 0                | 0                | 0                | 73            |
| ('Female', 'Adelie')    | flipper_length_mm | 65             | 5                | 3                | 8                | 73            |
| ('Female', 'Chinstrap') | bill_depth_mm     | 33             | 0                | 1                | 1                | 34            |
| ('Female', 'Chinstrap') | bill_length_mm    | 23             | 5                | 6                | 11               | 34            |
| ('Female', 'Chinstrap') | body_mass_g       | 31             | 2                | 1                | 3                | 34            |
| ('Female', 'Chinstrap') | flipper_length_mm | 33             | 1                | 0                | 1                | 34            |
| ('Female', 'Gentoo')    | bill_depth_mm     | 57             | 0                | 1                | 1                | 58            |
| ('Female', 'Gentoo')    | bill_length_mm    | 57             | 0                | 1                | 1                | 58            |
| ('Female', 'Gentoo')    | body_mass_g       | 57             | 1                | 0                | 1                | 58            |
| ('Female', 'Gentoo')    | flipper_length_mm | 56             | 1                | 1                | 2                | 58            |
| ('Male', 'Adelie')      | bill_depth_mm     | 64             | 3                | 6                | 9                | 73            |
| ('Male', 'Adelie')      | bill_length_mm    | 65             | 3                | 5                | 8                | 73            |
| ('Male', 'Adelie')      | body_mass_g       | 73             | 0                | 0                | 0                | 73            |
| ('Male', 'Adelie')      | flipper_length_mm | 67             | 4                | 2                | 6                | 73            |
| ('Male', 'Chinstrap')   | bill_depth_mm     | 33             | 1                | 0                | 1                | 34            |
| ('Male', 'Chinstrap')   | bill_length_mm    | 32             | 0                | 2                | 2                | 34            |
| ('Male', 'Chinstrap')   | body_mass_g       | 29             | 2                | 3                | 5                | 34            |
| ('Male', 'Chinstrap')   | flipper_length_mm | 32             | 1                | 1                | 2                | 34            |
| ('Male', 'Gentoo')      | bill_depth_mm     | 56             | 2                | 3                | 5                | 61            |
| ('Male', 'Gentoo')      | bill_length_mm    | 51             | 5                | 5                | 10               | 61            |
| ('Male', 'Gentoo')      | body_mass_g       | 59             | 1                | 1                | 2                | 61            |
| ('Male', 'Gentoo')      | flipper_length_mm | 59             | 2                | 0                | 2                | 61            |

## Outlier boundaries

`outlier_bounds()` returns the boundary values which any value below or above is considered an outlier:
```python
print(penguins.bbt.outlier_bounds(method='iqr', by=['sex', 'species'], factor=1))
```
|            |               | bill_length_mm | bill_length_mm | bill_depth_mm | bill_depth_mm | flipper_length_mm | flipper_length_mm | body_mass_g | body_mass_g |
|------------|---------------|----------------|----------------|---------------|---------------|-------------------|-------------------|-------------|-------------|
|            |               | lower          | upper          | lower         | upper         | lower             | upper             | lower       | upper       |
| **sex**    | **species**   |                |                |               |               |                   |                   |             |             |
| **Female** | **Adelie**    | 33             | 41.7           | 15.7          | 19.6          | 179               | 197               | 2800        | 3925        |
| **Female** | **Chinstrap** | 43.475         | 49.325         | 15.95         | 19.1          | 178.75            | 204.25            | 3031.25     | 4025        |
| **Female** | **Gentoo**    | 40.825         | 49.9           | 13            | 15.4          | 205               | 220               | 4050        | 5287.5      |
| **Male**   | **Adelie**    | 36.5           | 44             | 17.4          | 20.7          | 181               | 205               | 3300        | 4800        |
| **Male**   | **Chinstrap** | 48.125         | 53.9           | 17.8          | 20.8          | 189               | 210               | 3362.5      | 4468.75     |
| **Male**   | **Gentoo**    | 45.7           | 52.9           | 14.3          | 17            | 211               | 232               | 4900        | 6100        |
# Contributing

Contributions are more than welcome! You can contribute with several ways:

* Bug reports and bug fixes
* Recommendations for new features and implementation of those
* Writing and or improving existing tests, to ensure quality

Prior any contributions, opening an issue is recommended.

It is also recommended to install the package in ["development mode"](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#working-in-development-mode) while working on it. *When installed as editable, a project can be edited in-place without reinstallation.*

To install a Python package in "editable"/"development" mode change directory to the root of the project directory and run:

```bash
pip install -e .
pip install -r requirements-dev.txt # this will install the development dependencies (e.g. pytest)
```

In order to install the package and the development dependencies with a one liner, run the below:

```bash
pip install -e ".[dev]"
```

## General Guidelines

1. Fork the repository on GitHub.
2. Clone the forked repository to your local machine.
3. Make a new branch, from the `develop` branch for your feature or bug fix.
4. Implement your changes. 
   - It is recommended to write tests and examples for them in `tests\test_bambootols.py` and `bin\examples.py` respectively.
1. Create a Pull Request. Link it to the issue you have opened.

# Credits

Special thanks to [danikavu](https://github.com/danikavu) for the code reviews