# BambooTools

BambooTools is Python library designed to enhance your data analysis workflows. Built as an extension to the widely-used pandas library, BambooTools provides one liner methods for outlier detection and completeness summary in pandas datasets.

With BambooTools, you can easily identify and handle outliers in your data, enabling more accurate analyses and predictions. The library also offers a completeness summary feature, which provides a quick and efficient way to assess the completeness of your dataset.

## Installation

The package is in very early stage and yet not published in PyPi. 

```bash
pip install git+https://github.com/KwstasMCPU/BambooTools
```

OR simple download the project and:

```bash
pip install . 
```

# Usage

## Completeness summary

`completeness()` retuns a completeness summary table, stating the percentage and count of complete (not NULL) values:

```python
from bambootools import bambootools
import pandas as pd
import numpy as np

df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
                             'Parrot', 'Parrot',
                             'Lama', 'Falcon'],
                   'Max Speed': [380., 370., 
                                 24., 26., 
                                 np.nan, np.nan]
                   })

# check the completeness of the dataset per column
print(df.bbt.completeness())
```
|           | perc     | count |
|-----------|----------|-------|
| Animal    | 1        | 6     |
| Max Speed | 0.666667 | 4     |

Specifying a list of categorical columns would result the completeness per category:
```python
# check the completeness of the datataset per category
print(df.bbt.completeness(by=['Animal']))
```
| Animal | Max Speed   |           |
|--------|-------------|-----------|
|        | perc        | count     |
| Falcon | 0.666666667 | 2         |
| Lama   | 0           | 0         |
| Parrot | 1           | 2         |
## Outlier summary

`outlier_summary()` retuns a summary of the outliers found in the dataset based on a specific method (eg. IQR).
It returns the number of outliers below and above the boundaries calculated by the specific method.
```python
penguins = sns.load_dataset("penguins")
# identify outliers using the  Inter Quartile Range approach
print(penguins.bbt.outlier_summary('iqr', factor=1))
```
|                   | n_outliers_upper | n_outlier_lower | n_non_outliers | n_total_outliers |
|-------------------|------------------|-----------------|----------------|------------------|
| bill_depth_mm     | 0                | 0               | 342            | 0                |
| bill_length_mm    | 2                | 0               | 340            | 2                |
| body_mass_g       | 4                | 0               | 338            | 4                |
| flipper_length_mm | 0                | 0               | 342            | 0                |

You can also get the summary per group:

```python
# outliers per category
print(penguins.bbt.outlier_summary(method='iqr', by=['sex', 'species'], factor=1))
```
|                         |                   | lower | upper |
|-------------------------|-------------------|-------|-------|
| ('Female', 'Adelie')    | bill_depth_mm     | 1     | 1     |
| ('Female', 'Adelie')    | bill_length_mm    | 1     | 1     |
| ('Female', 'Adelie')    | body_mass_g       | 0     | 0     |
| ('Female', 'Adelie')    | flipper_length_mm | 5     | 3     |
| ('Female', 'Chinstrap') | bill_depth_mm     | 0     | 1     |
| ('Female', 'Chinstrap') | bill_length_mm    | 5     | 6     |
| ('Female', 'Chinstrap') | body_mass_g       | 2     | 1     |
| ('Female', 'Chinstrap') | flipper_length_mm | 1     | 0     |
| ('Female', 'Gentoo')    | bill_depth_mm     | 0     | 1     |
| ('Female', 'Gentoo')    | bill_length_mm    | 0     | 1     |
| ('Female', 'Gentoo')    | body_mass_g       | 1     | 0     |
| ('Female', 'Gentoo')    | flipper_length_mm | 1     | 1     |
| ('Male', 'Adelie')      | bill_depth_mm     | 3     | 6     |
| ('Male', 'Adelie')      | bill_length_mm    | 3     | 5     |
| ('Male', 'Adelie')      | body_mass_g       | 0     | 0     |
| ('Male', 'Adelie')      | flipper_length_mm | 4     | 2     |
| ('Male', 'Chinstrap')   | bill_depth_mm     | 1     | 0     |
| ('Male', 'Chinstrap')   | bill_length_mm    | 0     | 2     |
| ('Male', 'Chinstrap')   | body_mass_g       | 2     | 3     |
| ('Male', 'Chinstrap')   | flipper_length_mm | 1     | 1     |
| ('Male', 'Gentoo')      | bill_depth_mm     | 2     | 3     |
| ('Male', 'Gentoo')      | bill_length_mm    | 5     | 5     |
| ('Male', 'Gentoo')      | body_mass_g       | 1     | 1     |
| ('Male', 'Gentoo')      | flipper_length_mm | 2     | 0     |

## Outlier boundaries

`outlier_bounds()` returns the boundary values which any value below or above is considered an outlier:
```python
print(penguins.bbt.outlier_bounds(method='iqr',
                                  by=['sex', 'species'],
                                  factor=1))
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

Contributions are welcome! Contribution guidelines are pending.

# License

MIT