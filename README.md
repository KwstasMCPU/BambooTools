# BambooTools

BambooTools is a Python package that provides extensions of pandas functionalities.

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

```python
from bambootools import bambootools
import pandas as pd
import numpy as np

>>> df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
...                              'Parrot', 'Parrot', 
...                               'Lama', 'Falcon'],
...                  'Max Speed': [380., 370., 24., 26., np.nan, np.nan]})

# check the completeness of the dataset per column
>>> print(df.bbt.completeness())
               perc  count
Animal     1.000000      6
Max Speed  0.666667      4

# check the completeness of the datataset per category
>>> print(df.bbt.completeness(by=['Animal']))
        Max Speed      
            perc count
Animal                
Falcon  0.666667     2
Lama    0.000000     0
Parrot  1.000000     2
```
## Outlier summary

```python
>>> penguins = sns.load_dataset("penguins")
# identify outliers using the  Inter Quartile Range approach
>>> print(penguins.bbt.outlier_summary('iqr', factor=1))

                    n_outliers_upper  n_outlier_lower  n_non_outliers  n_total_outliers
bill_depth_mm                     0                0             342                 0
bill_length_mm                    2                0             340                 2
body_mass_g                       4                0             338                 4
flipper_length_mm                 0                0             342                 0

# outliers per category
>>> print(penguins.bbt.outlier_summary(method='iqr', by=['sex', 'species'], factor=1))

		                                lower	upper
('Female', 'Adelie')	bill_depth_mm	    1	1
('Female', 'Adelie')	bill_length_mm	    1	1
('Female', 'Adelie')	body_mass_g	        0	0
('Female', 'Adelie')	flipper_length_mm	5	3
('Female', 'Chinstrap')	bill_depth_mm	    0	1
('Female', 'Chinstrap')	bill_length_mm	    5	6
('Female', 'Chinstrap')	body_mass_g	        2	1
('Female', 'Chinstrap')	flipper_length_mm	1	0
('Female', 'Gentoo')	bill_depth_mm	    0	1
('Female', 'Gentoo')	bill_length_mm	    0	1
('Female', 'Gentoo')	body_mass_g	        1	0
('Female', 'Gentoo')	flipper_length_mm	1	1
('Male', 'Adelie')	bill_depth_mm	        3	6
('Male', 'Adelie')	bill_length_mm	        3	5
('Male', 'Adelie')	body_mass_g	            0	0
('Male', 'Adelie')	flipper_length_mm	    4	2
('Male', 'Chinstrap')	bill_depth_mm	    1	0
('Male', 'Chinstrap')	bill_length_mm	    0	2
('Male', 'Chinstrap')	body_mass_g	        2	3
('Male', 'Chinstrap')	flipper_length_mm	1	1
('Male', 'Gentoo')	bill_depth_mm   	    2	3
('Male', 'Gentoo')	bill_length_mm	        5	5
('Male', 'Gentoo')	body_mass_g	            1	1
('Male', 'Gentoo')	flipper_length_mm	    2	0
```

# Contributing

Contributions are welcome! Contribution guidelines are pending.

# License

MIT