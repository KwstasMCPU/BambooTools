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

```python
from bambootools import bambootools
import pandas as pd
import numpy as np

>>> df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
...                              'Parrot', 'Parrot', 
...                               'Lama', 'Falcon'],
...                  'Max Speed': [380., 370., 24., 26., np.nan, np.nan]})

# check the completeness of the dataset per column
>>> df.bbt.completeness()
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

# Documentation

Work in progress

# Contributing

Contributions are welcome! Contribution guidelines are pending.

# License

MIT