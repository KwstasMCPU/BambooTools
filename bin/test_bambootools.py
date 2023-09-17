from BambooTools import bambootools
import pandas as pd
import numpy as np

df = pd.DataFrame({'weight':[1, 2, 3, 4, 5, np.nan],
                   'animal':['cat', 'dog', 'dog', 'dog', 'cat', 'lama']})


print(df.bbt.pandas_obj)