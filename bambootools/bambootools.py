"""
This module implements the Requests API.

:copyright: (c) 2023 by Maravegias Konstantinos.
:license: MIT, see LICENSE for more details.
"""
import pandas as pd
import numpy as np


@pd.api.extensions.register_dataframe_accessor("bbt")
class BambooToolsAccessor:
    def __init__(self, pandas_obj) -> None:
        self._validate(pandas_obj)
        self._obj = pandas_obj
        
    @property
    def pandas_obj(self):
        return self._obj
    
    # @pandas_obj.setter
    # def pandas_obj(self, pandas_obj):
    #     if not isinstance(pandas_obj, pd.DataFrame):
    #         raise AttributeError("Must be a pandas DataFrame")
    @staticmethod    
    def _validate(obj):
        # verify this is a DataFrame
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a pandas DataFrame")
        
    def completeness(self):
        
        # if not isinstance(by, list):
        #     raise AttributeError("Must pass a list of columns")

        by = self._obj.columns.to_list()
        counts = self._obj.groupby(by, dropna=False).apply(lambda x: x.notnull().sum()).sum()
        perc = self._obj.groupby(by, dropna=False).apply(lambda x: x.notnull().sum()).sum() / self._obj.shape[0]

        _df = pd.concat([perc, counts], axis=1).\
            rename(columns={0:'percentage',
                            1:'count'
                            }
                   )
        return _df