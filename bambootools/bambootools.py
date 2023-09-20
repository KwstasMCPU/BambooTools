"""
:copyright: (c) 2023 by Maravegias Konstantinos.
:license: MIT, see LICENSE for more details.
"""
import pandas as pd
from typing import List, Tuple


@pd.api.extensions.register_dataframe_accessor("bbt")
class BambooToolsDfAccessor:
    def __init__(self, pandas_obj) -> None:
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @property
    def pandas_obj(self) -> pd.DataFrame:
        return self._obj

    @staticmethod    
    def _validate(obj) -> None:
        """Validate that the passed object is a pandas Dataframe

        Args:
            obj: Object passed

        Raises:
            AttributeError: Prompts the user that pd.DataFrame should be used
        """
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a pandas DataFrame")

    def completeness(self, by: List[str] = None) -> pd.DataFrame:
        """Returns the completeness table of a dataframe

        Returns:
            pd.DataFrame: The completeness table
        """

        if by is None:
            by = self._obj.columns.to_list()
            counts = self._obj.groupby(by, dropna=False).\
                apply(lambda x: x.notnull().sum()).sum()

            perc = self._obj.groupby(by, dropna=False).\
                apply(lambda x: x.notnull().sum()).sum() / self._obj.shape[0]

            _df = pd.concat([perc, counts], axis=1).\
                rename(columns={0: 'perc',
                                1: 'count'
                                }
                       )
        else:
            if not isinstance(by, List):
                raise AttributeError("`by` arg must be a list of strings")

            _df = self._obj.groupby(by, dropna=False).\
                agg([('perc', lambda x: x.notnull().sum()/x.shape[0]),
                     ('count', lambda x: x.notnull().sum())
                     ]
                    )

        return _df


@pd.api.extensions.register_series_accessor("bbt")
class BambooToolsSeriesAccessor:
    def __init__(self, series_obj) -> None:
        self._validate(series_obj)
        self._obj = series_obj

    @property
    def series_obj(self) -> pd.Series:
        return self._obj

    @staticmethod    
    def _validate(obj) -> None:
        """Validate that the passed object is a pandas Series

        Args:
            obj: Object passed

        Raises:
            AttributeError: Prompts the user that pd.Series should be used
        """
        if not isinstance(obj, pd.Series):
            raise AttributeError("Must be a pandas Series")

    def above(self, thresh: float,
              dropna: bool = False) -> Tuple[float, float]:
        """Calculates the number of values and their percentage which are
        above a specific threshold.

        Args:
            thresh (float): The threshold given by the user.
            dropna (bool, optional): If True drops the NULL records before
                the calculation of percentage of NULL values. Hence the total
                number of records equal to the number of non NULL values.
                Defaults to False.

        Returns:
            Tuple[float, float]: Counts and percentage of records above the
                given threshold.
        """
        if dropna:
            values_above = self._obj > thresh
        else:
            values_above = self._obj.dropna() > thresh
        count = values_above.sum()
        perc = values_above.sum() / len(values_above)

        return count, perc

    def below(self, thresh: float,
              dropna: bool = False) -> Tuple[float, float]:
        """Calculates the number of values and their percentage which are
        below a specific threshold.

        Args:
            thresh (float): The threshold given by the user.
            dropna (bool, optional): If True drops the NULL records before
                the calculation of percentage of NULL values. Hence the total
                number of records equal to the number of non NULL values.
                Defaults to False.

        Returns:
            Tuple[float, float]: Counts and percentage of records below the
                given threshold.
        """

        if dropna:
            values_above = self._obj < thresh
        else:
            values_above = self._obj.dropna() < thresh
        count = values_above.sum()
        perc = values_above.sum() / len(values_above)

        return count, perc
