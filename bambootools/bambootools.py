"""
:copyright: (c) 2023 by Maravegias Konstantinos.
:license: MIT, see LICENSE for more details.
"""
import pandas as pd
from typing import List, Tuple, Literal


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

    def outlier_summary(self, remover: Literal['std', 'iqr', 'percentiles'],
                        std_n: float = 3.0, factor: float = 1.5,
                        lower_thresh: float = 0.0, upper_thresh: float = 0.0
                        ) -> None:
        """Print an outlier summary.

        Args:
            remover (Literal[&#39;std&#39;, &#39;iqr&#39;, &#39;percentiles&#39;]): 
            _description_
            std_n (float, optional): _description_. Defaults to 3.0.
            factor (float, optional): _description_. Defaults to 1.5.
            lower_thresh (float, optional): _description_. Defaults to 0.0.
            upper_thresh (float, optional): _description_. Defaults to 0.0.
        """
        if remover == 'std':
            lower_limit, upper_limit = self.outlier_detector_std(std_n)
        if remover == 'iqr':
            lower_limit, upper_limit = self.outlier_detector_iqr(factor)
        if remover == 'percentiles':
            lower_limit, upper_limit = self.outlier_detector_percentiles(
                                                                lower_thresh,
                                                                upper_thresh)

        outliers_upper = self._obj[self._obj > upper_limit].notnull().sum()
        outliers_lower = self._obj[self._obj < lower_limit].notnull().sum()
        non_outliers = self._obj[(self._obj <= upper_limit)
                                & (self._obj >= lower_limit)
                                ].notnull().sum()
        print('Identification method: {}'.format(remover))
        print('---')
        print('>>> Identified outliers: {:,}'.format(outliers_upper.sum() +
                                                     outliers_lower.sum()
                                                     )
              )
        print('>>> Non-outliers: {:,}'.format(non_outliers.sum()))
        print('>>> Outlier%: {:.2%}'.format((outliers_upper.sum() + 
                                             outliers_lower.sum()
                                             ) / self._obj.notnull().sum().sum()
                                            )
              )

    def outlier_detector_std(self, std_n: float = 3.0):
        data_mean, data_std = self._obj.mean(), self._obj.std()
        cut_off = data_std * std_n
        lower_limit = data_mean - cut_off
        upper_limit = data_mean + cut_off
        return lower_limit, upper_limit

    def outlier_detector_iqr(self, factor: float = 1.5):
        q25 = self._obj.quantile(0.25, numeric_only=True)
        q75 = self._obj.quantile(0.75, numeric_only=True)
        iqr = q75 - q25
        cut_off = iqr * factor
        lower_limit = q25 - cut_off
        upper_limit = q75 + cut_off
        return lower_limit, upper_limit

    def outlier_detector_percentiles(self, lower_thresh, upper_thresh):
        lower_limit = self._obj.quantile(lower_thresh, numeric_only=True)
        upper_limit = self._obj.quantile(upper_thresh, numeric_only=True)
        return lower_limit, upper_limit


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
