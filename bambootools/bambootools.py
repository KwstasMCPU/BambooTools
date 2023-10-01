"""
:copyright: (c) 2023 by Maravegias Konstantinos.
:license: MIT, see LICENSE for more details.
"""
import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Tuple, Literal


@pd.api.extensions.register_dataframe_accessor("bbt")
class BambooToolsDfAccessor:
    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @property
    def pandas_obj(self) -> pd.DataFrame:
        return self._obj

    @staticmethod
    def _validate(obj) -> None:
        """Validate that the passed object is a pandas Dataframe

        Args:
            * obj: Object passed

        Raises:
            AttributeError: Prompts the user that pd.DataFrame should be used
        """
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a pandas DataFrame")

    def completeness(self, by: List[str] = None,
                     format: bool = False) -> pd.DataFrame:
        """Returns the completeness table of a dataframe. The returned columns
        are: `count`, `perc` and `total`.
            `count`: the non NULL values
            `perc`: the ratio of total records to the `count`
            `total`: the total records

        Args:
            * by (List[str], optional): List of columns to aggregate. If
                passed the completeness per column is measured per group.
                Defaults to None.
            * format (bool, optional): If True the perc column is formated
                to a percentage (eg 50.00%). Note that it returns a pandas
                Styler object instead of a DataFrame.

        Raises:
            AttributeError: Checks if by is a list of string

        Returns:
            pd.DataFrame: The completeness dataframe
        """
        if by is None:
            by = self._obj.columns.to_list()
            counts = self._obj.groupby(by, dropna=False).\
                apply(lambda x: x.notnull().sum()).sum()

            perc = self._obj.groupby(by, dropna=False).\
                apply(lambda x: x.notnull().sum()).sum() / self._obj.shape[0]

            _df = pd.concat([counts, perc], axis=1).\
                rename(columns={0: 'count',
                                1: 'perc'
                                }
                       )
            _df['total'] = self._obj.shape[0]

            if format:
                _df = _df.style.format({'perc': '{:.2%}'})
        else:
            if not isinstance(by, List):
                raise AttributeError("`by` arg must be a list of strings")

            _df = self._obj.groupby(by, dropna=False).\
                agg([('count', lambda x: x.notnull().sum()),
                     ('perc', lambda x: x.notnull().sum()/x.shape[0]),
                     ('total', 'size')
                     ]
                    )
            if format:
                format_dict = dict([(t, '{:.2%}') for t in
                                    _df.columns if t[1] == 'perc'])
                _df = _df.style.format(format_dict)
        return _df

    def missing_corr_matrix(self) -> pd.DataFrame:
        """Returns a missing correlations matrix. Calculates the conditional
        probability of a column being NULL given the fact another column is
        NULL.

        A missing correlation matrix is a table, which states for every column
        the probability of its values being NULL given the fact another's
        column values are NULL. In more details, if comparing columns A and B,
        it is the ratio between the number of records where both columns are
        NULL, and the total number of NULL values in column B.

        Note that the ratio between A and B is not equal to the ratio between
        B and A.

        `P(A is NULL | B is NULL) = P(A is NULL & B is NULL) / P(B is NULL)`

        Returns:
            pd.DataFrame: Returns an n x n matrix, with n equals the number of
                the initial dataframe's columns.
        """
        _df = self._obj
        columns_pairs_comb = list(combinations(_df.columns, 2))
        pairs_dict = {}
        for col_a, col_b in columns_pairs_comb:
            # calculate the conditional probabilities for each columns
            # pair
            result = self._conditional_probability(_df=_df,
                                                   col_a=col_a,
                                                   col_b=col_b)
            if col_a in pairs_dict:
                pairs_dict[col_a].update({col_b: result[0]})
            else:
                pairs_dict[col_a] = {col_b: result[0]}
            if col_b in pairs_dict:
                pairs_dict[col_b].update({col_a: result[1]})
            else:
                pairs_dict[col_b] = {col_a: result[1]}

        matrix = pd.DataFrame(pairs_dict)
        return matrix.reindex(matrix.columns)

    def outlier_bounds(self, method: Literal['std', 'iqr', 'percentiles'],
                       std_n: float = 3.0, factor: float = 1.5,
                       lower_thresh: float = 0.0, upper_thresh: float = 1.0,
                       by: List = None
                       ) -> pd.DataFrame:
        """Returns the outlier boundaries of the given dataframe for every
        numerical column. Outlier boundaries are defined as the values for
        which any other value above or below is considered as an outlier.

        Args:
            * method ({'std', 'iqr', 'percentiles'}): Which outlier method
                to be used.
                - 'std': Calculates the mean and standard deviation. A value
                becomes an outlier if exceeds the mean more than `std_n`
                standard deviations. Use if you assume that your data are
                normally distributed.
                - 'iqr': Calculates the IQR, then considers as an outlier
                every value being `factor`*IQR below or upper the 25%, 75%
                percentiles respectively.
                - 'percentiles': Detects as outliers, every value being below
                or upper the given percentiles.
            * std_n (float, optional): The number of standard deviations to be
                used in the `std` method. Defaults to 3.0.
            * factor (float, optional): The factor of IQR to be used in the
                `iqr` method. Defaults to 1.5.
            * lower_thresh (float, optional): The lower percentile threshold
                to be used in the `percentiles` method. Defaults to 0.0.
            * upper_thresh (float, optional): The upper percentile threshold
                to be used in the `percentiles` method. Defaults to 1.0.
            * by (List[str], optional): If the names of categorical columns are
                given, then the boundaries are calculated per group.

        Returns:
            pd.DataFrame: A table of the lower and upper boundary values
        """
        _df = self.pandas_obj.copy()
        # check if to group by per with any categorical column
        if by:
            # group by per categorical column
            if not isinstance(by, list):
                raise AttributeError("`by` arg must be a list of strings")
            cols = _df.select_dtypes(exclude=['category', 'object']).columns
            _df = _df.groupby(by)[cols]
            # select and call outlier method:
            if method == 'std':
                bounds = _df.apply(lambda group: group.apply(
                    self._outlier_detector_std, std_n=std_n)).unstack()
            if method == 'iqr':
                bounds = _df.apply(lambda group: group.apply(
                    self._outlier_detector_iqr, factor=factor)).unstack()
            if method == 'percentiles':
                bounds = _df.apply(lambda group: group.apply(
                    self._outlier_detector_percentiles,
                    lower_thresh=lower_thresh,
                    upper_thresh=upper_thresh
                    )).unstack()
            return bounds

        else:
            # do not group by per any categorical column
            # select and call outlier method:
            _df = _df.select_dtypes(exclude=['category', 'object'])
            if method == 'std':
                bounds = self._outlier_detector_std(_df, std_n)
            if method == 'iqr':
                bounds = self._outlier_detector_iqr(_df, factor)
            if method == 'percentiles':
                bounds = self._outlier_detector_percentiles(_df,
                                                            lower_thresh,
                                                            upper_thresh
                                                            )
            return pd.DataFrame({'lower': bounds['lower'],
                                 'upper': bounds['upper']
                                 }
                                )

    def outlier_summary(self, method: Literal['std', 'iqr', 'percentiles'],
                        std_n: float = 3.0, factor: float = 1.5,
                        lower_thresh: float = 0.0, upper_thresh: float = 1.0,
                        by: List = None
                        ) -> pd.DataFrame:
        """Returns an outlier summary table (counts of outliers) for upper and
        lower bounds. Utilises existing functions.

        Args:
            * method ({'std', 'iqr', 'percentiles'}): Which outlier method
                to be used.
                - 'std': Calculates the mean and standard deviation. A value
                becomes an outlier if exceeds the mean more than `std_n`
                standard deviations. Use if you assume that your data are
                normally distributed.
                - 'iqr': Calculates the IQR, then considers as an outlier
                every value being `factor`*IQR below or upper the 25%, 75%
                percentiles respectively.
                - 'percentiles': Detects as outliers, every value being below
                or upper the given percentiles.
            * std_n (float, optional): The number of standard deviations to be
                used in the `std` method. Defaults to 3.0.
            * factor (float, optional): The factor of IQR to be used in the
                `iqr` method. Defaults to 1.5.
            * lower_thresh (float, optional): The lower percentile threshold
                to be used in the `percentiles` method. Defaults to 0.0.
            * upper_thresh (float, optional): The upper percentile threshold
                to be used in the `percentiles` method. Defaults to 1.0.
            * by (List[str], optional): If the names of categorical columns are
                given, then the boundaries are calculated per group.

        Returns:
            pd.DataFrame: The outlier summary table
        """
        bounds = self.outlier_bounds(method, std_n, factor,
                                     lower_thresh, upper_thresh, by)
        outlier_counts = {}
        _df = self.pandas_obj.copy()
        cols = _df.select_dtypes(exclude=['category', 'object']).columns
        if by:
            # --> groupby summary table (group)
            for group in bounds.index:
                for col in cols:
                    lower_bound = bounds.loc[group, (col, 'lower')]
                    upper_bound = bounds.loc[group, (col, 'upper')]
                    # Count the outliers below/above the lower/upper bounds
                    lower_outliers = _df[((_df[by] == group).all(axis=1)
                                          & (_df[col] < lower_bound))]
                    upper_outliers = _df[((_df[by] == group).all(axis=1)
                                          & (_df[col] > upper_bound))]
                    # count the non outliers
                    non_outliers = _df[((_df[by] == group).all(axis=1)
                                        & (_df[col] <= upper_bound)
                                        & (_df[col] >= lower_bound))]
                    # Store the counts in the dictionary
                    outlier_counts[(group,
                                    col,
                                    'n_outliers_lower')] = len(lower_outliers)
                    outlier_counts[(group,
                                    col,
                                    'n_outliers_upper')] = len(upper_outliers)
                    outlier_counts[(group,
                                    col,
                                    'n_non_outliers')] = len(non_outliers)
            # generate the summary table
            summary_tbl = pd.Series(outlier_counts).unstack()
        else:
            # --> non groupby summary table
            _df = _df.select_dtypes(exclude=['category', 'object'])
            lower_outliers = _df.lt(bounds['lower'], axis=1).sum()
            upper_outliers = _df.gt(bounds['upper'], axis=1).sum()
            # count the non outliers
            non_outliers = (_df.le(bounds['upper'], axis=1)
                            & _df.ge(bounds['lower'], axis=1)).sum()
            # concat all the series into one dataframe
            summary_tbl = pd.concat([upper_outliers,
                                     lower_outliers,
                                     non_outliers], axis=1
                                    ).rename(columns={
                                        0: 'n_outliers_upper',
                                        1: 'n_outliers_lower',
                                        2: 'n_non_outliers'
                                        }
                                             )
        # add additional columns in the summary table
        qry_total_outliers = 'n_outliers_upper + n_outliers_lower'
        summary_tbl['n_total_outliers'] = summary_tbl.eval(qry_total_outliers)
        qry_total_records = 'n_non_outliers + n_total_outliers'
        summary_tbl['total_records'] = summary_tbl.eval(qry_total_records)
        return summary_tbl

    def _outlier_detector_std(self,
                              _df: pd.DataFrame = None,
                              std_n: float = 3.0
                              ) -> pd.Series:
        """Returns the upper and lower boundaries which are used to class a
        value as an outlier. The boundaries are caclulated as `std_n` times
        away from the mean.
        Note: Not suitable for data which do not follow normal distribution.

        Args:
            * _df (pd.DataFrame, optional): If called iternally it is passed as
                an argument. Defaults to None.
            * std_n (float, optional): The number of standard deviations to be
                used in the `std` method. Defaults to 3.0.

        Returns:
            pd.Series: The lower and upper bound
        """
        if _df is None:
            _df = self._obj
            _df = _df.select_dtypes(exclude=['category', 'object'])
        data_mean = _df.mean(numeric_only=True)
        data_std = _df.std(numeric_only=True)
        cut_off = data_std * std_n
        lower_bound = data_mean - cut_off
        upper_bound = data_mean + cut_off
        return pd.Series({'lower': lower_bound, 'upper': upper_bound})

    def _outlier_detector_iqr(self,
                              _df: pd.DataFrame = None,
                              factor: float = 1.5
                              ) -> pd.Series:
        """Returns the upper and lower boundaries which are used to class a
        value as an outlier. The boundaries are defined as the points, which
        are `factor` times the IQR below and above the Q1 and Q3 quartiles
        respectively.

        Args:
            * _df (pd.DataFrame, optional): If called iternally it is passed as
                an argument. Defaults to None.
            * std_n (float, factor): The multiplayer of the IQR.
                Defaults to 1.5.

        Returns:
            pd.Series: The lower and upper bound
        """
        if _df is None:
            _df = self._obj
            _df = _df.select_dtypes(exclude=['category', 'object'])
        q25 = _df.quantile(0.25)
        q75 = _df.quantile(0.75)
        iqr = q75 - q25
        cut_off = iqr * factor
        lower_bound = q25 - cut_off
        upper_bound = q75 + cut_off
        return pd.Series({'lower': lower_bound, 'upper': upper_bound})

    def _outlier_detector_percentiles(self,
                                      _df: pd.DataFrame = None,
                                      lower_thresh: float = 0.0,
                                      upper_thresh: float = 1.0
                                      ) -> pd.Series:
        """Returns the upper and lower boundaries which are used to class a
        value as an outlier. The boundaries are calculated as the returd values
        at the given `lower_thesh` and `upper_thresh` percentiles.

        Args:
            * _df (pd.DataFrame, optional): If called iternally it is passed as
                an argument. Defaults to None.
            * lower_thresh: Value between 0 <= q <= 1, the percentile to
                compute for the lower boundary value.
            * upper_thresh: Value between 0 <= q <= 1, the percentile to
                compute for the upper boundary value.

        Returns:
            pd.Series: The lower and upper bound
        """
        if _df is None:
            _df = self._obj
            _df = _df.select_dtypes(exclude=['category', 'object'])
        lower_bound = _df.quantile(lower_thresh)
        upper_bound = _df.quantile(upper_thresh)
        return pd.Series({'lower': lower_bound, 'upper': upper_bound})

    def _conditional_probability(self,
                                 _df: pd.DataFrame,
                                 col_a: str,
                                 col_b: str) -> Tuple[float, float]:
        """Calculates the probability of a column's value being NULL given the
        fact another's column value is NULL (conditional probability).

        Args:
            * _df (pd.DataFrame): The dataframe holding the both columns
            * col_a (str): Column name of the first column
            * col_b (str): Column name of the second column

        Returns:
            Tuple[float, float]: The conditional probabilities for a value of
                `col_a` and `col_b` being NULL respectively
        """
        col_a_mask = _df[col_a].isna()
        col_b_mask = _df[col_b].isna()
        both_na = (col_a_mask & col_b_mask).sum()
        p_a_being_na = col_a_mask.sum() / _df.shape[0]
        p_b_being_na = col_b_mask.sum() / _df.shape[0]
        p_both_being_na = both_na / _df.shape[0]
        # check for for possible division with zero warning
        msg = """`{}` has no NULL values.
        Conditional probability for `{}` set to NaN."""
        # column b has no NaN
        if p_b_being_na == 0:
            print(msg.format(col_b, col_a))
            p_a_conditional = np.nan
        else:
            p_a_conditional = p_both_being_na / p_b_being_na
        # column a has no NaN
        if p_a_being_na == 0:
            print(msg.format(col_a, col_b))
            p_b_conditional = np.nan
        else:
            p_b_conditional = p_both_being_na / p_a_being_na
        return p_a_conditional, p_b_conditional


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
            * obj: Object passed

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
            * thresh (float): The threshold given by the user.
            * dropna (bool, optional): If True drops the NULL records before
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
            * thresh (float): The threshold given by the user.
            * dropna (bool, optional): If True drops the NULL records before
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
