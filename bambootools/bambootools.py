"""
:copyright: (c) 2023 by Maravegias Konstantinos.
:license: MIT, see LICENSE for more details.
"""
from itertools import combinations
from typing import List, Tuple, Literal

import pandas as pd
import numpy as np


def _conditional_probability(
    data: pd.DataFrame, col_a: str, col_b: str
) -> Tuple[float, float]:
    """
    Calculates the probability of a column's value being NULL given the
    fact another's column value is NULL (conditional probability).

    Parameters
    ----------
    _df : DataFrame
        The dataset holding the column to examine.
    col_a : str
        Column name of the first column.
    col_b : str
        Column name of the second column.

    Returns
    -------
    Tuple[float, float]
        The conditional probabilities for the values of `col_a` and `col_b`
        being NULL respectively.
    """

    col_a_mask = data[col_a].isna()
    col_b_mask = data[col_b].isna()
    both_na = (col_a_mask & col_b_mask).sum()
    p_a_being_na = col_a_mask.sum() / data.shape[0]
    p_b_being_na = col_b_mask.sum() / data.shape[0]
    p_both_being_na = both_na / data.shape[0]
    # check for for possible division with zero warning
    # column b has no NaN
    if p_b_being_na == 0:
        p_a_conditional = np.nan
    else:
        p_a_conditional = p_both_being_na / p_b_being_na
    # column a has no NaN
    if p_a_being_na == 0:
        p_b_conditional = np.nan
    else:
        p_b_conditional = p_both_being_na / p_a_being_na
    return p_a_conditional, p_b_conditional


def _hash_table(df: pd.DataFrame, subset: List[str] = None) -> pd.Series:
    """Returns a data hash of the given DataFrame, excluding the index.

    Parameters
    ----------
    df : DataFrame
        The dataframe to be hashed
    subset :  list of a column label or sequence of labels, optinal
        The subset of columns to generate the hash series.

    Returns
    -------
    Series of uint64
        Series with same length as the object.

    Raises
    ------
    AttributeError
        Raises error if `subset` is not a list.
    """

    if not subset:
        subset = df.columns.to_list()
    else:
        if not isinstance(subset, List):
            raise AttributeError("`subset` arg must be a list of strings")

    hashed_series = pd.util.hash_pandas_object(df[subset], index=False)
    return hashed_series


@pd.api.extensions.register_dataframe_accessor("bbt")
class BambooToolsDfAccessor:
    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @property
    def pandas_obj(self) -> pd.DataFrame:
        return self._obj

    @staticmethod
    def _validate(obj: pd.DataFrame) -> None:
        """
        Validates if the passed object is a pandas Dataframe

        Parameters
        ----------
        obj : Dataframe

        Raises
        ------
        AttributeError
            Raises error if the object is not a pandas DataFrame.
        """

        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a pandas DataFrame")

    def completeness(self, by: List[str] = None) -> pd.DataFrame:
        """
        Returns the completeness table of a dataframe. The returned columns
        are:
            * `complete values`: the number non NULL values per column.
            * `completeness ratio`: the ratio of `complete values` to the
              total number of records (column's length).
            * `total`: the total number of records.

        Parameters
        ----------
        by : list of columns, default None
            The list of column to aggragate. Produces the completeness metrics
            per the groups specified on `by`.

        Returns
        -------
        DataFrame
            The completeness table.

        Raises
        ------
        AttributeError
            Raises error in case of wrong input for `by` argument.
        """

        if by is None:
            complete_values = self._obj.notnull()
            total_records = self._obj.shape[0]
            counts = complete_values.sum()
            counts.name = "complete values"
            perce = complete_values.sum() / total_records
            perce.name = "completeness ratio"
            output = counts.to_frame().join(perce)
            output["total"] = total_records
        else:
            if not isinstance(by, List):
                raise AttributeError("`by` arg must be a list of strings")

            output = self._obj.groupby(by, dropna=False).agg(
                [
                    ("complete values", lambda x: x.notnull().sum()),
                    (
                        "completeness ratio",
                        lambda x: x.notnull().sum() / x.shape[0],
                    ),
                    ("total", "size"),
                ]
            )
        return output

    def missing_corr_matrix(self) -> pd.DataFrame:
        """
        Returns the missing correlations matrix. Calculates the conditional
        probability of a record's value being NULL at a specific colunm given
        the fact, another's column value is missing for the same record.

        A missing correlation matrix is a table, which states for every column
        the above mentioned contidional probability. In more details, if
        examining the conditional probability of a record being NULL at column
        A compared to another column B, it is the ratio between the probability
        of the values of the same record being NULL in both the A and B columns
        (intersection), and the probability of a record being NULL at column B.

        `P(A is NULL | B is NULL) = P(A is NULL & B is NULL) / P(B is NULL)`

        Returns
        -------
        DataFrame
            Returns an `n x n` matrix, with `n` equals the number of
            the initial dataframe's columns.
        """

        _df = self._obj
        columns_pairs_comb = list(combinations(_df.columns, 2))
        pairs_dict = {}
        for col_a, col_b in columns_pairs_comb:
            # calculate the conditional probabilities for each columns
            # pair
            result = _conditional_probability(
                data=_df, col_a=col_a, col_b=col_b
            )
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

    def duplication_summary(self, subset: List[str] = None) -> pd.DataFrame:
        """
        Generates a duplication summary table. Calculates the number and
        percentage of duplicate rows.

        Summary table explained:

            * `total records`: Refers to the total row of the dataset.
            * `unique records`: Refers to the number of the unique records
            of the dataset.
            * `unique records without duplications`: Refers to the number of
            unique records which have no duplications.
            * `unique records with duplications`: Refers to the number of
            unique records which have duplications.
            * `total duplicated records`: Referns to the number of the total
            duplicated records.

        Parameters
        ----------
        subset : list of a column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates, by
            default use all of the columns.

        Returns
        -------
        DataFrame
            The duplication summary table.
        """

        _df = self.pandas_obj.copy()
        hashed_series = _hash_table(_df, subset)
        del _df

        total_records = len(hashed_series)
        n_unique_records = hashed_series.nunique()
        n_total_duplicate_records = hashed_series.duplicated(keep=False).sum()
        frequency_of_records = hashed_series.value_counts()
        n_unique_duplicate_records = (frequency_of_records > 1).sum()
        n_unique_non_duplicated_records = (frequency_of_records == 1).sum()

        output = pd.DataFrame(
            index=[
                "total records",
                "unique records",
                "unique records without duplications",
                "unique records with duplications",
                "total duplicated records",
            ],
            columns=["counts"],
            data=[
                total_records,
                n_unique_records,
                n_unique_non_duplicated_records,
                n_unique_duplicate_records,
                n_total_duplicate_records,
            ],
        )

        return output

    def duplication_frequency_table(
        self, subset: List[str] = None
    ) -> pd.DataFrame:
        """
        Generates a table which states the frequency of records with
        duplications. Categorizes the duplicated records according to their
        number of duplications, and reports the frequency of those categories.

        E.g.: if a record has 1 identical record (so 2 in including itself)
        it is classed in the `2` category (`n identical bins` column.) If there
        are 10 of those pairs, the frequency is `10` and they account for
        20 duplications (`sum of duplications` equals 20).

        Frequency table explained:

            * `n identical bins`: States the category of the duplicated
            records. `2` accounts for pairs of duplicates (two indentical
            records), `3` for triples, etc.
            * `frequency`: The frequency of the `d bins`.
            * `sum of duplications`: States for how many duplicated records
            those categories generate.
            * `percentage to total duplications`: Is ratio between the
            * `sum of duplications` and to total number of duplicated values.

        Parameters
        ----------
        subset : list of a column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates, by
            default use all of the columns.

        Returns
        -------
        DataFrame
            A table with metrics regarding the duplicated columns.
        """

        _df = self.pandas_obj.copy()
        hashed_series = _hash_table(_df, subset)
        del _df

        frquency_table = (
            hashed_series.value_counts()
            .value_counts()
            .sort_index()
            .to_frame(name="frequency")
        )
        frquency_table["sum of duplications"] = (
            frquency_table["frequency"] * frquency_table.index
        )
        frquency_table["n identical bins"] = pd.cut(
            frquency_table.index,
            bins=[2, 3, 4, 5, 6, 10, 15, 50, np.inf],
            include_lowest=True,
            labels=[
                "2",
                "3",
                "4",
                "5",
                "[6, 10)",
                "[10, 15)",
                "[15, 50)",
                "50>",
            ],
            right=False,
        )
        frquency_table.dropna(subset=["n identical bins"], inplace=True)

        output = frquency_table.groupby(
            ["n identical bins"], observed=False
        ).sum()
        output["percentage to total duplications"] = (
            output["sum of duplications"] / output["sum of duplications"].sum()
        )
        return output

    def outlier_bounds(
        self,
        method: Literal["std", "iqr", "percentiles"],
        std_n: float = 3.0,
        factor: float = 1.5,
        lower_thresh: float = 0.0,
        upper_thresh: float = 1.0,
        by: List["str"] = None,
    ) -> pd.DataFrame:
        """
        Returns the outlier boundaries of the given dataframe for every
        numerical column. Outlier boundaries are defined as the values for
        which any other value above or below is considered an outlier.

        Parameters
        ----------
        method : {'std', 'iqr', 'percentiles'}
            Determines which outlier method to be used.

            * `std`: Calculates the mean and standard deviation. A value
            becomes an outlier if exceeds the mean for more than `std_n`
            standard deviations. Use if you assume that your data are
            normally distributed.
            * `iqr`: Calculates the IQR, then considers as an outlier
            every value being `factor`*IQR below or upper the 25%, 75%
            percentiles respectively.
            * `percentiles`: Detects as outliers, every value being below
            or upper the given percentiles.

        std_n : float, default 3.0
            The number of standard deviations to be used in order to determine
            if a value is an outlier. Used in the `std` method.
        factor : float, default 1.5
            The factor of IQR to be used in order to determine if a value is
            and outlier. Used in the `iqr` method.
        lower_thresh : float, default 0.0
            The lower percentile threshold to be used in the `percentiles`
            method.
        upper_thresh : float, default 1.0
            The upper percentile threshold to be used in the `percentiles`
            method.
        by : list of columns, default None
            The list of column to aggragate. Produces the outlier bounds
            metrics per the groups specified on `by`.

        Returns
        -------
        DataFrame
            The dataframe table with the boundaries per column.
        """

        _df = self.pandas_obj.copy()
        # check if to group by per with any categorical column
        if by:
            # group by per categorical column
            if not isinstance(by, list):
                raise AttributeError("`by` arg must be a list of strings")
            cols = _df.select_dtypes(exclude=["category", "object"]).columns
            _df = _df.groupby(by)[cols]
            # select and call outlier method:
            if method == "std":
                bounds = _df.apply(
                    lambda group: group.apply(
                        self._outlier_detector_std, std_n=std_n
                    )
                ).unstack()
            if method == "iqr":
                bounds = _df.apply(
                    lambda group: group.apply(
                        self._outlier_detector_iqr, factor=factor
                    )
                ).unstack()
            if method == "percentiles":
                bounds = _df.apply(
                    lambda group: group.apply(
                        self._outlier_detector_percentiles,
                        lower_thresh=lower_thresh,
                        upper_thresh=upper_thresh,
                    )
                ).unstack()
            return bounds

        else:
            # do not group by per any categorical column
            # select and call outlier method:
            _df = _df.select_dtypes(exclude=["category", "object"])
            if method == "std":
                bounds = self._outlier_detector_std(_df, std_n)
            if method == "iqr":
                bounds = self._outlier_detector_iqr(_df, factor)
            if method == "percentiles":
                bounds = self._outlier_detector_percentiles(
                    _df, lower_thresh, upper_thresh
                )
            return pd.DataFrame(
                {"lower": bounds["lower"], "upper": bounds["upper"]}
            )

    def outlier_summary(
        self,
        method: Literal["std", "iqr", "percentiles"],
        std_n: float = 3.0,
        factor: float = 1.5,
        lower_thresh: float = 0.0,
        upper_thresh: float = 1.0,
        by: List = None,
    ) -> pd.DataFrame:
        """
        Generates an outlier summary table. The outlier summary table produces
        metrics regarding the outliers values of the dataset. It uses different
        methods to define a values as an outlier.

        Outlier summary table explained:

            * n_outliers_upper: The number of outliers existing above the upper
            limit.
            * n_outliers_lower: The number of outliers existing below the lower
            limit.
            * n_non_outliers: The number of non outlier values.
            * total_records: The total records of the dataset.

        Parameters
        ----------
        method : {'std', 'iqr', 'percentiles'}
            Determines which outlier method to be used.

            * `std`: Calculates the mean and standard deviation. A value
            becomes an outlier if exceeds the mean for more than `std_n`
            standard deviations. Use if you assume that your data are
            normally distributed.
            * `iqr`: Calculates the IQR, then considers as an outlier
            every value being `factor`*IQR below or upper the 25%, 75%
            percentiles respectively.
            * `percentiles`: Detects as outliers, every value being below
            or upper the given percentiles.

        std_n : float, default 3.0
            The number of standard deviations to be used in order to determine
            if a value is an outlier. Used in the `std` method.
        factor : float, default 1.5
            The factor of IQR to be used in order to determine if a value is
            and outlier. Used in the `iqr` method.
        lower_thresh : float, default 0.0
            The lower percentile threshold to be used in the `percentiles`
            method.
        upper_thresh : float, default 1.0
            The upper percentile threshold to be used in the `percentiles`
            method.
        by : list of columns, default None
            The list of column to aggragate. Produces the outlier bounds
            metrics per the groups specified on `by`.

        Returns
        -------
        DataFrame
            The outlier summary dataframe.
        """

        bounds = self.outlier_bounds(
            method, std_n, factor, lower_thresh, upper_thresh, by
        )
        outlier_counts = {}
        _df = self.pandas_obj.copy()
        cols = _df.select_dtypes(exclude=["category", "object"]).columns
        if by:
            # --> groupby summary table (group)
            for group in bounds.index:
                for col in cols:
                    lower_bound = bounds.loc[group, (col, "lower")]
                    upper_bound = bounds.loc[group, (col, "upper")]
                    # Count the outliers below/above the lower/upper bounds
                    lower_outliers = _df[
                        (
                            (_df[by] == group).all(axis=1)
                            & (_df[col] < lower_bound)
                        )
                    ]
                    upper_outliers = _df[
                        (
                            (_df[by] == group).all(axis=1)
                            & (_df[col] > upper_bound)
                        )
                    ]
                    # count the non outliers
                    non_outliers = _df[
                        (
                            (_df[by] == group).all(axis=1)
                            & (_df[col] <= upper_bound)
                            & (_df[col] >= lower_bound)
                        )
                    ]
                    # Store the counts in the dictionary
                    outlier_counts[(group, col, "n_outliers_lower")] = len(
                        lower_outliers
                    )
                    outlier_counts[(group, col, "n_outliers_upper")] = len(
                        upper_outliers
                    )
                    outlier_counts[(group, col, "n_non_outliers")] = len(
                        non_outliers
                    )
            # generate the summary table
            summary_tbl = pd.Series(outlier_counts).unstack()
        else:
            # --> non groupby summary table
            _df = _df.select_dtypes(exclude=["category", "object"])
            lower_outliers = _df.lt(bounds["lower"], axis=1).sum()
            upper_outliers = _df.gt(bounds["upper"], axis=1).sum()
            # count the non outliers
            non_outliers = (
                _df.le(bounds["upper"], axis=1)
                & _df.ge(bounds["lower"], axis=1)
            ).sum()
            # concat all the series into one dataframe
            summary_tbl = pd.concat(
                [upper_outliers, lower_outliers, non_outliers], axis=1
            ).rename(
                columns={
                    0: "n_outliers_upper",
                    1: "n_outliers_lower",
                    2: "n_non_outliers",
                }
            )
        # add additional columns in the summary table
        qry_total_outliers = "n_outliers_upper + n_outliers_lower"
        summary_tbl["n_total_outliers"] = summary_tbl.eval(qry_total_outliers)
        qry_total_records = "n_non_outliers + n_total_outliers"
        summary_tbl["total_records"] = summary_tbl.eval(qry_total_records)
        return summary_tbl

    def _outlier_detector_std(
        self, _df: pd.DataFrame = None, std_n: float = 3.0
    ) -> pd.Series:
        """
        Returns the upper and lower boundaries which are used to class a
        value as an outlier. The boundaries are caclulated as `std_n` times
        away from the mean.

        Note: It is not suitable for data which do not follow normal
        distribution.

        Parameters
        ----------
        _df : DataFrame, default None
            If called iternally it is passed as an argument.
        std_n : float, default 3.0
            The number of standard deviations to be used in the `std` method.

        Returns
        -------
        Series
            The lower and upper outlier boundaries.
        """

        if _df is None:
            _df = self._obj
            _df = _df.select_dtypes(exclude=["category", "object"])
        data_mean = _df.mean(numeric_only=True)
        data_std = _df.std(numeric_only=True)
        cut_off = data_std * std_n
        lower_bound = data_mean - cut_off
        upper_bound = data_mean + cut_off
        return pd.Series({"lower": lower_bound, "upper": upper_bound})

    def _outlier_detector_iqr(
        self, _df: pd.DataFrame = None, factor: float = 1.5
    ) -> pd.Series:
        """
        Returns the upper and lower boundaries which are used to class a
        value as an outlier. The boundaries are defined as the values, which
        are `factor` times the IQR below and above the Q1 and Q3 quartiles
        respectively.

        Parameters
        ----------
        _df : DataFrame, default None
            If called iternally it is passed as an argument.
        factor : float, default 1.5
            The multiplayer of the IQR.

        Returns
        -------
        Series
            The lower and upper outlier boundaries.
        """

        if _df is None:
            _df = self._obj
            _df = _df.select_dtypes(exclude=["category", "object"])
        q25 = _df.quantile(0.25)
        q75 = _df.quantile(0.75)
        iqr = q75 - q25
        cut_off = iqr * factor
        lower_bound = q25 - cut_off
        upper_bound = q75 + cut_off
        return pd.Series({"lower": lower_bound, "upper": upper_bound})

    def _outlier_detector_percentiles(
        self,
        _df: pd.DataFrame = None,
        lower_thresh: float = 0.0,
        upper_thresh: float = 1.0,
    ) -> pd.Series:
        """
        Returns the upper and lower boundaries which are used to class a
        value as an outlier. The boundaries are defined as the values
        at the given `lower_thesh` and `upper_thresh` percentiles.

        Parameters
        ----------
        _df : DataFrame, default None
            If called iternally it is passed as an argument.
        lower_thresh : float, between [0, 1], default 0.0
            The percentile to compute for the lower boundary value.
        upper_thresh : float, between [0, 1], default 0.0
            The percentile to compute for the upper boundary value.

        Returns
        -------
        Series
            The lower and upper outlier boundaries.
        """

        if _df is None:
            _df = self._obj
            _df = _df.select_dtypes(exclude=["category", "object"])
        if lower_thresh < 0 | lower_thresh > 1:
            raise ValueError("Lower threshold should be within [0, 1]")
        if upper_thresh < 0 | upper_thresh > 1:
            raise ValueError("Upper threshold should be within [0, 1]")

        lower_bound = _df.quantile(lower_thresh)
        upper_bound = _df.quantile(upper_thresh)
        return pd.Series({"lower": lower_bound, "upper": upper_bound})


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
        """
        Validates if the passed object is a pandas Series

        Parameters
        ----------
        obj : Series

        Raises
        ------
        AttributeError
            Raises error if the object is not a pandas Series.
        """

        if not isinstance(obj, pd.Series):
            raise AttributeError("Must be a pandas Series")

    def above(
        self, thresh: float, dropna: bool = False
    ) -> Tuple[float, float]:
        """
        Calculates the number of values and their percentage which are
        above a specific threshold.

        Parameters
        ----------
        thresh : float
            The threshold given by the user.
        dropna : bool, default False
             If True drops the NULL records before the calculation of
             percentage of NULL values. Hence the total number of records
             equal to the number of non NULL values.

        Returns
        -------
        Tuple[float, float]
            Counts and percentage of records above the given threshold.
        """

        if dropna:
            values_above = self._obj > thresh
        else:
            values_above = self._obj.dropna() > thresh
        count = values_above.sum()
        perc = values_above.sum() / len(values_above)

        return count, perc

    def below(
        self, thresh: float, dropna: bool = False
    ) -> Tuple[float, float]:
        """
        Calculates the number of values and their percentage which are
        beloq a specific threshold.

        Parameters
        ----------
        thresh : float
            The threshold given by the user.
        dropna : bool, default False
             If True drops the NULL records before the calculation of
             percentage of NULL values. Hence the total number of records
             equal to the number of non NULL values.

        Returns
        -------
        Tuple[float, float]
            Counts and percentage of records below the given threshold.
        """

        if dropna:
            values_above = self._obj < thresh
        else:
            values_above = self._obj.dropna() < thresh
        count = values_above.sum()
        perc = values_above.sum() / len(values_above)

        return count, perc
