"""
This module provides a convenient feature construction.

Included classes support creation such properties as backward time lag for a few steps,
aggregating functions over grouped data and share of a target in different groups.
"""
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from datetime import datetime
from numpy import nan
from pandas import DataFrame, Series, Timestamp
from pandas import DatetimeIndex, PeriodIndex, MultiIndex
from sklearn.feature_selection import RFECV
from typing import Union, List, ClassVar

fb_dict = namedtuple('fb_dict', ['builder', 'params'])


class FeatureBasic:
    """
    An interface for each feature builder.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def build(self, df: DataFrame, idx=None, **kwargs) -> DataFrame:
        """
        Constructs the feature and add it to the given DataFrame.

        :param df: basic table to design a new property.
        :type df: DataFrame

        :param idx: index to insert a new property. A new column creates if none.
        :type idx: Timestamp

        :param kwargs: other arguments.
        :type kwargs: dict

        :return: incoming table with new or augmented column.
        :rtype: DataFrame
        """
        pass

    @classmethod
    def get_column_name(cls, **kwargs) -> str:
        """
        Global class method defines name of a future column. Built on arguments from a function "build".

        :param kwargs: other arguments.
        :type kwargs: dict

        :return: feature's name.
        :rtype: str
        """
        pass


class Lag(FeatureBasic):
    """
    Backward shift of the target column to some steps.
    """
    def build(self, df: DataFrame, idx=None,
              target_name: str = 'Count',
              target_group: str = 'Nomenclature_size',
              i: int = 1, **kwargs) -> DataFrame:
        """
        Shift target column to the i steps back.

        :param df: basic table to design a new property.
        :type df: DataFrame

        :param idx: index to insert a new property. A new column creates if none.
        :type idx: Timestamp

        :param target_name: name of shifting column.
        :type target_name: str

        :param target_group: column defines target multivariance.
        :type target_group: str

        :param i: number of steps.
        :type i: int

        :param kwargs: other arguments.
        :type kwargs: dict

        :return: incoming table with new or augmented column.
        :rtype: DataFrame
        """
        if target_group in df.columns:
            df.loc[idx:, self.get_column_name(target_name, i)] = df.groupby(target_group
                                                                            )[target_name].shift(i).loc[idx:]
        else:
            df.loc[idx:, self.get_column_name(target_name, i)] = df.loc[:idx, target_name].shift(i).loc[idx:]
        return df

    @classmethod
    def get_column_name(cls, target_name: str, i: int, **kwargs) -> str:
        """This method overrides :meth:`FeatureBasic.get_column_name`."""
        return "lag_{}_{}".format(i, target_name.lower())


class AvgGroup(FeatureBasic):
    """
    Get some grouped target statistic calculated in a window with specified length.
    """
    def build(self, df: DataFrame, idx=None,
              func: str = 'min',
              group_names: List[str] = None,
              target_name: str = 'Count',
              rolling_window: int = 3, **kwargs) -> DataFrame:
        """
        Groups target by group_names columns, then take one step back and count some statistic inside a
        specified window.

        :param df: basic table to design a new property.
        :type df: DataFrame

        :param idx: index to insert a new property. A new column creates if none.
        :type idx: Timestamp

        :param func: name of the aggregating function. Use pandas function from the 'groupby' block.
        :type func: str

        :param group_names: columns to group by.
        :type group_names: list

        :param target_name: name of column for calculating statistics
        :type target_name: str

        :param rolling_window: size of window to roll in
        :type rolling_window: int

        :param kwargs: other arguments.
        :type kwargs: dict

        :return: incoming table with new or augmented column.
        :rtype: DataFrame
        """
        if group_names is None:
            group_names = ['Nomenclature_size']
        rolled_df = df.loc[:idx, :].groupby(group_names
                                            )[target_name].transform(lambda x: x.shift(1).rolling(rolling_window
                                                                                                  ).aggregate(func))
        df.loc[idx:, self.get_column_name(func, group_names, rolling_window)] = rolled_df.loc[idx:]
        return df

    @classmethod
    def get_column_name(cls, func: str, group_names: list,
                        rolling_window: int, **kwargs) -> str:
        """This method overrides :meth:`FeatureBasic.get_column_name`."""
        return 'grp_{}_{}_{}'.format(rolling_window, str(func),
                                     ''.join([name[:2] for name in group_names]).lower())


class Avg(FeatureBasic):
    """
    Get some target statistic calculated in a window with specified length.
    """
    def build(self, df: DataFrame, idx=None,
              func: str = 'min',
              target_name: str = 'Count',
              rolling_window: int = 3, **kwargs) -> DataFrame:
        """
        Groups target by group_names columns, then take one step back and count some statistic inside a
        specified window.

        :param df: basic table to design a new property.
        :type df: DataFrame

        :param idx: index to insert a new property. A new column creates if none.
        :type idx: Timestamp

        :param func: name of the aggregating function. Use pandas function from the 'groupby' block.
        :type func: str

        :param target_name: name of column for calculating statistics
        :type target_name: str

        :param rolling_window: size of window to roll in
        :type rolling_window: int

        :param kwargs: other arguments.
        :type kwargs: dict

        :return: incoming table with new or augmented column.
        :rtype: DataFrame
        """
        right_par = df.loc[:idx, target_name].shift(1).rolling(rolling_window)
        df.loc[idx:, self.get_column_name(str(func), target_name,
                                          rolling_window)] = getattr(right_par, func)().loc[idx:]
        return df

    @classmethod
    def get_column_name(cls, func: str,
                        target_name: str,
                        rolling_window: int, **kwargs) -> str:
        """This method overrides :meth:`FeatureBasic.get_column_name`."""
        return 'avg_{}_{}_{}'.format(rolling_window, str(func), target_name[:2])


class DateIndex(FeatureBasic):
    """
    Extract various parts of date.
    """
    def build(self, df: DataFrame, idx=None, feature_name: str = 'year', **kwargs) -> DataFrame:
        """
        Take from DatetimeIndex or PeriodIndex different information about dates. Other indexes provide zeros.

        :param df: basic table to design a new property.
        :type df: DataFrame

        :param idx: index to insert a new property. A new column creates if none.
        :type idx: Timestamp

        :param feature_name: name of extracting part of date.
        :type feature_name: str

        :param kwargs: other arguments.
        :type kwargs: dict

        :return: incoming table with new or augmented column.
        :rtype: DataFrame
        """
        if isinstance(df.index, (DatetimeIndex, PeriodIndex)):
            df.loc[idx:, self.get_column_name(feature_name)] = getattr(df.loc[idx:, :].index, feature_name)
        else:
            df.loc[idx:, self.get_column_name(feature_name)] = 0
        return df

    @classmethod
    def get_column_name(cls, feature_name: str, **kwargs) -> str:
        """This method overrides :meth:`FeatureBasic.get_column_name`."""
        return feature_name


class Covid(FeatureBasic):
    """Flag of COVID-19"""
    def build(self, df: DataFrame, idx=None, **kwargs):
        """
        Set information about the start and, in perspective, the end of COVID-19.
        :param df: basic table to design a new property.
        :type df: DataFrame

        :param idx: index to insert a new property. A new column creates if none.
        :type idx: Timestamp

        :param kwargs: other arguments.
        :type kwargs: dict

        :return: incoming table with new or augmented column.
        :rtype: DataFrame
        """
        df[self.get_column_name()] = 0
        df.loc[df.index >= '2020-04-30', 'Covid'] = 1
        return df

    @classmethod
    def get_column_name(cls, **kwargs) -> str:
        """This method overrides :meth:`FeatureBasic.get_column_name`."""
        return 'Covid'


class Part(FeatureBasic):
    """
    Count multivariate target proportion in a certain group.
    """
    def build(self, df: DataFrame, idx=None,
              group_names: List[str] = None,
              col_name: str = 'new_col',
              date_threshold: Union[Timestamp, str] = None,
              target_group: str = 'Nomenclature_size',
              target_name: str = 'Count',
              **kwargs) -> DataFrame:
        """
        First step this function cuts data before defined threshold to avoid using incomplete records.
        Then groups multivariance target by dimensions from group_names and count share of each variance in
        target_group.

        :param df: basic table to design a new property.
        :type df: DataFrame

        :param idx: index to insert a new property. A new column creates if none.
        :type idx: Timestamp

        :param group_names: name of columns to group target by.
        :type group_names: List[str]

        :param col_name: new/updating column's name.
        :type col_name: str

        :param date_threshold: parameter defines right bound of time series using in calculation.
        If parameter contains date-like value, function use it as max date of series directly. String usage has
        two values: 'last_month' and any other. This option allows to use the first day of the month/year from
         last observation as right non-included bound.
        :type date_threshold: Union[Timestamp, str]

        :param target_group: column defines target multivariance.
        :type target_group: str

        :param kwargs: other arguments.
        :type kwargs: other arguments.

        :return: incoming table with new or augmented column.
        :rtype: DataFrame
        """
        if group_names is None:
            group_names = ['year']

        if isinstance(date_threshold, str):
            if date_threshold == 'last_month':
                date_threshold = datetime(df.index[-1].year, df.index[-1].month, 1)
            else:
                date_threshold = datetime(df.index[-1].year, 1, 1)
        elif date_threshold is None:
            date_threshold = datetime(df.index[-1].year, 1, 1)

        def count_part(x):
            """Calculate share of multivariate target."""
            return x.groupby([target_group])[target_name].sum() / x[target_name].sum()
        part_nom = df[(df.index < date_threshold)].groupby(group_names).apply(lambda x: count_part(x))

        if not isinstance(part_nom.index, MultiIndex) == 1:
            part_nom = [part_nom.loc[int(row[group_names[0]]) if isinstance(row[group_names[0]],
                                                                            float) else row[group_names[0]],
                                     int(row[target_group]) if isinstance(row[target_group], float)
                                     else row[target_group]] if int(row[group_names[0]]) in part_nom.index else nan
                        for _, row in df.loc[idx:, :].iterrows()]

        elif isinstance(part_nom, Series):
            part_nom = [part_nom.loc[tuple(row[group_names + [target_group]].tolist()  # .astype(int).tolist()
                                           )] if tuple(row[group_names + [target_group]].tolist()
                                                       ) in part_nom.index else nan
                        for _, row in df.loc[idx:, :].iterrows()]

        else:
            part_nom = [part_nom.loc[tuple(row[group_names].tolist()), int(row[target_group])
                                     if isinstance(row[target_group], float) else row[target_group]]
                        if tuple(row[group_names].tolist()) in part_nom.index else nan
                        for _, row in df.loc[idx:, :].iterrows()]

        df.loc[idx:, self.get_column_name(col_name)] = part_nom
        return df

    @classmethod
    def get_column_name(cls, col_name: str, **kwargs) -> str:
        """This method overrides :meth:`FeatureBasic.get_column_name`."""
        return col_name


class Shifted(FeatureBasic):
    """
    Move values from target column for a several steps up or down relative to another column.

    Parameters:
        id_columns: list
            Primary keys using to merge target table values with additional.

    Attributes:
        id_columns: list
            Primary keys using to merge target table values with additional.
    """
    def __init__(self, id_columns=None):
        self.id_columns = ['Nomenclature_size', 'year', 'month'] if id_columns is None else id_columns

    def build(self, df: DataFrame, idx=None,
              col_name: str = 'new_col',
              shifting_name: str = 'month',
              shifter_name: str = 'year',
              shift_size: int = 1,
              **kwargs) -> DataFrame:
        """

        :param df: basic table to design a new property.
        :type df: DataFrame

        :param idx: index to insert a new property. A new column creates if none.
        :type idx: Timestamp

        :param col_name: new/updating column's name.
        :type col_name: str

        :param shifting_name: name of moving column.
        :type shifting_name: str

        :param shifter_name: name of the column relative to which te movement will be performed.
        :type shifter_name: str

        :param shift_size: number of steps to move.
        :type shift_size: int

        :param kwargs: other arguments.
        :type kwargs: dict

        :return: incoming table with new or augmented column.
        :rtype: DataFrame
        """
        idx_df = df.index
        data_shift = df.loc[:idx, self.id_columns + [shifting_name]].drop_duplicates()
        data_shift.loc[:, shifter_name] = data_shift.loc[:, shifter_name] + shift_size
        if col_name in df.columns:
            df = df.drop(columns=[col_name])
        df = df.merge(data_shift, on=self.id_columns, how='left').rename(
            columns={'{}_x'.format(shifting_name): shifting_name, '{}_y'.format(shifting_name
                                                                                ): self.get_column_name(col_name)})
        df.index = idx_df
        return df

    @classmethod
    def get_column_name(cls, col_name, **kwargs) -> str:
        """This method overrides :meth:`FeatureBasic.get_column_name`."""
        return col_name


def create_builder_record(classname: ClassVar, params: dict) -> dict:
    """
    Auxiliary function creating a new record for later use in specialized feature builder.

    :param classname: Name of class implementing FeatureBasic interface.
    :type classname: ClassVar FeatureBasic

    :param params: parameters for 'build' function.
    :type params: dict

    :return: feature name + a tuple with FeatureBasic class and key parameters.
    :rtype: dict
    """
    return {classname.get_column_name(**params): fb_dict(classname(), params)}


def search_best_features(x: DataFrame, y: Series, estimators: list) -> dict:
    """
    Function take the list of estimators and the train data with a plenty of columns.
    And do a feature selection with RFECV from sklearn library. Minimal number of features is two.

    :param x: train data
    :type x: DataFrame

    :param y: train target
    :type y: Series

    :param estimators: list of estimators to do a feature searching.
    :type estimators: list

    :return: estimators with theirs optimal features
    :rtype: dict
    """
    return {estimator: x.columns[RFECV(estimator(), step=1,
                                       cv=5, verbose=0,
                                       min_features_to_select=2).fit(x, y).support_] for estimator in estimators}
