"""
Wrapper for different grid searchers from external libraries.
Support kats and sktime.
"""
from __future__ import annotations
import mock
import timelibs

from abc import ABCMeta, abstractmethod
from kats.consts import SearchMethodEnum, TimeSeriesData
from kats.utils.time_series_parameter_tuning import SearchMethodFactory
from numpy import array
from pandas import DataFrame, Series
from sklearn.metrics import mean_absolute_error
from sktime.forecasting.model_selection import ExpandingWindowSplitter, ForecastingGridSearchCV
from sktime.forecasting.model_selection._tune import BaseGridSearch
from sktime.performance_metrics.forecasting import MeanAbsoluteError
from typing import TypeVar, Tuple, List, Union

T = TypeVar('T')


class GridTuner:
    """
    Interface for classes realizing the grid search.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def search_best_parameters(self, *args) -> Tuple[T, float]:
        """
        Function searches the best parameters for some estimator defined inside the class initialization.
        The main scorer is mean_absolute_error from sklearn lib.

        :param args: additional arguments
        :type args: list

        :return: estimator with optimal parameters and the best mae value
        :rtype: (any, float)
        """
        pass


class ForecasterTuner(GridTuner):
    """
    Wrapper for sktime grid search functions. Works with specific estimators implementing class BaseForecaster.
    Basic ForecastingGridSearchCV falls if forecaster fit/predict raise an exception so there's tiny monkey patching
    solving this problem. Also all predictions rounded to positive integer values.

    Parameters:
        x: DataFrame
            Univariate time series. Forecast builds for a some steps forward.

        forecaster: BaseForecaster
            Tuning forecaster.

        param_grid: dict
            Dictionary with parameters to search.

    Attributes:
        x: DataFrame
            Univariate time series. Forecast builds for a some steps forward.

        forecaster: BaseForecaster
            Tuning forecaster.

        param_grid: dict
            Dictionary with parameters to search.
    """

    def __init__(self, x: DataFrame, forecaster: T, param_grid: dict):
        self.x = x
        self.forecaster = forecaster
        self.param_grid = param_grid

    def search_best_parameters(self, initial_window) -> Tuple[T, float]:
        """
        Wrapper for ForecastingGridSearchCV function from sktime lib. Searches optimal
        parameters with expanding train window for a few steps forward.

        :param initial_window: max window size in expanding window splitter.
        :type initial_window: int

        :return: estimator with optimal parameters and the best mae value
        :rtype: (any, float)
        """
        cv = ExpandingWindowSplitter(initial_window=initial_window, fh=1)
        with mock.patch.object(BaseGridSearch, '_fit', timelibs.utils.new_fit):
            gscv = ForecastingGridSearchCV(self.forecaster, strategy='refit',
                                           cv=cv, scoring=MeanAbsoluteError(),
                                           param_grid=self.param_grid)
            gscv.fit(self.x)
        return gscv.best_forecaster_, gscv.best_score_


class RegressorTuner(GridTuner):
    """
    Grid search for a table-like data. Use SearchMethodFactory from kats to optimize search performance.

    Parameters:
        x: DataFrame, TimeSeriesData
            Table-like data to do a search.

        y: Series
            Target.

        estimator: BaseForecaster
            Tuning estimators.

        param_grid: list
            Dictionary with parameters to search.

    Attributes:
        x: DataFrame
            Table-like data to do a search.

        y: Series
            Target.

        estimator: BaseForecaster
            Tuning estimators.

        param_grid: list
            Dictionary with parameters to search.
    """

    def __init__(self, x: Union[DataFrame, TimeSeriesData], y: Series, estimator: T, param_grid: list):
        self.x = x
        self.y = y
        self.estimator = estimator
        self.param_grid = param_grid

    def search_best_parameters(self) -> Tuple[T, float]:
        """
        Wrapper for SearchMethodFactory from kats.

        :return: estimator initiated with the best found parameters and the best score.
        :rtype: Tuple[T, float]
        """
        tuner = SearchMethodFactory.create_search_method(
            objective_name="evaluation_metric",
            parameters=self.param_grid,
            selected_search_method=SearchMethodEnum.GRID_SEARCH)
        tuner.generate_evaluate_new_parameter_values(evaluation_function=self.eval_function)
        tuner_result = (tuner.list_parameter_value_scores())
        if tuner_result.query('mean >= 0').empty:
            return self.estimator(**tuner_result.loc[0, 'parameters']), 100000
        else:
            idx = tuner_result.query('mean >= 0')['mean'].idxmin()
            return self.estimator(**tuner_result.loc[idx, 'parameters']), tuner_result.loc[idx, 'mean']

    def eval_function(self, params: dict):
        """
        Function iterates throw data with expanding window and count mean absolute error by one-step prediction.

        :param params: estimator's parameters
        :type params: dict

        :return: averaged value of all MAEs.
        :rtype: float
        """

        def inner_for(train_index: list, test_index: list) -> float:
            """
            One iteration of search function. Count MAE for a one train/test values from a global split.
            Also all predictions rounded to positive integer values.

            :param train_index: list of train indexes
            :type train_index: list

            :param test_index: list of test indexes
            :type test_index: list

            :return: mean absolute error of true target and predicted values.
            :rtype: float
            """
            train_index, test_index = self.x.index.unique()[train_index], self.x.index.unique()[test_index]
            model = self.estimator(**params)
            model.fit(self.x.loc[train_index, :], self.y.loc[train_index])
            return mean_absolute_error(self.y.loc[test_index], timelibs.positive_round(
                model.predict(self.x.loc[test_index, :])))

        # noinspection PyBroadException
        try:
            tscv = ExpandingWindowSplitter(initial_window=len(self.x.index.unique()) - 4, fh=1)
            losses = [inner_for(train_index, test_index
                                ) for train_index, test_index in tscv.split(self.x.index.unique())]
        except Exception as e:
            print(e)
            return -1
        return array(losses).mean()

    def eval_function_VAR(self, params) -> List[float]:
        """
        Specific function to VAR forecaster.
        Function iterates throw data with expanding window and count mean absolute error by one-step prediction.

        :param params: estimator's parameters
        :type params: VARParams

        :return: averaged value of all MAEs.
        :rtype: List[float]
        """

        def inner_for(train_index, test_index):
            """
            One iteration of search function. Count MAE for a one train/test values from a global split.
            Also all predictions rounded to positive integer values.

            :param train_index: list of train indexes
            :type train_index: list

            :param test_index: list of test indexes
            :type test_index: list

            :return: mean absolute error of true target and predicted values.
            :rtype: List[float]
            """
            model = self.estimator(self.x[:train_index[-1]], params)
            model.fit()
            model_pred = model.predict(len(test_index))
            return [mean_absolute_error(self.x.value.loc[test_index, key].tolist(),
                                        timelibs.positive_round(array(value.value.fcst))
                                        ) for key, value in model_pred.items()]
        try:
            tscv = ExpandingWindowSplitter(initial_window=len(self.x) - 2, fh=1)
            losses = [inner_for(train_index, test_index) for train_index, test_index
                      in tscv.split(Series(self.x.time.unique()))]
        except Exception as e:
            print(e)
            return [-1]*self.x.values.shape[1]
        return array(losses).mean(axis=0)
