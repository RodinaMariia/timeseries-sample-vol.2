"""
Different forecasting model supporting univariate and multivariate time series.

Contains more complex methods then previous version of the 'timelibs' library, like
vector autoregressive model from kats lib, vector decomposition to a seasonal part and residuals,
building a regression on statistical characteristic of series and other.
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from catboost import CatBoostRegressor
from copy import deepcopy
from dateutil.relativedelta import relativedelta
from itertools import product
from kats.models.var import VARModel, VARParams
from kats.consts import TimeSeriesData
from lightgbm import LGBMRegressor
from numpy import any, array, around, ndarray
from pandas import DataFrame, Series
from pandas import PeriodIndex, Timestamp
from typing import TypeVar, Tuple, Any, Union
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, ARDRegression
from sktime.forecasting.fbprophet import Prophet
from sktime.transformations.series.detrend import Deseasonalizer
from sktime.forecasting.compose import TransformedTargetForecaster, make_reduction
from timelibs.utils import get_param_grid, RegressorTuner, ForecasterTuner, featurebuilder as fbuilder
from utils4sales import StandardScalerExt

T = TypeVar('T')


class SalesPredictor:
    """
    Interface to the all forecasting models.

    Provide getting model's score, calculated with cross-validation, to compare various models between each others.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, x: DataFrame, *args) -> SalesPredictor:
        """
        Fit model to provide future forecasting.

        :param x: univariate or multivariate time series.
        :type x: DataFrame

        :param args: variable length argument list.
        :type args: list

        :return: self
        :rtype: SalesPredictor
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> ndarray:
        """
        Build a forecast to a several steps forward.

        :param args: variable length argument list.
        :type args: list

        :param kwargs: arbitrary keyword arguments.
        :type kwargs: dict

        :return: prediction by the some horizon.
        :rtype: ndarray
        """
        pass

    @abstractmethod
    def search_best_estimator(self, x: DataFrame, **kwargs) -> Tuple[Any, float]:
        """
        Cross-validated search for a nested estimator. Select both a specific predictor and its parameters.
        Also calculate scores to the picked up values.

        :param x: validation data.
        :type x: DataFrame

        :param kwargs: arbitrary keyword arguments.
        :type kwargs: dict

        :return: optimal estimator initiated with the best parameters and it's score.
        :rtype: Tuple[Any, float]
        """
        pass

    @abstractmethod
    def get_score(self, *args) -> float:
        """
        Get score of the nested estimator calculated at the validation data.

        :param args: variable length argument list.
        :type args: list

        :return: score of the nested estimator.
        :rtype: float
        """
        pass


class PredictorResiduals(SalesPredictor):
    # noinspection PyUnresolvedReferences
    """
    This predictor works with two stages. First it builds main forecast with basic autoregressive model
    (uses facebook Prophet) and after that solve a regression on residuals. Regressor and it's features
    generates and selects from predetermined list automatically.

    Attributes:
        IDX_SPLIT: float,
            Size of train part to fit the basic forecaster (default is 0.2).

        TRAIN_IDX: list
            Train indexes.

        PREDICT_IDX: list
            Predict indexes to fit a regressor.

        x: DataFrame
            Univariate time series(default is None).

        forecaster: Prophet
            Nested forecaster to build the basic prediction (default is None).

        estimator: Any
            Simple regressor predicting residuals between forecaster's calculation and
            the true value (default is None).

        columns_builder: dict
            Automatically created statistical features from the basic time series.

        best_score: float
            Best score of the nested estimator calculated during cross-validation.

        scaler: StandardScalerExt
            Standard scaler from sklearn lib optimized o work with named tables.

        estimators: list
            Potential regressor's classes.

    .. note:: Using a small part of data to fit the forecaster leads to uncertain prediction at the
    latest data which is inherited by regressor. Therefore forecaster use the whole data assigned by PREDICT_IDX
    both to fit and predict.
    """
    estimators = [LGBMRegressor, RandomForestRegressor, GradientBoostingRegressor,
                  CatBoostRegressor, ElasticNet, ARDRegression]

    def __init__(self):
        super().__init__()
        self.IDX_SPLIT = 0.2
        self.TRAIN_IDX = []
        self.PREDICT_IDX = []
        self.x = None
        self.forecaster = None
        self.estimator = None
        self.columns_builder = {}
        self.best_score = None
        self.scaler = StandardScalerExt()

    def fit(self, x: DataFrame, search_estimator: bool = True, *args, **kwargs) -> PredictorResiduals:
        """
        Fit model to provide future forecasting.

        :param x: univariate time series.
        :type x: DataFrame

        :param search_estimator: search the optimal estimator with features if enabled.
        Also recalculated the best score value.
        :type search_estimator: bool

        :param args: variable length argument list.
        :type args: list

        :param kwargs: arbitrary keyword arguments.
        :type kwargs: dict

        :return: self
        :rtype: PredictorResiduals
        """
        x = x.copy()
        self.TRAIN_IDX, self.PREDICT_IDX = x.index[:int(len(x.index) * self.IDX_SPLIT
                                                        )], x.index[int(len(x.index) * self.IDX_SPLIT):]
        # search and fit best forecaster
        if search_estimator or self.forecaster is None:
            self.forecaster, _ = self.search_best_forecaser(x)
        self.forecaster.fit(x.loc[self.PREDICT_IDX, 'Count'])
        preds, residuals = self._get_residuals(x, self.PREDICT_IDX)
        x.loc[self.PREDICT_IDX, 'Forecasted'] = preds
        x.loc[self.PREDICT_IDX, 'Count'] = residuals

        # search and fit best regressor
        if search_estimator or self.estimator is None:
            self._create_columns_builder_full()
            x_train = self._create_columns(x.loc[self.PREDICT_IDX, ['Forecasted', 'Count']])
            self.estimator, self.best_score = self.search_best_estimator(x_train.drop(columns=['Count']), x_train.Count)
            x = x_train.loc[:, list(self.columns_builder.keys()) + ['Count', 'Forecasted']]
        else:
            x = self._create_columns(x.loc[self.PREDICT_IDX, ['Forecasted', 'Count']])
        x = x.reindex(sorted(x.columns), axis=1)
        self.x = x.copy()

        # scale fit data
        self.scaler.fit(x.drop(columns=['Count']))
        x.loc[:, x.drop(columns=['Count']).columns] = self.scaler.transform(x.drop(columns=['Count']))
        x, y = x.drop(columns=['Count']), x.Count
        self.estimator.fit(x, y)
        return self

    def predict(self, h: int = 1, *args, **kwargs) -> ndarray:
        """
        Make a forecast to h steps forward.

        :param h: forecast horizon
        :type h: int

        :param args: variable length argument list.
        :type args: list

        :param kwargs: arbitrary keyword arguments.
        :type kwargs: dict

        :return: list of predictions.
        :rtype: ndarray
        """
        x = self.x.copy(deep=True)
        idx_start = x.index[-1]
        for i in range(1, h + 1):
            idx = idx_start + relativedelta(months=i)
            x.loc[idx, :] = 0
            x.loc[idx, 'Forecasted'] = self.forecaster.predict(i)[-1]
            x = self._create_columns(x, idx).reindex(sorted(x.columns), axis=1)

            # scale columns before predict
            x_predict = x.copy().drop(columns=['Count'])
            x_predict.loc[:, :] = self.scaler.transform(x_predict)
            x.loc[idx, 'Count'] = self.estimator.predict(x_predict.loc[idx, :].to_frame().T)[0]
        return positive_round((x.loc[idx_start +
                                     relativedelta(months=1):, 'Forecasted'] +
                               x.loc[idx_start + relativedelta(months=1):, 'Count']).values)

    def get_score(self, *args) -> float:
        """Get the best score of tne nested estimator."""
        return self.best_score

    def _get_residuals(self, x: DataFrame, predict_idx: list) -> Tuple[list, list]:
        """
        Calculate forecast and residuals between this prediction and the true values. Forecast build for a one
        step ahead within the list of indices 'predict_idx'.

        :param x: univariate time series.
        :type x: DataFrame

        :param predict_idx: indices of the original data to calculate the forecasts and residuals.
        :type predict_idx:

        :return: list of predictions and residuals counted at the given indices.
        :rtype: Tuple[list, list]
        """
        x = x.copy()
        forecaster = deepcopy(self.forecaster)
        preds = []
        residuals = []
        for idx in predict_idx:
            forecaster.fit(x.loc[x.index < idx, 'Count'])
            pred = forecaster.predict(1)
            preds.append(pred)
            residuals.append(x.loc[x.index == idx, 'Count'] - pred)
        return preds, residuals

    def _create_columns_builder_minimal(self):
        """Create default columns set."""
        self.columns_builder.clear()
        self.columns_builder.update(fbuilder.create_builder_record(fbuilder.DateIndex,
                                                                   {'feature_name': 'year'}))
        self.columns_builder.update(fbuilder.create_builder_record(fbuilder.DateIndex,
                                                                   {'feature_name': 'month'}))
        self.columns_builder.update(fbuilder.create_builder_record(fbuilder.Covid, {}))
        [self.columns_builder.update(fbuilder.create_builder_record(fbuilder.Lag, {'target_name': 'Count',
                                                                                   'i': i})
                                     ) for i in range(2, 12)]
        self.columns_builder.update(fbuilder.create_builder_record(fbuilder.Avg, {'func': 'mean',
                                                                                  'target_name': 'Forecasted',
                                                                                  'rolling_window': 6}))
        self.columns_builder.update(fbuilder.create_builder_record(fbuilder.Avg, {'func': 'std',
                                                                                  'target_name': 'Forecasted',
                                                                                  'rolling_window': 6}))
        self.columns_builder.update(fbuilder.create_builder_record(fbuilder.Avg, {'func': 'min',
                                                                                  'target_name': 'Forecasted',
                                                                                  'rolling_window': 6}))
        self.columns_builder.update(fbuilder.create_builder_record(fbuilder.Avg, {'func': 'max',
                                                                                  'target_name': 'Forecasted',
                                                                                  'rolling_window': 6}))

    def _create_columns_builder_full(self):
        """Create full columns set for later feature searching."""
        self.columns_builder.clear()
        self.columns_builder.update(fbuilder.create_builder_record(fbuilder.DateIndex,
                                                                   {'feature_name': 'year'}))
        self.columns_builder.update(fbuilder.create_builder_record(fbuilder.DateIndex,
                                                                   {'feature_name': 'month'}))
        self.columns_builder.update(fbuilder.create_builder_record(fbuilder.Covid, {}))
        [self.columns_builder.update(fbuilder.create_builder_record(fbuilder.Lag, {'target_name': 'Count',
                                                                                   'i': i})
                                     ) for i in range(2, 12)]

        params = product(['mean', 'std', 'min', 'max'], ['Forecasted', 'Count'], [3, 6, 12])
        [self.columns_builder.update(fbuilder.create_builder_record(fbuilder.Avg, {'func': p1,
                                                                                   'target_name': p2,
                                                                                   'rolling_window': p3}
                                                                    )) for p1, p2, p3 in params]

    def _create_columns(self, x: DataFrame, idx: Union[Timestamp, str] = None) -> DataFrame:
        """
        Consistently add columns to the given dataframe. The sequence and composition of columns
        determines by the 'columns_builder'.

        :param x: univariate time series.
        :type x: DataFrame

        :param idx: index to place the new columns. Create a new column if None,
        otherwise set calculations to the existing column from index to end.
        :type idx: Timestamp, str

        :return: incoming dataframe with new columns.
        :rtype: DataFrame
        """
        df = x.copy()
        for value in self.columns_builder.values():
            df = value.builder.build(df=df, idx=idx, **value.params)

        # [df := value.builder.build(df=df, idx=idx, **value.params) for value in self.columns_builder.values()]
        df = df.fillna(0).loc[df.index >= '2017-01-31'].sort_index()
        return df

    def search_best_estimator(self, x: DataFrame, y: Series = None) -> Tuple[Any, float]:
        """
        Search an optimal estimator from the predefined list. Besides, for every estimator picking up
        the most helpful features and features from the best estimator removes to the column_builder.

        :param x: time series features.
        :type x: DataFrame

        :param y: target.
        :type y: Series

        :return: The best estimator with it's score.
        :rtype:  Tuple[Any, float]
        """
        y = y if y is not None else x.loc[:, 'Count']
        estimators = self.estimators.copy()
        estimators_cols = fbuilder.search_best_features(x, y, estimators)

        df_metrics = DataFrame(None, columns=['estimator', 'metric'])
        for estimator in estimators:
            tune1, tune2 = RegressorTuner(x.loc[:, estimators_cols.get(estimator)],
                                          y, estimator, get_param_grid(estimator(), 'Kats')).search_best_parameters()
            df_metrics.loc[df_metrics.shape[0], ['estimator', 'metric']] = [tune1, tune2]

        idx = df_metrics.loc[:, 'metric'].astype(float).idxmin()
        cols = estimators_cols.get(df_metrics.loc[idx, 'estimator'].__class__)
        self.columns_builder = dict(filter(lambda key: key[0] in cols,
                                           self.columns_builder.items()))
        self._check_columns()
        return df_metrics.loc[idx, 'estimator'], df_metrics.loc[idx, 'metric']

    def search_best_forecaser(self, x) -> Tuple[Prophet, float]:
        """
        Search optimal parameters to the nested forecaster.

        :param x: univariate time series.
        :type x: DataFrame

        :return: forecaster initialized with an optimal parameters and it's score.
        :rtype: Tuple[Prophet, float]
        """
        data = x.copy().loc[:, 'Count']
        forecaster = Prophet()
        param_grid = {'yearly_seasonality': [True, False],
                      'weekly_seasonality': [False],
                      'daily_seasonality': [False],
                      'seasonality_mode': ['additive'],
                      'growth': ['linear', 'flat']
                      }
        initial_window = len(data) - 4
        tuner = ForecasterTuner(data, forecaster, param_grid)
        forecaster, fc_score = tuner.search_best_parameters(initial_window=initial_window)
        return forecaster, fc_score

    def _check_columns(self):
        """Columns can be linked to each other. So it's necessary to add columns which are important to other
         features included in columns_builder."""
        if any(['year' in name for name in self.columns_builder.keys()]) and (self.columns_builder.get('year') is None):
            new_entry = fbuilder.create_builder_record(fbuilder.DateIndex, {'feature_name': 'year'})
            new_entry.update(self.columns_builder)
            self.columns_builder = new_entry.copy()

        if any(['month' in name for name in self.columns_builder.keys()]) and (
                self.columns_builder.get('month') is None):
            new_entry = fbuilder.create_builder_record(fbuilder.DateIndex, {'feature_name': 'month'})
            new_entry.update(self.columns_builder)
            self.columns_builder = new_entry.copy()


class PredictorDeseasonal(SalesPredictor):
    # noinspection PyUnresolvedReferences
    """
    Predictor use two-steps pipeline. At first, forecaster delete season from the univariate time series. There's no
    trend in our target data, so we can build the forecasting without removing a trend.
    Therefore we treat remaining sequence as the stationary and make a prediction on it.

    Attributes:
        estimator: TransformedTargetForecaster
            Pipeline containing sktime Deseasonalizer and RecursiveTimeSeriesRegressionForecaster (default is None).

        best_score: float
            Best score of the nested estimator calculated during cross-validation.

        estimators: list
            Potential regressor's classes uses at a RecursiveTimeSeriesRegressionForecaster.
    """
    estimators = [LGBMRegressor(), RandomForestRegressor(), GradientBoostingRegressor(),
                  ElasticNet(), ARDRegression()]

    def __init__(self):
        super().__init__()
        self.estimator = None
        self.best_score = None

    def fit(self, x: DataFrame, search_estimator: bool = True, *args, **kwargs) -> PredictorDeseasonal:
        """
        Fit model to provide future forecasting.

        :param x: univariate time series.
        :type x: DataFrame

        :param search_estimator: search the optimal estimator with features if enabled.
        Also recalculated the best score value.
        :type search_estimator: bool

        :param args: variable length argument list.
        :type args: list

        :param kwargs: arbitrary keyword arguments.
        :type kwargs: dict

        :return: self
        :rtype: PredictorDeseasonal
        """
        x = x.copy(deep=True).loc[:, 'Count']
        x.index = PeriodIndex(x.index.to_list(), freq='M')
        if search_estimator or self.estimator is None:
            self.estimator, self.best_score = self.search_best_estimator(x)
        self.estimator.fit(x)
        return self

    def predict(self, h: int, *args, **kwargs) -> ndarray:
        """
        Make a forecast to h steps forward.

        :param h: forecast horizon
        :type h: int

        :param args: variable length argument list.
        :type args: list

        :param kwargs: arbitrary keyword arguments.
        :type kwargs: dict

        :return: list of predictions.
        :rtype: ndarray
        """
        return positive_round(array([self.estimator.predict(i)[0] for i in range(1, h + 1)]))

    def search_best_estimator(self, x, **kwargs) -> Tuple[Any, float]:
        """
        Search an optimal estimator from the predefined list and it's parameters.

        :param x: time series features.
        :type x: DataFrame

        :param kwargs: arbitrary keyword arguments.
        :type kwargs: dict

        :return: The best estimator with it's score.
        :rtype:  Tuple[Any, float]
        """
        estimators = self.estimators.copy()
        data = x.copy()
        forecaster = TransformedTargetForecaster([
            ('deseasonalize', Deseasonalizer(model='additive', sp=6)),
            ('forecast', make_reduction(estimators[0], window_length=12, strategy='recursive'))
        ])

        initial_window = int(len(data) * 0.6)
        param_grid = {'forecast__window_length': [3, 4, 6] + list(range(12, initial_window, 6)),
                      'forecast__estimator': estimators,
                      'deseasonalize__sp': list(filter(lambda el: initial_window >= 2 * el, [3, 4, 6, 12]))
                      }
        forecaster, fc_score = ForecasterTuner(data, forecaster, param_grid).search_best_parameters(initial_window)
        del estimators

        regressor = forecaster.named_steps.get('forecast')
        regressor, fc_score = ForecasterTuner(data, regressor, get_param_grid(regressor.estimator, 'Sktime')
                                              ).search_best_parameters(initial_window)
        forecaster.named_steps['forecast'] = regressor
        return forecaster, fc_score

    def get_score(self, *args) -> float:
        """Get the best score of tne nested estimator."""
        return self.best_score


class PredictorVAR(SalesPredictor):
    """
    Vector autoregressive model from the kats lib. Works with multivariate series.

    Attributes:
        estimator: VARModel
            Vector autoregressive model to forecast all rows at once.

        best_score: dict
            Best score of the nested estimator calculated for each row from
            multivariate series separately.
    """

    def __init__(self):
        super().__init__()
        self.estimator = None
        self.best_score = None

    def fit(self, x: DataFrame, *args, **kwargs) -> PredictorVAR:
        """
        Fit model to provide future forecasting. Also calculate scores to all sequence in multivariate series.

        :param x: multivariate time series.
        :type x: DataFrame

        :param args: variable length argument list.
        :type args: list

        :param kwargs: arbitrary keyword arguments.
        :type kwargs: dict

        :return: self
        :rtype: PredictorResiduals
        """
        x = self._create_multiseries(x.copy())
        if isinstance(x, DataFrame):
            x = TimeSeriesData(x.reset_index(drop=False).rename(columns={'Date': 'time'}))
        self._count_score(x)
        self.estimator = VARModel(x, VARParams())
        self.estimator.fit()
        return self

    def predict(self, h: int, nomenclature_size: int, *args, **kwargs) -> ndarray:
        """
        Make a forecast to h steps forward for the one particular sequence from multivariate series.

        :param h: forecast horizon
        :type h: int

        :param nomenclature_size: encoded number of sequence from the whole time series.
        :type nomenclature_size: int

        :param args: variable length argument list.
        :type args: list

        :param kwargs: arbitrary keyword arguments.
        :type kwargs: dict

        :return: list of predictions.
        :rtype: ndarray
        """
        return positive_round(self.estimator.predict(steps=h).get(nomenclature_size).value.fcst.values)

    def search_best_estimator(self, x, **kwargs):
        """Plug. There's nothing to optimize."""
        pass

    def get_score(self, nomenclature_size: int) -> float:
        """
        Get the best score to the one particular sequence from multivariate series.

        :param nomenclature_size: encoded number of sequence from the whole time series.
        :type nomenclature_size: int

        :return: the best score to the one univariate sequence calculated during the cross-validation.
        :rtype: float
        """
        return 10000 if self.best_score.get(nomenclature_size) is None else self.best_score.get(nomenclature_size)

    def _create_multiseries(self, data: DataFrame) -> DataFrame:
        """Transform one long miltivariate sequence to the table containing
        univariate series in the each row."""
        ts_dict = {col_name: data.loc[data.Nomenclature_size == col_name,
                                      'Count'].tolist() for col_name in data['Nomenclature_size'].unique()}
        data_multi = DataFrame(ts_dict, index=data.index.unique())
        return data_multi.sort_index(axis=1).asfreq('M').fillna(0)

    def _count_score(self, x: TimeSeriesData):
        """
        Calculate nested estimators average error using cross-validation with an expanding window.

        :param x: multivariate time series formatted in the specific notation.
        :type x: TimeSeriesData
        """
        tuner = RegressorTuner(x, Series(), VARModel, [None])
        self.best_score = dict(zip(x.value.columns, tuner.eval_function_VAR(VARParams())))


class PredictorRegression(SalesPredictor):
    # noinspection PyUnresolvedReferences
    """
    Classic regression. Predictor creates a plenty of features from the original multivariate time series,
    then select the optimal set of columns to the best scored nested regressor.
    The following calculation carry out on this features set.

    Attributes:
        x: DataFrame
            Multivariate time series (default is None).

        estimator: Any
            Simple regressor predicting residuals between forecaster's calculation and
            the true value (default is None).

        columns_builder: dict
            Automatically created statistical features from the basic time series.

        best_score: float
            Best score of the nested estimator calculated during cross-validation.

        scaler: StandardScalerExt
            Standard scaler from sklearn lib optimized o work with named tables.

        estimators: list
            Potential regressor's classes.
    """
    estimators = [LGBMRegressor, RandomForestRegressor, GradientBoostingRegressor,
                  CatBoostRegressor, ElasticNet, ARDRegression]

    def __init__(self):
        super().__init__()
        self.estimator = None
        self.x = None
        self.columns_builder = {}
        self.additional_columns_builder = {}
        self.best_score = None
        self.scaler = StandardScalerExt()

    def fit(self, x: DataFrame, search_estimator: bool = True,
            nomenclature_size: int = -1, *args, **kwargs) -> PredictorRegression:
        """
        Fit model to provide future forecasting.

        :param x: univariate time series.
        :type x: DataFrame

        :param search_estimator: search the optimal estimator with features if enabled.
        Also recalculated the best score value.
        :type search_estimator: bool

        :param nomenclature_size: encoded number of sequence from the whole time series.
        :type nomenclature_size: int

        :param args: variable length argument list.
        :type args: list

        :param kwargs: arbitrary keyword arguments.
        :type kwargs: dict

        :return: self
        :rtype: PredictorRegression
        """
        self._check_nomenclature(nomenclature_size, 0)
        x = x.copy()
        # search and fit best regressor
        if search_estimator or self.estimator is None:
            self._create_columns_builder_full()
            x_train = self._create_columns(x)
            self.estimator, self.best_score = \
                self.search_best_estimator(x_train.query('Nomenclature_size=={}'.format(nomenclature_size)
                                                         ).drop(columns=['Count']),
                                           x_train.query('Nomenclature_size=={}'.format(nomenclature_size)).Count)
            x = x_train.loc[:, list(self.columns_builder.keys()) + ['Count', 'Nomenclature_size',
                                                                    'Nomenclature_group']]
        else:
            x = self._create_columns(x)
        x = x.reindex(sorted(x.columns), axis=1)
        self.x = x.copy()

        # scale fit data
        self.scaler.fit(x.drop(columns=['Count']))
        x = x.drop(columns=list(self.additional_columns_builder.keys()))
        x.loc[:, x.drop(columns=['Count']).columns] = self.scaler.transform(x.drop(columns=['Count']))
        x, y = x.drop(columns=['Count']), x.Count
        self.estimator.fit(x, y)
        return self

    def predict(self, h: int, nomenclature_size: int, nomenclature_group: int, *args, **kwargs) -> ndarray:
        """
        Make a forecast to h steps forward.

        :param h: forecast horizon
        :type h: int

        :param nomenclature_size: encoded number of sequence from the whole time series.
        :type nomenclature_size: int

        :param nomenclature_group: auxiliary parameter for some additional features.
        :type nomenclature_group: int

        :param args: variable length argument list.
        :type args: list

        :param kwargs: arbitrary keyword arguments.
        :type kwargs: dict

        :return: list of predictions.
        :rtype: ndarray
        """
        self._check_nomenclature(nomenclature_size, nomenclature_group)
        x = self.x.copy(deep=True)
        idx_start = x.index[-1]
        for i in range(1, h + 1):
            idx = idx_start + relativedelta(months=i)
            x.loc[idx, :] = 0
            x.loc[idx, 'Nomenclature_size'] = nomenclature_size
            x.loc[idx, 'Nomenclature_group'] = nomenclature_group
            x = self._create_columns(x, idx).reindex(sorted(x.columns), axis=1)

            # scale columns before predict
            x_predict = x.copy().drop(columns=['Count'])
            x_predict.loc[:, :] = self.scaler.transform(x_predict)
            x_predict = x_predict.drop(columns=list(self.additional_columns_builder.keys()))
            x.loc[idx, 'Count'] = self.estimator.predict(x_predict.loc[idx, :].to_frame().T)[0]
        return positive_round(x.loc[idx_start + relativedelta(months=1):, 'Count'].values)

    def search_best_estimator(self, x: DataFrame, y: Series = None) -> Tuple[Any, float]:
        """
        Search an optimal estimator from the predefined list. Besides, for every estimator picking up
        the most helpful features, after that features from the best estimator transferred to the column_builder.

        :param x: table-like data with time series features.
        :type x: DataFrame

        :param y: target.
        :type y: Series

        :return: The best estimator with it's score.
        :rtype:  Tuple[Any, float]
        """
        y = y if y is not None else x.loc[:, 'Count']
        estimators = self.estimators.copy()

        scaler = StandardScalerExt().fit(x)
        x.loc[:, :] = scaler.transform(x)
        estimators_cols = fbuilder.search_best_features(x, y, estimators)

        # regressor grid-search
        df_metrics = DataFrame(None, columns=['estimator', 'metric'])
        for estimator in estimators:
            tune1, tune2 = RegressorTuner(x.loc[:, estimators_cols.get(estimator)],
                                          y, estimator, get_param_grid(estimator(), 'Kats')).search_best_parameters()
            df_metrics.loc[df_metrics.shape[0], ['estimator', 'metric']] = [tune1, tune2]

        idx = df_metrics.loc[:, 'metric'].astype(float).idxmin()
        cols = estimators_cols.get(df_metrics.loc[idx, 'estimator'].__class__)

        # transfer optimal columns set to the builder
        self._check_columns(cols.tolist())
        basic_builder = dict(filter(lambda key: key[0] in cols,
                                    self.columns_builder.items()))
        self.columns_builder = self.additional_columns_builder.copy()
        self.columns_builder.update(basic_builder)
        return df_metrics.loc[idx, 'estimator'], df_metrics.loc[idx, 'metric']

    def get_score(self, *args) -> float:
        """Get the best score of tne nested estimator."""
        return self.best_score

    def _create_columns_builder_minimal(self):
        """Create default columns set."""
        self.columns_builder.clear()
        self.columns_builder.update(fbuilder.create_builder_record(fbuilder.DateIndex,
                                                                   {'feature_name': 'year'}))
        self.columns_builder.update(fbuilder.create_builder_record(fbuilder.DateIndex,
                                                                   {'feature_name': 'month'}))
        self.columns_builder.update(fbuilder.create_builder_record(fbuilder.Covid, {}))

        [self.columns_builder.update(fbuilder.create_builder_record(fbuilder.AvgGroup,
                                                                    {'func': 'mean',
                                                                     'target_name': 'Count',
                                                                     'group_names': ['Nomenclature_size'],
                                                                     'rolling_window': p1}
                                                                    )) for p1 in [3, 12]]

        params = zip([['month'], ['year', 'month'],
                      ['Nomenclature_group', 'month'],
                      ['Nomenclature_group', 'year', 'month']],
                     ['month_part', 'year_month_part', 'grp_month_part', 'grp_year_month_part'],
                     ['last_year', 'last_month', 'last_year', 'last_month'])

        [self.columns_builder.update(fbuilder.create_builder_record(fbuilder.Part,
                                                                    {'group_names': p1,
                                                                     'col_name': p2,
                                                                     'date_threshold': p3,
                                                                     'target_group': 'Nomenclature_size',
                                                                     'target_name': 'Count'})) for p1, p2, p3 in params]

        self.columns_builder.update(fbuilder.create_builder_record(fbuilder.Shifted,
                                                                   {'col_name': 'year_month_part_shift',
                                                                    'shifting_name': 'year_month_part',
                                                                    'shifter_name': 'year',
                                                                    'shift_size': 1}))
        self.columns_builder.update(fbuilder.create_builder_record(fbuilder.Shifted,
                                                                   {'col_name': 'grp_year_month_part_shift',
                                                                    'shifting_name': 'grp_year_month_part',
                                                                    'shifter_name': 'year',
                                                                    'shift_size': 1}))

    def _create_columns_builder_full(self):
        """Create full columns set for later feature searching."""
        self.columns_builder.clear()
        self.columns_builder.update(fbuilder.create_builder_record(fbuilder.DateIndex,
                                                                   {'feature_name': 'year'}))
        self.columns_builder.update(fbuilder.create_builder_record(fbuilder.DateIndex,
                                                                   {'feature_name': 'month'}))
        self.columns_builder.update(fbuilder.create_builder_record(fbuilder.Covid, {}))
        [self.columns_builder.update(fbuilder.create_builder_record(fbuilder.Lag, {'target_name': 'Count',
                                                                                   'target_group': 'Nomenclature_size',
                                                                                   'i': i})
                                     ) for i in range(2, 12)]

        params = product(['mean', 'std', 'min', 'max'],
                         ['Nomenclature_size', 'Nomenclature_group'], [3, 6, 12])
        [self.columns_builder.update(fbuilder.create_builder_record(fbuilder.AvgGroup,
                                                                    {'func': p1,
                                                                     'target_name': 'Count',
                                                                     'group_names': [p2],
                                                                     'rolling_window': p3}
                                                                    )) for p1, p2, p3 in params]

        params = zip([['year', 'month']],
                     ['year_month_part'],
                     ['last_month'])
        [self.columns_builder.update(fbuilder.create_builder_record(fbuilder.Part,
                                                                    {'group_names': p1,
                                                                     'col_name': p2,
                                                                     'date_threshold': p3,
                                                                     'target_group': 'Nomenclature_size',
                                                                     'target_name': 'Count'})) for p1, p2, p3 in params]

        params = zip(['year_month_part_shift', 'year_month_part_shift'],
                     ['year_month_part', 'year_month_part'],
                     [1, 2, 1, 2])
        [self.columns_builder.update(fbuilder.create_builder_record(fbuilder.Shifted,
                                                                    {'col_name': p1,
                                                                     'shifting_name': p2,
                                                                     'shifter_name': 'year',
                                                                     'shift_size': p3})) for p1, p2, p3 in params]

    def _create_columns(self, x: DataFrame, idx: Union[Timestamp, str] = None) -> DataFrame:
        """
        Consistently add columns to the given dataframe. The sequence and composition of columns
        determines by the 'columns_builder'.

        :param x: multivariate time series.
        :type x: DataFrame

        :param idx: index to place the new columns. Create a new column if None,
        otherwise set calculations to the existing column from index to end.
        :type idx: Timestamp, str

        :return: incoming dataframe with new columns.
        :rtype: DataFrame
        """
        df = x.copy()
        fill_index = df.index
        [df := value.builder.build(df=df, idx=idx, **value.params) for value in self.columns_builder.values()]
        df.index = fill_index
        df = df.fillna(0).loc[df.index >= '2017-01-31'].sort_index()
        return df

    def _check_nomenclature(self, nomenclature_size: int, nomenclature_group: int):
        """Check incoming specifications of the time series."""
        if (nomenclature_size < 0) or (nomenclature_group < 0):
            raise ValueError("Nomenclature's size and group must be meaningful.")

    def _check_columns(self, final_columns: list):
        """Columns can be linked to each other. So it's necessary to add columns which are important to other
        features included in columns_builder."""
        if any(['year' in name for name in final_columns]) and ('year' not in final_columns):
            self.additional_columns_builder['year'] = self.columns_builder.get('year')

        if any(['month' in name for name in final_columns]) and ('month' not in final_columns):
            self.additional_columns_builder['month'] = self.columns_builder.get('month')

        [self.additional_columns_builder.update({name[:idx - 1]: self.columns_builder.get(name[:idx - 1])})
         for name in final_columns if ((idx := name.find("shift")) > -1) and (name[:idx-1] not in final_columns)]

        # for name in final_columns:
        #     idx = name.find("shift")
        #     t2 = name[:idx - 1]
        #     if (idx > -1) and (name[:idx-1] not in final_columns):
        #         t1 = self.columns_builder.get(name[:idx - 1])
        #         self.additional_columns_builder.update({t2: t1})


class PredictorEnsemble(SalesPredictor):
    # noinspection PyUnresolvedReferences
    """
    Simple ensemble. Combines forecasts from other SalesPredictor and make a prediction at this data without
    any additional features.

    Attributes:
        estimator: Any
            Simple regressor using for ensemble (default is None).

        forecasters: dict
            A few forecasters with it's names.

        best_score: float
            Best score of the nested estimator calculated during cross-validation (default is None).

        estimators: list
            Potential regressor's classes.

    Properties:
        forecasters: dict
            A few forecasters with it's names.

    """
    estimators = [LGBMRegressor, RandomForestRegressor, GradientBoostingRegressor,
                  CatBoostRegressor, ElasticNet, ARDRegression]

    def __init__(self, forecasters: dict):
        super().__init__()
        self.estimator = None
        self.forecasters = forecasters
        self.best_score = None

    def fit(self, x: DataFrame, search_estimator: bool = True, nomenclature_size: int = -1,
            nomenclature_group: int = -1, *args, **kwargs) -> PredictorEnsemble:
        """
        Fit model to provide future forecasting.

        :param x: univariate time series.
        :type x: DataFrame

        :param search_estimator: search the optimal estimator with features if enabled.
        Also recalculated the best score value.
        :type search_estimator: bool

        :param nomenclature_size: encoded number of sequence from the whole time series.
        :type nomenclature_size: int

        :param nomenclature_group: analytical groups of the nomenclature,
        auxiliary parameter for some additional features.
        :type nomenclature_group: int

        :param args: variable length argument list.
        :type args: list

        :param kwargs: arbitrary keyword arguments.
        :type kwargs: dict

        :return: self
        :rtype: PredictorEnsemble
        """

        self._check_nomenclature(nomenclature_size, nomenclature_group)
        x = x.copy()
        # search and fit best regressor
        if search_estimator or self.estimator is None:
            self.estimator, self.best_score = self.search_best_estimator(x, None, nomenclature_size, nomenclature_group)

        _, predict_idx = self._get_train_predict_indexes(x, 0.5)
        train_idx = x.index.unique()
        x_train = self._fit_predict_forecasters(x.copy(), self.forecasters, train_idx, predict_idx,
                                                nomenclature_size, nomenclature_group)
        x_train.index = predict_idx
        x, y = x_train, x.query('(Nomenclature_size=={})&(Nomenclature_group=={})'.format(nomenclature_size,
                                                                                          nomenclature_group)
                                ).loc[predict_idx, 'Count']
        self.estimator.fit(x.reindex(sorted(x.columns), axis=1), y)
        return self

    def predict(self, h: int, nomenclature_size: int, nomenclature_group: int) -> ndarray:
        """
        Make a forecast to h steps forward.

        :param h: forecast horizon
        :type h: int

        :param nomenclature_size: encoded number of sequence from the whole time series.
        :type nomenclature_size: int

        :param nomenclature_group: an auxiliary parameter for some additional features.
        :type nomenclature_group: int

        :return: list of predictions.
        :rtype: ndarray
        """
        self._check_nomenclature(nomenclature_size, nomenclature_group)
        x = self._predict_forecasters(forecasters=self.forecasters, h=h,
                                      nomenclature_size=nomenclature_size,
                                      nomenclature_group=nomenclature_group)
        return positive_round(self.estimator.predict(x.reindex(sorted(x.columns), axis=1)))

    def search_best_estimator(self, x: DataFrame, y: Series = None, nomenclature_size: int = -1,
                              nomenclature_group: int = -1):
        """
        Search an optimal estimator to make an ensemble.

        :param x: time series.
        :type x: DataFrame

        :param y: target.
        :type y: Series

        :param nomenclature_size: encoded number of sequence from the whole time series.
        :type nomenclature_size: int

        :param nomenclature_group: an auxiliary parameter for some additional features.
        :type nomenclature_group: int

        :return: The best estimator with it's score.
        :rtype:
        """
        self._check_nomenclature(nomenclature_size, nomenclature_group)
        y = y if y is not None else x.query('Nomenclature_size == {}'.format(nomenclature_size)).loc[:, 'Count']
        estimators = self.estimators.copy()
        forecasters = self.forecasters.copy()
        train_idx, predict_idx = self._get_train_predict_indexes(x, 0.7)

        # fit forecasters at the train data and create predictions
        x_train = self._fit_predict_forecasters(x, forecasters, train_idx, predict_idx,
                                                nomenclature_size, nomenclature_group)
        y_train = y.loc[predict_idx]
        y_train.index = x_train.index

        # grid-search the best scored estimator with it's parameters.
        df_metrics = DataFrame(None, columns=['estimator', 'metric'])
        for estimator in estimators:
            tune1, tune2 = RegressorTuner(x_train, y_train, estimator,
                                          get_param_grid(estimator(), 'Kats')).search_best_parameters()
            df_metrics.loc[df_metrics.shape[0], ['estimator', 'metric']] = [tune1, tune2]

        idx = df_metrics.loc[:, 'metric'].astype(float).idxmin()
        return df_metrics.loc[idx, 'estimator'], df_metrics.loc[idx, 'metric']

    def get_score(self, *args) -> float:
        """Get the best score of tne nested estimator."""
        return self.best_score

    def _get_train_predict_indexes(self, x: DataFrame, idx_split: float) -> Tuple[list, list]:
        """
        Split data's indices to a train and test parts.

        :param x: the whole data to split.
        :type x: DataFrame

        :param idx_split: share of a train part.
        :type idx_split: float

        :return: lists of train ant test indices.
        :rtype: Tuple[list, list]
        """
        idxs = x.index.unique().sort_values()
        return idxs[:int(len(idxs) * idx_split)], idxs[int(len(idxs) * idx_split):]

    def _fit_forecasters(self, x: DataFrame, forecasters: dict,
                         train_idx: list, nomenclature_size: int) -> PredictorEnsemble:
        """Fit nested forecasters with the given data."""
        [forecaster.fit(fit_preparation(forecaster, x.loc[train_idx, :], nomenclature_size),
                        search_estimator=False, nomenclature_size=nomenclature_size)
         for _, forecaster in forecasters.items()]
        return self

    def _predict_forecasters(self, forecasters: dict, h: int,
                             nomenclature_size: int, nomenclature_group: int) -> DataFrame:
        """Each forecaster builds a prediction to h steps forward. This information combines in one table and
        inherit to the main estimator."""
        x_train = DataFrame(None, columns=forecasters.keys())
        for key, forecaster in forecasters.items():
            x_train.loc[:, key] = forecaster.predict(h, nomenclature_size, nomenclature_group)
        return x_train

    def _fit_predict_forecasters(self, x, forecasters, train_idx, predict_idx,
                                 nomenclature_size, nomenclature_group):
        """Combination of the fit and predict functions"""
        return self._fit_forecasters(x, forecasters, train_idx, nomenclature_size
                                     )._predict_forecasters(forecasters, len(predict_idx),
                                                            nomenclature_size, nomenclature_group)
        # TODO: Make PredictorRegression get correct predictions in ensemble.
        # In the one hand, PredictorRegression needs the whole data to make a correct prediction to the
        # next step. On the other hand, in production it's impossible, so PredictorEnsemble can
        # makes this regressor too important, cause in the train data it's predictions are too good.

        # x_train = DataFrame(None, columns=forecasters.keys(), index=predict_idx)
        # for key, forecaster in forecasters.items():
        #     if isinstance(forecaster, PredictorRegression):
        #         for idx in predict_idx:
        #             forecaster.fit(fit_preparation(forecaster,
        #                                            x.loc[x.index < idx, :], nomenclature_size),
        #                            search_estimator=False, nomenclature_size=nomenclature_size)
        #             x_train.at[idx, key] = forecaster.predict(1, nomenclature_size, nomenclature_group)[0]
        #     else:
        #         forecaster.fit(fit_preparation(forecaster, x.loc[train_idx, :], nomenclature_size),
        #                        search_estimator=False, nomenclature_size=nomenclature_size)
        #         x_train.loc[:, key] = forecaster.predict(len(predict_idx),
        #                                                  nomenclature_size, nomenclature_group)
        # return x_train.astype(float)

    def _check_nomenclature(self, nomenclature_size: int, nomenclature_group: int = -1):
        """Check incoming specifications of the time series."""
        if (nomenclature_size < 0) or (nomenclature_group < 0):
            raise ValueError("Nomenclature's size and group must be meaningful.")


def fit_preparation(predictor: SalesPredictor, x: DataFrame,
                    nomenclature_size: int = -1) -> DataFrame:
    """
    Prepare data according to the expected behaviour of the predictor.
    May transform multivariate series to a univariate.

    :param predictor: data consumer.
    :type predictor: SalesPredictor

    :param x: data to transform, multivariate time series.
    :type x: DataFrame

    :param nomenclature_size: the serial number of the one sequence from the whole multivariate series.
    :type nomenclature_size: int

    :return: transformed data.
    :rtype: DataFrame
    """
    if isinstance(predictor, (PredictorResiduals, PredictorDeseasonal)):
        return x.query('Nomenclature_size == {}'.format(nomenclature_size)).sort_index().asfreq('M').fillna(0)
    else:
        return x


def positive_round(x: ndarray) -> ndarray:
    """
    Change incoming value from float to positive integer.

    :param x: list of floats.
    :type x: list-like

    :return: list of positive integers.
    :rtype: ndarray
    """
    return array(list(map(lambda value: max(value, 0), around(x))))
