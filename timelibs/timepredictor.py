"""
Main manager for multivariate time series forecasting.

Basic manager class works with different predictive methods implemented SalesPredictor interface.
For each entry of multivariate time series manager builds set of competing strategies and
cross-validate them with the recent records of series. Then choose the best scoring model and use it for a
future forecasting.

Finally we keep a set of associated values of multivariate keys (Nomenclature_size) and it's forecasters.
"""
from __future__ import annotations
import os
import dill as pickle
from datetime import datetime
from pandas import DataFrame
from timelibs.app import app
from timelibs.timemodel import (PredictorDeseasonal, PredictorResiduals, PredictorRegression,
                                PredictorVAR, PredictorEnsemble, fit_preparation)
from utils4sales.utils import LabelEncoderExt, preprocess_multivariate_data


class SalesCompound:
    """
    Class works with multivariate time series. The whole data splits at the
    separate sequences and to each univariate sequence builds it's own forecaster.
    At first it's necessary to initialize manager: search the best forecaster from
    the proposed options, create and fit encoders to categorical features,
    make bindings inside incoming data. Then linked with specific sequences
    forecasters can be fitted and ba able to make a prediction.

    Attributes:
        encoders: dict[str: LabelEncoderExt],
            Encoders to categorical features with feature's names.

        size_predictors: dict[int: SalesPredictor],
            Encoded serial numbers of nomenclatures with predictors.

        all_sizes: DataFrame (default is None),
            Binding between nomenclatures and analytic groups.
    """
    def __init__(self):
        self.encoders = {}
        self.size_predictors = {}
        self.all_sizes = None

    def initial(self, data: DataFrame) -> SalesCompound:
        """
        Function makes basic preparation to a future work: select the best scored estimator implementing
        SalesPredictor interface, choose optimal parameters and features (if necessary), create and fit encoders.

        :param data: raw multivariate data.
        :type data: DataFrame

        :return: self
        :rtype: SalesCompound
        """
        data = self._create_encoders(self.preprocess(data.copy()))
        all_models = dict(zip(['Deseasonal', 'Residuals', 'Var'],
                              [PredictorDeseasonal, PredictorResiduals,  PredictorVAR]))
        #
        # all_models = dict(zip(['Regression'],
        #                       [PredictorRegression]))
        for idx, row in self.all_sizes.iterrows():
            # noinspection PyBroadException
            try:
                df_metrics = DataFrame(None, columns=['estimator', 'metric'])
                print(row.Nomenclature_size)
                # search scores of basic predictors
                for _, model in all_models.items():
                    try:
                        print(model)
                        estimator = model()
                        estimator.fit(fit_preparation(estimator, data, row.Nomenclature_size),
                                      search_estimator=True,
                                      nomenclature_size=row.Nomenclature_size)
                        df_metrics.loc[df_metrics.shape[0], ['estimator', 'metric']] = \
                            [estimator, estimator.get_score(row.Nomenclature_size)]
                    except Exception as e:
                        app.logger.debug(
                            "{}: model {} search error at size {}, {}".format(
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model, row.Nomenclature_size,  e))
                # search score of an ensemble method
                print('PredictorEnsemble')
                estimator = PredictorEnsemble(dict(zip(all_models.keys(),
                                                       df_metrics.loc[:, 'estimator'].copy(deep=True))))
                estimator.fit(data, search_estimator=True, nomenclature_size=row.Nomenclature_size,
                              nomenclature_group=row.Nomenclature_group)
                df_metrics.loc[df_metrics.shape[0], ['estimator', 'metric']] = [estimator, estimator.get_score()]

                # set the best estimator as main to the current time sequence
                idx = df_metrics.loc[:, 'metric'].astype(float).idxmin()
                self.size_predictors[row.Nomenclature_size] = df_metrics.loc[idx, 'estimator']
            except Exception as e:
                app.logger.debug("{}: initial error at size {}. {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                                           row.Nomenclature_size, e))
        return self

    def preprocess(self, data: DataFrame) -> DataFrame:
        """
        Standard data preparation: creates time-indices, smooths negative values,
        fills missing dates.

        :param data: raw multivariate data.
        :type data: DataFrame

        :return: transformed data.
        :rtype: DataFrame
        """
        return preprocess_multivariate_data(data, freq='M', value=0)

    def fit(self, x: DataFrame) -> SalesCompound:
        """
        Fit previously initialized estimators with a new data.

        :param x: a new multivariate data to fit.
        :type x: DataFrame

        :return: self
        :rtype: SalesCompound
        """
        if len(self.size_predictors) == 0:
            raise AttributeError("Class must be initialized manually by 'initial' function!")

        x = self._create_encoders(self.preprocess(x))
        [estimator.fit(fit_preparation(estimator, x, key), search_estimator=False,
                       nomenclature_size=key,
                       nomenclature_group=self.all_sizes.query('Nomenclature_size=={}'.format(key)
                                                               ).Nomenclature_group.values[0])
         for key, estimator in self.size_predictors.items()]
        return self

    def predict(self, h: int) -> dict:
        """
        Make predictions to h steps forward to the each sequence from the whole data.

        :param h: number of steps
        :type h: int

        :return: nomenclature's names with predictions
        :rtype: dict
        """
        if len(self.size_predictors) == 0:
            raise AttributeError("Class must be initialized manually by 'initial' function!")
        enc = self.encoders.get('Nomenclature_size')
        preds = {}
        for key, estimator in self.size_predictors.items():
            try:
                preds[enc.inverse_transform([key])[0]] = \
                    estimator.predict(h, key, self.all_sizes.query('Nomenclature_size=={}'.format(key)
                                                                   ).Nomenclature_group.values[0])
            except Exception as e:
                app.logger.debug("{}: predict error at size {}. {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                                           enc.inverse_transform([key])[0], e))
        return preds

    def _create_encoders(self, data: DataFrame) -> DataFrame:
        """Create encoders to categorical data then replace original information with transformed.
        Also memorize bounds between nomenclatures and analytic groups."""
        enc_names = ['Nomenclature_size', 'Nomenclature_group']
        for name in enc_names:
            self.encoders[name] = LabelEncoderExt().fit(data[name].unique())
            data.loc[:, name] = self.encoders.get(name).transform(data[name])
        self.all_sizes = data.loc[:, ['Nomenclature_size', 'Nomenclature_group']].drop_duplicates()
        return data


def save_pickle(file, name):
    """Util for saving pickle-files"""
    with open(os.path.join(os.path.dirname(__file__), '..', 'data/', name + '.pickle'), 'wb') as f:
        pickle.dump(file, f)


def load_pickle(name: str):
    """Util for loading pickle-files"""
    with open(os.path.join(os.path.dirname(__file__), '..', 'data/', name + '.pickle'), 'rb') as f:
        return pickle.load(f)
