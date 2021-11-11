"""
Grid searching attendant module. Contain parameters to tune in different notations.
Module support searching with libs: Kats and Sktime.

It's important to modify the global getter when the new notation adds.
"""
from __future__ import annotations
from abc import ABCMeta, abstractmethod
from math import exp
from typing import TypeVar, Union

T = TypeVar('T')


class ParameterBridge:
    """
    Main interface for all parameter handlers.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_param_grid(self) -> Union[list, dict]:
        """
        Return estimator's tuning parameters in specific notation.

        :return: formatted parameters kit.
        :rtype: list, dict
        """
        pass


class ParameterSktimeLGBMRegressor(ParameterBridge):
    """
    Parameters grid to the LGBMRegressor class with function ForecastingGridSearchCV from sktime.
    """
    def get_param_grid(self) -> dict:
        """This method overrides :meth:`ParameterBridge.get_param_grid`."""
        return {'estimator__boosting_type': ['gbdt', 'dart', 'goss', 'rf'],
                'estimator__num_leaves': range(5, 35, 5),
                'estimator__max_depth': [-1, 5, 10],        
                #
                # Block with some additional parameters to search.
                #
                }


class ParameterSktimeGradientBoostingRegressor(ParameterBridge):
    """
    Parameters grid to the GradientBoostingRegressor class with function ForecastingGridSearchCV from sktime.
    """
    def get_param_grid(self) -> dict:
        """This method overrides :meth:`ParameterBridge.get_param_grid`."""
        return {
                'estimator__n_estimators': [10, 50, 100, 300],
                'estimator__max_depth': [2, 5, 10],
                #
                # Block with some additional parameters to search.
                #
               }


class ParameterSktimeRandomForestRegressor(ParameterBridge):
    """
    Parameters grid to the RandomForestRegressor class with function ForecastingGridSearchCV from sktime.
    """
    def get_param_grid(self) -> dict:
        """This method overrides :meth:`ParameterBridge.get_param_grid`."""
        return {'estimator__n_estimators': [10, 50, 100, 300],
                'estimator__min_samples_split': [0.01, 0.2, 0.3],
                'estimator__min_samples_leaf': [0.001, 0.01, 0.1, 1]
                }


class ParameterSktimeElasticNet(ParameterBridge):
    """
    Parameters grid to the ElasticNet class with function ForecastingGridSearchCV from sktime.
    """
    def get_param_grid(self) -> dict:
        """This method overrides :meth:`ParameterBridge.get_param_grid`."""
        return {'estimator__alpha': [0.1, 0.5, 1.0, 1.5, 5],
                'estimator__l1_ratio': [0, 0.2, 0.5, 0.8, 1],
                #
                # Block with some additional parameters to search.
                #
                }


class ParameterSktimeARDRegression(ParameterBridge):
    """
    Parameters grid to the ARDRegression class with function ForecastingGridSearchCV from sktime.
    """
    def get_param_grid(self) -> dict:
        """This method overrides :meth:`ParameterBridge.get_param_grid`."""
        return {'estimator__n_iter': [10, 100, 300, 500],
                'estimator__tol': [exp(-3), exp(-2)],
                #
                # Block with some additional parameters to search.
                #
                }


# Kats block

class ParameterKatsLGBMRegressor(ParameterBridge):
    """
    Parameters grid to the LGBMRegressor class with SearchMethodFactory functions from kats.
    """
    def get_param_grid(self) -> list:
        """This method overrides :meth:`ParameterBridge.get_param_grid`."""
        return [
            {'name': 'boosting_type',
             'type': 'choice',
             'value_type': 'str',
             'values': ['gbdt', 'dart', 'goss', 'rf']
             },
             #
             # Block with some additional parameters to search.
             #
        ]


class ParameterKatsGradientBoostingRegressor(ParameterBridge):
    """
    Parameters grid to the GradientBoostingRegressor class with SearchMethodFactory functions from kats.
    """
    def get_param_grid(self) -> list:
        """This method overrides :meth:`ParameterBridge.get_param_grid`."""
        return [
                {'name': 'n_estimators',
                 'type': 'choice',
                 'value_type': 'int',
                 'values': [10, 50, 100, 300]},
                {'name': 'max_depth',
                 'type': 'choice',
                 'value_type': 'int',
                 'values': [2, 5, 10]},
                #
                # Block with some additional parameters to search.
                #
                ]


class ParameterKatsRandomForestRegressor(ParameterBridge):
    """
    Parameters grid to the RandomForestRegressor class with SearchMethodFactory functions from kats.
    """
    def get_param_grid(self) -> list:
        """This method overrides :meth:`ParameterBridge.get_param_grid`."""
        return [
            {'name': 'n_estimators',
             'type': 'choice',
             'value_type': 'int',
             'values': [10, 50, 100, 300]
             },
             #
             # Block with some additional parameters to search.
             #
        ]


class ParameterKatsElasticNet(ParameterBridge):
    """
    Parameters grid to the ElasticNet class with SearchMethodFactory functions from kats.
    """
    def get_param_grid(self) -> list:
        """This method overrides :meth:`ParameterBridge.get_param_grid`."""
        return [
            {'name': 'alpha',
             'type': 'choice',
             'value_type': 'float',
             'values': [0.1, 0.5, 1.0]
             },
            {'name': 'l1_ratio',
             'type': 'choice',
             'value_type': 'float',
             'values': [0, 0.2, 0.5, 0.8, 1]
             },
             #
             # Block with some additional parameters to search.
             #
        ]


class ParameterKatsARDRegression(ParameterBridge):
    """
    Parameters grid to the ARDRegression class with SearchMethodFactory functions from kats.
    """
    def get_param_grid(self) -> list:
        """This method overrides :meth:`ParameterBridge.get_param_grid`."""
        return [
            {'name': 'n_iter',
             'type': 'choice',
             'value_type': 'int',
             'values': [10, 100, 300, 500]
             },
            {'name': 'tol',
             'type': 'choice',
             'value_type': 'float',
             'values': [exp(-3), exp(-2)]
             },
             #
             # Block with some additional parameters to search.
             #
        ]


class ParameterKatsCatBoostRegressor(ParameterBridge):
    """
    Parameters grid to the CatBoostRegressor class with SearchMethodFactory functions from kats.
    """
    def get_param_grid(self) -> list:
        """This method overrides :meth:`ParameterBridge.get_param_grid`."""
        return [
            {'name': 'bootstrap_type',
             'type': 'choice',
             'value_type': 'str',
             'values': ["Bayesian", "Bernoulli", "MVS"]
             },
             #
             # Block with some additional parameters to search.
             #
            {'name': 'verbose',
             'type': 'choice',
             'value_type': 'bool',
             'values':  [False]
             },
        ]


def get_param_grid(estimator: T, optimizer: str) -> Union[list, dict]:
    """
    Global getter redirecting estimators to possible parameters.

    :param estimator: estimator for further tuning.
    :type estimator: highly likely RegressorMixin

    :param optimizer: lib contains optimizing function.
    :type optimizer: str

    :return: estimator's parameters in some notation
    :rtype: list, dict
    """
    if optimizer not in ['Kats', 'Sktime']:
        raise ValueError('optimizer can be "Kats" or "Sktime"')
    klass = globals()['Parameter{}{}'.format(optimizer, estimator.__class__.__name__)]
    return klass().get_param_grid()
