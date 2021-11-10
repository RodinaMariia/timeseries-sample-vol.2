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
                # 'estimator__learning_rate': [0.1, 0.01, 0.001],
                # 'estimator__n_estimators': [100, 1000, 10000],
                # 'estimator__reg_alpha': [0, 0.3, 0.6, 0.9],
                # 'estimator__reg_lambda': [0, 0.3, 0.6, 0.9],
                # 'estimator__min_child_weight': [math.exp(-1), math.exp(-3), math.exp(-5)],
                # #'estimator__min_child_samples': range(10, 40, 10)
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
                # 'estimator__learning_rate': [0.001, 0.01, 0.1],
                # 'estimator__subsample': [0.1, 0.5, 1, 2],
                # 'estimator__min_samples_split': [0.0001, 0.01, 0.1, 0.2, 1],
                # 'estimator__min_samples_leaf': [0.0001, 0.01, 0.1, 0.5, 1],
                # 'estimator__min_impurity_decrease': [0.0001, 0.01, 0.1, 0.9, 0],
                # 'estimator__alpha': [0.2, 0.5, 0.9],
                # 'estimator__criterion': ['friedman_mse', 'mse'],
                # 'estimator__loss': ['ls', 'lad', 'huber', 'quantile']
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
                # 'estimator__max_iter': [10, 50, 100],
                # 'estimator__tol': [exp(-2), exp(-4), exp(-6)],
                # 'estimator__random_state': [42],
                # 'estimator__selection': ['cyclic', 'random']
                }


class ParameterSktimeARDRegression(ParameterBridge):
    """
    Parameters grid to the ARDRegression class with function ForecastingGridSearchCV from sktime.
    """
    def get_param_grid(self) -> dict:
        """This method overrides :meth:`ParameterBridge.get_param_grid`."""
        return {'estimator__n_iter': [10, 100, 300, 500],
                'estimator__tol': [exp(-3), exp(-2)],
                # 'alpha_1': [exp(-7), exp(-6), exp(-5), exp(-4)],
                # 'alpha_2': [exp(-7), exp(-6), exp(-5), exp(-4)],
                # 'lambda_1': [exp(-7), exp(-6), exp(-5), exp(-4)],
                # 'lambda_2': [exp(-7), exp(-6), exp(-5), exp(-4)],
                # 'fit_intercept': [False]
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
            # {'name': 'num_leaves',
            #  'type': 'choice',
            #  'value_type': 'int',
            #  'values': [5, 15, 30]
            #  },
            # {'name': 'max_depth',
            #  'type': 'choice',
            #  'value_type': 'int',
            #  'values': [-1, 5, 10]
            #  },
            # {'name': 'learning_rate',
            #  'type': 'choice',
            #  'value_type': 'float',
            #  'values': [0.1, 0.01, 0.001]
            #  },
            # {'name': 'n_estimators',
            #  'type': 'choice',
            #  'value_type': 'int',
            #  'values': [100, 1000, 10000]
            #  },
            # {'name': 'reg_alpha',
            #  'type': 'choice',
            #  'value_type': 'float',
            #  'values': [0, 0.3, 0.6, 0.9]
            #  },
            # {'name': 'reg_lambda',
            #  'type': 'choice',
            #  'value_type': 'float',
            #  'values': [0, 0.3, 0.6, 0.9]
            #  },
            # {'name': 'min_child_weight',
            #  'type': 'choice',
            #  'value_type': 'float',
            #  'values': [math.exp(-1), math.exp(-3), math.exp(-5)]
            #  },
            # {'name': 'min_child_samples',
            #  'type': 'choice',
            #  'value_type': 'int',
            #  'values': [5, 15, 35]
            #  },
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
                # {'name': 'learning_rate',
                #  'type': 'choice',
                #  'value_type': 'float',
                #  'values': [0.001, 0.01, 0.1]},
                # {'name': 'subsample',
                #  'type': 'choice',
                #  'value_type': 'float',
                #  'values': [0.1, 0.5, 1, 2]},
                # {'name': 'min_samples_split',
                #  'type': 'choice',
                #  'value_type': 'float',
                #  'values': [0.0001, 0.01, 0.1, 0.2, 1]},
                # {'name': 'min_samples_leaf',
                #  'type': 'choice',
                #  'value_type': 'float',
                #  'values': [0.0001, 0.01, 0.1, 0.5, 1]},
                # {'name': 'min_impurity_decrease',
                #  'type': 'choice',
                #  'value_type': 'float',
                #  'values': [0.0001, 0.01, 0.1, 0.9, 0]},
                # {'name': 'alpha',
                #  'type': 'choice',
                #  'value_type': 'int',
                #  'values': [0.2, 0.5, 0.9]},
                # {'name': 'criterion',
                #  'type': 'choice',
                #  'value_type': 'str',
                #  'values': ['friedman_mse', 'mse']},
                # {'name': 'loss',
                #  'type': 'choice',
                #  'value_type': 'int',
                #  'values': ['ls', 'lad', 'huber', 'quantile']}
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
            # {'name': 'criterion',
            #  'type': 'choice',
            #  'value_type': 'str',
            #  'values': ['gini', 'entropy']
            #  },
            # {'name': 'min_samples_split',
            #  'type': 'choice',
            #  'value_type': 'float',
            #  'values': [0.01, 0.2, 0.3]
            #  },
            # {'name': 'min_samples_leaf',
            #  'type': 'choice',
            #  'value_type': 'float',
            #  'values': [0.001, 0.01, 0.1, 1]
            #  }
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
            # {'name': 'max_iter',
            #  'type': 'choice',
            #  'value_type': 'int',
            #  'values': [10, 50, 100]
            #  },
            # {'name': 'tol',
            #  'type': 'choice',
            #  'value_type': 'float',
            #  'values': [exp(-2), exp(-4), exp(-6)]
            #  },
            # {'name': 'random_state',
            #  'type': 'choice',
            #  'value_type': 'int',
            #  'values': [42]
            #  },
            # {'name': 'selection',
            #  'type': 'choice',
            #  'value_type': 'str',
            #  'values': ['cyclic', 'random']
            #  }
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
            # {'name': 'alpha_1',
            #  'type': 'choice',
            #  'value_type': 'float',
            #  'values': [exp(-7), exp(-6), exp(-5), exp(-4)]
            #  },
            # {'name': 'alpha_2',
            #  'type': 'choice',
            #  'value_type': 'float',
            #  'values': [exp(-7), exp(-6), exp(-5), exp(-4)]
            #  },
            # {'name': 'lambda_1',
            #  'type': 'choice',
            #  'value_type': 'float',
            #  'values': [exp(-7), exp(-6), exp(-5), exp(-4)]
            #  },
            # {'name': 'lambda_2',
            #  'type': 'choice',
            #  'value_type': 'float',
            #  'values': [exp(-7), exp(-6), exp(-5), exp(-4)]
            #  },
            # {'name': 'fit_intercept',
            #  'type': 'choice',
            #  'value_type': 'bool',
            #  'values': [False]
            #  },
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
            # {'name': 'depth',
            #  'type': 'choice',
            #  'value_type': 'int',
            #  'values': [2, 5, 10, 20]
            #  },
            # {'name': 'iterations',
            #  'type': 'choice',
            #  'value_type': 'int',
            #  'values': [10, 100, 300, 1000]
            #  },
            # {'name': 'grow_policy',
            #  'type': 'choice',
            #  'value_type': 'str',
            #  'values': ["SymmetricTree", "Depthwise", "Lossguide"]
            #  },
            # {'name': 'loss_function',
            #  'type': 'choice',
            #  'value_type': 'str',
            #  'values': ['Quantile:alpha=0.7', 'Quantile:alpha=0.2', 'RMSE']
            #  },
            # {'name': 'learning_rate',
            #  'type': 'choice',
            #  'value_type': 'float',
            #  'values': [0.1, 0.001, 0.0001]
            #  },
            # {'name': 'l2_leaf_reg',
            #  'type': 'choice',
            #  'value_type': 'int',
            #  'values': [0, 2, 4, 6]
            #  },
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
