"""
Monkey pathing module.

Contains patches to the sktime library in the part of cross-validation strategy.
"""
import time
import timelibs
from joblib import Parallel, delayed
from numpy import nan
from pandas import DataFrame
from sklearn.base import clone
from sklearn.model_selection import check_cv as check_cv_sklearn
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_evaluation._functions import _check_strategy, _split
from sktime.utils.validation.forecasting import check_scoring, check_cv, check_y_X


def new_fit(self, y, X=None, fh=None, **fit_params):
    """
    Patch for BaseGridSearch._fit function. Catch errors from the 'evaluate' function and return a default DataFrame to
    ensure that calculations are continue.

    Fit to training data.

    Parameters
    ----------
    y : pd.Series
        Target time series to which to fit the forecaster.
    fh : int, list or np.array, optional (default=None)
        The forecasters horizon with the steps ahead to to predict.
    X : pd.DataFrame, optional (default=None)
        Exogenous variables are ignored

    Returns
    -------
    self : returns an instance of self.
    """
    cv = check_cv_sklearn(self.cv)

    scoring = check_scoring(self.scoring)
    scoring_name = f"test_{scoring.name}"

    parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

    def _fit_and_score(params):
        # Clone forecaster.
        forecaster = clone(self.forecaster)

        # Set parameters.
        forecaster.set_params(**params)

        # Evaluate.
        try:
            out = rounded_evaluate(
                forecaster,
                cv,
                y,
                X,
                strategy=self.strategy,
                scoring=scoring,
                fit_params=fit_params,
            )
        except Exception as e:
            print(e)
            out = DataFrame({scoring_name: [nan], 'fit_time': [1], 'pred_time': [1]})

        # Filter columns.
        out = out.filter(items=[scoring_name, "fit_time", "pred_time"], axis=1)

        # Aggregate results.
        out = out.mean()
        out = out.add_prefix("mean_")

        # Add parameters to output table.
        out["params"] = params

        return out

    def evaluate_candidates(candidate_params):
        candidate_params = list(candidate_params)

        if self.verbose > 0:
            n_candidates = len(candidate_params)
            n_splits = cv.get_n_splits(y)
            print(  # noqa
                "Fitting {0} folds for each of {1} candidates,"
                " totalling {2} fits".format(
                    n_splits, n_candidates, n_candidates * n_splits
                )
            )

        out = parallel(
            delayed(_fit_and_score)(params) for params in candidate_params
        )

        if len(out) < 1:
            raise ValueError(
                "No fits were performed. "
                "Was the CV iterator empty? "
                "Were there no candidates?"
            )

        return out

    # Run grid-search cross-validation.
    results = self._run_search(evaluate_candidates)

    results = DataFrame(results)

    # Rank results, according to whether greater is better for the given scoring.
    results[f"rank_{scoring_name}"] = results.loc[:, f"mean_{scoring_name}"].rank(
        ascending=not scoring.greater_is_better
    )

    self.cv_results_ = results

    # Select best parameters.
    self.best_index_ = results.loc[:, f"rank_{scoring_name}"].argmin()
    self.best_score_ = results.loc[self.best_index_, f"mean_{scoring_name}"]
    self.best_params_ = results.loc[self.best_index_, "params"]
    self.best_forecaster_ = clone(self.forecaster).set_params(**self.best_params_)

    # Refit model with best parameters.
    if self.refit:
        self.best_forecaster_.fit(y, X, fh)

    # Sort values according to rank
    results = results.sort_values(
        by=f"rank_{scoring_name}", ascending=not scoring.greater_is_better
    )
    # Select n best forecaster
    self.n_best_forecasters_ = []
    self.n_best_scores_ = []
    for i in range(self.return_n_best_forecasters):
        params = results["params"].iloc[i]
        rank = results[f"rank_{scoring_name}"].iloc[i]
        rank = str(int(rank))
        forecaster = clone(self.forecaster).set_params(**params)
        # Refit model with best parameters.
        if self.refit:
            forecaster.fit(y, X, fh)
        self.n_best_forecasters_.append((rank, forecaster))
        # Save score
        score = results[f"mean_{scoring_name}"].iloc[i]
        self.n_best_scores_.append(score)

    return self


def rounded_evaluate(
    forecaster,
    cv,
    y,
    X=None,
    strategy="refit",
    scoring=None,
    fit_params=None,
    return_data=False,
):
    """
    Patch for BaseGridSearch - evaluate function. Makes predictions positive integer to ensure finding the optimal
    parameters with this condition.

    Evaluate forecaster using timeseries cross-validation.

    Parameters
    ----------
    forecaster : sktime.forecaster
        Any forecaster
    cv : Temporal cross-validation splitter
        Splitter of how to split the data into test data and train data
    y : pd.Series
        Target time series to which to fit the forecaster.
    X : pd.DataFrame, default=None
        Exogenous variables
    strategy : {"refit", "update"}
        Must be "refit" or "update". The strategy defines whether the `forecaster` is
        only fitted on the first train window data and then updated, or always refitted.
    scoring : subclass of sktime.performance_metrics.BaseMetric, default=None.
        Used to get a score function that takes y_pred and y_test arguments
        and accept y_train as keyword argument.
        If None, then uses scoring = MeanAbsolutePercentageError(symmetric=True).
    fit_params : dict, default=None
        Parameters passed to the `fit` call of the forecaster.
    return_data : bool, default=False
        Returns three additional columns in the DataFrame, by default False.
        The cells of the columns contain each a pd.Series for y_train,
        y_pred, y_test.

    Returns
    -------
    pd.DataFrame
        DataFrame that contains several columns with information regarding each
        refit/update and prediction of the forecaster.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.model_evaluation import evaluate
    >>> from sktime.forecasting.model_selection import ExpandingWindowSplitter
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster(strategy="mean", sp=12)
    >>> cv = ExpandingWindowSplitter(
    ...     initial_window=24,
    ...     step_length=12,
    ...     fh=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    >>> results = evaluate(forecaster=forecaster, y=y, cv=cv)
    """
    _check_strategy(strategy)
    cv = check_cv(cv, enforce_start_with_window=True)
    scoring = check_scoring(scoring)
    y, X = check_y_X(y, X)
    fit_params = {} if fit_params is None else fit_params

    # Define score name.
    score_name = "test_" + scoring.name

    # Initialize dataframe.
    results = DataFrame()

    # Run temporal cross-validation.
    for i, (train, test) in enumerate(cv.split(y)):
        # split data
        y_train, y_test, X_train, X_test = _split(y, X, train, test, cv.fh)

        # create forecasting horizon
        fh = ForecastingHorizon(y_test.index, is_relative=False)

        # fit/update
        start_fit = time.time()
        if i == 0 or strategy == "refit":
            forecaster.fit(y_train, X_train, fh=fh, **fit_params)

        else:  # if strategy == "update":
            forecaster.update(y_train, X_train)
        fit_time = time.time() - start_fit

        # predict
        start_pred = time.time()
        y_pred = forecaster.predict(fh, X=X_test)
        y_pred.loc[:] = timelibs.positive_round(y_pred.values)
        pred_time = time.time() - start_pred

        # score
        score = scoring(y_test, y_pred, y_train=y_train)

        # save results
        results = results.append(
            {
                score_name: score,
                "fit_time": fit_time,
                "pred_time": pred_time,
                "len_train_window": len(y_train),
                "cutoff": forecaster.cutoff,
                "y_train": y_train if return_data else nan,
                "y_test": y_test if return_data else nan,
                "y_pred": y_pred if return_data else nan,
            },
            ignore_index=True,
        )

    # post-processing of results
    if not return_data:
        results = results.drop(columns=["y_train", "y_test", "y_pred"])
    results["len_train_window"] = results["len_train_window"].astype(int)

    return results
