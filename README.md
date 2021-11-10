## Second look at the multinomial time series.

A second approach to the problem of predicting multiple series, the first one is [here](https://github.com/RodinaMariia/timeseries-sample).
As before it was desided to build independent model to the each sequence of the whole dataset. It's possible to split all series to a several groups and create one model to a one cluster, but it provides decreasing the forecast accuracy.

Module *timemodel* implements different forecast models, more complex than before. Such approaches are used as 
* Making basic forecast and predict residuals separetely for a one time sequense;
* Detrend and deseasonal univariate time series and on the remaining data calculate a regression;
* Using vector autoregressive model to the multivariate series;
* Extract statistical features from the whole data and build a simple regression on it;
* Ensemble from the above models;

