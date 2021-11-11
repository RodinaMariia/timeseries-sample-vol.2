## Second look at the multinomial time series.

A second approach to the problem of predicting multiple series, the first one is [here](https://github.com/RodinaMariia/timeseries-sample).
As before it was desided to build independent model to the each sequence of the whole dataset. It's possible to split all series to a several groups and create to the one cluster it's own model, but it provides decreasing the forecast accuracy with a small gain in time.

The first step is to create the main manager from *timepredictor*, which in turn splits the whole set of sequenses into seperate parts and search an optimal forecasting model for an each one. When all univariate time series bounded with its forecasters, there's become possible to built separate predictions or fit forecasters with a new data. To find the optimal forecaster, manager uses competing models from the *timemodel* module which implements different forecast models, more complex than before. Such approaches are used as 
* Making basic forecast and predict residuals separetely for a one time sequense;
* Detrend and deseasonal univariate time series and on the remaining data calculate a regression;
* Using vector autoregressive model to the multivariate series;
* Extract statistical features from the whole data and build a simple regression on it;
* Making an ensemble from the previous models.

Some parts of the code have been removed.

**For reference only.** 

