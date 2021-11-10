"""
API for multivariate forecast.

The main function: make predictions to a several steps forward from the last date in the fitted data.
"""
import os
import logging
import warnings
import pandas as pd
from config import settings
from datetime import datetime
from timelibs.app import app
from timelibs.timepredictor import SalesCompound, load_pickle, save_pickle

if __name__ == '__main__':
    #
    #cmp = SalesCompound()
    #cmp.initial(pd.read_csv(os.path.join('data/', "sales.csv"),
    #                    sep=";", parse_dates=['Date'], dayfirst=True))
    #save_pickle(cmp, 'compound_short')

    cmp = load_pickle('compound_short')
    # for key, value in cmp.size_predictors.items():
    #     t = cmp.encoders.get('Nomenclature_size').inverse_transform([key])
    #     print('{}: {}'.format(cmp.encoders.get('Nomenclature_size').inverse_transform([key])[0],
    #                           value))
    #preds = cmp.size_predictors.get(24).predict(3, 24, cmp.all_sizes.query('Nomenclature_size==24').Nomenclature_group.values[0])

    # cmp.fit(pd.read_csv(os.path.join('data/', "sales.csv"),
    #                     sep=";", parse_dates=['Date'], dayfirst=True))
    preds = cmp.predict(3)
    print(preds)
