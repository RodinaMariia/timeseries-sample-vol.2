"""
Entrypoint for a fast calculations. Current example initialize manager with a some data and make predictions to a three steps forward.
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
    cmp = SalesCompound()
    cmp.initial(pd.read_csv(os.path.join('data/', "sales.csv"),
                        sep=";", parse_dates=['Date'], dayfirst=True))
    save_pickle(cmp, 'compound_short')    
    print(cmp.predict(3))
