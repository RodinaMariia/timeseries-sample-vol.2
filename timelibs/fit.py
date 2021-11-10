"""
Wrapper for fitting and initializing functions.
"""
import os
from pandas import read_csv
from timelibs.timepredictor import SalesCompound, load_pickle, save_pickle


def initial():
    """Create new manager and initialize it."""
    manager = SalesCompound()
    manager.initial(read_csv(os.path.join('data/', "sales.csv"),
                             sep=";",
                             parse_dates=['Date'],
                             dayfirst=True)
                    )
    save_pickle(manager, 'compound')
    return manager


def fit():
    """Fit existing manager. New manages creates if the old one can't be fitted."""
    # noinspection PyBroadException
    try:
        manager = load_pickle('compound')
        manager.fit(read_csv(os.path.join('data/', "sales.csv"),
                             sep=";",
                             parse_dates=['Date'],
                             dayfirst=True))
        save_pickle(manager, 'compound')
    except Exception as e:
        print(e)
        initial()
