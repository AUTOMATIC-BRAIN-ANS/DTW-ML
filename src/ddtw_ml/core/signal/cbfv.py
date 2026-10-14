"""
CBFV signal from the flow velocities in the left and right cerebral arteries.
"""

import pandas as pd

from ddtw_ml.core.signal.artefacts import filter_fv


def calculate_cbfv(df, col_fvl, col_fvr):
    """
    Function to calculate the CBFV signal based on the FV from left and right side.
    :param df: data in the DataFrame format.
    :param col_fvl: FV from the left side.
    :param col_fvr: FV from the right side.
    :return: CBFV signal.
    """
    fvl, fvr = filter_fv(df, col_fvl), filter_fv(df, col_fvr)
    length = min(len(fvl), len(fvr))
    cbfv = []
    for i in range(0, length):
        if fvl[i] > fvr[i]:
            cbfv.append(fvl[i])
        else:
            cbfv.append(fvr[i])
    return pd.Series(cbfv)
