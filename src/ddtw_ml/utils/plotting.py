"""
Common settings of the plots.
"""

import matplotlib.pyplot as plt


def use_latex():
    """
    Function to use LaTeX formatting for plots.
    """
    # use LaTeX for text rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({
        'text.latex.preamble': r'\usepackage[utf8]{inputenc} \usepackage[T1]{fontenc}'
    })
