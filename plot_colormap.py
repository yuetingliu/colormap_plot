import os
import sys
import logging

import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set the sans-serif, Arial fonts,
# Set the default mathtext as bold
from matplotlib import rcParams
rcParams['backend'] = 'agg'
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['axes.linewidth']=3
rcParams['mathtext.default']='bf'

# Set fontproperties
from matplotlib.font_manager import FontProperties
#     font1 = FontProperties()
#     font1.set_weight('bold')
#     font1.set_size(font_size)


log = logging.getLogger(__name__)


def get_config():
    """Get config file."""
    with open('config.yml', 'r') as stream:
        config = yaml.load(stream)
    return config


def load_excel(fn):
    """Load excel data file to a ordereddict."""
    odict = pd.read_excel(fn, sheet_name=None)
    data_odict = odict.copy()
    num_sheets = len(odict)
    # filter real data, some sheet might contain figures
    for sname, df in odict.items():
        if (not hasattr(df, 'columns')) or (df.columns[0] != 'Wavenumber'):
            data_odict.pop(sname)
            log.info("Remove non-data sheet '{}'".format(sname))
    log.info("{:d} sheets, {:d} data sheets: {}".format(
                 num_sheets, len(data_odict), data_odict.keys()))
    return data_odict


def preprocess(df, lower=None, upper=None):
    '''slice the wanted range.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to be plotted
    lower : int
        lower range of wavenumber
    upper : int
        upper range of wavenumber

    Returns
    -------
    X, Y, Z : tuple of float32
    '''
    k = df[df['Wavenumber']>=lower]      # Slice the wavenumber range using a, b
    k = k[k['Wavenumber']<=upper]
    x = k['Wavenumber']                  # Get X axis
    y = np.array(df.columns[1:], dtype='float32')  # Get y axis
    X, Y= np.meshgrid(x,y)               # Set the grid with meshgrid
    Z = k.iloc[:, 1:].astype(float)      # Get values and explictly set the data type as float values
    Z = Z.T                              # Transpose Z because X and Y are transposed
    return X, Y, Z                       # Return X,Y,Z


def plot_(X, Y, Z, figsize, font_size= 26, save=True,
          fn='plot', vmax=None):
    '''Plot pocolormesh figure,
    X,Y: the axis
    Z: the values
    xaxis,yaxis the figure size'''
    # set font for both x and y label
    font0 = FontProperties()
    font0.set_style('italic')
    font0.set_size(font_size)
    font0.set_weight('bold')

    fig = plt.figure(figsize=figsize)
    plt.pcolormesh(X, Y, Z, cmap=plt.cm.jet, vmax=vmax)
    plt.axis([X.max(), X.min(), Y.min(), Y.max()])
    plt.colorbar()
    plt.xlabel('Wavenumber  /  $cm^{-1}$', fontproperties=font0)
    plt.ylabel('Time / $s$', fontproperties=font0)
    plt.tick_params(axis='both', labelsize=26, top='off', right='off')
#   plt.yticks(fontweight='bold')
    if save == True:
        plt.savefig('{}.png'.format(fn), dpi=300)


def plot_fl(fl, save=True):
    config = get_config()
    x1, x2 = config['range']
    figsize = config['figsize']
    font_size = config['font_size']
    fn, _ = os.path.splitext(fl)

    # load excel to dataframe
    odict = load_excel(fl)
    for sn, df in odict.items():
        log.info("Plot sheet {}".format(sn))
        X, Y, Z = preprocess(df, x1, x2)
        save_fn = '{}_{}'.format(fn, sn)
        plot_(X, Y, Z, figsize, font_size, save=save, fn=save_fn, vmax=Z.values.max()*0.85)


def main():
    logging.basicConfig(
        stream=sys.stdout,
        format='[%(asctime)s %(levelname)s] %(message)s',
        level=logging.INFO
    )

    if len(sys.argv) == 1:
        log.warning("add excel file, Yixiao!")
        sys.exit()
    fl = sys.argv[1]
    plot_fl(fl)
    log.info("Complete")


if __name__ == "__main__":
    main()
