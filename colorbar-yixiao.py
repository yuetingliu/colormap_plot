
# coding: utf-8

# In[82]:

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlrd
get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u"config Inline_backend.figure_format='retina'")


# In[83]:

def preprocess(fl, lower=None, upper=None):
    '''load the file and slice the wanted range.
    lower and upper are the lower and upper range of the wavenumber,
    return the grid X, Y, and the values Z'''
    df = pd.read_excel(fl, header=None)  # Load the data into a dataframe
    k = df[df[0]>=lower]                     # Slice the wavenumber range using a, b 
    k = k[k[0]<=upper]
    x = k[0]                             # Get X axis
    y = df.iloc[0, 1:]                   # Get y axis   
    X, Y= np.meshgrid(x,y)               # Set the grid with meshgrid
    Z = k.iloc[:, 1:].astype(float)      # Get values and explictly set the data type as float values  
    Z = Z.T                              # Transpose Z because X and Y are transposed 
    return X, Y, Z                       # Return X,Y,Z


# In[92]:

def plot_(X, Y, Z, x_size=11.36, y_size=7.93, font_size= 26, save=True):
    '''Plot pocolormesh figure,
    X,Y: the axis
    Z: the values
    xaxis,yaxis the figure size'''
    # Set the sans-serif, Arial fonts,
    # Set the default mathtext as bold
    from matplotlib import rcParams
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['axes.linewidth']=3
    rcParams['mathtext.default']='bf'
    
    # Set fontproperties 
    from matplotlib.font_manager import FontProperties
    font0 = FontProperties()
    font0.set_style('italic')
    font0.set_size(font_size)
    font0.set_weight('bold')
#     font1 = FontProperties()
#     font1.set_weight('bold')
#     font1.set_size(font_size)

    fig = plt.figure(figsize=(x_size, y_size))
    plt.pcolormesh(X, Y, Z, cmap=plt.cm.jet)
    plt.axis([X.max(), X.min(), Y.min(), Y.max()])
    plt.colorbar()
    plt.xlabel('Wavenumber  /  $cm^{-1}$', fontproperties=font0)
    plt.ylabel('Time / $min$', fontproperties=font0)
    plt.tick_params(axis='both', labelsize=26, top='off', right='off')
    plt.show()
#   plt.yticks(fontweight='bold')
    if save == True:
        plt.savefig('{}.png'.format(fl), dpi=300)


# In[94]:

fl = 'Rh-NO.xlsx'    # File name
w_num = [1500, 2500]   # Wavenumber range
f_size = [11.36, 7.93]  # Figure size
font_size = 36
X, Y, Z = preprocess(fl, w_num[0], w_num[1])
plot_(X, Y, Z, f_size[0], f_size[1], font_size, save=True)


# In[ ]:



