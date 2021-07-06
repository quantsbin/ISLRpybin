# Let's import libraries required for this chapter
import sys
sys.path.append(r'../../..')

#Conventional Package
import pandas as pd
import seaborn as sns
import numpy as np
import warnings
import itertools
import math
from itertools import product
#Plot related package
import plotly
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
#ML related packages
import statsmodels.api as sm
import scipy.stats as stats
import scipy as sp
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.regressionplots import plot_leverage_resid2
#User defined function file
from yellowbrick.regressor import CooksDistance
from yellowbrick.datasets import load_concrete
import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"]=10,10

plotly.offline.init_notebook_mode()
warnings.filterwarnings('ignore')

from islrpybin.src.linearRegeressionAnalysis import DiagnosticPlots
