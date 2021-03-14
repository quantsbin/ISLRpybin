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
from itertools import combinations
#Plot related package
import plotly
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
#ML related packages
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV

#User defined function file
import matplotlib.pyplot as plt

plotly.offline.init_notebook_mode()
warnings.filterwarnings('ignore')