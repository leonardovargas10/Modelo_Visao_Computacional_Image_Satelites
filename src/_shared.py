"""Implementações preservadas do arquivo exemplo_funcoes.md.

Arquivo gerado mecanicamente. As assinaturas, corpos, retornos e estilo foram mantidos.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import builtins
import logging
import os
import pickle
import random
import re
import sys
import warnings
import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import seaborn as sns
from dateutil.relativedelta import relativedelta
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import FuncFormatter
from numpy import interp
from scipy.stats import *
# O wildcard do SciPy também exporta um nome legado ``stats``; restaura o
# módulo público para chamadas como ``stats.probplot`` e ``stats.mannwhitneyu``.
import scipy.stats as stats
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.weightstats import ztest

try:
    from IPython.display import Image, display
except ImportError:  # IPython não é obrigatório para uso como biblioteca.
    Image = None
    display = builtins.print

try:
    from tabulate import tabulate
except ImportError:  # Usado apenas para apresentação tabular em notebooks.
    tabulate = None

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, chi2, mutual_info_classif
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import *
# Necessário antes de importar HalvingGridSearchCV/HalvingRandomSearchCV,
# que fazem parte do __all__ de sklearn.model_selection em algumas versões.
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import *
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
import shap
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from category_encoders import BinaryEncoder, CatBoostEncoder

# Dependência opcional usada somente quando a busca Bayesiana for solicitada.
try:
    from skopt import BayesSearchCV
except ImportError:
    BayesSearchCV = None

# Regressão probabilística e conformal. Permanecem opcionais para que métricas,
# visualizações e modelos clássicos possam ser usados sem essas instalações.
try:
    import skpro
    from skpro.distributions import Gamma, LogNormal, Normal
    from skpro.regression.residual import ResidualDouble
except ImportError:
    skpro = None
    Gamma = LogNormal = Normal = ResidualDouble = None

try:
    import mapie
    from mapie.regression import SplitConformalRegressor
    try:
        from mapie.metrics.regression import regression_coverage_score
    except ImportError:
        regression_coverage_score = None
    try:
        from mapie.utils import train_conformalize_test_split
    except ImportError:
        train_conformalize_test_split = None
except ImportError:
    mapie = None
    SplitConformalRegressor = regression_coverage_score = train_conformalize_test_split = None

SEED = RANDOM_STATE = 42
