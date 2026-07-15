"""Implementações preservadas do arquivo exemplo_funcoes.md.

Arquivo gerado mecanicamente. As assinaturas, corpos, retornos e estilo foram mantidos.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import builtins, os, pickle, random, warnings
import joblib
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from numpy import interp
from scipy.stats import *
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, chi2, mutual_info_classif
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
import shap
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from category_encoders import BinaryEncoder, CatBoostEncoder
SEED = RANDOM_STATE = 42
