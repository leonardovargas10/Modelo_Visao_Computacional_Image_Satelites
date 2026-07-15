## Bibliotecas De Manipulação de Dados e Visualização
import pandas as pd 
import builtins as builtins
import matplotlib.pyplot as plt
import seaborn as sns 
from IPython.display import display, Image
from tabulate import tabulate
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from dateutil.relativedelta import relativedelta

## Bibliotecas de Modelagem Matemática e Estatística
import numpy as np
import scipy as sp 
import scipy.stats as stats
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import normaltest, ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, kruskal, uniform, chi2_contingency
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.diagnostic import lilliefors
from numpy import interp

# Bibliotecas de Seleção de Modelos
from skopt import BayesSearchCV
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, cross_val_predict
from sklearn.feature_selection import VarianceThreshold, chi2, mutual_info_classif

# Bibliotecas de Pré-Processamento e Pipeline
from category_encoders import BinaryEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer

# Bibliotecas de Modelos de Machine Learning
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import shap

# Bibliotecas de Métricas de Machine Learning
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, precision_score, recall_score, precision_recall_curve, average_precision_score, f1_score, log_loss, brier_score_loss, confusion_matrix, silhouette_score

# Parâmetros de Otimização
import warnings
# %matplotlib inline
# sns.set(style="whitegrid", font_scale=1.2)
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.size'] = '14'
# plt.rcParams['figure.figsize'] = [10, 5]
# pd.set_option('display.max_rows', 100)
# pd.set_option('display.max_columns', 100)
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('display.float_format', lambda x: '%.2f' % x) # Tira os números do formato de Notação Científica
# np.set_printoptions(suppress=True) # Tira os números do formato de Notação Científica em Numpy Arrays
# warnings.filterwarnings('ignore')
# warnings.simplefilter(action='ignore', category=FutureWarning) # Retira Future Warnings
def plota_barras_agrupadas(df, x, y, titulo):
    ax = sns.barplot(data = df, x = x, y = y)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')
    ax.set_title(f'{titulo}')
    ax.set_xlabel(f'{x}', fontsize = 14)
    ax.set_ylabel(f'Quantidade de Inadimplentes', fontsize = 14)
    ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
    plt.show()

def plota_barras(lista_variaveis, hue, df, linhas, colunas, titulo, rotation):
    if hue != False:
        if (linhas == 1) and (colunas == 1):
            k = 0
            ax = sns.countplot(x = lista_variaveis[k], data = df, orient = 'h', hue = hue)
            ax.set_title(f'{titulo}')
            ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
            ax.set_ylabel(f'Quantidade', fontsize = 14)
            total = []
            for bar in ax.patches:
                height = bar.get_height()
                total.append(height)
            total = builtins.sum(total)
            
            sizes = []
            for bar in ax.patches:
                height = bar.get_height()
                sizes.append(height)
                ax.text(bar.get_x() + bar.get_width()/1.6,
                        height,
                        f'{builtins.round((height/total)*100, 2)}%',
                        ha = 'center',
                        fontsize = 12
                )

            ax.set_ylim(0, builtins.max(sizes)*1.1)
            ax.set_xticklabels(df[lista_variaveis[k]].unique(), rotation = rotation, ha='right', fontsize=10)
            # Formatação manual dos rótulos do eixo y para remover a notação científica
            ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
            # Adicionamos os nomes das categorias no eixo x
            ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
            plt.show()

        elif linhas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize=(14, 7), sharey=True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.countplot(x = lista_variaveis[k], data = df, orient = 'h', ax = axis[j], hue = hue)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Quantidade', fontsize = 14)
                    total = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        total.append(height)
                    total = builtins.sum(total)
                    
                    sizes = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        sizes.append(height)
                        ax.text(bar.get_x() + bar.get_width()/1.6,
                                height,
                                f'{builtins.round((height/total)*100, 2)}%',
                                ha = 'center',
                                fontsize = 12
                        )
                    ax.set_ylim(0, builtins.max(sizes)*1.1)
                    ax.set_xticklabels(df[lista_variaveis[k]].unique(), rotation = rotation, ha='right', fontsize=10)
                    # Formatação manual dos rótulos do eixo y para remover a notação científica
                    ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
                    # Adicionamos os nomes das categorias no eixo x
                    ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
                    k = k + 1
                    
        elif colunas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize=(14, 7), sharey=True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.countplot(x = lista_variaveis[k], data = df, orient = 'h', ax = axis[i], hue = hue)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Quantidade', fontsize = 14)
                    total = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        total.append(height)
                    total = builtins.sum(total)
                    
                    sizes = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        sizes.append(height)
                        ax.text(bar.get_x() + bar.get_width()/1.6,
                                height,
                                f'{builtins.round((height/total)*100, 2)}%',
                                ha = 'center',
                                fontsize = 12
                        )
                    ax.set_ylim(0, builtins.max(sizes)*1.1)
                    ax.set_xticklabels(df[lista_variaveis[k]].unique(), rotation = rotation, ha='right', fontsize=10)
                    # Formatação manual dos rótulos do eixo y para remover a notação científica
                    ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
                    # Adicionamos os nomes das categorias no eixo x
                    ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
                    k = k + 1
            
        else: 
            fig, axis = plt.subplots(linhas, colunas, figsize=(14, 7), sharey=True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.countplot(x = lista_variaveis[k], data = df, orient = 'h', ax = axis[i, j], hue = hue)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Quantidade', fontsize = 14)
                    total = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        total.append(height)
                    total = builtins.sum(total)
                    
                    sizes = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        sizes.append(height)
                        ax.text(bar.get_x() + bar.get_width()/1.6,
                                height,
                                f'{builtins.round((height/total)*100, 2)}%',
                                ha = 'center',
                                fontsize = 12
                        )
                    ax.set_ylim(0, builtins.max(sizes)*1.1)
                    ax.set_xticklabels(df[lista_variaveis[k]].unique(), rotation = rotation, ha='right', fontsize=10)
                    # Formatação manual dos rótulos do eixo y para remover a notação científica
                    ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
                    # Adicionamos os nomes das categorias no eixo x
                    ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
                    k = k + 1
           
    else:
        if (linhas == 1) and (colunas == 1):
            k = 0
            ax = sns.countplot(x = lista_variaveis[k], data = df, orient = 'h', color='#1FB3E5')
            ax.set_title(f'{titulo}')
            ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
            ax.set_ylabel(f'Quantidade', fontsize = 14)
            total = []
            for bar in ax.patches:
                height = bar.get_height()
                total.append(height)
            total = builtins.sum(total)
            
            sizes = []
            for bar in ax.patches:
                height = bar.get_height()
                sizes.append(height)
                ax.text(bar.get_x() + bar.get_width()/1.6,
                        height,
                        f'{builtins.round((height/total)*100, 2)}%',
                        ha = 'center',
                        fontsize = 12
                )
            ax.set_ylim(0, builtins.max(sizes)*1.1)
            ax.set_xticklabels(df[lista_variaveis[k]].unique(), rotation = rotation, ha='right', fontsize=10)
            # Formatação manual dos rótulos do eixo y para remover a notação científica
            ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
            # Adicionamos os nomes das categorias no eixo x
            ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
            plt.show()

        elif linhas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize=(14, 7), sharey=True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.countplot(x = lista_variaveis[k], data = df, orient = 'h', ax = axis[j], color='#1FB3E5')
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Quantidade', fontsize = 14)
                    total = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        total.append(height)
                    total = builtins.sum(total)
                    
                    sizes = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        sizes.append(height)
                        ax.text(bar.get_x() + bar.get_width()/1.6,
                                height,
                                f'{builtins.round((height/total)*100, 2)}%',
                                ha = 'center',
                                fontsize = 12
                        )
                    ax.set_ylim(0, builtins.max(sizes)*1.1)
                    ax.set_xticklabels(df[lista_variaveis[k]].unique(), rotation = rotation, ha='right', fontsize=10)
                    # Formatação manual dos rótulos do eixo y para remover a notação científica
                    ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
                    # Adicionamos os nomes das categorias no eixo x
                    ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
                    k = k + 1
            

        elif colunas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize=(14, 7), sharey=True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.countplot(x = lista_variaveis[k], data = df, orient = 'h', ax = axis[i], color='#1FB3E5')
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Quantidade', fontsize = 14)
                    total = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        total.append(height)
                    total = builtins.sum(total)
                    
                    sizes = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        sizes.append(height)
                        ax.text(bar.get_x() + bar.get_width()/1.6,
                                height,
                                f'{builtins.round((height/total)*100, 2)}%',
                                ha = 'center',
                                fontsize = 12
                        )
                    ax.set_ylim(0, builtins.max(sizes)*1.1)
                    ax.set_xticklabels(df[lista_variaveis[k]].unique(), rotation = rotation, ha='right', fontsize=10)
                    # Formatação manual dos rótulos do eixo y para remover a notação científica
                    ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
                    # Adicionamos os nomes das categorias no eixo x
                    ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
                    k = k + 1
            

        else:
            fig, axis = plt.subplots(linhas, colunas, figsize=(14, 7), sharey=True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.countplot(x = lista_variaveis[k], data = df, orient = 'h', ax = axis[i, j], color='#1FB3E5')
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Quantidade', fontsize = 14)
                    total = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        total.append(height)
                    total = builtins.sum(total)
                    
                    sizes = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        sizes.append(height)
                        ax.text(bar.get_x() + bar.get_width()/1.6,
                                height,
                                f'{builtins.round((height/total)*100, 2)}%',
                                ha = 'center',
                                fontsize = 12
                        )
                    ax.set_ylim(0, builtins.max(sizes)*1.1)
                    ax.set_xticklabels(df[lista_variaveis[k]].unique(), rotation = rotation, ha='right', fontsize=10)
                    # Formatação manual dos rótulos do eixo y para remover a notação científica
                    ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
                    # Adicionamos os nomes das categorias no eixo x
                    ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
                    k = k + 1

def plota_histograma(lista_variaveis, hue, df, linhas, colunas, titulo):
    if hue != False:

        if (linhas == 1) and (colunas == 1): 
            k = 0

            df_good = df.loc[df['hue'] == 'GOOD']

            mediana = df[lista_variaveis[k]].median()
            media = df[lista_variaveis[k]].mean()
            plt.figure(figsize = (14, 5))
            ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', bins = 30, hue = hue)
            ax.set_title(f'{titulo}')
            ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
            ax.set_ylabel(f'Frequência', fontsize = 14)
            ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
            ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
            plt.ticklabel_format(style='plain')
            plt.legend(loc = 'best')
            plt.ticklabel_format(style='plain', axis='both')
            plt.show()
            
        elif linhas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize = (14, 5), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    mediana = df[lista_variaveis[k]].median()
                    media = df[lista_variaveis[k]].mean().round()
                    ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[j], bins = 30, hue = hue)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Frequência', fontsize = 14)
                    ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
                    ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
                    ax.ticklabel_format(style='plain')
                    ax.legend(loc = 'best')
                    ax.ticklabel_format(style='plain', axis='both')
                    k = k + 1
            

        elif colunas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize = (14, 5), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    mediana = df[lista_variaveis[k]].median()
                    media = df[lista_variaveis[k]].mean().round()
                    ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[i], bins = 30, hue = hue)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Frequência', fontsize = 14)
                    ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
                    ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
                    ax.ticklabel_format(style='plain')
                    ax.legend(loc = 'best')
                    ax.ticklabel_format(style='plain', axis='both')
                    k = k + 1
            
        else:
            fig, axis = plt.subplots(linhas, colunas, figsize = (14, 5), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    mediana = df[lista_variaveis[k]].median()
                    media = df[lista_variaveis[k]].mean().round()
                    ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[i, j], bins = 30, hue = hue)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Frequência', fontsize = 14)
                    ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
                    ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
                    ax.ticklabel_format(style='plain')
                    ax.legend(loc = 'best')
                    ax.ticklabel_format(style='plain', axis='both')
                    k = k + 1
            
    else:
    
        if (linhas == 1) and (colunas == 1): 
            k = 0
            mediana = df[lista_variaveis[k]].median()
            media = df[lista_variaveis[k]].mean()
            plt.figure(figsize = (14, 5))
            ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', bins = 30)
            ax.set_title(f'{titulo}')
            ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
            ax.set_ylabel(f'Frequência', fontsize = 14)
            ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
            ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
            plt.ticklabel_format(style='plain')
            plt.legend(loc = 'best')
            plt.ticklabel_format(style='plain', axis='both')
            plt.show()

        elif linhas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize = (14, 5), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    mediana = df[lista_variaveis[k]].median()
                    media = df[lista_variaveis[k]].mean().round()
                    ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[j], bins = 30)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Frequência', fontsize = 14)
                    ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
                    ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
                    ax.ticklabel_format(style='plain')
                    ax.legend(loc = 'best')
                    ax.ticklabel_format(style='plain', axis='both')
                    k = k + 1
            
        elif colunas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize = (14, 5), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    mediana = df[lista_variaveis[k]].median()
                    media = df[lista_variaveis[k]].mean().round()
                    ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[i], bins = 30)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Frequência', fontsize = 14)
                    ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
                    ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
                    ax.ticklabel_format(style='plain')
                    ax.legend(loc = 'best')
                    ax.ticklabel_format(style='plain', axis='both')
                    k = k + 1
            

        else:
            fig, axis = plt.subplots(linhas, colunas, figsize = (14, 5), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    mediana = df[lista_variaveis[k]].median()
                    media = df[lista_variaveis[k]].mean().round()
                    ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[i, j], bins = 30)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Frequência', fontsize = 14)
                    ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
                    ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
                    ax.ticklabel_format(style='plain')
                    ax.legend(loc = 'best')
                    ax.ticklabel_format(style='plain', axis='both')
                    k = k + 1

def plota_boxplot(lista_variaveis, hue, df, linhas, colunas, titulo):
    if hue != False:
        if (linhas == 1) and (colunas == 1): 
            k = 0
            plt.figure(figsize = (10, 7))
            ax = sns.boxplot(x = lista_variaveis[k], data = df, palette = ['#1FB3E5', '#64ED8F', '#B864ED'], orient = 'h', y = hue)
            ax.set_title(f'{titulo}')
            ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 10)
            ax.set_ylabel(f'Frequência', fontsize = 10)
            ax.xaxis.set_major_formatter('{:.0f}'.format)
            plt.show()

        elif linhas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize = (10, 7), sharey = True)
            fig.suptitle(f'{titulo}', fontsize = 10)
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.boxplot(x = lista_variaveis[k], data = df, palette = ['#1FB3E5', '#64ED8F', '#B864ED'], ax = axis[j], orient = 'h', y = hue)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 10)
                    ax.set_ylabel(f'Frequência', fontsize = 10)
                    ax.set_xticklabels(ax.get_xticks(), fontsize=7) 
                    k = k + 1
            
        elif colunas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize = (10, 7), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.boxplot(x = lista_variaveis[k], data = df, palette = ['#1FB3E5', '#64ED8F', '#B864ED'], ax = axis[i], orient = 'h', y = hue)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 10)
                    ax.set_ylabel(f'Frequência', fontsize = 10)
                    ax.set_xticklabels(ax.get_xticks(), fontsize=7) 
                    k = k + 1
            
        else:
            fig, axis = plt.subplots(linhas, colunas, figsize = (10, 7), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.boxplot(x = lista_variaveis[k], data = df, palette = ['#1FB3E5', '#64ED8F', '#B864ED'], ax = axis[i, j], orient = 'h', y = hue)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 10)
                    ax.set_ylabel(f'Frequência', fontsize = 10)
                    ax.set_xticklabels(ax.get_xticks(), fontsize=7) 
                    k = k + 1

    else:
        if (linhas == 1) and (colunas == 1): 
            k = 0
            plt.figure(figsize = (10, 7))
            ax = sns.boxplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', orient = 'h')
            ax.set_title(f'{titulo}')
            ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 10)
            ax.set_ylabel(f'Frequência', fontsize = 10)
            ax.xaxis.set_major_formatter('{:.0f}'.format)
            plt.show()

        elif linhas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize = (10, 7), sharey = True)
            fig.suptitle(f'{titulo}', fontsize = 10)
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.boxplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[j], orient = 'h')
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 10)
                    ax.set_ylabel(f'Frequência', fontsize = 10)
                    ax.set_xticklabels(ax.get_xticks(), fontsize=7) 
                    k = k + 1

        elif colunas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize = (10, 7), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.boxplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[i], orient = 'h')
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 10)
                    ax.set_ylabel(f'Frequência', fontsize = 10)
                    ax.set_xticklabels(ax.get_xticks(), fontsize=7) 
                    k = k + 1
    
        else:
            fig, axis = plt.subplots(linhas, colunas, figsize = (10, 7), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.boxplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[i, j], orient = 'h')
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 10)
                    ax.set_ylabel(f'Frequência', fontsize = 10)
                    ax.set_xticklabels(ax.get_xticks(), fontsize=7) 
                    k = k + 1

def plota_dispersao(df, titulo,  x, y, metodo):
    plt.figure(figsize = (10, 5))
    sns.set(style = 'whitegrid')
    corr1 = str(df[[x, y]].corr(method = metodo).iloc[1, 0].round(2))
    sns.scatterplot(data = df, x = x, y = y, color = '#1FB3E5', sizes = 1, alpha = 0.50, marker = '.')
    plt.text(1, 1, f'Correlacao: {corr1}', fontsize = 12)
    plt.title(f'{titulo}', fontsize = 14)
    plt.xlabel(f'{x}', fontsize = 14)
    plt.ylabel(f'{y}', fontsize = 14)
    plt.ticklabel_format(style = 'plain')
    plt.grid(True, linestyle=':')
    sns.despine()
    plt.tight_layout()
    plt.show()

def auc_ks_juntos(classificador, target, 
                                    y_train, y_predict_train, 
                                    y_test, y_predict_test, 
                                    y_predict_proba_train, y_predict_proba_test, 
                                    cv_results):

    predict_proba_train = pd.DataFrame(y_predict_proba_train.tolist(), columns=['predict_proba_0', 'predict_proba_1'])
    predict_proba_test = pd.DataFrame(y_predict_proba_test.tolist(), columns=['predict_proba_0', 'predict_proba_1'])

    # Inicialize as variáveis x_max_ks e y_max_ks fora dos blocos condicionais
    x_max_ks_train, y_max_ks_train = 0, 0
    x_max_ks_test, y_max_ks_test = 0, 0
    x_max_ks_cv, y_max_ks_cv = 0, 0

    ### Treino
    results_train = y_train[[target]].copy()
    results_train['y_predict_train'] = y_predict_train
    results_train['predict_proba_0'] = list(predict_proba_train['predict_proba_0']) # Probabilidade de ser Bom (classe 0)
    results_train['predict_proba_1'] = list(predict_proba_train['predict_proba_1']) # Probabilidade de ser Mau (classe 1)

    results_train_sorted = results_train.sort_values(by='predict_proba_1', ascending=False)
    results_train_sorted['Cumulative N Population'] = range(1, results_train_sorted.shape[0] + 1)
    results_train_sorted['Cumulative N Good'] = results_train_sorted[target].cumsum()
    results_train_sorted['Cumulative N Bad'] = results_train_sorted['Cumulative N Population'] - results_train_sorted['Cumulative N Good']
    results_train_sorted['Cumulative Perc Population'] = results_train_sorted['Cumulative N Population'] / results_train_sorted.shape[0]
    results_train_sorted['Cumulative Perc Good'] = results_train_sorted['Cumulative N Good'] / results_train_sorted[target].sum()
    results_train_sorted['Cumulative Perc Bad'] = results_train_sorted['Cumulative N Bad'] / (results_train_sorted.shape[0] - results_train_sorted[target].sum())

    max_ks_index_train = np.argmax(results_train_sorted['Cumulative Perc Good'] - results_train_sorted['Cumulative Perc Bad'])
    x_max_ks_train = results_train_sorted['Cumulative Perc Population'].iloc[max_ks_index_train]
    y_max_ks_train = results_train_sorted['Cumulative Perc Good'].iloc[max_ks_index_train]
    y_min_ks_train = results_train_sorted['Cumulative Perc Bad'].iloc[max_ks_index_train]

        ###### Calculate AUC and ROC for the training set
    y_true_train = results_train[target]
    y_scores_train = results_train['predict_proba_1']
    auc_train = roc_auc_score(y_true_train, y_scores_train)
    fpr_train, tpr_train, thresholds_train = roc_curve(y_true_train, y_scores_train)
        ###### Calculate KS curve for the training set
    KS_train = round(np.max(results_train_sorted['Cumulative Perc Good'] - results_train_sorted['Cumulative Perc Bad']), 2)

    ### Teste
    results_test = y_test[[target]].copy()
    results_test['y_predict_test'] = y_predict_test
    results_test['predict_proba_0'] = list(predict_proba_test['predict_proba_0']) # Probabilidade de ser Bom (classe 0)
    results_test['predict_proba_1'] = list(predict_proba_test['predict_proba_1']) # Probabilidade de ser Mau (classe 1)

    results_test_sorted = results_test.sort_values(by='predict_proba_1', ascending=False)
    results_test_sorted['Cumulative N Population'] = range(1, results_test_sorted.shape[0] + 1)
    results_test_sorted['Cumulative N Good'] = results_test_sorted[target].cumsum()
    results_test_sorted['Cumulative N Bad'] = results_test_sorted['Cumulative N Population'] - results_test_sorted['Cumulative N Good']
    results_test_sorted['Cumulative Perc Population'] = results_test_sorted['Cumulative N Population'] / results_test_sorted.shape[0]
    results_test_sorted['Cumulative Perc Good'] = results_test_sorted['Cumulative N Good'] / results_test_sorted[target].sum()
    results_test_sorted['Cumulative Perc Bad'] = results_test_sorted['Cumulative N Bad'] / (results_test_sorted.shape[0] - results_test_sorted[target].sum())

    max_ks_index_test = np.argmax(results_test_sorted['Cumulative Perc Good'] - results_test_sorted['Cumulative Perc Bad'])
    x_max_ks_test = results_test_sorted['Cumulative Perc Population'].iloc[max_ks_index_test]
    y_max_ks_test = results_test_sorted['Cumulative Perc Good'].iloc[max_ks_index_test]
    y_min_ks_test = results_test_sorted['Cumulative Perc Bad'].iloc[max_ks_index_test]


            ###### Calculate AUC and ROC for the test set
    y_true_test = results_test[target]
    y_scores_test = results_test['predict_proba_1']
    auc_test = roc_auc_score(y_true_test, y_scores_test)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_true_test, y_scores_test)
            ###### Calculate KS curve for the test set
    KS_test = round(np.max(results_test_sorted['Cumulative Perc Good'] - results_test_sorted['Cumulative Perc Bad']), 2)

    # Cross-validation set
    auc_scores_cv = []
    ks_scores_cv = []
    roc_curves_cv = []
    ks_curves_cv = []
    for fold_results in cv_results:
        results_cv = fold_results[[target]].copy()
        results_cv['y_predict_cv'] = fold_results['y_predict']
        results_cv['predict_proba_0'] = fold_results['predict_proba_0']
        results_cv['predict_proba_1'] = fold_results['predict_proba_1']

        y_true_cv = results_cv[target]
        y_scores_cv = results_cv['predict_proba_1']

        # Aggregate ROC curves
        fpr_cv, tpr_cv, _ = roc_curve(y_true_cv, y_scores_cv)
        roc_curves_cv.append((fpr_cv, tpr_cv))

        # Aggregate AUC scores
        auc_cv = roc_auc_score(y_true_cv, y_scores_cv)
        auc_scores_cv.append(auc_cv)

        # Aggregate KS scores
        results_cv_sorted = results_cv.sort_values(by='predict_proba_1', ascending=False)
        results_cv_sorted['Cumulative N Population'] = range(1, results_cv_sorted.shape[0] + 1)
        results_cv_sorted['Cumulative N Good'] = results_cv_sorted[target].cumsum()
        results_cv_sorted['Cumulative N Bad'] = results_cv_sorted['Cumulative N Population'] - results_cv_sorted['Cumulative N Good']
        results_cv_sorted['Cumulative Perc Population'] = results_cv_sorted['Cumulative N Population'] / results_cv_sorted.shape[0]
        results_cv_sorted['Cumulative Perc Good'] = results_cv_sorted['Cumulative N Good'] / results_cv_sorted[target].sum()
        results_cv_sorted['Cumulative Perc Bad'] = results_cv_sorted['Cumulative N Bad'] / (results_cv_sorted.shape[0] - results_cv_sorted[target].sum())
        ks_cv = np.max(results_cv_sorted['Cumulative Perc Good'] - results_cv_sorted['Cumulative Perc Bad'])
        ks_scores_cv.append(ks_cv)

    # Calculate average ROC, AUC and KS scores across folds
    auc_cv_mean = np.mean(auc_scores_cv)
    ks_cv_mean = np.mean(ks_scores_cv)
    mean_fpr_cv = np.linspace(0, 1, 100)  # You can adjust the number of points for a smoother curve
    mean_tpr_cv = np.mean([interp(mean_fpr_cv, fpr, tpr) for fpr, tpr in roc_curves_cv], axis=0)
    mean_ks_cv = np.mean(ks_curves_cv, axis=0)

    max_ks_index_cv = np.argmax(results_cv_sorted['Cumulative Perc Good'] - results_cv_sorted['Cumulative Perc Bad'])
    x_max_ks_cv = results_cv_sorted['Cumulative Perc Population'].iloc[max_ks_index_cv]
    y_max_ks_cv = results_cv_sorted['Cumulative Perc Good'].iloc[max_ks_index_cv]
    y_min_ks_cv = results_cv_sorted['Cumulative Perc Bad'].iloc[max_ks_index_cv]

    KS_cv = round(np.max(results_cv_sorted['Cumulative Perc Good'] - results_cv_sorted['Cumulative Perc Bad']), 2)

    # Plot ROC and KS curves side by side
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Training set ROC curve
    axs[0].plot(fpr_train, tpr_train, label='Train ROC Curve (AUC = {:.2f})'.format(auc_train), color='blue')
    axs[0].fill_between(fpr_train, 0, tpr_train, color='gray', alpha=0.3)  # Preencha a área sob a curva ROC
    axs[0].plot([0, 1], [0, 1], linestyle='--', color='black')
    axs[0].set_xlabel('False Positive Rate', fontsize = 14)
    axs[0].set_ylabel('True Positive Rate', fontsize = 14)
    axs[0].set_title(f'ROC Curve - {classificador}', fontsize = 14)

    # Test set ROC curve
    axs[0].plot(fpr_test, tpr_test, label='Test ROC Curve (AUC = {:.2f})'.format(auc_test), color='red')
    axs[0].fill_between(fpr_test, 0, tpr_test, color='gray', alpha=0.3)  # Preencha a área sob a curva ROC

    # Cross-validation set ROC cruve
    axs[0].plot(mean_fpr_cv, mean_tpr_cv, label='CV ROC Curve (AUC = {:.2f})'.format(auc_cv_mean), color='green')
    axs[0].fill_between(mean_fpr_cv, 0, mean_tpr_cv, color='gray', alpha=0.3)

    # Adicione a legenda personalizada com cores para a curva ROC
    roc_legend_labels = [
        {'label': 'Train ROC Curve (AUC = {:.2f})'.format(auc_train), 'color': 'blue', 'marker': 'o'},
        {'label': 'Test ROC Curve (AUC = {:.2f})'.format(auc_test), 'color': 'red', 'marker': 's'},
        {'label': 'CV ROC Curve (AUC = {:.2f})'.format(auc_test), 'color': 'green', 'marker': '^'}
    ]

    # Criar marcadores personalizados para a legenda ROC
    roc_legend_handles = [Line2D([0], [0], marker=label_info['marker'], color='w', markerfacecolor=label_info['color'], markersize=10) for label_info in roc_legend_labels]

    # Adicione a legenda personalizada ao gráfico da curva ROC
    roc_legend = axs[0].legend(handles=roc_legend_handles, labels=[label_info['label'] for label_info in roc_legend_labels], loc='upper right', bbox_to_anchor=(0.9, 0.4), fontsize='11')
    roc_legend.set_title('ROC AUC', prop={'size': '11'})


    # Train set KS curve
    axs[1].plot(results_train_sorted['Cumulative Perc Population'], results_train_sorted['Cumulative Perc Good'], label='Train Positive Class (Class 1)', color='blue')
    axs[1].plot(results_train_sorted['Cumulative Perc Population'], results_train_sorted['Cumulative Perc Bad'], label='Train Negative Class (Class 0)', color='blue')
    axs[1].plot([x_max_ks_train, x_max_ks_train], [y_min_ks_train, y_max_ks_train], color='black', linestyle='--')
    axs[1].fill_between(results_train_sorted['Cumulative Perc Population'], results_train_sorted['Cumulative Perc Good'], results_train_sorted['Cumulative Perc Bad'], color='gray', alpha=0.5)
    axs[1].text(x=results_train_sorted['Cumulative Perc Population'].iloc[np.argmax(results_train_sorted['Cumulative Perc Good'] - results_train_sorted['Cumulative Perc Bad'])],
                y=(y_min_ks_train + results_train_sorted['Cumulative Perc Good'].iloc[np.argmax(results_train_sorted['Cumulative Perc Good'] - results_train_sorted['Cumulative Perc Bad'])]) / 2,
                s=str(KS_train), fontsize = 14, color='blue', ha='left', va='center', rotation=45)
    axs[1].set_xlabel('Cumulative Percentage of Population', fontsize = 14)
    axs[1].set_ylabel('Cumulative Percentage', fontsize = 14)
    axs[1].set_title(f'KS Plot - {classificador}', fontsize = 14)

    # Test set KS curve
    axs[1].plot(results_test_sorted['Cumulative Perc Population'], results_test_sorted['Cumulative Perc Good'], label='Test Positive Class (Class 1)', color='red')
    axs[1].plot(results_test_sorted['Cumulative Perc Population'], results_test_sorted['Cumulative Perc Bad'], label='Test Negative Class (Class 0)', color='red')
    axs[1].plot([x_max_ks_test, x_max_ks_test], [y_min_ks_test, y_max_ks_test], color='black', linestyle='--')
    axs[1].fill_between(results_test_sorted['Cumulative Perc Population'], results_test_sorted['Cumulative Perc Good'], results_test_sorted['Cumulative Perc Bad'], color='gray', alpha=0.5)
    axs[1].text(x=results_test_sorted['Cumulative Perc Population'].iloc[np.argmax(results_test_sorted['Cumulative Perc Good'] - results_test_sorted['Cumulative Perc Bad'])],
                y=(y_min_ks_test + results_test_sorted['Cumulative Perc Good'].iloc[np.argmax(results_test_sorted['Cumulative Perc Good'] - results_test_sorted['Cumulative Perc Bad'])]) / 2,
                s=str(KS_test), fontsize = 14, color='red', ha='left', va='center', rotation=45)
    axs[1].set_xlabel('Cumulative Percentage of Population', fontsize = 14)
    axs[1].set_ylabel('Cumulative Percentage', fontsize = 14)
    axs[1].set_title(f'KS Plot - {classificador}', fontsize = 14)

    # Cross-validation set KS curve
    axs[1].plot(results_cv_sorted['Cumulative Perc Population'], results_cv_sorted['Cumulative Perc Good'], label='CV Positive Class (Class 1)', color='green')
    axs[1].plot(results_cv_sorted['Cumulative Perc Population'], results_cv_sorted['Cumulative Perc Bad'], label='CV Negative Class (Class 0)', color='green')
    axs[1].plot([x_max_ks_cv, x_max_ks_cv], [y_min_ks_cv, y_max_ks_cv], color='black', linestyle='--')
    axs[1].fill_between(results_cv_sorted['Cumulative Perc Population'], results_cv_sorted['Cumulative Perc Good'], results_cv_sorted['Cumulative Perc Bad'], color='gray', alpha=0.5)
    axs[1].text(x=x_max_ks_cv,
                y=(y_min_ks_cv + y_max_ks_cv) / 2,
                s=str(KS_cv), fontsize = 14, color='green', ha='left', va='center', rotation=45)
    axs[1].set_xlabel('Cumulative Percentage of Population', fontsize = 14)
    axs[1].set_ylabel('Cumulative Percentage', fontsize = 14)
    axs[1].set_title(f'KS Plot - {classificador}', fontsize = 14)


    # Adicione a legenda personalizada com cores
    ks_legend_labels = [
        {'label': f'Treino (KS: {KS_train})', 'color': 'blue', 'marker': 'o'},
        {'label': f'Teste (KS: {KS_test})', 'color': 'red', 'marker': 's'},
        {'label': f'CV (KS: {KS_cv})', 'color': 'green', 'marker': '^'}
    ]

    # Criar marcadores personalizados para a legenda
    legend_handles = [Line2D([0], [0], marker=label_info['marker'], color='w', markerfacecolor=label_info['color'], markersize=10) for label_info in ks_legend_labels]

    ks_legend = axs[1].legend(handles=legend_handles, labels=[label_info['label'] for label_info in ks_legend_labels], loc='upper right', bbox_to_anchor=(0.9, 0.4), fontsize='11')
    ks_legend.set_title('KS', prop={'size': '11'})

    plt.tight_layout()
    plt.show()

def auc_ks_final(classificador, target, y, y_predict, y_predict_proba):

    predict_proba = pd.DataFrame(y_predict_proba.tolist(), columns=['predict_proba_0', 'predict_proba_1'])

    # Inicialize as variáveis x_max_ks e y_max_ks fora dos blocos condicionais
    x_max_ks, y_max_ks = 0, 0

    ### Etapa Final
    results = y[[target]].copy()
    results['y_predict'] = y_predict
    results['predict_proba_0'] = list(predict_proba['predict_proba_0']) # Probabilidade de ser Bom (classe 0)
    results['predict_proba_1'] = list(predict_proba['predict_proba_1']) # Probabilidade de ser Mau (classe 1)

    results_sorted = results.sort_values(by='predict_proba_1', ascending=False)
    results_sorted['Cumulative N Population'] = range(1, results_sorted.shape[0] + 1)
    results_sorted['Cumulative N Good'] = results_sorted[target].cumsum()
    results_sorted['Cumulative N Bad'] = results_sorted['Cumulative N Population'] - results_sorted['Cumulative N Good']
    results_sorted['Cumulative Perc Population'] = results_sorted['Cumulative N Population'] / results_sorted.shape[0]
    results_sorted['Cumulative Perc Good'] = results_sorted['Cumulative N Good'] / results_sorted[target].sum()
    results_sorted['Cumulative Perc Bad'] = results_sorted['Cumulative N Bad'] / (results_sorted.shape[0] - results_sorted[target].sum())

    max_ks_index = np.argmax(results_sorted['Cumulative Perc Good'] - results_sorted['Cumulative Perc Bad'])
    x_max_ks = results_sorted['Cumulative Perc Population'].iloc[max_ks_index]
    y_max_ks = results_sorted['Cumulative Perc Good'].iloc[max_ks_index]
    y_min_ks = results_sorted['Cumulative Perc Bad'].iloc[max_ks_index]

        ###### Calculate AUC and ROC for the training set
    y_true = results[target]
    y_scores = results['predict_proba_1']
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        ###### Calculate KS curve for the training set
    KS = round(np.max(results_sorted['Cumulative Perc Good'] - results_sorted['Cumulative Perc Bad']), 2)

    # Plot ROC and KS curves side by side
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Training set ROC curve
    axs[0].plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc), color='blue')
    axs[0].fill_between(fpr, 0, tpr, color='gray', alpha=0.3)  # Preencha a área sob a curva ROC
    axs[0].plot([0, 1], [0, 1], linestyle='--', color='black')
    axs[0].set_xlabel('False Positive Rate', fontsize = 14)
    axs[0].set_ylabel('True Positive Rate', fontsize = 14)
    axs[0].set_title(f'ROC Curve - {classificador}', fontsize = 14)

    # Adicione a legenda personalizada com cores para a curva ROC
    roc_legend_labels = [
        {'label': 'ROC Curve (AUC = {:.2f})'.format(auc), 'color': 'blue', 'marker': 'o'},
    ]

    # Criar marcadores personalizados para a legenda ROC
    roc_legend_handles = [Line2D([0], [0], marker=label_info['marker'], color='w', markerfacecolor=label_info['color'], markersize=10) for label_info in roc_legend_labels]

    # Adicione a legenda personalizada ao gráfico da curva ROC
    roc_legend = axs[0].legend(handles=roc_legend_handles, labels=[label_info['label'] for label_info in roc_legend_labels], loc='upper right', bbox_to_anchor=(0.9, 0.4), fontsize='11')
    roc_legend.set_title('ROC AUC', prop={'size': '11'})


    # Train set KS curve
    axs[1].plot(results_sorted['Cumulative Perc Population'], results_sorted['Cumulative Perc Good'], label='Train Positive Class (Class 1)', color='blue')
    axs[1].plot(results_sorted['Cumulative Perc Population'], results_sorted['Cumulative Perc Bad'], label='Train Negative Class (Class 0)', color='blue')
    axs[1].plot([x_max_ks, x_max_ks], [y_min_ks, y_max_ks], color='black', linestyle='--')
    axs[1].fill_between(results_sorted['Cumulative Perc Population'], results_sorted['Cumulative Perc Good'], results_sorted['Cumulative Perc Bad'], color='gray', alpha=0.5)
    axs[1].text(x=results_sorted['Cumulative Perc Population'].iloc[np.argmax(results_sorted['Cumulative Perc Good'] - results_sorted['Cumulative Perc Bad'])],
                y=(y_min_ks + results_sorted['Cumulative Perc Good'].iloc[np.argmax(results_sorted['Cumulative Perc Good'] - results_sorted['Cumulative Perc Bad'])]) / 2,
                s=str(KS), fontsize = 14, color='blue', ha='left', va='center', rotation=45)
    axs[1].set_xlabel('Cumulative Percentage of Population', fontsize = 14)
    axs[1].set_ylabel('Cumulative Percentage', fontsize = 14)
    axs[1].set_title(f'KS Plot - {classificador}', fontsize = 14)

    # Adicione a legenda personalizada com cores
    ks_legend_labels = [
        {'label': f'(KS: {KS})', 'color': 'blue', 'marker': 'o'},
    ]

    # Criar marcadores personalizados para a legenda
    legend_handles = [Line2D([0], [0], marker=label_info['marker'], color='w', markerfacecolor=label_info['color'], markersize=10) for label_info in ks_legend_labels]

    ks_legend = axs[1].legend(handles=legend_handles, labels=[label_info['label'] for label_info in ks_legend_labels], loc='upper right', bbox_to_anchor=(0.9, 0.4), fontsize='11')
    ks_legend.set_title('KS', prop={'size': '11'})

    plt.tight_layout()
    plt.show()


def verifica_tipo_variavel(df):
    analytics = df.copy()

    qualitativas = [column for column in analytics.columns if 
            (analytics[column].dtype.name == 'object') 
         or ('bad_rate' not in str(analytics[column].name.lower()) and 'mean' not in str(analytics[column].name.lower()) and analytics[column].nunique() <= 2)
         or ('delinq_2yrs' in str(analytics[column].name.lower()))
        ]
    quantitativas = [column for column in analytics.columns if column not in qualitativas]
    # discretas = [column for column in analytics.columns if 
    # (
    #     analytics[column].dtype.name != 'object' 
    #     and analytics[column].nunique() > 2 
    #     and analytics[column].nunique() <= 50 
    #     and 'bad_rate' not in str(analytics[column].name.lower()) 
    #     and 'mean' not in str(analytics[column].name.lower()))
    # ] 
    # quantitativas = [column for column in analytics.columns if 
    # (analytics[column].dtype.name != 'object' and analytics[column].nunique() > 50) 
    #     or ('bad_rate' in str(analytics[column].name.lower())) 
    #     or ('mean' in str(analytics[column].name.lower()))
    # ] 

    qualitativas = pd.DataFrame({'variaveis':qualitativas, 'tipo':'qualitativa'})
    quantitativas = pd.DataFrame({'variaveis':quantitativas, 'tipo':'quantitativas'})
    #discretas = pd.DataFrame({'variaveis':discretas, 'tipo':'discreta'})
    #continuas = pd.DataFrame({'variaveis':continuas, 'tipo':'continua'})

    #variaveis = pd.concat([qualitativas, discretas, continuas])
    variaveis = pd.concat([qualitativas, quantitativas])

    return variaveis

def analisa_correlacao(metodo, df):
    plt.figure(figsize=(14, 7))
    heatmap = sns.heatmap(df.corr(method=metodo), vmin=-1, vmax=1, cmap='magma', annot = True)
    heatmap.set_title(f"Analisando Correlação de {metodo}")
    plt.grid(False)
    plt.box(False)
    plt.tight_layout()
    plt.grid(False)
    plt.show()

def analisa_normalidade(amostra, variavel):

    normaltest_amostra = normaltest(amostra[variavel])
    if normaltest_amostra[1] < 0.05:
        print(f'Pelo Teste de Hipótese, A Hipótese Nula de que a variável "{variavel}" segue uma Distribuição Normal é REJEITADA!')
    else:
        print(f'Pelo Teste de Hipótese, A Hipótese Nula de  que a variável "{variavel}" segue uma Distribuição Normal é ACEITA')

    plt.figure(figsize = (5, 3))
    stats.probplot(amostra[variavel], dist = 'norm', plot = plt)
    plt.title(f'Amostra 1', fontsize = 14)
    plt.grid(False)
    plt.box(False)
    plt.tight_layout()
    plt.show()

def analisa_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IIQ = Q3 - Q1

    outlier_inferior = Q1 - 1.5*IIQ 
    outlier_superior = Q3 + 1.5*IIQ

    return outlier_inferior, outlier_superior

def remove_features_baixa_variancia(target, df, threshold):
    target_column = df[target]
    features = df.drop(target, axis=1)

    selector = VarianceThreshold(threshold=threshold)
    features_filtered = selector.fit_transform(features)

    feature_indices = selector.get_support(indices=True)
    selected_features = features.columns[feature_indices]
    selected_features = selected_features.append(pd.Index([target]))

    return selected_features.tolist()

def remove_features_mutual_information(target, df, threshold):
    x_train, y_train = separa_feature_target(target, df)

    # Calcular a informação mútua entre cada variável e a variável de saída
    mutual_info = mutual_info_classif(x_train, y_train, random_state = 42)

    # Criar um DataFrame com o nome da feature e sua mutual information
    features_selected = pd.DataFrame({'Feature': x_train.columns, 'Mutual Information': mutual_info})
    features_selected = features_selected.loc[features_selected['Mutual Information'] > threshold]
    features_selected = features_selected.sort_values(by='Mutual Information', ascending=False).reset_index(drop=True)

    selected_features = list(features_selected['Feature'])
    selected_features.append(target)
    
    return features_selected, selected_features

def remove_features_feature_importance(target, df, class_weight, threshold):

    # Separa entre Features e Target
    x, y = separa_feature_target(target, df)

    # Criar o modelo de Random Forest
    model = RandomForestClassifier(random_state=42, criterion='entropy', n_estimators=20, class_weight={0:1, 1:class_weight})

    # Treinar o modelo
    model.fit(x, y)

    # Obter as importâncias das features
    feature_importances = model.feature_importances_

    # Selecionar as features com importância maior que zero
    selected_features = list(x.columns[feature_importances > threshold])
    selected_features.append(target)

    feature_importance_df = pd.DataFrame({
        'feature': x.columns,
        'importance': feature_importances
    }).sort_values(by = 'importance', ascending = False)
    feature_importance_df = feature_importance_df.loc[feature_importance_df['importance'] > 0]
    feature_importance_df['importance'] = feature_importance_df['importance']*100

    return selected_features, feature_importance_df



def teste_hipotese_duas_amostras_independentes(parametrico, amostra1, amostra2, variavel):
    media_amostra_1 = amostra1[variavel].mean()
    media_amostra_2 = amostra2[variavel].mean()
    mediana_amostra_1 = amostra1[variavel].median()
    mediana_amostra_2 = amostra2[variavel].median()

    if parametrico == True: 
        print(f'Média Amostra 1: {media_amostra_1}')
        print(f'Média Amostra 2: {media_amostra_2}')
        stat, p_value = ztest(amostra1[variavel], amostra2[variavel]) 
        if p_value > 0.05:
            print(f'Pelo Teste de Hipótese Z, não há diferença significativa entre as médias da Amostra 1 e Amostra 2')
        else:
            print(f'Pelo Teste de Hipótese Z, há diferença significativa entre as médias da Amostra 1 e Amostra 2')
    else:
        print(f'Mediana Amostra 1: {mediana_amostra_1}')
        print(f'Mediana Amostra 2: {mediana_amostra_2}')
        stat, p_value = stats.mannwhitneyu(amostra1[variavel], amostra2[variavel]) 
        if p_value > 0.05:
            print(f'Pelo Teste de Hipótese de Mann Whitney, não há diferença significativa entre as medianas da Amostra 1 e Amostra 2')
        else:
            print(f'Pelo Teste de Hipótese de Mann Whitney, há diferença significativa entre as medianas da Amostra 1 e Amostra 2')


def teste_hipotese_muitas_amostras_independentes(amostras, variavel):
    medianas = []
    
    for i, amostra in enumerate(amostras):
        mediana_amostra = amostra[variavel].median()
        medianas.append(mediana_amostra)
        print(f'Mediana Amostra {i+1}: {mediana_amostra}')

    stat, p_value = kruskal(*[amostra[variavel] for amostra in amostras])
    
    if p_value > 0.05:
        print(f'Pelo teste de Kruskal-Wallis, não há diferença significativa entre as medianas das amostras')
    else:
        print(f'Pelo teste de Kruskal-Wallis, há diferença significativa entre as medianas das amostras')

def teste_hipotese_duas_variaveis_categoricas(df, variavel1, variavel2):
    # Crie tabelas de contingência
    crosstab = pd.crosstab(df[variavel1], df[variavel2])
    
    # Realize o teste qui-quadrado
    chi2, p, _, _ = chi2_contingency(crosstab)
    
    # Verifique o valor-p
    if p > 0.05:
        print(f'Pelo Teste Qui-Quadrado, não há associação significativa entre {variavel1} e {variavel2}.')
    else:
        print(f'Pelo Teste Qui-Quadrado, há associação significativa entre {variavel1} e {variavel2}.')

def ks_test(y_proba_0, y_proba_1):
    KS, p_value = stats.ks_2samp(y_proba_0, y_proba_1)

    if p_value > 0.05:
        ks_message = 'Pelo Teste de KS, não há diferença significativa entre as amostras'
    else:
        ks_message = 'Pelo Teste de KS, há diferença significativa entre as amostras'

    return KS, ks_message

def plot_linear_separability(feature_1, feature_2, x_train, y_train, target):
    # Crie um DataFrame para facilitar a visualização com seaborn
    df = pd.DataFrame(x_train, columns=[feature_1, feature_2])
    df['Target'] = y_train[target]

    # Configure o estilo do seaborn para uma boa estética
    sns.set(style="whitegrid")

    # Plote um gráfico de dispersão com diferentes cores para cada classe
    sns.scatterplot(x=feature_1, y=feature_2, hue='Target', data=df, palette="bright")

    # Adicione uma linha de separação linear (hiperplano)
    plt.title('Linear Separability')
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.legend(loc='best')
    plt.show()


def woe(df, feature, target):
    good = df.loc[df[target] == 'BAD'].groupby(feature, as_index = False)[target].count().rename({target:'good'}, axis = 1)
    bad = df.loc[df[target] == 'GOOD'].groupby(feature, as_index = False)[target].count().rename({target:'bad'}, axis = 1)

    woe = good.merge(bad, on = feature, how = 'left')
    woe['percent_good'] = woe['good']/woe['good'].sum()
    woe['percent_bad'] = woe['bad']/woe['bad'].sum()
    woe['woe'] = round(np.log(woe['percent_good']/woe['percent_bad']), 3)
    woe['iv'] = ((woe['percent_good'] - woe['percent_bad'])*np.log(woe['percent_good']/woe['percent_bad'])).sum()

    woe['woe'].fillna(0, inplace = True)
    woe['iv'].fillna(0, inplace = True)

    woe.sort_values(by = 'woe', ascending = True, inplace = True)

    weight_of_evidence = woe['woe'].unique()
    iv = round(woe['iv'].max(), 2)

    x = list(df[feature].unique())
    x.sort()
    y = list(woe['woe'].values)
    plt.figure(figsize = (10, 4))
    plt.plot(x, y, marker = 'o', linestyle = '--', linewidth=2, color = '#1FB3E5')
    for label, value in zip(x, y):
        plt.text(x = label, y = value, s = str(value), fontsize = 20, color = 'red', ha='left', va='center', rotation = 45)
    # plt.title(f'WOE of "{feature}" with an Information Value {iv} ', fontsize=14)
    plt.title(f'Weight of Evidence da variável "{feature}"', fontsize=14)
    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Weight of Evidence', fontsize=14)
    plt.xticks(ha='right', fontsize = 10, rotation = 45)

def iv(df, feature, target):
    good = df.loc[df[target] == 0].groupby(feature, as_index = False)[target].count().rename({target:'good'}, axis = 1)
    bad = df.loc[df[target] == 1].groupby(feature, as_index = False)[target].count().rename({target:'bad'}, axis = 1)

    woe = good.merge(bad, on = feature, how = 'left')
    woe['percent_good'] = woe['good']/woe['good'].sum()
    woe['percent_bad'] = woe['bad']/woe['bad'].sum()
    woe['woe'] = round(np.log(woe['percent_good']/woe['percent_bad']), 3)
    woe['iv'] = ((woe['percent_good'] - woe['percent_bad'])*np.log(woe['percent_good']/woe['percent_bad'])).sum()

    woe['woe'].fillna(0, inplace = True)
    woe['iv'].fillna(0, inplace = True)

    weight_of_evidence = woe['woe'].unique()
    iv = round(woe['iv'].max(), 2)

    dicionario = {feature:iv}

    iv_df = pd.DataFrame(list(dicionario.items()), columns=['Feature', 'IV'])
    
    return iv_df

def metricas_classificacao(classificador, y_train, y_predict_train, y_test, y_predict_test, y_predict_proba_train, y_predict_proba_test):

    predict_proba_train = pd.DataFrame(y_predict_proba_train.tolist(), columns=['predict_proba_0', 'predict_proba_1'])
    predict_proba_test = pd.DataFrame(y_predict_proba_test.tolist(), columns=['predict_proba_0', 'predict_proba_1'])

    # Treino
    accuracy_train = accuracy_score(y_train, y_predict_train)
    precision_train = precision_score(y_train, y_predict_train)
    recall_train = recall_score(y_train, y_predict_train)
    f1_train = f1_score(y_train, y_predict_train)
    roc_auc_train = roc_auc_score(y_train['situacao_do_emprestimo'], predict_proba_train['predict_proba_1'])
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train['situacao_do_emprestimo'], predict_proba_train['predict_proba_1'])
    ks_train = max(tpr_train - fpr_train)
    metricas_treino = pd.DataFrame({'Acuracia': accuracy_train, 'Precisao': precision_train, 'Recall': recall_train, 'F1-Score': f1_train, 'AUC': roc_auc_train, 'KS': ks_train, 'Etapa': 'treino', 'Classificador': classificador}, index=[0])
    
    # Teste
    accuracy_test = accuracy_score(y_test, y_predict_test)
    precision_test = precision_score(y_test, y_predict_test)
    recall_test = recall_score(y_test, y_predict_test)
    f1_test = f1_score(y_test, y_predict_test)
    roc_auc_test = roc_auc_score(y_test['situacao_do_emprestimo'], predict_proba_test['predict_proba_1'])
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test['situacao_do_emprestimo'], predict_proba_test['predict_proba_1'])
    ks_test = max(tpr_test - fpr_test)
    metricas_teste = pd.DataFrame({'Acuracia': accuracy_test, 'Precisao': precision_test, 'Recall': recall_test, 'F1-Score': f1_test, 'AUC': roc_auc_test, 'KS': ks_test, 'Etapa': 'teste', 'Classificador': classificador}, index=[0])
    
    # Consolidando
    metricas_finais = pd.concat([metricas_treino, metricas_teste])

    return metricas_finais

def metricas_classificacao_modelos_juntos(lista_modelos):
    if len(lista_modelos) > 0:
        metricas_modelos = pd.concat(lista_modelos)#.set_index('Classificador')
    else:
        metricas_modelos = lista_modelos[0]
    # Redefina o índice para torná-lo exclusivo
    df = metricas_modelos.reset_index(drop=True)
    df = df.round(2)

    # Função para formatar as células com base na Etapa
    def color_etapa(val):
        color = 'black'
        if val == 'treino':
            color = 'blue'
        elif val == 'teste':
            color = 'red'
        return f'color: {color}; font-weight: bold;'

    # Função para formatar os valores com até duas casas decimais
    def format_values(val):
        if isinstance(val, (int, float)):
            return f'{val:.2f}'
        return val

    # Estilizando o DataFrame
    styled_df = df.style\
        .format(format_values)\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px', subset=pd.IndexSlice[:, :])\
        .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px', subset=pd.IndexSlice[:, 'Acuracia':'F1-Score'])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px', subset=pd.IndexSlice[:, 'Etapa'])\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    # Mostrando o DataFrame estilizado
    styled_df
    return styled_df

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

def metricas_classificacao_final(classificador, df, y, y_predict, y_predict_proba):

    predict_proba = pd.DataFrame(y_predict_proba.tolist(), columns=['predict_proba_0', 'predict_proba_1'])

    # Amostra Final
    accuracy = accuracy_score(y, y_predict)
    precision = precision_score(y, y_predict)
    recall = recall_score(y, y_predict)
    f1 = f1_score(y, y_predict)
    roc_auc = roc_auc_score(y['situacao_do_emprestimo'], predict_proba['predict_proba_1'])
    fpr, tpr, thresholds = roc_curve(y['situacao_do_emprestimo'], predict_proba['predict_proba_1'])
    ks = max(tpr - fpr)
    total, retorno_financeiro_por_caso, valor_de_exposicao_total, return_on_portfolio = retorno_financeiro(df, y_predict)
    total = 'R$' + str(int(round(total/1000000, 0))) + ' MM'
    valor_de_exposicao_total = 'R$' + str(float(round(valor_de_exposicao_total/1000000000, 3))) + 'B'
    rocp = str(return_on_portfolio) + '%'
    metricas_finais = pd.DataFrame({
        # 'Acuracia': accuracy, 
        # 'Precisao': precision, 
        # 'Recall': recall, 
        # 'F1-Score': f1, 
        # 'AUC': roc_auc, 
        # 'KS': ks, 
        'Etapa': 'Amostra Final', 
        'Método': classificador, 
        'Valor Total de Exposição': valor_de_exposicao_total,
        'Retorno Financeiro': total,
        'Return on Credit Portfolio (ROCP)': rocp
    }, index=[0])

    df = metricas_finais.reset_index(drop=True)
    df = df.round(2)

    # Função para formatar as células com base na Etapa
    def color_etapa(val):
        color = 'black'
        if val == 'treino':
            color = 'blue'
        elif val == 'teste':
            color = 'red'
        return f'color: {color}; font-weight: bold;'

    # Função para formatar os valores com até duas casas decimais
    def format_values(val):
        if isinstance(val, (int, float)):
            return f'{val:.2f}'
        return val

    # Estilizando o DataFrame
    styled_df = df.style\
        .format(format_values)\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px', subset=pd.IndexSlice[:, 'Etapa':])\
        .applymap(color_etapa, subset=pd.IndexSlice[:, 'Etapa':])\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    # Mostrando o DataFrame estilizado
    return styled_df


# Exemplo de chamada da função
# Substitua os argumentos pelos seus dados reais
# result = metricas_classificacao_final(classificador, df, y, y_predict, y_predict_proba)
# result.show()  # Certifique-se de ajustar conforme necessário, dependendo do ambiente em que você está executando o código.

def retorno_financeiro(df, y_predict):

    df_aux = df.copy()
    df_aux['qt_parcelas'] = np.where(df_aux['qt_parcelas'] == ' 36 months', 36, 60)
    df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] = df_aux['qt_parcelas'] * df_aux['pagamento_mensal']
    df_aux['y_predict'] = y_predict

    TN = df_aux.loc[(df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 0)].shape[0] # O CARA É BOM E MEU MODELO FALA QUE ELE É BOM
    FN = df_aux.loc[(df_aux['situacao_do_emprestimo'] == 1) & (df_aux['y_predict'] == 0)].shape[0] # O CARA É MAU E MEU MODELO FALA QUE ELE É BOM
    FP = df_aux.loc[(df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 1)].shape[0] # O CARA É BOM E MEU MODELO FALA QUE É MAU
    TP = df_aux.loc[(df_aux['situacao_do_emprestimo'] == 1) & (df_aux['y_predict'] == 1)].shape[0] # O CARA É MAU E O MEU MODELO FALA QUE É MAU

    df_aux['caso'] = np.where((df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 0), 'Verdadeiro Negativo (Cliente Bom | Modelo classifica como Bom) - Ganho a Diferença entre Valor Bruto e Valor com Juros', # Ganha a Diferença entre Valor Bruto e Valor com Juros
                        np.where((df_aux['situacao_do_emprestimo'] == 1) & (df_aux['y_predict'] == 0), 'Falso Negativo (Cliente Mau | Modelo classifica como Bom) - Perco o valor emprestado', # Perde o valor emprestado
                        np.where((df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 1), 'Falso Positivo (Cliente Bom | Modelo classifica como Mau) - Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros', # Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros
                        'Verdadeiro Positivo (Cliente Mau | Modelo classifica como Mau) - Não ganho nada' # Não ganho nada
    )))

    df_aux['retorno_financeiro'] = np.where((df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 0), df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] - df_aux['valor_emprestimo_solicitado'], # Ganha a Diferença entre Valor Bruto e Valor com Juros
                        np.where((df_aux['situacao_do_emprestimo'] == 1) & (df_aux['y_predict'] == 0), df_aux['valor_emprestimo_solicitado']*(-1), # Perde o valor emprestado
                        np.where((df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 1), 0, # Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros (df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] - df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'])*(-1)
                        0 # Não ganho nada
    )))

    valor_de_exposicao_total = int(df_aux['valor_emprestimo_solicitado'].sum())
    retorno_financeiro = int(df_aux['retorno_financeiro'].sum())
    valor_conquistado = valor_de_exposicao_total + retorno_financeiro
    return_on_portfolio = round((retorno_financeiro/valor_de_exposicao_total)*100, 2)
    retorno_financeiro_por_caso = df_aux.groupby('caso', as_index = False)['retorno_financeiro'].sum().sort_values(by = 'retorno_financeiro', ascending = False)

    # Crie um DataFrame a partir dos hiperparâmetros
    df = retorno_financeiro_por_caso.reset_index(drop=True)
    df = df.round(2)

    def color_etapa(val):
        color = 'black'
        if val == 'treino':
            color = 'blue'
        elif val == 'teste':
            color = 'red'
        return f'color: {color}; font-weight: bold;'

    # Função para formatar os valores com até duas casas decimais
    def format_values(val):
        if isinstance(val, (int, float)):
            return f'{val:.2f}'
        return val

    # Estilizando o DataFrame
    styled_df = df.style\
        .format(format_values)\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px')\
        .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    return retorno_financeiro, styled_df, valor_de_exposicao_total, return_on_portfolio


# def retorno_financeiro(df, target, y, y_predict):

#     df_aux = df.copy()

#     TN = df_aux.loc[(df_aux['y_true'] == 0) & (df_aux['y_predict'] == 0)].shape[0] # O CARA É BOM E MEU MODELO FALA QUE ELE É BOM
#     FN = df_aux.loc[(df_aux['y_true'] == 1) & (df_aux['y_predict'] == 0)].shape[0] # O CARA É MAU E MEU MODELO FALA QUE ELE É BOM
#     FP = df_aux.loc[(df_aux['y_true'] == 0) & (df_aux['y_predict'] == 1)].shape[0] # O CARA É BOM E MEU MODELO FALA QUE É MAU
#     TP = df_aux.loc[(df_aux['y_true'] == 1) & (df_aux['y_predict'] == 1)].shape[0] # O CARA É MAU E O MEU MODELO FALA QUE É MAU

#     df_aux['caso'] = np.where((df_aux['y_true'] == 0) & (df_aux['y_predict'] == 0), 'Verdadeiro Negativo (Cliente Bom | Modelo classifica como Bom) - Ganho a Diferença entre Valor Bruto e Valor com Juros', # Ganha a Diferença entre Valor Bruto e Valor com Juros
#                         np.where((df_aux['y_true'] == 1) & (df_aux['y_predict'] == 0), 'Falso Negativo (Cliente Mau | Modelo classifica como Bom) - Perco o valor emprestado', # Perde o valor emprestado
#                         np.where((df_aux['y_true'] == 0) & (df_aux['y_predict'] == 1), 'Falso Positivo (Cliente Bom | Modelo classifica como Mau) - Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros', # Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros
#                         'Verdadeiro Positivo (Cliente Mau | Modelo classifica como Mau) - Não ganho nada' # Não ganho nada
#     )))

#     df_aux['retorno_financeiro'] = np.where((df_aux['y_true'] == 0) & (df_aux['y_predict'] == 0), df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] - df_aux['valor_emprestimo_solicitado'], # Ganha a Diferença entre Valor Bruto e Valor com Juros
#                         np.where((df_aux['y_true'] == 1) & (df_aux['y_predict'] == 0), df_aux['valor_emprestimo_solicitado']*(-1), # Perde o valor emprestado
#                         np.where((df_aux['y_true'] == 0) & (df_aux['y_predict'] == 1), 0, # Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros (df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] - df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'])*(-1)
#                         0 # Não ganho nada
#     )))

#     valor_de_exposicao_total = int(df_aux['valor_emprestimo_solicitado'].sum())
#     retorno_financeiro = int(df_aux['retorno_financeiro'].sum())
#     valor_conquistado = valor_de_exposicao_total + retorno_financeiro
#     return_on_portfolio = round((retorno_financeiro/valor_de_exposicao_total)*100, 2)
#     retorno_financeiro_por_caso = df_aux.groupby('caso', as_index = False)['retorno_financeiro'].sum().sort_values(by = 'retorno_financeiro', ascending = False)

#     # Crie um DataFrame a partir dos hiperparâmetros
#     df = retorno_financeiro_por_caso.reset_index(drop=True)
#     df = df.round(2)

#     def color_etapa(val):
#         color = 'black'
#         if val == 'treino':
#             color = 'blue'
#         elif val == 'teste':
#             color = 'red'
#         return f'color: {color}; font-weight: bold;'

#     # Função para formatar os valores com até duas casas decimais
#     def format_values(val):
#         if isinstance(val, (int, float)):
#             return f'{val:.2f}'
#         return val

#     # Estilizando o DataFrame
#     styled_df = df.style\
#         .format(format_values)\
#         .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px')\
#         .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
#         .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
#         .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
#         .set_table_styles([
#             {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
#         ])

#     return retorno_financeiro, styled_df, valor_de_exposicao_total, return_on_portfolio


def separa_feature_target(target, dados):
    x = dados.drop(target, axis = 1)
    y = dados[[target]]

    return x, y


def separa_treino_teste(target, dados, size):
    x = dados.drop(target, axis = 1)
    y = dados[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= size, random_state = 42)

    df_train = pd.concat([y_train, x_train], axis = 1)
    df_test = pd.concat([y_test, x_test], axis = 1)

    return df_train, df_test

def discretiza_variavel(df, variavel_quant, variavel_qualit, bins, labels, right):
    df[variavel_qualit] = pd.cut(
        df[variavel_quant], 
        bins= bins, 
        labels= labels, 
        right = right
    )
    df.drop(variavel_quant, axis = 1, inplace = True)

def transform_to_deciles(df, variavel_continua):
    # Calcula os limites dos deciles
    decile_limits = [i / 10 for i in range(11)]  # [0.0, 0.1, 0.2, ..., 1.0]
    
    # Aplica a função qcut para transformar a variável em deciles
    deciles = pd.qcut(df[variavel_continua], q=10, labels=False, duplicates='drop')
    
    return deciles
    
def transform_to_quantiles(df, variavel_continua):

    decile_limits = [df[variavel_continua].quantile(i / 10) for i in range(10)]  # Calcula os pontos de corte
    
    return decile_limits

# Exemplo de uso:
# Suponha que você tenha um DataFrame df e uma coluna chamada 'variavel_contínua'
# df['decile_values'] = transform_to_decile_values(df, 'variavel_contínua')


def transform_to_percentiles(df, variavel_continua):
    # Calcula os limites dos percentis de 0,01 a 0,95 em incrementos de 0,01
    percentile_limits = [i / 100 for i in range(1, 100, 10)]  # [0.01, 0.02, ..., 0.95]
    
    # Aplica a função qcut para transformar a variável em percentis
    percentiles = pd.qcut(df[variavel_continua], q=percentile_limits, labels=False, duplicates='drop')
    
    return percentiles

def simple_imputer(df):

    df_aux = df.copy()
    imputer = SimpleImputer(strategy = 'median')
    imputer.fit(df_aux)

    return imputer


def Classificador(classificador, x_train, y_train, x_test, y_test, class_weight):

    def simple_imputer(df):

        df_aux = df.copy()
        imputer = SimpleImputer(strategy = 'median')
        imputer.fit(df_aux)

        return imputer
    
    cols = list(x_train.columns)
    imputer = simple_imputer(x_train)
    x_train = pd.DataFrame(imputer.transform(x_train), columns = x_train.columns)
    x_test = pd.DataFrame(imputer.transform(x_test), columns = x_test.columns)

    # Define as colunas categóricas e numéricas
    models = {
        'Regressão Logística': make_pipeline(
            ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                ('scaler', make_pipeline(MinMaxScaler()), cols)
            ]),
            LogisticRegression(
                random_state=42, # Semente aleatória para reproducibilidade dos resultados
                class_weight={0: 1, 1: class_weight}, # Peso atribuído às classes. Pode ser útil para lidar com conjuntos de dados desbalanceados.
                C=1, # Parâmetro de regularização inversa. Controla a força da regularização.
                penalty='l2', # Tipo de regularização. 'l1', 'l2', 'elasticnet', ou 'none'.
                max_iter=50, # Número máximo de iterações para a convergência do otimizador.
                solver='liblinear' # Algoritmo de otimização. 'newton-cg', 'lbfgs', 'liblinear' (gradiente descendente), 'sag' (Stochastic gradient descent), 'saga' (Stochastic gradient descent que suporta reg L1).
                )
        ),
        'Naive Bayes': make_pipeline(
            ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols)
            ]),
            GaussianNB(priors = [0.88, 0.12]) # Probabilidade a Priori
        ),
        'KNN Classifier': make_pipeline(
            ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                ('scaler', make_pipeline(MinMaxScaler()), cols)
            ]),
            KNeighborsClassifier(n_neighbors=3)  # Escolha o número adequado de vizinhos
        ),
        'SVM': make_pipeline(
            ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                ('scaler', make_pipeline(MinMaxScaler()), cols)
            ]),
            SVC(kernel='linear', class_weight={0: 1, 1: class_weight}, cache_size=1000, probability=True, random_state=42)
        ),
        'Random Forest': make_pipeline(
            ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols)
            ]),
        RandomForestClassifier(
            random_state=42,            # Semente aleatória para reproducibilidade dos resultados
            criterion='entropy',       # Critério usado para medir a qualidade de uma divisão
            n_estimators=50,           # Número de árvores na floresta (equivalente ao n_estimators no XGBoost)
            max_depth = 6,                # Profundidade máxima de cada árvore
            class_weight={0:1, 1:class_weight},  # Peso das classes em casos desequilibrados
            min_samples_split=2,        # O número mínimo de amostras necessárias para dividir um nó interno
            min_samples_leaf=1,         # O número mínimo de amostras necessárias para ser um nó folha
            max_features='auto',        # O número máximo de características a serem consideradas para a melhor divisão
            max_leaf_nodes=None,        # O número máximo de folhas que uma árvore pode ter
            bootstrap=True               # Se deve ou não amostrar com substituição ao construir árvores
            )
        ),
        'XGBoost': make_pipeline(
            ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols)
            ]),
        XGBClassifier(
            random_state=42,            # Semente aleatória para reproducibilidade dos resultados
            n_estimators=50,           # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
            max_depth = 6,                # Profundidade máxima de cada árvore
            learning_rate = 0.04,         # Taxa de aprendizado - controla a contribuição de cada árvore
            eval_metric='logloss',      # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
            objective='binary:logistic',# Define o objetivo do modelo, 'binary:logistic' para classificação binária
            scale_pos_weight=class_weight,  # Peso das classes positivas em casos desequilibrados
            reg_alpha=1,                # Termo de regularização L1 (penalidade nos pesos)
            reg_lambda=0,               # Termo de regularização L2 (penalidade nos quadrados dos pesos)
            gamma=1,                    # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
            colsample_bytree=0.5,       # Fração de características a serem consideradas ao construir cada árvore
            subsample=0.5,              # Fração de amostras a serem usadas para treinar cada árvore
            base_score=0.5              # Threshold de Probabilidade de Decisão do Classificador (geralmente é 0.5 para problemas de classificação binária)
            )
        )
        # 'MLP': make_pipeline(
        #     ColumnTransformer([
        #         ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
        #         ('scaler', make_pipeline(MinMaxScaler()), cols)
        #     ]),
        #     MLPClassifier(
        #         hidden_layer_sizes=(100,),  # Número de neurônios em cada camada oculta, ajuste conforme necessário
        #         activation='relu',  # Função de ativação para as camadas ocultas
        #         solver='adam',  # Algoritmo de otimização
        #         alpha=0.0001,  # Termo de regularização
        #         batch_size='auto',  # Tamanho do lote para otimização em lote, 'auto' ajusta automaticamente
        #         learning_rate='constant',  # Taxa de aprendizado
        #         learning_rate_init=0.001,  # Taxa de aprendizado inicial
        #         max_iter=200,  # Número máximo de iterações
        #         random_state=42
        #     )
        # )
    }

    if classificador in models:
        model = models[classificador]
    else:
        print('Utilize Regressão Logística, Random Forest ou XGBoost como opções de Classificadores!')

    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    y_proba_train = model.predict_proba(x_train)
    y_proba_test = model.predict_proba(x_test)

    return model, y_pred_train, y_pred_test, y_proba_train, y_proba_test


def validacao_cruzada_classificacao(classificador, df, target_column, n_splits, class_weight):

    def numero_de_anos_emprego_atual(df):
        df['qt_anos_mesmo_emprego'] = (df['qt_anos_mesmo_emprego'].replace({'< 1 year':0, '1 year':1, '2 years':2, '3 years':3, '4 years':4, '5 years':5, '6 years':6, '7 years':7, '8 years':8, '9 years':9,'10+ years':10}).fillna(0))
        df['qt_anos_mesmo_emprego'] = df['qt_anos_mesmo_emprego'].apply(lambda x:int(x))
        df['qt_anos_mesmo_emprego'] = np.where(df['qt_anos_mesmo_emprego'] <= 3, '3_YEARS', 
                            np.where(df['qt_anos_mesmo_emprego'] <= 6, '6_YEARS',
                            np.where(df['qt_anos_mesmo_emprego'] <= 9, '9_YEARS',
                            '10_YEARS+')))
        return df['qt_anos_mesmo_emprego']

    def numero_de_registros_negativos(df):

        df = df[['situacao_do_emprestimo', 'registros_publicos_depreciativos']].copy()
        df[['registros_publicos_depreciativos']] = np.where(df[['registros_publicos_depreciativos']] == 0, 'sem_registros_negativos', 'com_registros_negativos')

        return df['registros_publicos_depreciativos']

    def consulta_de_credito_nos_ultimos_6_meses(df):
        df = df[['situacao_do_emprestimo', 'consultas_credito_6meses']].copy()
        df[['consultas_credito_6meses']] = np.where(df[['consultas_credito_6meses']] == 0, 'sem_consultas', 'com_consultas')

        return df['consultas_credito_6meses']

    def compromento_de_renda(df): 
        df_aux = df[['faturamento_anual', 'pagamento_mensal', 'valor_emprestimo_solicitado', 'qt_parcelas', 'taxa_de_juros', 'situacao_do_emprestimo']].copy()
        df_aux['qt_parcelas'] = np.where(df_aux['qt_parcelas'] == ' 36 months', 36, 60)
        df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] = df_aux['pagamento_mensal']*df_aux['qt_parcelas']
        df_aux['pagamento_anual'] = np.where(df_aux['qt_parcelas'] == ' 36 months', df_aux['valor_emprestimo_solicitado_com_taxa_de_juros']/3, df_aux['valor_emprestimo_solicitado_com_taxa_de_juros']/5)
        df_aux['comprometimento_de_renda_anual'] = ((df_aux['pagamento_anual']/df_aux['faturamento_anual'])*100).round(2)
        
        return df_aux['comprometimento_de_renda_anual']

    def numero_incidencias_inadimplencia_vencidas_30d(df):
        df_aux = df[['situacao_do_emprestimo', 'inadimplencia_vencida_30dias']].copy()
        df_aux['inadimplencia_vencida_30dias'] = np.where(df_aux[['inadimplencia_vencida_30dias']] == 0, 'sem_inadimplencia_vencida', 'com_inadimplencia_vencida')

        return df_aux['inadimplencia_vencida_30dias']

    def n_meses_produto_credito_atual(df):
        df = df.copy()
        df['data_financiamento_emprestimo'] = pd.to_datetime(df['data_financiamento_emprestimo'], format = '%b-%y')
        df['mths_since_data_financiamento_emprestimo'] = round(pd.to_numeric((pd.to_datetime('2023-09-20') - df['data_financiamento_emprestimo'])/np.timedelta64(1, 'M')))
        df['mths_since_data_financiamento_emprestimo'] = df['mths_since_data_financiamento_emprestimo'].fillna(df['mths_since_data_financiamento_emprestimo'].median())
        df['mths_since_data_financiamento_emprestimo'] = np.where(df['mths_since_data_financiamento_emprestimo'] < 0, df['mths_since_data_financiamento_emprestimo'].median(), df['mths_since_data_financiamento_emprestimo'])
        df['mths_since_data_financiamento_emprestimo'] = df['mths_since_data_financiamento_emprestimo'].apply(lambda x:int(x))
        df['data_financiamento_emprestimo'] = df['mths_since_data_financiamento_emprestimo']

        return df['data_financiamento_emprestimo']

    def n_meses_primeiro_produto_credito(df):
        df = df.copy()
        df['data_contratacao_primeiro_produto_credito'] = pd.to_datetime(df['data_contratacao_primeiro_produto_credito'], format = '%b-%y')
        df['mths_since_data_contratacao_primeiro_produto_credito'] = round(pd.to_numeric((pd.to_datetime('2023-09-20') - df['data_contratacao_primeiro_produto_credito'])/np.timedelta64(1, 'M')))
        df['mths_since_data_contratacao_primeiro_produto_credito'] = df['mths_since_data_contratacao_primeiro_produto_credito'].fillna(df['mths_since_data_contratacao_primeiro_produto_credito'].median())
        df['mths_since_data_contratacao_primeiro_produto_credito'] = np.where(df['mths_since_data_contratacao_primeiro_produto_credito'] < 0, df['mths_since_data_contratacao_primeiro_produto_credito'].median(), df['mths_since_data_contratacao_primeiro_produto_credito'])
        df['mths_since_data_contratacao_primeiro_produto_credito'] = df['mths_since_data_contratacao_primeiro_produto_credito'].apply(lambda x:int(x))
        df['data_contratacao_primeiro_produto_credito'] = df['mths_since_data_contratacao_primeiro_produto_credito']
        
        return df['data_contratacao_primeiro_produto_credito']

    def produto_disponivel_publicamente(df):
        df_aux = df[['situacao_do_emprestimo', 'produto_disponivel_publicamente']].copy()
        df_aux['produto_disponivel_publicamente'] = np.where(df_aux[['produto_disponivel_publicamente']] == 0, 'sem_disponibilidade_publica', 'com_disponibilidade_publica')

        return df_aux['produto_disponivel_publicamente']

    def formato_features_binarias(df):
        df['qt_parcelas'] = np.where(df['qt_parcelas'] == ' 36 months', 0, 1)
        df['registros_publicos_depreciativos'] = np.where(df['qt_parcelas'] == 'sem_registros_negativos', 0, 1)
        df['inadimplencia_vencida_30dias'] = np.where(df['inadimplencia_vencida_30dias'] == 'sem_inadimplencia_vencida', 0, 1)
        df['tipo_de_concessao_do_credor'] = np.where(df['tipo_de_concessao_do_credor'] == 'f', 0, 1)
        df['plano_de_pagamento'] = np.where(df['plano_de_pagamento'] == 'n', 0, 1)
        df['renda_comprovada'] = np.where(df['renda_comprovada'] == 'Source Verified', 0, 1)
        df['consultas_credito_6meses'] = np.where(df['consultas_credito_6meses'] == 'sem_consultas', 0, 1)
        df['produto_disponivel_publicamente'] = np.where(df['consultas_credito_6meses'] == 'sem_disponibilidade_publica', 0, 1)

        return df

    def target_encoder_bad_rate(df, tipo):
        categoricas = ['grau_de_emprestimo', 'subclasse_de_emprestimo', 'produto_de_credito', 'qt_anos_mesmo_emprego', 'status_propriedade_residencial', 'estado']
        df_aux_2 = df.copy()
        if tipo == 'Criação':
            for cat in categoricas:
                df_aux = df[[f'{cat}', 'situacao_do_emprestimo']].copy()
                good = pd.DataFrame(df_aux.loc[df_aux['situacao_do_emprestimo'] == 0].groupby(f'{cat}', as_index = False)['situacao_do_emprestimo'].count()).rename({'situacao_do_emprestimo':'qt_good'}, axis = 1)
                bad = pd.DataFrame(df_aux.loc[df_aux['situacao_do_emprestimo'] == 1].groupby(f'{cat}', as_index = False)['situacao_do_emprestimo'].count()).rename({'situacao_do_emprestimo':'qt_bad'}, axis = 1)
                df_aux = good.merge(bad, on = f'{cat}', how = 'left')
                df_aux['qt_total'] = df_aux['qt_good'] + df_aux['qt_bad']
                df_aux[f'{cat}_enc'] = ((df_aux['qt_bad']/df_aux['qt_total'])*100).round(2)
                df_aux[f'{cat}_enc'] = df_aux[f'{cat}_enc'].apply(lambda x:float(x))
                df_aux = df_aux[[f'{cat}', f'{cat}_enc']].drop_duplicates().sort_values(by = f'{cat}_enc', ascending = True)
                df_aux.to_csv(f'features/{cat}_enc.csv', index = False)
                df_aux_2 = df_aux_2.merge(df_aux[[f'{cat}', f'{cat}_enc']], on = f'{cat}', how = 'left')
                df_aux_2.drop(f'{cat}', axis = 1, inplace = True)
        else:
            for cat in categoricas:
                ft = pd.read_csv(f'features/{cat}_enc.csv')
                replace_dict = dict(zip(ft[f'{cat}'], ft[f'{cat}_enc']))
                df_aux_2[f'{cat}_enc'] = df_aux_2[f'{cat}'].replace(replace_dict)

        return df_aux_2


    def simple_imputer(df):

        df_aux = df.copy()
        imputer = SimpleImputer(strategy = 'median')
        imputer.fit(df_aux)

        return imputer

    columns_selected = ['situacao_do_emprestimo', 'qt_parcelas','grau_de_emprestimo','subclasse_de_emprestimo','produto_de_credito', 'inadimplencia_vencida_30dias', 'valor_emprestimo_solicitado','taxa_de_juros','data_financiamento_emprestimo','produto_disponivel_publicamente','plano_de_pagamento','tipo_de_concessao_do_credor','pagamento_mensal','qt_anos_mesmo_emprego','status_propriedade_residencial',
    'renda_comprovada','faturamento_anual','estado', 'limite_total_produtos_credito','limite_total_rotativos','limite_rotativos_utilizado','taxa_utilizacao_limite_rotativos','qt_produtos_credito_contratados_atualmente','qt_produtos_credito_contratados_historicamente','registros_publicos_depreciativos','consultas_credito_6meses','data_contratacao_primeiro_produto_credito','qt_meses_desde_ultimo_registro_publico', 'qt_meses_classificacao_mais_recente_90dias',
    'qt_meses_ultima_inadimplencia']

    df_raw = df[columns_selected].copy()

    # Feature Selection

    features_selected = pd.read_csv('features/features_selected.csv')
    features_selected = features_selected.loc[features_selected['importance'] > 1] 
    features_selected = list(features_selected['feature'].unique()) + ['situacao_do_emprestimo']

    # Inicializar o KFold para dividir os dados
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Listas para armazenar as métricas para cada fold
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_scores = []  # Lista para armazenar os valores de AUC
    ks_scores = []   # Lista para armazenar os valores de KS
    cv_results = []  # Lista para armazenar os resultados de validação cruzada

    # Loop pelos folds
    for train_idx, test_idx in kfold.split(df_raw):
        # Criar DataFrames de treino e teste
        df_train = df_raw.iloc[train_idx]
        df_test = df_raw.iloc[test_idx]

        # Criação das Features sem Data Leakage
        df_train['qt_anos_mesmo_emprego'] = numero_de_anos_emprego_atual(df_train)
        df_train['registros_publicos_depreciativos'] = numero_de_registros_negativos(df_train)
        df_train['consultas_credito_6meses'] = consulta_de_credito_nos_ultimos_6_meses(df_train)
        df_train['comprometimento_de_renda_anual'] = compromento_de_renda(df_train)
        df_train['inadimplencia_vencida_30dias'] = numero_incidencias_inadimplencia_vencidas_30d(df_train)
        df_train['data_financiamento_emprestimo'] = n_meses_produto_credito_atual(df_train)
        df_train['data_contratacao_primeiro_produto_credito'] = n_meses_primeiro_produto_credito(df_train)
        df_train = formato_features_binarias(df_train)
        df_train = target_encoder_bad_rate(df_train, 'escoragem')

        df_test['qt_anos_mesmo_emprego'] = numero_de_anos_emprego_atual(df_test)
        df_test['registros_publicos_depreciativos'] = numero_de_registros_negativos(df_test)
        df_test['consultas_credito_6meses'] = consulta_de_credito_nos_ultimos_6_meses(df_test)
        df_test['comprometimento_de_renda_anual'] = compromento_de_renda(df_test)
        df_test['inadimplencia_vencida_30dias'] = numero_incidencias_inadimplencia_vencidas_30d(df_test)
        df_test['data_financiamento_emprestimo'] = n_meses_produto_credito_atual(df_test)
        df_test['data_contratacao_primeiro_produto_credito'] = n_meses_primeiro_produto_credito(df_test)
        df_test = formato_features_binarias(df_test)
        df_test = target_encoder_bad_rate(df_test, 'escoragem')

        # Filtragem das Features que passaram no Feature Selection
        df_train = df_train[features_selected]
        df_test = df_test[features_selected]

        # Separação Feature e Target
        x_train, y_train = separa_feature_target('situacao_do_emprestimo', df_train)
        x_test, y_test = separa_feature_target('situacao_do_emprestimo', df_test)
        
        # Imputer
        cols = list(x_train.columns)
        imputer = simple_imputer(x_train)
        x_train = pd.DataFrame(imputer.transform(x_train), columns = x_train.columns)
        x_test = pd.DataFrame(imputer.transform(x_test), columns = x_test.columns)

    # Define as colunas categóricas e numéricas
        models = {
            'Regressão Logística': make_pipeline(
                ColumnTransformer([
                    ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                    ('scaler', make_pipeline(MinMaxScaler()), cols)
                ]),
                LogisticRegression(
                    random_state=42, # Semente aleatória para reproducibilidade dos resultados
                    class_weight={0: 1, 1: class_weight}, # Peso atribuído às classes. Pode ser útil para lidar com conjuntos de dados desbalanceados.
                    C=1, # Parâmetro de regularização inversa. Controla a força da regularização.
                    penalty='l2', # Tipo de regularização. 'l1', 'l2', 'elasticnet', ou 'none'.
                    max_iter=50, # Número máximo de iterações para a convergência do otimizador.
                    solver='liblinear' # Algoritmo de otimização. 'newton-cg', 'lbfgs', 'liblinear' (gradiente descendente), 'sag' (Stochastic gradient descent), 'saga' (Stochastic gradient descent que suporta reg L1).
                    )
            ),
            'Naive Bayes': make_pipeline(
                ColumnTransformer([
                    ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols)
                ]),
                GaussianNB(priors = [0.88, 0.12]) # Probabilidade a Priori
            ),
            'KNN Classifier': make_pipeline(
                ColumnTransformer([
                    ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                    ('scaler', make_pipeline(MinMaxScaler()), cols)
                ]),
                KNeighborsClassifier(n_neighbors=3)  # Escolha o número adequado de vizinhos
            ),
            'SVM': make_pipeline(
                ColumnTransformer([
                    ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                    ('scaler', make_pipeline(MinMaxScaler()), cols)
                ]),
                SVC(kernel='linear', 
                    class_weight={0: 1, 1: class_weight}, 
                    cache_size=1000,
                    probability=True,
                    random_state=42
                    )
            ),
            'Random Forest': make_pipeline(
                ColumnTransformer([
                    ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols)
                ]),
            RandomForestClassifier(
                random_state=42,            # Semente aleatória para reproducibilidade dos resultados
                criterion='entropy',       # Critério usado para medir a qualidade de uma divisão
                n_estimators=50,           # Número de árvores na floresta (equivalente ao n_estimators no XGBoost)
                max_depth = 6,                # Profundidade máxima de cada árvore
                class_weight={0:1, 1:class_weight},  # Peso das classes em casos desequilibrados
                min_samples_split=2,        # O número mínimo de amostras necessárias para dividir um nó interno
                min_samples_leaf=1,         # O número mínimo de amostras necessárias para ser um nó folha
                max_features='auto',        # O número máximo de características a serem consideradas para a melhor divisão
                max_leaf_nodes=None,        # O número máximo de folhas que uma árvore pode ter
                bootstrap=True               # Se deve ou não amostrar com substituição ao construir árvores
                )
            ),
            'XGBoost': make_pipeline(
                ColumnTransformer([
                    ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols)
                ]),
            XGBClassifier(
                random_state=42,            # Semente aleatória para reproducibilidade dos resultados
                tree_method = 'gpu_hist',
                n_estimators=50,           # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
                max_depth = 6,                # Profundidade máxima de cada árvore
                learning_rate = 0.04,         # Taxa de aprendizado - controla a contribuição de cada árvore
                eval_metric='logloss',      # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
                objective='binary:logistic',# Define o objetivo do modelo, 'binary:logistic' para classificação binária
                scale_pos_weight=class_weight,  # Peso das classes positivas em casos desequilibrados
                reg_alpha=1,                # Termo de regularização L1 (penalidade nos pesos)
                reg_lambda=0,               # Termo de regularização L2 (penalidade nos quadrados dos pesos)
                gamma=1,                    # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
                colsample_bytree=0.5,       # Fração de características a serem consideradas ao construir cada árvore
                subsample=0.5,              # Fração de amostras a serem usadas para treinar cada árvore
                base_score=0.5              # Threshold de Probabilidade de Decisão do Classificador (geralmente é 0.5 para problemas de classificação binária)
                )
            )
            # 'MLP': make_pipeline(
            #     ColumnTransformer([
            #         ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
            #         ('scaler', make_pipeline(MinMaxScaler()), cols)
            #     ]),
            #     MLPClassifier(
            #         hidden_layer_sizes=(100,),  # Número de neurônios em cada camada oculta, ajuste conforme necessário
            #         activation='relu',  # Função de ativação para as camadas ocultas
            #         solver='adam',  # Algoritmo de otimização
            #         alpha=0.0001,  # Termo de regularização
            #         batch_size='auto',  # Tamanho do lote para otimização em lote, 'auto' ajusta automaticamente
            #         learning_rate='constant',  # Taxa de aprendizado
            #         learning_rate_init=0.001,  # Taxa de aprendizado inicial
            #         max_iter=200,  # Número máximo de iterações
            #         random_state=42
            #     )
            # )
        }

        if classificador in models:
            model = models[classificador]
        else:
            print('Utilize Regressão Logística, Random Forest ou XGBoost como opções de Classificadores!')

        # Treinar o modelo usando os dados de treinamento
        model.fit(x_train, y_train)

        # Obter as probabilidades previstas para ambas as classes
        y_proba = model.predict_proba(x_test)

        # Fazer as previsões usando o modelo nos dados de teste
        y_pred = model.predict(x_test)

        # Calcular as métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        y_proba = model.predict_proba(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ks = max(tpr - fpr)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(roc_auc)
        ks_scores.append(ks)

        # Adicionar resultados de validação cruzada ao DataFrame
        fold_results = pd.DataFrame({
            'situacao_do_emprestimo': y_test['situacao_do_emprestimo'].values,
            'y_predict': y_pred,
            'predict_proba_0': y_proba[:, 0],  # Probabilidade da classe 0
            'predict_proba_1': y_proba[:, 1]  # Probabilidade da classe 1
        })
        cv_results.append(fold_results)


    # Calcular a média das métricas para todos os folds
    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    mean_auc = np.mean(auc_scores),
    mean_ks = np.mean(ks_scores)

    # Criar um DataFrame com as métricas
    metricas_finais = pd.DataFrame({
        'Acuracia': mean_accuracy,
        'Precisao': mean_precision,
        'Recall': mean_recall,
        'F1-Score': mean_f1,
        'AUC':mean_auc,
        'KS': mean_ks,
        'Etapa': 'validacao_cruzada',
        'Classificador': classificador
    }, index=[1])

    return metricas_finais, cv_results

# def modelo_otimizado_hyperopt_(classificador, target, x_train, y_train, x_test, y_test):

#     cols = list(x_train.columns)

#     # Define o ColumnTransformer
#     preprocessor = ColumnTransformer([
#         ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
#         ('scaler', make_pipeline(MinMaxScaler()), cols)
#     ])

#     # Divide o conjunto de treinamento em treinamento e validação
#     x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

#     # Função de avaliação para o Hyperopt
#     def objective(params):
#         model = make_pipeline(
#             preprocessor,
#             XGBClassifier(
#                 random_state=42,
#                 eval_metric='logloss',
#                 objective='binary:logistic',
#                 **params
#             )
#         )

#         # Treina o modelo com Early Stopping usando o conjunto de validação
#         model.fit(
#             x_train, 
#             y_train.values.ravel(),
#             eval_set=[(x_val, y_val.values.ravel())],  # Conjunto de dados utilizado para otimização
#             early_stopping_rounds=10,  # Número de iterações sem melhoria no conjunto de validação para parar o treinamento
#             verbose=False
#         )

#         # Obtém a melhor iteração do modelo
#         best_iteration = model.named_steps['xgbclassifier'].best_iteration

#         # Use a métrica de validação cruzada para otimização
#         score = model.named_steps['xgbclassifier'].best_score
#         return -score  # Hyperopt minimiza a função objetivo, então multiplicamos por -1

#     # Espaço de busca para os hiperparâmetros
#     space = {
#         'n_estimators': hp.choice('n_estimators', [10, 20, 50, 100]),
#         'max_depth': hp.choice('max_depth', [4, 5, 7, 8, 9, 10]),
#         'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
#         'reg_alpha': hp.uniform('reg_alpha', 0.5, 1),
#         'reg_lambda': hp.uniform('reg_lambda', 0.5, 1),
#         'gamma': hp.uniform('gamma', 0.5, 1),
#         'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
#         'subsample': hp.uniform('subsample', 0.5, 1),
#         'scale_pos_weight': hp.choice('scale_pos_weight', [3, 5, 8, 10, 12, 14]),
#         'base_score': hp.uniform('base_score', 0.30, 0.90)
#     }

#     # Executa a otimização
#     trials = Trials()
#     best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials, verbose=0)


#     # Cria o modelo final com os melhores hiperparâmetros encontrados
#     best_model = XGBClassifier(
#         n_estimators=int(best_params['n_estimators']),
#         max_depth=int(best_params['max_depth']),
#         learning_rate=best_params['learning_rate'],
#         subsample=best_params['subsample'],
#         reg_alpha=best_params['reg_alpha'],
#         reg_lambda=best_params['reg_lambda'],
#         gamma=best_params['gamma'],
#         colsample_bytree=best_params['colsample_bytree'],
#         scale_pos_weight=int(best_params['scale_pos_weight']),
#         base_score=best_params['base_score'],
#         random_state=42
#     )

#     # Treina o modelo final no conjunto completo de treinamento
#     best_model.fit(x_train, y_train.values.ravel())

#     y_pred_train = best_model.predict(x_train)
#     y_pred_test = best_model.predict(x_test)

#     y_proba_train = best_model.predict_proba(x_train)
#     y_proba_test = best_model.predict_proba(x_test)

#     return best_model, y_pred_train, y_pred_test, y_proba_train, y_proba_test, best_params


def otimizacao(classificador, x_train, y_train, x_test, y_test):
    def simple_imputer(df):

        df_aux = df.copy()
        imputer = SimpleImputer(strategy = 'median')
        imputer.fit(df_aux)

        return imputer
    
    cols = list(x_train.columns)
    imputer = simple_imputer(x_train)
    x_train = pd.DataFrame(imputer.transform(x_train), columns = x_train.columns)
    x_test = pd.DataFrame(imputer.transform(x_test), columns = x_test.columns)

    # Define o ColumnTransformer
    preprocessor = ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                ('scaler', make_pipeline(MinMaxScaler()), cols)
            ])

    # Define o modelo de XGBoost com a otimização de hiperparâmetros via BayesSearch
    model = make_pipeline(
        preprocessor,
        BayesSearchCV(
            XGBClassifier(random_state=42, tree_method = 'gpu_hist', eval_metric='logloss', objective='binary:logistic'),
            {
                'n_estimators': (99, 100), # Número de Árvores construídas
                'max_depth': (7, 8, 9), # Profundidade Máxima de cada Árvore
                'learning_rate': (0.03, 0.05), # Tamanho do passo utilizado no Método do Gradiente Descendente
                'reg_alpha':(0.5, 1), # Valor do Alpha aplicado durante a Regularização Lasso L1 
                'reg_lambda':(0.5, 1), # Valor do Lambda aplicado durante a Regularização Ridge L2
                'gamma':(0.5, 1), # Valor mínimo permitido para um Nó de Árvore ser aceito. Ajuda a controlar o crescimento das Árvores, evitando divisões insignificantes
                'colsample_bytree':(0.5, 1), # Porcentagem de Colunas utilizada para a amostragem aleatória durante a criação das Árvores
                'subsample':(0.5, 1), # Porcentagem de Linhas utilizada para a amostragem aleatória durante a criação das Árvores
                'scale_pos_weight':(4, 5, 6, 7, 8), # Peso atribuído a classe positiva, aumentando a importância da classe minoritária
                #'base_score':(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) # Threshold de Probabilidade de decisão do modelo
            },
            n_iter=10,
            random_state=42,
            n_jobs=-1,
            scoring='recall',
            cv=5
        )
    )

    # Treina o modelo
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    y_proba_train = model.predict_proba(x_train)
    y_proba_test = model.predict_proba(x_test)

    melhores_hiperparametros = model.named_steps['bayessearchcv'].best_params_
    hiperparametros = pd.DataFrame([melhores_hiperparametros])

    best_hiperpams = []
    for chave, valor in melhores_hiperparametros.items():
        best_hiperpams.append([chave, valor])

    pivot = pd.DataFrame(best_hiperpams).T
    pivot.columns = pivot.iloc[0]
    pivot = pivot.drop(0)

    # Crie um DataFrame a partir dos hiperparâmetros
    df = hiperparametros.reset_index(drop=True)
    df = df.round(2)

    def color_etapa(val):
        color = 'black'
        if val == 'treino':
            color = 'blue'
        elif val == 'teste':
            color = 'red'
        return f'color: {color}; font-weight: bold;'

    # Função para formatar os valores com até duas casas decimais
    def format_values(val):
        if isinstance(val, (int, float)):
            return f'{val:.2f}'
        return val

    # Estilizando o DataFrame
    styled_df = df.style\
        .format(format_values)\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px')\
        .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    return model, y_pred_train, y_pred_test, y_proba_train, y_proba_test, styled_df, pivot

def validacao_cruzada_classificacao_otimizada(classificador, df, target_column, n_splits, best_hiperpams):

    def numero_de_anos_emprego_atual(df):
        df['qt_anos_mesmo_emprego'] = (df['qt_anos_mesmo_emprego'].replace({'< 1 year':0, '1 year':1, '2 years':2, '3 years':3, '4 years':4, '5 years':5, '6 years':6, '7 years':7, '8 years':8, '9 years':9,'10+ years':10}).fillna(0))
        df['qt_anos_mesmo_emprego'] = df['qt_anos_mesmo_emprego'].apply(lambda x:int(x))
        df['qt_anos_mesmo_emprego'] = np.where(df['qt_anos_mesmo_emprego'] <= 3, '3_YEARS', 
                            np.where(df['qt_anos_mesmo_emprego'] <= 6, '6_YEARS',
                            np.where(df['qt_anos_mesmo_emprego'] <= 9, '9_YEARS',
                            '10_YEARS+')))
        return df['qt_anos_mesmo_emprego']

    def numero_de_registros_negativos(df):

        df = df[['situacao_do_emprestimo', 'registros_publicos_depreciativos']].copy()
        df[['registros_publicos_depreciativos']] = np.where(df[['registros_publicos_depreciativos']] == 0, 'sem_registros_negativos', 'com_registros_negativos')

        return df['registros_publicos_depreciativos']

    def consulta_de_credito_nos_ultimos_6_meses(df):
        df = df[['situacao_do_emprestimo', 'consultas_credito_6meses']].copy()
        df[['consultas_credito_6meses']] = np.where(df[['consultas_credito_6meses']] == 0, 'sem_consultas', 'com_consultas')

        return df['consultas_credito_6meses']

    def compromento_de_renda(df): 
        df_aux = df[['faturamento_anual', 'pagamento_mensal', 'valor_emprestimo_solicitado', 'qt_parcelas', 'taxa_de_juros', 'situacao_do_emprestimo']].copy()
        df_aux['qt_parcelas'] = np.where(df_aux['qt_parcelas'] == ' 36 months', 36, 60)
        df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] = df_aux['pagamento_mensal']*df_aux['qt_parcelas']
        df_aux['pagamento_anual'] = np.where(df_aux['qt_parcelas'] == ' 36 months', df_aux['valor_emprestimo_solicitado_com_taxa_de_juros']/3, df_aux['valor_emprestimo_solicitado_com_taxa_de_juros']/5)
        df_aux['comprometimento_de_renda_anual'] = ((df_aux['pagamento_anual']/df_aux['faturamento_anual'])*100).round(2)
        
        return df_aux['comprometimento_de_renda_anual']

    def numero_incidencias_inadimplencia_vencidas_30d(df):
        df_aux = df[['situacao_do_emprestimo', 'inadimplencia_vencida_30dias']].copy()
        df_aux['inadimplencia_vencida_30dias'] = np.where(df_aux[['inadimplencia_vencida_30dias']] == 0, 'sem_inadimplencia_vencida', 'com_inadimplencia_vencida')

        return df_aux['inadimplencia_vencida_30dias']

    def n_meses_produto_credito_atual(df):
        df = df.copy()
        df['data_financiamento_emprestimo'] = pd.to_datetime(df['data_financiamento_emprestimo'], format = '%b-%y')
        df['mths_since_data_financiamento_emprestimo'] = round(pd.to_numeric((pd.to_datetime('2023-09-20') - df['data_financiamento_emprestimo'])/np.timedelta64(1, 'M')))
        df['mths_since_data_financiamento_emprestimo'] = df['mths_since_data_financiamento_emprestimo'].fillna(df['mths_since_data_financiamento_emprestimo'].median())
        df['mths_since_data_financiamento_emprestimo'] = np.where(df['mths_since_data_financiamento_emprestimo'] < 0, df['mths_since_data_financiamento_emprestimo'].median(), df['mths_since_data_financiamento_emprestimo'])
        df['mths_since_data_financiamento_emprestimo'] = df['mths_since_data_financiamento_emprestimo'].apply(lambda x:int(x))
        df['data_financiamento_emprestimo'] = df['mths_since_data_financiamento_emprestimo']

        return df['data_financiamento_emprestimo']

    def n_meses_primeiro_produto_credito(df):
        df = df.copy()
        df['data_contratacao_primeiro_produto_credito'] = pd.to_datetime(df['data_contratacao_primeiro_produto_credito'], format = '%b-%y')
        df['mths_since_data_contratacao_primeiro_produto_credito'] = round(pd.to_numeric((pd.to_datetime('2023-09-20') - df['data_contratacao_primeiro_produto_credito'])/np.timedelta64(1, 'M')))
        df['mths_since_data_contratacao_primeiro_produto_credito'] = df['mths_since_data_contratacao_primeiro_produto_credito'].fillna(df['mths_since_data_contratacao_primeiro_produto_credito'].median())
        df['mths_since_data_contratacao_primeiro_produto_credito'] = np.where(df['mths_since_data_contratacao_primeiro_produto_credito'] < 0, df['mths_since_data_contratacao_primeiro_produto_credito'].median(), df['mths_since_data_contratacao_primeiro_produto_credito'])
        df['mths_since_data_contratacao_primeiro_produto_credito'] = df['mths_since_data_contratacao_primeiro_produto_credito'].apply(lambda x:int(x))
        df['data_contratacao_primeiro_produto_credito'] = df['mths_since_data_contratacao_primeiro_produto_credito']
        
        return df['data_contratacao_primeiro_produto_credito']

    def produto_disponivel_publicamente(df):
        df_aux = df[['situacao_do_emprestimo', 'produto_disponivel_publicamente']].copy()
        df_aux['produto_disponivel_publicamente'] = np.where(df_aux[['produto_disponivel_publicamente']] == 0, 'sem_disponibilidade_publica', 'com_disponibilidade_publica')

        return df_aux['produto_disponivel_publicamente']

    def formato_features_binarias(df):
        df['qt_parcelas'] = np.where(df['qt_parcelas'] == ' 36 months', 0, 1)
        df['registros_publicos_depreciativos'] = np.where(df['qt_parcelas'] == 'sem_registros_negativos', 0, 1)
        df['inadimplencia_vencida_30dias'] = np.where(df['inadimplencia_vencida_30dias'] == 'sem_inadimplencia_vencida', 0, 1)
        df['tipo_de_concessao_do_credor'] = np.where(df['tipo_de_concessao_do_credor'] == 'f', 0, 1)
        df['plano_de_pagamento'] = np.where(df['plano_de_pagamento'] == 'n', 0, 1)
        df['renda_comprovada'] = np.where(df['renda_comprovada'] == 'Source Verified', 0, 1)
        df['consultas_credito_6meses'] = np.where(df['consultas_credito_6meses'] == 'sem_consultas', 0, 1)
        df['produto_disponivel_publicamente'] = np.where(df['consultas_credito_6meses'] == 'sem_disponibilidade_publica', 0, 1)

        return df

    def target_encoder_bad_rate(df, tipo):
        categoricas = ['grau_de_emprestimo', 'subclasse_de_emprestimo', 'produto_de_credito', 'qt_anos_mesmo_emprego', 'status_propriedade_residencial', 'estado']
        df_aux_2 = df.copy()
        if tipo == 'Criação':
            for cat in categoricas:
                df_aux = df[[f'{cat}', 'situacao_do_emprestimo']].copy()
                good = pd.DataFrame(df_aux.loc[df_aux['situacao_do_emprestimo'] == 0].groupby(f'{cat}', as_index = False)['situacao_do_emprestimo'].count()).rename({'situacao_do_emprestimo':'qt_good'}, axis = 1)
                bad = pd.DataFrame(df_aux.loc[df_aux['situacao_do_emprestimo'] == 1].groupby(f'{cat}', as_index = False)['situacao_do_emprestimo'].count()).rename({'situacao_do_emprestimo':'qt_bad'}, axis = 1)
                df_aux = good.merge(bad, on = f'{cat}', how = 'left')
                df_aux['qt_total'] = df_aux['qt_good'] + df_aux['qt_bad']
                df_aux[f'{cat}_enc'] = ((df_aux['qt_bad']/df_aux['qt_total'])*100).round(2)
                df_aux[f'{cat}_enc'] = df_aux[f'{cat}_enc'].apply(lambda x:float(x))
                df_aux = df_aux[[f'{cat}', f'{cat}_enc']].drop_duplicates().sort_values(by = f'{cat}_enc', ascending = True)
                df_aux.to_csv(f'features/{cat}_enc.csv', index = False)
                df_aux_2 = df_aux_2.merge(df_aux[[f'{cat}', f'{cat}_enc']], on = f'{cat}', how = 'left')
                df_aux_2.drop(f'{cat}', axis = 1, inplace = True)
        else:
            for cat in categoricas:
                ft = pd.read_csv(f'features/{cat}_enc.csv')
                replace_dict = dict(zip(ft[f'{cat}'], ft[f'{cat}_enc']))
                df_aux_2[f'{cat}_enc'] = df_aux_2[f'{cat}'].replace(replace_dict)

        return df_aux_2



    def simple_imputer(df):

        df_aux = df.copy()
        imputer = SimpleImputer(strategy = 'median')
        imputer.fit(df_aux)

        return imputer

    columns_selected = ['situacao_do_emprestimo', 'qt_parcelas','grau_de_emprestimo','subclasse_de_emprestimo','produto_de_credito', 'inadimplencia_vencida_30dias', 'valor_emprestimo_solicitado', 'taxa_de_juros','data_financiamento_emprestimo','produto_disponivel_publicamente','plano_de_pagamento','tipo_de_concessao_do_credor','pagamento_mensal','qt_anos_mesmo_emprego','status_propriedade_residencial',
    'renda_comprovada','faturamento_anual','estado', 'limite_total_produtos_credito','limite_total_rotativos','limite_rotativos_utilizado','taxa_utilizacao_limite_rotativos','qt_produtos_credito_contratados_atualmente','qt_produtos_credito_contratados_historicamente','registros_publicos_depreciativos','consultas_credito_6meses','data_contratacao_primeiro_produto_credito','qt_meses_desde_ultimo_registro_publico', 'qt_meses_classificacao_mais_recente_90dias',
    'qt_meses_ultima_inadimplencia']


    df_raw = df[columns_selected].copy()

    # Feature Selection

    features_selected = pd.read_csv('features/features_selected.csv')
    features_selected = features_selected.loc[features_selected['importance'] > 1] 
    features_selected = list(features_selected['feature'].unique()) + ['situacao_do_emprestimo']

    # Inicializar o KFold para dividir os dados
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Listas para armazenar as métricas para cada fold
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_scores = []  # Lista para armazenar os valores de AUC
    ks_scores = []   # Lista para armazenar os valores de KS
    cv_results = []  # Lista para armazenar os resultados de validação cruzada

    # Loop pelos folds
    for train_idx, test_idx in kfold.split(df_raw):
        # Criar DataFrames de treino e teste
        df_train = df_raw.iloc[train_idx]
        df_test = df_raw.iloc[test_idx]

        # Criação das Features sem Data Leakage
        df_train['qt_anos_mesmo_emprego'] = numero_de_anos_emprego_atual(df_train)
        df_train['registros_publicos_depreciativos'] = numero_de_registros_negativos(df_train)
        df_train['consultas_credito_6meses'] = consulta_de_credito_nos_ultimos_6_meses(df_train)
        df_train['comprometimento_de_renda_anual'] = compromento_de_renda(df_train)
        df_train['inadimplencia_vencida_30dias'] = numero_incidencias_inadimplencia_vencidas_30d(df_train)
        df_train['data_financiamento_emprestimo'] = n_meses_produto_credito_atual(df_train)
        df_train['data_contratacao_primeiro_produto_credito'] = n_meses_primeiro_produto_credito(df_train)
        df_train = formato_features_binarias(df_train)
        df_train = target_encoder_bad_rate(df_train, 'escoragem')

        df_test['qt_anos_mesmo_emprego'] = numero_de_anos_emprego_atual(df_test)
        df_test['registros_publicos_depreciativos'] = numero_de_registros_negativos(df_test)
        df_test['consultas_credito_6meses'] = consulta_de_credito_nos_ultimos_6_meses(df_test)
        df_test['comprometimento_de_renda_anual'] = compromento_de_renda(df_test)
        df_test['inadimplencia_vencida_30dias'] = numero_incidencias_inadimplencia_vencidas_30d(df_test)
        df_test['data_financiamento_emprestimo'] = n_meses_produto_credito_atual(df_test)
        df_test['data_contratacao_primeiro_produto_credito'] = n_meses_primeiro_produto_credito(df_test)
        df_test = formato_features_binarias(df_test)
        df_test = target_encoder_bad_rate(df_test, 'escoragem')

        # Filtragem das Features que passaram no Feature Selection
        df_train = df_train[features_selected]
        df_test = df_test[features_selected]

        # Separação Feature e Target
        x_train, y_train = separa_feature_target('situacao_do_emprestimo', df_train)
        x_test, y_test = separa_feature_target('situacao_do_emprestimo', df_test)
        
        # Imputer
        cols = list(x_train.columns)
        imputer = simple_imputer(x_train)
        x_train = pd.DataFrame(imputer.transform(x_train), columns = x_train.columns)
        x_test = pd.DataFrame(imputer.transform(x_test), columns = x_test.columns)


        # Melhores Hiperparâmetros
        melhores_hiperparametros = best_hiperpams
        colsample_bytree = melhores_hiperparametros['colsample_bytree'][1]
        gamma = melhores_hiperparametros['gamma'][1]
        learning_rate = melhores_hiperparametros['learning_rate'][1]
        max_depth = melhores_hiperparametros['max_depth'][1]
        n_estimators = melhores_hiperparametros['n_estimators'][1]
        reg_alpha = melhores_hiperparametros['reg_alpha'][1]
        reg_lambda = melhores_hiperparametros['reg_lambda'][1]
        scale_pos_weight = melhores_hiperparametros['scale_pos_weight'][1]
        subsample = melhores_hiperparametros['subsample'][1]

        # Define as colunas categóricas e numéricas
        model = make_pipeline(
                ColumnTransformer([
                    ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols)
                ]),
                
                # XGBClassifier(
                #     random_state=42, # Semente aleatória para reproducibilidade dos resultados
                #     tree_method = 'gpu_hist',
                #     eval_metric='logloss', # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
                #     objective='binary:logistic', # Define o objetivo do modelo, 'binary:logistic' para classificação binária
                #     n_estimators = n_estimators, # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
                #     max_depth = max_depth, # Profundidade máxima de cada árvore
                #     learning_rate = learning_rate, # Taxa de aprendizado - controla a contribuição de cada árvore
                #     reg_alpha = reg_alpha, # Termo de regularização L1 (penalidade nos pesos)
                #     reg_lambda = reg_lambda, # Termo de regularização L2 (penalidade nos quadrados dos pesos)
                #     gamma = gamma, # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
                #     colsample_bytree = colsample_bytree, # Fração de características a serem consideradas ao construir cada árvore
                #     subsample = subsample, # Fração de amostras a serem usadas para treinar cada árvore
                #     scale_pos_weight = scale_pos_weight, # Peso das classes positivas em casos desequilibrados
                #     base_score = 0.5 # Threshold de Probabilidade de Decisão do Classificador (geralmente é 0.5 para problemas de classificação binária)
                # )
            XGBClassifier(
                    random_state=42, # Semente aleatória para reproducibilidade dos resultados
                    tree_method = 'gpu_hist', # Treino usando GPU
                    eval_metric='logloss', # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
                    objective='binary:logistic', # Define o objetivo do modelo, 'binary:logistic' para classificação binária
                    n_estimators = 99, # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
                    max_depth = 8, # Profundidade máxima de cada árvore
                    learning_rate = 0.03209718317105407, # Taxa de aprendizado - controla a contribuição de cada árvore
                    reg_alpha = 0.7268326719031495, # Termo de regularização L1 (penalidade nos pesos)
                    reg_lambda = 0.5777240270252717, # Termo de regularização L2 (penalidade nos quadrados dos pesos)
                    gamma = 0.9593612608346885, # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
                    colsample_bytree = 0.7224162561505759, # Fração de características a serem consideradas ao construir cada árvore
                    subsample = 0.7786702115169006, # Fração de amostras a serem usadas para treinar cada árvore
                    scale_pos_weight = 7, # Peso das classes positivas em casos desequilibrados
                    base_score = 0.5 # Threshold de Probabilidade de Decisão do Classificador (geralmente é 0.5 para problemas de classificação binária)
                )
            )

        # Treinar o modelo usando os dados de treinamento
        model.fit(x_train, y_train)

        # Obter as probabilidades previstas para ambas as classes
        y_proba = model.predict_proba(x_test)

        # Fazer as previsões usando o modelo nos dados de teste
        y_pred = model.predict(x_test)

        # Calcular as métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        y_proba = model.predict_proba(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ks = max(tpr - fpr)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(roc_auc)
        ks_scores.append(ks)

        # Adicionar resultados de validação cruzada ao DataFrame
        fold_results = pd.DataFrame({
            'situacao_do_emprestimo': y_test['situacao_do_emprestimo'].values,
            'y_predict': y_pred,
            'predict_proba_0': y_proba[:, 0],  # Probabilidade da classe 0
            'predict_proba_1': y_proba[:, 1]  # Probabilidade da classe 1
        })
        cv_results.append(fold_results)


    # Calcular a média das métricas para todos os folds
    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    mean_auc = np.mean(auc_scores),
    mean_ks = np.mean(ks_scores)

    # Criar um DataFrame com as métricas
    metricas_finais = pd.DataFrame({
        'Acuracia': mean_accuracy,
        'Precisao': mean_precision,
        'Recall': mean_recall,
        'F1-Score': mean_f1,
        'AUC':mean_auc,
        'KS': mean_ks,
        'Etapa': 'validacao_cruzada',
        'Classificador': classificador
    }, index=[1])

    return metricas_finais, cv_results



def modelo_corte_probabilidade(df_model, df_retorno_financeiro, target, x, y):

    def simple_imputer(df_model):

        df_aux = df_model.copy()
        imputer = SimpleImputer(strategy = 'median')
        imputer.fit(df_aux)

        return imputer
    
    cols = list(x.columns)
    imputer = simple_imputer(x)
    x = pd.DataFrame(imputer.transform(x), columns = x.columns)
    
    list_threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    list_lucro = []
    for threshold in list_threshold:
        # Define o ColumnTransformer
        preprocessor = ColumnTransformer([
                    ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                    ('scaler', make_pipeline(MinMaxScaler()), cols)
                ])

        model = make_pipeline(
        preprocessor,
            XGBClassifier(
            random_state=42, # Semente aleatória para reproducibilidade dos resultados
            tree_method = 'gpu_hist', # Treino usando GPU
            eval_metric='logloss', # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
            objective='binary:logistic', # Define o objetivo do modelo, 'binary:logistic' para classificação binária
            n_estimators = 99, # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
            max_depth = 8, # Profundidade máxima de cada árvore
            learning_rate = 0.03209718317105407, # Taxa de aprendizado - controla a contribuição de cada árvore
            reg_alpha = 0.7268326719031495, # Termo de regularização L1 (penalidade nos pesos)
            reg_lambda = 0.5777240270252717, # Termo de regularização L2 (penalidade nos quadrados dos pesos)
            gamma = 0.9593612608346885, # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
            colsample_bytree = 0.7224162561505759, # Fração de características a serem consideradas ao construir cada árvore
            subsample = 0.7786702115169006, # Fração de amostras a serem usadas para treinar cada árvore
            scale_pos_weight = 7, # Peso das classes positivas em casos desequilibrados
            base_score = 0.5 # Threshold de Probabilidade de Decisão do Classificador (geralmente é 0.5 para problemas de classificação binária)
            )
        )
        
        model.fit(x, y)

        y_pred = model.predict(x)
        y_predict_proba = model.predict_proba(x)

        teste_threshold = pd.DataFrame({'y_predict':y_pred, 'Proba Good':y_predict_proba[:, 0]})
        teste_threshold['y_predict_threshold'] = np.where(teste_threshold['Proba Good'] <= threshold, 1, 0)
        y_pred = teste_threshold['y_predict_threshold'].values

        lucro = retorno_financeiro(df_retorno_financeiro, y_pred)[0]
        list_lucro.append(lucro)
    
    corte_probabilidade = pd.DataFrame({'threshold':list_threshold, 'lucro':list_lucro})


    # teste_threshold = pd.DataFrame({'y_true':y_test['situacao_do_emprestimo'].values, 'y_predict_test':y_predict_test_otimizado, 'Proba Good':y_proba_test_otimizado[:, 0], 'Proba Bad':y_proba_test_otimizado[:, 1]})
    # teste_threshold['y_predict_threshold'] = np.where(teste_threshold['Proba Good'] <= 0.4, 1, 0)


    # display(teste_threshold.groupby('y_true', as_index = False)['Proba Good'].count())
    # display(teste_threshold.groupby('y_predict_test', as_index = False)['Proba Good'].count())
    # display(teste_threshold.groupby('y_predict_threshold', as_index = False)['Proba Good'].count())
    return corte_probabilidade



def modelo_oficial(classificador, x, y):
    def simple_imputer(df_model):

        df_aux = df_model.copy()
        imputer = SimpleImputer(strategy = 'median')
        imputer.fit(df_aux)

        return imputer
    
    cols = list(x.columns)
    imputer = simple_imputer(x)
    x = pd.DataFrame(imputer.transform(x), columns = x.columns)

    # Define o ColumnTransformer
    preprocessor = ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                ('scaler', make_pipeline(MinMaxScaler()), cols)
            ])

    model = make_pipeline(
    preprocessor,
        XGBClassifier(
        random_state=42, # Semente aleatória para reproducibilidade dos resultados
        tree_method = 'gpu_hist', # Treino usando GPU
        eval_metric='logloss', # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
        objective='binary:logistic', # Define o objetivo do modelo, 'binary:logistic' para classificação binária
        n_estimators = 99, # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
        max_depth = 8, # Profundidade máxima de cada árvore
        learning_rate = 0.03209718317105407, # Taxa de aprendizado - controla a contribuição de cada árvore
        reg_alpha = 0.7268326719031495, # Termo de regularização L1 (penalidade nos pesos)
        reg_lambda = 0.5777240270252717, # Termo de regularização L2 (penalidade nos quadrados dos pesos)
        gamma = 0.9593612608346885, # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
        colsample_bytree = 0.7224162561505759, # Fração de características a serem consideradas ao construir cada árvore
        subsample = 0.7786702115169006, # Fração de amostras a serem usadas para treinar cada árvore
        scale_pos_weight = 7, # Peso das classes positivas em casos desequilibrados
        base_score = 0.5 # Threshold de Probabilidade de Decisão do Classificador (geralmente é 0.5 para problemas de classificação binária)
        )
    )

    # Treina o modelo oficial
    model.fit(x, y)
    salvar_modelo_pickle(model, 'models/clf_final.pkl')

def escoragem(x, y):
    clf_final = carregar_modelo_pickle('models/clf_final.pkl')

    y_pred = clf_final.predict(x)
    y_proba= clf_final.predict_proba(x)
    teste_threshold = pd.DataFrame({'y_predict':y_pred, 'Proba Good':y_proba[:, 0]})
    teste_threshold['y_predict_threshold'] = np.where(teste_threshold['Proba Good'] <= 0.3, 1, 0)
    y_pred = teste_threshold['y_predict_threshold'].values

    return y_pred, y_proba


def salvar_modelo_pickle(modelo, caminho_arquivo):
    """
    Salva um modelo em um arquivo pickle.

    Parâmetros:
    - modelo: O modelo treinado que você deseja salvar.
    - caminho_arquivo: O caminho do arquivo onde o modelo será salvo.
    """
    with open(caminho_arquivo, 'wb') as arquivo:
        pickle.dump(modelo, arquivo)
    print(f"Modelo salvo em {caminho_arquivo}")

def carregar_modelo_pickle(caminho_arquivo):
    """
    Carrega um modelo salvo de um arquivo pickle.

    Parâmetros:
    - caminho_arquivo: O caminho do arquivo onde o modelo foi salvo.

    Retorna:
    - O modelo carregado.
    """
    with open(caminho_arquivo, 'rb') as arquivo:
        modelo_carregado = pickle.load(arquivo)
    print(f"Modelo carregado de {caminho_arquivo}")
    return modelo_carregado


def calibracao_probabilidade():  
    # Modelo Otimizado
    model_otimizado, y_predict_train_otimizado, y_predict_test_otimizado, y_predict_proba_train_otimizado, y_predict_proba_test_otimizado = modelo_otimizado('Bayes Search + Threshold Proba + XGBoost', x_train, y_train, x_test, y_test)

    # Modelo de Calibração
    calibrated_clf = CalibratedClassifierCV(model_otimizado, cv=5, method='isotonic')
    calibrated_clf.fit(x_train, y_train)

    y_predict_proba_ajustada_train = calibrated_clf.predict_proba(x_train)
    y_predict_proba_ajustada_test = calibrated_clf.predict_proba(x_test)

    predict_proba_train = pd.DataFrame(y_predict_proba_train_otimizado.tolist(), columns=['predict_proba_0', 'predict_proba_1'])
    predict_proba_test = pd.DataFrame(y_predict_proba_test_otimizado.tolist(), columns=['predict_proba_0', 'predict_proba_1'])

    predict_proba_ajustada_train = pd.DataFrame(y_predict_proba_ajustada_train.tolist(), columns=['predict_proba_0', 'predict_proba_1'])
    predict_proba_ajustada_test = pd.DataFrame(y_predict_proba_ajustada_test.tolist(), columns=['predict_proba_0', 'predict_proba_1'])

    # Definição das Probabilidades

    probabilities_train = predict_proba_train['predict_proba_1'] # Obtenha as probabilidades previstas
    prob_true_train, prob_pred_train = calibration_curve(y_train, probabilities_train, n_bins=10) # Calcule a curva de calibração
    brier_score_train = brier_score_loss(y_train, probabilities_train) # Calcule o Brier Score (uma métrica de calibração)

    probabilities_test = predict_proba_test['predict_proba_1'] # Obtenha as probabilidades previstas
    prob_true_test, prob_pred_test = calibration_curve(y_test, probabilities_test, n_bins=10) # Calcule a curva de calibração
    brier_score_test = brier_score_loss(y_test, probabilities_test) # Calcule o Brier Score (uma métrica de calibração)

    probabilities_ajustada_train = predict_proba_ajustada_train['predict_proba_1'] # Obtenha as probabilidades previstas
    prob_true_ajustada_train, prob_pred_ajustada_train = calibration_curve(y_train, probabilities_ajustada_train, n_bins=10) # Calcule a curva de calibração
    brier_score_ajustada_train = brier_score_loss(y_train, probabilities_ajustada_train) # Calcule o Brier Score (uma métrica de calibração)

    probabilities_ajustada_test = predict_proba_ajustada_test['predict_proba_1'] # Obtenha as probabilidades previstas
    prob_true_ajustada_test, prob_pred_ajustada_test = calibration_curve(y_test, probabilities_ajustada_test, n_bins=10) # Calcule a curva de calibração
    brier_score_ajustada_test = brier_score_loss(y_test, probabilities_ajustada_test) # Calcule o Brier Score (uma métrica de calibração)

    y_predict_ajustada_train_best_clf = calibrated_clf.predict(x_train)
    y_predict_ajustada_test_best_clf = calibrated_clf.predict(x_test)

    y_predict_proba_ajustada_train_best_clf = calibrated_clf.predict_proba(x_train)
    y_predict_proba_ajustada_test_best_clf = calibrated_clf.predict_proba(x_test)


    # Métricas Otimizadas

    metricas_before_calibration_ajustada = metricas_classificacao('Threshold Proba (0.1) + Bayes Search + XGBoost', y_train, y_predict_train_otimizado, y_test, y_predict_test_otimizado, y_predict_proba_train_otimizado, y_predict_proba_test_otimizado)
    metricas_after_calibration_ajustada = metricas_classificacao('Calibration + Threshold Proba (0.1) + Bayes Search + XGBoost', y_train, y_predict_ajustada_train_best_clf, y_test, y_predict_ajustada_test_best_clf, y_predict_proba_ajustada_train_best_clf, y_predict_proba_ajustada_test_best_clf)


    print('Métricas Finais')
    metricas_finais = metricas_classificacao_modelos_juntos(
        [
            metricas_before_calibration_ajustada,
            metricas_after_calibration_ajustada
        ]
    )
    display(metricas_finais)

    # Plote a curva de calibração
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred_train, prob_true_train, marker='o', label='Probability Curve Before Calibration- Train', color = 'blue')
    plt.plot(prob_pred_test, prob_true_test, marker='o', label='Probability Curve Before Calibration - Test', color = 'red')
    plt.plot(prob_pred_ajustada_train, prob_true_ajustada_train, marker='o', label='Probability Curve After Calibration - Train', color = 'green')
    plt.plot(prob_pred_ajustada_test, prob_true_ajustada_test, marker='o', label='Probability Curve After Calibration - Test', color = 'purple')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.title(f'Calibration Curve')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.show()


def politica_de_credito(df):    
    df_aux = df.copy()
    df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] = df_aux['pagamento_mensal']*df_aux['qt_parcelas']

    df_aux = df_aux[['situacao_do_emprestimo', 'estado', 'valor_emprestimo_solicitado', 'valor_emprestimo_solicitado_com_taxa_de_juros',
                    'taxa_de_juros', 'faturamento_anual', 'comprometimento_de_renda_anual', 'subclasse_de_emprestimo', 'grau_de_emprestimo']].copy()

    for num in df_aux.drop(['situacao_do_emprestimo', 'estado', 'valor_emprestimo_solicitado', 'valor_emprestimo_solicitado_com_taxa_de_juros', 'subclasse_de_emprestimo', 'grau_de_emprestimo'], axis = 1):
        df_aux[f'{num}'].fillna(df_aux[num].median(), inplace = True)
    df_aux.rename({'estado':'qt_clientes'}, axis = 1, inplace = True)


    for col in ['comprometimento_de_renda_anual', 'faturamento_anual', 'taxa_de_juros']:
        df_bad_rate = pd.read_excel(f"Credit_Policy/{col}_value_pair.xlsx").sort_values(by = f'{col}_value', ascending = True)
        df_aux[f'{col}'] = np.where(df_aux[f'{col}'] <= df_bad_rate[f'{col}_value'].values[0], df_bad_rate[f'{col}_enc'].values[0], 
                           np.where(df_aux[f'{col}'] <= df_bad_rate[f'{col}_value'].values[1], df_bad_rate[f'{col}_enc'].values[1], 
                           np.where(df_aux[f'{col}'] <= df_bad_rate[f'{col}_value'].values[2], df_bad_rate[f'{col}_enc'].values[2], 
                           np.where(df_aux[f'{col}'] <= df_bad_rate[f'{col}_value'].values[3], df_bad_rate[f'{col}_enc'].values[3], 
                           np.where(df_aux[f'{col}'] <= df_bad_rate[f'{col}_value'].values[4], df_bad_rate[f'{col}_enc'].values[4], 
                           np.where(df_aux[f'{col}'] <= df_bad_rate[f'{col}_value'].values[5], df_bad_rate[f'{col}_enc'].values[5], 
                           np.where(df_aux[f'{col}'] <= df_bad_rate[f'{col}_value'].values[6], df_bad_rate[f'{col}_enc'].values[6], 
                           np.where(df_aux[f'{col}'] <= df_bad_rate[f'{col}_value'].values[7], df_bad_rate[f'{col}_enc'].values[7], 
                           np.where(df_aux[f'{col}'] <= df_bad_rate[f'{col}_value'].values[8], df_bad_rate[f'{col}_enc'].values[8], 
                           df_bad_rate[f'{col}_enc'].values[9])))))))))

    for col in ['grau_de_emprestimo']:
        df_bad_rate = pd.read_excel(f"Credit_Policy/{col}_value_pair.xlsx").sort_values(by = f'{col}_value', ascending = True)
        df_aux[f'{col}'] = np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[0], df_bad_rate[f'{col}_enc'].values[0], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[1], df_bad_rate[f'{col}_enc'].values[1], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[2], df_bad_rate[f'{col}_enc'].values[2], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[3], df_bad_rate[f'{col}_enc'].values[3], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[4], df_bad_rate[f'{col}_enc'].values[4], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[5], df_bad_rate[f'{col}_enc'].values[5],         
                           df_bad_rate[f'{col}_enc'].values[6]))))))

    for col in ['subclasse_de_emprestimo']:
        df_bad_rate = pd.read_excel(f"Credit_Policy/{col}_value_pair.xlsx").sort_values(by = f'{col}_value', ascending = True)
        df_aux[f'{col}'] = np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[0], df_bad_rate[f'{col}_enc'].values[0], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[1], df_bad_rate[f'{col}_enc'].values[1], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[2], df_bad_rate[f'{col}_enc'].values[2], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[3], df_bad_rate[f'{col}_enc'].values[3], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[4], df_bad_rate[f'{col}_enc'].values[4], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[5], df_bad_rate[f'{col}_enc'].values[5], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[6], df_bad_rate[f'{col}_enc'].values[6], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[7], df_bad_rate[f'{col}_enc'].values[7], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[8], df_bad_rate[f'{col}_enc'].values[8], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[9], df_bad_rate[f'{col}_enc'].values[9], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[10], df_bad_rate[f'{col}_enc'].values[10], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[11], df_bad_rate[f'{col}_enc'].values[11], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[12], df_bad_rate[f'{col}_enc'].values[12], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[13], df_bad_rate[f'{col}_enc'].values[13], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[14], df_bad_rate[f'{col}_enc'].values[14], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[15], df_bad_rate[f'{col}_enc'].values[15], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[16], df_bad_rate[f'{col}_enc'].values[16], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[17], df_bad_rate[f'{col}_enc'].values[17], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[18], df_bad_rate[f'{col}_enc'].values[18], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[19], df_bad_rate[f'{col}_enc'].values[19], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[20], df_bad_rate[f'{col}_enc'].values[20], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[21], df_bad_rate[f'{col}_enc'].values[21], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[22], df_bad_rate[f'{col}_enc'].values[22], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[23], df_bad_rate[f'{col}_enc'].values[23], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[24], df_bad_rate[f'{col}_enc'].values[24], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[25], df_bad_rate[f'{col}_enc'].values[25],   
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[26], df_bad_rate[f'{col}_enc'].values[26], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[27], df_bad_rate[f'{col}_enc'].values[27], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[28], df_bad_rate[f'{col}_enc'].values[28], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[29], df_bad_rate[f'{col}_enc'].values[29], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[30], df_bad_rate[f'{col}_enc'].values[30], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[31], df_bad_rate[f'{col}_enc'].values[31], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[32], df_bad_rate[f'{col}_enc'].values[32], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[33], df_bad_rate[f'{col}_enc'].values[33],            
                           df_bad_rate[f'{col}_enc'].values[34]))))))))))))))))))))))))))))))))))

    df_aux['predict_proba_1'] = df_aux[[
        'taxa_de_juros', 'subclasse_de_emprestimo',
        'grau_de_emprestimo',
        'comprometimento_de_renda_anual', 'faturamento_anual']].sum(axis = 1)
    df_aux['predict_proba_1'] = np.where(df_aux['predict_proba_1'] >= 100, 100, df_aux['predict_proba_1'])
    df_aux['predict_proba_1'] = df_aux['predict_proba_1']/100
    df_aux['predict_proba_0'] = 1 - df_aux['predict_proba_1']
    df_aux['y_predict'] = np.where(df_aux['predict_proba_0']  <= 0.3, 1, 0)
    df_aux.rename({'situacao_do_emprestimo':'y_true'}, axis = 1, inplace = True)

    df_aux.head()

    y_true = df_aux[['y_true']]
    y_predict = df_aux[['y_predict']]
    y_predict_proba_0 = df_aux['predict_proba_0'].values
    y_predict_proba_1 = df_aux['predict_proba_1'].values
    y_predict_proba = pd.DataFrame({'predict_proba_0':y_predict_proba_0, 'predict_proba_1':y_predict_proba_1})

    df_predict = pd.DataFrame(
            {
                'y_true':df_aux['y_true'].values,
                'y_predict':df_aux['y_predict'].values,
                'y_predict_proba_0':y_predict_proba_0,
                'y_predict_proba_1':y_predict_proba_1,
            }
        )

    return y_true, y_predict, y_predict_proba, df_predict

def corte_probabilidade_politica(df_politica):
    list_threshold = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    list_lucro = []
    for threshold in list_threshold:
        df_politica['predict_proba_1'] = df_politica[[
            'taxa_de_juros_enc', 'subclasse_de_emprestimo_enc',
            'grau_de_emprestimo_enc',
            'comprometimento_de_renda_anual_enc', 'faturamento_anual_enc']].sum(axis = 1)
        df_politica['predict_proba_1'] = np.where(df_politica['predict_proba_1'] >= 100, 100, df_politica['predict_proba_1'])
        df_politica['predict_proba_1'] = df_politica['predict_proba_1']/100
        df_politica['predict_proba_0'] = 1 - df_politica['predict_proba_1']
        df_politica['y_predict'] = np.where(df_politica['predict_proba_0']  <= threshold, 1, 0)
        y_predict = df_politica['y_predict'].values

        lucro = retorno_financeiro_politica_credito(df_politica, y_predict)[0]
        list_lucro.append(lucro)
        df_politica.drop(['predict_proba_1', 'predict_proba_0', 'y_predict'], axis = 1, inplace = True)
    
    corte_probabilidade = pd.DataFrame({'threshold':list_threshold, 'lucro':list_lucro})
    return corte_probabilidade

def metricas_politica_credito(Politica, y_train, y_predict_train, y_test, y_predict_test, y_predict_proba_train, y_predict_proba_test):

    predict_proba_train = y_predict_proba_train
    predict_proba_test = y_predict_proba_test

    # Treino
    accuracy_train = accuracy_score(y_train, y_predict_train)
    precision_train = precision_score(y_train, y_predict_train)
    recall_train = recall_score(y_train, y_predict_train)
    f1_train = f1_score(y_train, y_predict_train)
    roc_auc_train = roc_auc_score(y_train['y_true'], predict_proba_train['predict_proba_1'])
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train['y_true'], predict_proba_train['predict_proba_1'])
    ks_train = max(tpr_train - fpr_train)
    metricas_treino = pd.DataFrame({'Acuracia': accuracy_train, 'Precisao': precision_train, 'Recall': recall_train, 'F1-Score': f1_train, 'AUC': roc_auc_train, 'KS': ks_train, 'Etapa': 'treino', 'Politica': Politica}, index=[0])
    
    # Teste
    accuracy_test = accuracy_score(y_test, y_predict_test)
    precision_test = precision_score(y_test, y_predict_test)
    recall_test = recall_score(y_test, y_predict_test)
    f1_test = f1_score(y_test, y_predict_test)
    roc_auc_test = roc_auc_score(y_test['y_true'], predict_proba_test['predict_proba_1'])
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test['y_true'], predict_proba_test['predict_proba_1'])
    ks_test = max(tpr_test - fpr_test)
    metricas_teste = pd.DataFrame({'Acuracia': accuracy_test, 'Precisao': precision_test, 'Recall': recall_test, 'F1-Score': f1_test, 'AUC': roc_auc_test, 'KS': ks_test, 'Etapa': 'teste', 'Politica': Politica}, index=[0])
    
    # Consolidando
    metricas_finais = pd.concat([metricas_treino, metricas_teste])

    return metricas_finais

def auc_ks_politica(Politica, target, 
                                    y_train, y_predict_train, 
                                    y_test, y_predict_test, 
                                    predict_proba_train, predict_proba_test):

    # Inicialize as variáveis x_max_ks e y_max_ks fora dos blocos condicionais
    x_max_ks_train, y_max_ks_train = 0, 0
    x_max_ks_test, y_max_ks_test = 0, 0

    ### Treino
    results_train = y_train[[target]].copy()
    results_train['y_predict_train'] = y_predict_train
    results_train['predict_proba_0'] = list(predict_proba_train['predict_proba_0']) # Probabilidade de ser Bom (classe 0)
    results_train['predict_proba_1'] = list(predict_proba_train['predict_proba_1']) # Probabilidade de ser Mau (classe 1)

    results_train_sorted = results_train.sort_values(by='predict_proba_1', ascending=False)
    results_train_sorted['Cumulative N Population'] = range(1, results_train_sorted.shape[0] + 1)
    results_train_sorted['Cumulative N Good'] = results_train_sorted[target].cumsum()
    results_train_sorted['Cumulative N Bad'] = results_train_sorted['Cumulative N Population'] - results_train_sorted['Cumulative N Good']
    results_train_sorted['Cumulative Perc Population'] = results_train_sorted['Cumulative N Population'] / results_train_sorted.shape[0]
    results_train_sorted['Cumulative Perc Good'] = results_train_sorted['Cumulative N Good'] / results_train_sorted[target].sum()
    results_train_sorted['Cumulative Perc Bad'] = results_train_sorted['Cumulative N Bad'] / (results_train_sorted.shape[0] - results_train_sorted[target].sum())

    max_ks_index_train = np.argmax(results_train_sorted['Cumulative Perc Good'] - results_train_sorted['Cumulative Perc Bad'])
    x_max_ks_train = results_train_sorted['Cumulative Perc Population'].iloc[max_ks_index_train]
    y_max_ks_train = results_train_sorted['Cumulative Perc Good'].iloc[max_ks_index_train]
    y_min_ks_train = results_train_sorted['Cumulative Perc Bad'].iloc[max_ks_index_train]

        ###### Calculate AUC and ROC for the training set
    y_true_train = results_train[target]
    y_scores_train = results_train['predict_proba_1']
    auc_train = roc_auc_score(y_true_train, y_scores_train)
    fpr_train, tpr_train, thresholds_train = roc_curve(y_true_train, y_scores_train)
        ###### Calculate KS curve for the training set
    KS_train = round(np.max(results_train_sorted['Cumulative Perc Good'] - results_train_sorted['Cumulative Perc Bad']), 2)

    ### Teste
    results_test = y_test[[target]].copy()
    results_test['y_predict_test'] = y_predict_test
    results_test['predict_proba_0'] = list(predict_proba_test['predict_proba_0']) # Probabilidade de ser Bom (classe 0)
    results_test['predict_proba_1'] = list(predict_proba_test['predict_proba_1']) # Probabilidade de ser Mau (classe 1)

    results_test_sorted = results_test.sort_values(by='predict_proba_1', ascending=False)
    results_test_sorted['Cumulative N Population'] = range(1, results_test_sorted.shape[0] + 1)
    results_test_sorted['Cumulative N Good'] = results_test_sorted[target].cumsum()
    results_test_sorted['Cumulative N Bad'] = results_test_sorted['Cumulative N Population'] - results_test_sorted['Cumulative N Good']
    results_test_sorted['Cumulative Perc Population'] = results_test_sorted['Cumulative N Population'] / results_test_sorted.shape[0]
    results_test_sorted['Cumulative Perc Good'] = results_test_sorted['Cumulative N Good'] / results_test_sorted[target].sum()
    results_test_sorted['Cumulative Perc Bad'] = results_test_sorted['Cumulative N Bad'] / (results_test_sorted.shape[0] - results_test_sorted[target].sum())

    max_ks_index_test = np.argmax(results_test_sorted['Cumulative Perc Good'] - results_test_sorted['Cumulative Perc Bad'])
    x_max_ks_test = results_test_sorted['Cumulative Perc Population'].iloc[max_ks_index_test]
    y_max_ks_test = results_test_sorted['Cumulative Perc Good'].iloc[max_ks_index_test]
    y_min_ks_test = results_test_sorted['Cumulative Perc Bad'].iloc[max_ks_index_test]


            ###### Calculate AUC and ROC for the test set
    y_true_test = results_test[target]
    y_scores_test = results_test['predict_proba_1']
    auc_test = roc_auc_score(y_true_test, y_scores_test)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_true_test, y_scores_test)
            ###### Calculate KS curve for the test set
    KS_test = round(np.max(results_test_sorted['Cumulative Perc Good'] - results_test_sorted['Cumulative Perc Bad']), 2)

    # Plot ROC and KS curves side by side
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Training set ROC curve
    axs[0].plot(fpr_train, tpr_train, label='Train ROC Curve (AUC = {:.2f})'.format(auc_train), color='purple')
    axs[0].fill_between(fpr_train, 0, tpr_train, color='gray', alpha=0.3)  # Preencha a área sob a curva ROC
    axs[0].plot([0, 1], [0, 1], linestyle='--', color='black')
    axs[0].set_xlabel('False Positive Rate', fontsize = 14)
    axs[0].set_ylabel('True Positive Rate', fontsize = 14)
    axs[0].set_title(f'ROC Curve - {Politica}', fontsize = 14)

    # Test set ROC curve
    axs[0].plot(fpr_test, tpr_test, label='Test ROC Curve (AUC = {:.2f})'.format(auc_test), color='orange')
    axs[0].fill_between(fpr_test, 0, tpr_test, color='gray', alpha=0.3)  # Preencha a área sob a curva ROC

    # Adicione a legenda personalizada com cores para a curva ROC
    roc_legend_labels = [
        {'label': 'Train ROC Curve (AUC = {:.2f})'.format(auc_train), 'color': 'purple', 'marker': 'o'},
        {'label': 'Test ROC Curve (AUC = {:.2f})'.format(auc_test), 'color': 'orange', 'marker': 's'},
    ]

    # Criar marcadores personalizados para a legenda ROC
    roc_legend_handles = [Line2D([0], [0], marker=label_info['marker'], color='w', markerfacecolor=label_info['color'], markersize=10) for label_info in roc_legend_labels]

    # Adicione a legenda personalizada ao gráfico da curva ROC
    roc_legend = axs[0].legend(handles=roc_legend_handles, labels=[label_info['label'] for label_info in roc_legend_labels], loc='upper right', bbox_to_anchor=(0.9, 0.4), fontsize='11')
    roc_legend.set_title('ROC AUC', prop={'size': '11'})


    # Train set KS curve
    axs[1].plot(results_train_sorted['Cumulative Perc Population'], results_train_sorted['Cumulative Perc Good'], label='Train Positive Class (Class 1)', color='purple')
    axs[1].plot(results_train_sorted['Cumulative Perc Population'], results_train_sorted['Cumulative Perc Bad'], label='Train Negative Class (Class 0)', color='purple')
    axs[1].plot([x_max_ks_train, x_max_ks_train], [y_min_ks_train, y_max_ks_train], color='black', linestyle='--')
    axs[1].fill_between(results_train_sorted['Cumulative Perc Population'], results_train_sorted['Cumulative Perc Good'], results_train_sorted['Cumulative Perc Bad'], color='gray', alpha=0.5)
    axs[1].text(x=results_train_sorted['Cumulative Perc Population'].iloc[np.argmax(results_train_sorted['Cumulative Perc Good'] - results_train_sorted['Cumulative Perc Bad'])],
                y=(y_min_ks_train + results_train_sorted['Cumulative Perc Good'].iloc[np.argmax(results_train_sorted['Cumulative Perc Good'] - results_train_sorted['Cumulative Perc Bad'])]) / 2,
                s=str(KS_train), fontsize = 14, color='purple', ha='left', va='center', rotation=45)
    axs[1].set_xlabel('Cumulative Percentage of Population', fontsize = 14)
    axs[1].set_ylabel('Cumulative Percentage', fontsize = 14)
    axs[1].set_title(f'KS Plot - {Politica}', fontsize = 14)

    # Test set KS curve
    axs[1].plot(results_test_sorted['Cumulative Perc Population'], results_test_sorted['Cumulative Perc Good'], label='Test Positive Class (Class 1)', color='orange')
    axs[1].plot(results_test_sorted['Cumulative Perc Population'], results_test_sorted['Cumulative Perc Bad'], label='Test Negative Class (Class 0)', color='orange')
    axs[1].plot([x_max_ks_test, x_max_ks_test], [y_min_ks_test, y_max_ks_test], color='black', linestyle='--')
    axs[1].fill_between(results_test_sorted['Cumulative Perc Population'], results_test_sorted['Cumulative Perc Good'], results_test_sorted['Cumulative Perc Bad'], color='gray', alpha=0.5)
    axs[1].text(x=results_test_sorted['Cumulative Perc Population'].iloc[np.argmax(results_test_sorted['Cumulative Perc Good'] - results_test_sorted['Cumulative Perc Bad'])],
                y=(y_min_ks_test + results_test_sorted['Cumulative Perc Good'].iloc[np.argmax(results_test_sorted['Cumulative Perc Good'] - results_test_sorted['Cumulative Perc Bad'])]) / 2,
                s=str(KS_test), fontsize = 14, color='orange', ha='left', va='center', rotation=45)
    axs[1].set_xlabel('Cumulative Percentage of Population', fontsize = 14)
    axs[1].set_ylabel('Cumulative Percentage', fontsize = 14)
    axs[1].set_title(f'KS Plot - {Politica}', fontsize = 14)

    # Adicione a legenda personalizada com cores
    ks_legend_labels = [
        {'label': f'Treino (KS: {KS_train})', 'color': 'purple', 'marker': 'o'},
        {'label': f'Teste (KS: {KS_test})', 'color': 'orange', 'marker': 's'},
    ]

    # Criar marcadores personalizados para a legenda
    legend_handles = [Line2D([0], [0], marker=label_info['marker'], color='w', markerfacecolor=label_info['color'], markersize=10) for label_info in ks_legend_labels]

    ks_legend = axs[1].legend(handles=legend_handles, labels=[label_info['label'] for label_info in ks_legend_labels], loc='upper right', bbox_to_anchor=(0.9, 0.4), fontsize='11')
    ks_legend.set_title('KS', prop={'size': '11'})

    plt.tight_layout()
    plt.show()

def retorno_financeiro_politica_credito(df, y_predict):

    df_aux = df.copy()
    df_aux['y_predict'] = y_predict

    TN = df_aux.loc[(df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 0)].shape[0] # O CARA É BOM E MEU MODELO FALA QUE ELE É BOM
    FN = df_aux.loc[(df_aux['situacao_do_emprestimo'] == 1) & (df_aux['y_predict'] == 0)].shape[0] # O CARA É MAU E MEU MODELO FALA QUE ELE É BOM
    FP = df_aux.loc[(df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 1)].shape[0] # O CARA É BOM E MEU MODELO FALA QUE É MAU
    TP = df_aux.loc[(df_aux['situacao_do_emprestimo'] == 1) & (df_aux['y_predict'] == 1)].shape[0] # O CARA É MAU E O MEU MODELO FALA QUE É MAU

    df_aux['caso'] = np.where((df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 0), 'Verdadeiro Negativo (Cliente Bom | Modelo classifica como Bom) - Ganho a Diferença entre Valor Bruto e Valor com Juros', # Ganha a Diferença entre Valor Bruto e Valor com Juros
                        np.where((df_aux['situacao_do_emprestimo'] == 1) & (df_aux['y_predict'] == 0), 'Falso Negativo (Cliente Mau | Modelo classifica como Bom) - Perco o valor emprestado', # Perde o valor emprestado
                        np.where((df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 1), 'Falso Positivo (Cliente Bom | Modelo classifica como Mau) - Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros', # Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros
                        'Verdadeiro Positivo (Cliente Mau | Modelo classifica como Mau) - Não ganho nada' # Não ganho nada
    )))

    df_aux['retorno_financeiro'] = np.where((df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 0), df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] - df_aux['valor_emprestimo_solicitado'], # Ganha a Diferença entre Valor Bruto e Valor com Juros
                        np.where((df_aux['situacao_do_emprestimo'] == 1) & (df_aux['y_predict'] == 0), df_aux['valor_emprestimo_solicitado']*(-1), # Perde o valor emprestado
                        np.where((df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 1), 0, # Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros (df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] - df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'])*(-1)
                        0 # Não ganho nada
    )))

    valor_de_exposicao_total = int(df_aux['valor_emprestimo_solicitado'].sum())
    retorno_financeiro = int(df_aux['retorno_financeiro'].sum())
    valor_conquistado = valor_de_exposicao_total + retorno_financeiro
    return_on_portfolio = round((retorno_financeiro/valor_de_exposicao_total)*100, 2)
    retorno_financeiro_por_caso = df_aux.groupby('caso', as_index = False)['retorno_financeiro'].sum().sort_values(by = 'retorno_financeiro', ascending = False)

    # Crie um DataFrame a partir dos hiperparâmetros
    df = retorno_financeiro_por_caso.reset_index(drop=True)
    df = df.round(2)

    def color_etapa(val):
        color = 'black'
        if val == 'treino':
            color = 'blue'
        elif val == 'teste':
            color = 'red'
        return f'color: {color}; font-weight: bold;'

    # Função para formatar os valores com até duas casas decimais
    def format_values(val):
        if isinstance(val, (int, float)):
            return f'{val:.2f}'
        return val

    # Estilizando o DataFrame
    styled_df = df.style\
        .format(format_values)\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px')\
        .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    return retorno_financeiro, styled_df, valor_de_exposicao_total, return_on_portfolio

def retorno_financeiro_politica(df, y_true, y_predict):

    df_aux = df[['qt_parcelas', 'pagamento_mensal', 'valor_emprestimo_solicitado']].copy()
    df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] = df_aux['pagamento_mensal']*df_aux['qt_parcelas']
    df_aux['y_true'] = y_true['y_true'].values
    df_aux['y_predict'] = y_predict['y_predict'].values

    df_aux['caso'] = np.where((df_aux['y_true'].values == 0) & (df_aux['y_predict'].values == 0),'Verdadeiro Negativo (Cliente Bom | Modelo classifica como Bom) - Ganho a Diferença entre Valor Bruto e Valor com Juros',
                    np.where((df_aux['y_true'].values == 1) & (df_aux['y_predict'].values == 0),'Falso Negativo (Cliente Mau | Modelo classifica como Bom) - Perco o valor emprestado',
                    np.where((df_aux['y_true'].values == 0) & (df_aux['y_predict'].values == 1),'Falso Positivo (Cliente Bom | Modelo classifica como Mau) - Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros','Verdadeiro Positivo (Cliente Mau | Modelo classifica como Mau) - Não ganho nada'
    )))


    df_aux['retorno_financeiro'] = np.where((df_aux['y_true'].values == 0) & (df_aux['y_predict'].values == 0),df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] - df_aux['valor_emprestimo_solicitado'],
                                np.where((df_aux['y_true'].values == 1) & (df_aux['y_predict'].values == 0),df_aux['valor_emprestimo_solicitado'] * (-1),
                                np.where((df_aux['y_true'].values == 0) & (df_aux['y_predict'].values == 1),0,0
    )))

    valor_de_exposicao_total = int(df_aux['valor_emprestimo_solicitado'].sum())
    retorno_financeiro = int(df_aux['retorno_financeiro'].sum())
    valor_conquistado = valor_de_exposicao_total + retorno_financeiro
    return_on_portfolio = round((retorno_financeiro/valor_de_exposicao_total)*100, 2)
    retorno_financeiro_por_caso = df_aux.groupby('caso', as_index = False)['retorno_financeiro'].sum().sort_values(by = 'retorno_financeiro', ascending = False)

    # Crie um DataFrame a partir dos hiperparâmetros
    df = retorno_financeiro_por_caso.reset_index(drop=True)
    df = df.round(2)

    def color_etapa(val):
        color = 'black'
        if val == 'treino':
            color = 'blue'
        elif val == 'teste':
            color = 'red'
        return f'color: {color}; font-weight: bold;'

    # Função para formatar os valores com até duas casas decimais
    def format_values(val):
        if isinstance(val, (int, float)):
            return f'{val:.2f}'
        return val

    # Estilizando o DataFrame
    styled_df = df.style\
        .format(format_values)\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px')\
        .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    return retorno_financeiro, styled_df, valor_de_exposicao_total, return_on_portfolio

def metricas_politica_final(Politica, df, df_grana, y, y_predict, y_predict_proba):

    # Amostra Final
    accuracy = accuracy_score(y, y_predict)
    precision = precision_score(y, y_predict)
    recall = recall_score(y, y_predict)
    f1 = f1_score(y, y_predict)
    roc_auc = roc_auc_score(y['y_true'], y_predict_proba['y_predict_proba_1'])
    fpr, tpr, thresholds = roc_curve(y['y_true'], y_predict_proba['y_predict_proba_1'])
    ks = max(tpr - fpr)
    total, retorno_financeiro_por_caso, valor_de_exposicao_total, return_on_portfolio = retorno_financeiro_politica(df_grana, y, y_predict)
    total = 'R$' + str(int(round(total/1000000, 0))) + ' MM'
    valor_de_exposicao_total = 'R$' + str(float(round(valor_de_exposicao_total/1000000000, 3))) + 'B'
    rocp = str(return_on_portfolio) + '%'
    metricas_finais = pd.DataFrame({
        # 'Acuracia': accuracy, 
        # 'Precisao': precision, 
        # 'Recall': recall, 
        # 'F1-Score': f1, 
        # 'AUC': roc_auc, 
        # 'KS': ks, 
        'Etapa': 'Amostra Final', 
        'Método': Politica, 
        'Valor Total de Exposição': valor_de_exposicao_total,
        'Retorno Financeiro': total,
        'Return on Credit Portfolio (ROCP)': rocp
    }, index=[0])

    df = metricas_finais.reset_index(drop=True)
    df = df.round(2)

    # Função para formatar as células com base na Etapa
    def color_etapa(val):
        color = 'black'
        if val == 'treino':
            color = 'blue'
        elif val == 'teste':
            color = 'red'
        return f'color: {color}; font-weight: bold;'

    # Função para formatar os valores com até duas casas decimais
    def format_values(val):
        if isinstance(val, (int, float)):
            return f'{val:.2f}'
        return val

    # Estilizando o DataFrame
    styled_df = df.style\
        .format(format_values)\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px', subset=pd.IndexSlice[:, 'Etapa':])\
        .applymap(color_etapa, subset=pd.IndexSlice[:, 'Etapa':])\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    # Mostrando o DataFrame estilizado
    return styled_df

def retorno_financeiro_swap_in_swap_out(df):

    df_aux = df.copy()

    TN = df_aux.loc[(df_aux['y_true'] == 0) & (df_aux['y_predict_test_best_clf'] == 0)].shape[0] # O CARA É BOM E MEU MODELO FALA QUE ELE É BOM
    FN = df_aux.loc[(df_aux['y_true'] == 1) & (df_aux['y_predict_test_best_clf'] == 0)].shape[0] # O CARA É MAU E MEU MODELO FALA QUE ELE É BOM
    FP = df_aux.loc[(df_aux['y_true'] == 0) & (df_aux['y_predict_test_best_clf'] == 1)].shape[0] # O CARA É BOM E MEU MODELO FALA QUE É MAU
    TP = df_aux.loc[(df_aux['y_true'] == 1) & (df_aux['y_predict_test_best_clf'] == 1)].shape[0] # O CARA É MAU E O MEU MODELO FALA QUE É MAU

    df_aux['caso'] = np.where((df_aux['y_true'] == 0) & (df_aux['y_predict_test_best_clf'] == 0), 'Verdadeiro Negativo (Cliente Bom | Modelo classifica como Bom) - Ganho a Diferença entre Valor Bruto e Valor com Juros', # Ganha a Diferença entre Valor Bruto e Valor com Juros
                        np.where((df_aux['y_true'] == 1) & (df_aux['y_predict_test_best_clf'] == 0), 'Falso Negativo (Cliente Mau | Modelo classifica como Bom) - Perco o valor emprestado', # Perde o valor emprestado
                        np.where((df_aux['y_true'] == 0) & (df_aux['y_predict_test_best_clf'] == 1), 'Falso Positivo (Cliente Bom | Modelo classifica como Mau) - Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros', # Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros
                        'Verdadeiro Positivo (Cliente Mau | Modelo classifica como Mau) - Não ganho nada' # Não ganho nada
    )))

    df_aux['retorno_financeiro'] = np.where((df_aux['y_true'] == 0) & (df_aux['y_predict_test_best_clf'] == 0), df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] - df_aux['valor_emprestimo_solicitado'], # Ganha a Diferença entre Valor Bruto e Valor com Juros
                        np.where((df_aux['y_true'] == 1) & (df_aux['y_predict_test_best_clf'] == 0), df_aux['valor_emprestimo_solicitado']*(-1), # Perde o valor emprestado
                        np.where((df_aux['y_true'] == 0) & (df_aux['y_predict_test_best_clf'] == 1), 0, # Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros (df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] - df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'])*(-1)
                        0 # Não ganho nada
    )))

    valor_de_exposicao_total = int(df_aux['valor_emprestimo_solicitado'].sum())
    retorno_financeiro = int(df_aux['retorno_financeiro'].sum())
    valor_conquistado = valor_de_exposicao_total + retorno_financeiro
    return_on_portfolio = round((retorno_financeiro/valor_de_exposicao_total)*100, 2)
    retorno_financeiro_por_caso = df_aux.groupby('caso', as_index = False)['retorno_financeiro'].sum().sort_values(by = 'retorno_financeiro', ascending = False)

    # Crie um DataFrame a partir dos hiperparâmetros
    df = retorno_financeiro_por_caso.reset_index(drop=True)
    df = df.round(2)

    def color_etapa(val):
        color = 'black'
        if val == 'treino':
            color = 'blue'
        elif val == 'teste':
            color = 'red'
        return f'color: {color}; font-weight: bold;'

    # Função para formatar os valores com até duas casas decimais
    def format_values(val):
        if isinstance(val, (int, float)):
            return f'{val:.2f}'
        return val

    # Estilizando o DataFrame
    styled_df = df.style\
        .format(format_values)\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px')\
        .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    return retorno_financeiro, styled_df, valor_de_exposicao_total, return_on_portfolio

def plot_shap_beeswarm(modelo_pipeline, X, titulo="SHAP Beeswarm"):
    """
    Plota apenas o gráfico Beeswarm de valores SHAP para um modelo XGBoost dentro de um Pipeline sklearn.
    """
    # --- 1) Extrai o modelo final (XGBoost) ---
    model = modelo_pipeline.named_steps["xgbclassifier"]

    # --- 2) Extrai o ColumnTransformer e aplica transformação ---
    X_transformado = modelo_pipeline.named_steps["columntransformer"].transform(X)
    feature_names = modelo_pipeline.named_steps["columntransformer"].get_feature_names_out()

    # --- 3) Cria o SHAP Explainer e calcula valores ---
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_transformado)

    # --- 4) Plota apenas o Beeswarm ---
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_transformado, feature_names=feature_names, show=False, plot_size=(12, 6))
    plt.title(titulo, fontsize=14)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Funções gerais de rating, migração e análise de risco
# Rotinas extraídas e generalizadas a partir do notebook de modelagem.
# -----------------------------------------------------------------------------

def target_encoder_bad_rate(
    df,
    colunas_categoricas,
    target,
    tabelas_encoding=None,
    percentual=True,
    sufixo='_bad_rate',
):
    """Cria ou aplica target encoding pela taxa média do evento.

    Se ``tabelas_encoding`` não for informado, as tabelas são aprendidas a
    partir de ``df`` e devolvidas junto com os dados transformados. Para aplicar
    o mesmo encoding em outra amostra, passe as tabelas retornadas anteriormente.
    """
    resultado = df.copy()
    tabelas = {} if tabelas_encoding is None else tabelas_encoding
    aprender = tabelas_encoding is None
    escala = 100 if percentual else 1

    for coluna in colunas_categoricas:
        nome_encoding = f'{coluna}{sufixo}'
        if aprender:
            tabela = (
                df.groupby(coluna, dropna=False)[target]
                .mean().mul(escala)
                .rename(nome_encoding).reset_index()
            )
            tabelas[coluna] = tabela
        else:
            tabela = tabelas[coluna]
        resultado = resultado.merge(tabela, on=coluna, how='left').drop(columns=coluna)
    return resultado, tabelas


def formatar_valor_milhoes(valor, pos=None):
    """Formata um valor monetário em milhões de reais."""
    return f'R$ {valor / 1e6:.2f} MM'


def abbreviate_number(number):
    """Abrevia números com os sufixos K, MM e B."""
    for fator, sufixo in ((1e9, 'B'), (1e6, 'MM'), (1e3, 'K')):
        if abs(number) >= fator:
            return f'{number / fator:.1f}'.rstrip('0').rstrip('.') + sufixo
    return str(int(number))


def calcular_rating(probabilidade_default, n_ratings=10, maior_rating_menor_risco=True):
    """Converte probabilidades de default em ratings de frequência semelhante."""
    probabilidades = pd.Series(probabilidade_default).astype(float)
    if n_ratings < 2:
        raise ValueError('n_ratings deve ser maior ou igual a 2.')
    if probabilidades.notna().sum() < n_ratings:
        raise ValueError('Não há observações suficientes para formar os ratings.')

    # O rank evita a perda de faixas quando várias probabilidades são iguais.
    decis = pd.qcut(
        probabilidades.rank(method='first'),
        q=n_ratings,
        labels=False,
    ).astype('Int64')
    rating = n_ratings - 1 - decis if maior_rating_menor_risco else decis
    return rating.rename('rating')


def criar_base_rating(
    df,
    probabilidade_default,
    nome_rating='rating_model',
    target=None,
    coluna_exposicao=None,
    n_ratings=10,
):
    """Cria a base analítica de rating, incluindo volumetria e perda esperada."""
    resultado = df.copy()
    resultado['Probability of Default'] = np.asarray(probabilidade_default)
    resultado[nome_rating] = calcular_rating(
        resultado['Probability of Default'], n_ratings=n_ratings
    ).to_numpy()
    resultado['qt_pessoas_rating'] = resultado.groupby(nome_rating)[nome_rating].transform('size')
    if coluna_exposicao is not None:
        resultado['expected_loss'] = (
            resultado['Probability of Default'] * resultado[coluna_exposicao]
        )
    if target is not None:
        resultado['bad_rate_rating'] = resultado.groupby(nome_rating)[target].transform('mean')
    return resultado


def comparar_ratings(
    df,
    probabilidade_modelo,
    probabilidade_politica,
    n_ratings=10,
):
    """Atribui ratings do modelo e da política e classifica cada migração."""
    resultado = df.copy()
    resultado['Probability of Default - Modelo'] = np.asarray(probabilidade_modelo)
    resultado['Probability of Default - Política'] = np.asarray(probabilidade_politica)
    resultado['rating_model'] = calcular_rating(
        resultado['Probability of Default - Modelo'], n_ratings
    ).to_numpy()
    resultado['rating_politica'] = calcular_rating(
        resultado['Probability of Default - Política'], n_ratings
    ).to_numpy()
    resultado['mesmo_resultado'] = resultado['rating_model'].eq(resultado['rating_politica'])
    resultado['performance_swap_in_swap_out'] = np.select(
        [
            resultado['rating_model'] < resultado['rating_politica'],
            resultado['rating_model'] > resultado['rating_politica'],
        ],
        ['Downgrade', 'Upgrade'],
        default='Manutencao',
    )
    return resultado


def matriz_migracao_rating(
    df,
    rating_origem='rating_model',
    rating_destino='rating_politica',
    normalizar='total',
    segmento=None,
    coluna_segmento=None,
):
    """Calcula a matriz de migração de ratings, geral ou por produto.

    ``normalizar`` aceita ``None`` (contagens), ``'total'`` (percentual da
    carteira) ou ``'linha'`` (percentual dentro de cada rating de origem).
    """
    if segmento is not None and coluna_segmento is None:
        raise ValueError('Informe coluna_segmento ao filtrar um segmento.')
    dados = df if segmento is None else df.loc[df[coluna_segmento].eq(segmento)]
    matriz = pd.crosstab(dados[rating_origem], dados[rating_destino], dropna=False)
    ratings = sorted(set(dados[rating_origem].dropna()) | set(dados[rating_destino].dropna()))
    matriz = matriz.reindex(index=ratings, columns=ratings, fill_value=0)
    if normalizar is None:
        return matriz
    if normalizar == 'total':
        total = matriz.to_numpy().sum()
        return (matriz / total * 100).round(2) if total else matriz.astype(float)
    if normalizar == 'linha':
        return matriz.div(matriz.sum(axis=1).replace(0, np.nan), axis=0).mul(100).fillna(0).round(2)
    raise ValueError("normalizar deve ser None, 'total' ou 'linha'.")


def resumo_migracao_rating(df, coluna='performance_swap_in_swap_out'):
    """Resume a volumetria absoluta e percentual de upgrades/downgrades."""
    resumo = df[coluna].value_counts().rename('quantidade').to_frame()
    resumo['percentual'] = (resumo['quantidade'] / len(df) * 100).round(2)
    return resumo.rename_axis('migracao').reset_index()


def plot_matriz_migracao_rating(matriz, titulo='Matriz de Migração de Rating', fmt='.2f'):
    """Plota uma matriz de migração já calculada."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matriz, annot=True, cmap='Blues', fmt=fmt, linewidths=.5, ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel('Rating de destino')
    ax.set_ylabel('Rating de origem')
    fig.tight_layout()
    return fig, ax


def plot_rating_risco(
    df,
    target,
    perda_esperada,
    rating='rating_model',
    titulo='Risco de Crédito por Rating',
):
    """Plota volumetria, maus pagadores e perda esperada por rating."""
    resumo = df.groupby(rating).agg(
        total_pessoas=(rating, 'size'),
        maus_pagadores=(target, 'sum'),
        perda_esperada=(perda_esperada, 'sum'),
    ).reset_index()
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=resumo, x=rating, y='total_pessoas', color='#1FB3E5', ax=ax1)
    ax1.set(title=titulo, xlabel='Rating', ylabel='Total de pessoas')
    ax2 = ax1.twinx()
    sns.lineplot(data=resumo, x=rating, y='maus_pagadores', marker='o', color='blue', ax=ax2)
    sns.lineplot(data=resumo, x=rating, y='perda_esperada', marker='o', color='red', ax=ax2)
    ax2.set_ylabel('Maus pagadores / Perda esperada')
    fig.tight_layout()
    return fig, (ax1, ax2)


"""Funções reutilizáveis extraídas do notebook do case Data Masters."""

import builtins
import pickle
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from IPython.display import display
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

try:
    from skopt import BayesSearchCV
except ImportError:  # Necessário apenas nas rotinas de otimização.
    BayesSearchCV = None


def plota_barras(lista_variaveis, df, titulo, rotation=0):        
    k = 0
    # Ordena os dados para garantir que as labels correspondam corretamente às barras
    df_sorted = df[lista_variaveis[k]].value_counts().index
    ax = sns.countplot(x=lista_variaveis[k], data=df, order=df_sorted, color='#1FB3E5')
    
    ax.set_title(f'{titulo}')
    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize=14)
    ax.set_ylabel('Quantidade', fontsize=14)
    
    # Calcular o total para obter os percentuais
    total = sum([p.get_height() for p in ax.patches])
    
    sizes = []
    for bar in ax.patches:
        height = bar.get_height()
        sizes.append(height)
        ax.text(bar.get_x() + bar.get_width()/2,
                height,
                f'{builtins.round((height/total)*100, 2)}%',
                ha='center',
                fontsize=12
        )
    
    ax.set_ylim(0, max(sizes) * 1.1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right', fontsize=10)
    ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)

    plt.tight_layout()
    plt.show()


def plota_grafico_linhas(df, x, y, nao_calcula_media, title):

    if nao_calcula_media:
        # Criando o gráfico de linha
        plt.figure(figsize=(18, 6))
        plt.plot(df[x], df[y], marker='o', linestyle='-', color='#1FB3E5')

        # Adicionando títulos e rótulos aos eixos
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)

        for i, txt in enumerate(df[y]):
            plt.annotate(f'{txt:.1f}', (df[x][i], df[y][i]), textcoords="offset points", xytext=(0,1), ha='center')

        # Exibindo o gráfico
        plt.grid(True)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        media = df[y].mean()
        # Criando o gráfico de linha
        plt.figure(figsize=(18, 6))
        plt.plot(df[x], df[y], marker='o', linestyle='-', color='#1FB3E5')

        # Adicionando linha da média
        plt.axhline(y=media, color='r', linestyle='--', linewidth=1, label=f'Média: {media:.2f}')
        plt.legend()

        # Adicionando títulos e rótulos aos eixos
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)

        for i, txt in enumerate(df[y]):
            plt.annotate(f'{txt:.1f}', (df[x][i], df[y][i]), textcoords="offset points", xytext=(0,1), ha='center')

        # Exibindo o gráfico
        plt.grid(True)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


def muda_tipagem_variavel(df, feature, type):

    if type == "int":
        df[feature] = df[feature].apply(lambda x: int(x) if pd.notnull(x) else 999999)
    else:
        df[feature] = df[feature].apply(lambda x: float(x) if pd.notnull(x) else 999999)

    df.replace(999999, np.nan, inplace=True)

    return df[feature]


def analisa_distribuicao_via_percentis(df, variaveis):
    def sublinha_percentis(s):
        is_1_percentile = s.name == '1%'
        is_99_8_percentile = s.name == '99.8%'
        if is_1_percentile or is_99_8_percentile:
            return ['background-color: blue'] * len(s)
        else:
            return [''] * len(s)

    percentis = df[variaveis].describe(percentiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.998]).style.apply(sublinha_percentis, axis=1)    

    return percentis


def compara_medias_amostras(df, variaveis_continuas):  
    num_variaveis = len(variaveis_continuas)
    num_pares = (num_variaveis + 1) // 2  # Número de pares de variáveis para subplots
    fig, axes = plt.subplots(num_pares, 2, figsize=(14, 4 * num_pares))

    # Ajusta para o caso onde há apenas uma variável
    if num_pares == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for i in range(num_pares):
        if 2 * i < num_variaveis:
            variavel1 = variaveis_continuas[2 * i]
            percentis1 = df[variavel1].describe(percentiles=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
            p1_1 = percentis1['1%']
            p99_1 = percentis1['99%']
            df_raw1 = df.loc[(df[variavel1] > p1_1) & (df[variavel1] < p99_1)].copy()
            df_com_churn1 = df_raw1.loc[df_raw1["churn"] == 1]
            df_sem_churn1 = df_raw1.loc[df_raw1["churn"] == 0]
            
            medias_amostrais_com_churn1 = []
            medias_amostrais_sem_churn1 = []
            
            for j in range(5000):
                amostra_churn1 = random.choices(df_com_churn1[variavel1].values, k=1000)
                media_amostra_churn1 = np.mean(amostra_churn1)
                medias_amostrais_com_churn1.append(media_amostra_churn1)

                amostra_sem_churn1 = random.choices(df_sem_churn1[variavel1].values, k=1000)
                media_amostra_sem_churn1 = np.mean(amostra_sem_churn1)
                medias_amostrais_sem_churn1.append(media_amostra_sem_churn1)

            ax_hist1 = axes[i, 0]
            ax_hist1.hist(medias_amostrais_com_churn1, bins=30, alpha=0.5, label='Churn', linewidth=5, color="red")
            ax_hist1.hist(medias_amostrais_sem_churn1, bins=30, alpha=0.5, label='Sem Churn', linewidth=5, color="green")
            ax_hist1.legend(loc='upper right')
            ax_hist1.set_xlabel('Valores')
            ax_hist1.set_ylabel('Frequência')
            ax_hist1.set_title(f'Distribuição das Médias Amostrais de "{variavel1}" ')
            ax_hist1.grid(True)
        
        if 2 * i + 1 < num_variaveis:
            variavel2 = variaveis_continuas[2 * i + 1]
            percentis2 = df[variavel2].describe(percentiles=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
            p1_2 = percentis2['1%']
            p99_2 = percentis2['99%']
            df_raw2 = df.loc[(df[variavel2] > p1_2) & (df[variavel2] < p99_2)].copy()
            df_com_churn2 = df_raw2.loc[df_raw2["churn"] == 1]
            df_sem_churn2 = df_raw2.loc[df_raw2["churn"] == 0]
            
            medias_amostrais_com_churn2 = []
            medias_amostrais_sem_churn2 = []
            
            for j in range(5000):
                amostra_churn2 = random.choices(df_com_churn2[variavel2].values, k=1000)
                media_amostra_churn2 = np.mean(amostra_churn2)
                medias_amostrais_com_churn2.append(media_amostra_churn2)

                amostra_sem_churn2 = random.choices(df_sem_churn2[variavel2].values, k=1000)
                media_amostra_sem_churn2 = np.mean(amostra_sem_churn2)
                medias_amostrais_sem_churn2.append(media_amostra_sem_churn2)

            ax_hist2 = axes[i, 1]
            ax_hist2.hist(medias_amostrais_com_churn2, bins=30, alpha=0.5, label='Churn', linewidth=5, color="red")
            ax_hist2.hist(medias_amostrais_sem_churn2, bins=30, alpha=0.5, label='Sem Churn', linewidth=5, color="green")
            ax_hist2.legend(loc='upper right')
            ax_hist2.set_xlabel('Valores')
            ax_hist2.set_ylabel('Frequência')
            ax_hist2.set_title(f'Distribuição das Médias Amostrais de "{variavel2}" ')
            ax_hist2.grid(True)

    plt.tight_layout()
    plt.show()


def woe(df, feature, target):
    churn = df.loc[df[target] == 1].groupby(feature, as_index = False)[target].count().rename({target:'churn'}, axis = 1)
    sem_churn = df.loc[df[target] == 0].groupby(feature, as_index = False)[target].count().rename({target:'sem_churn'}, axis = 1)

    woe = churn.merge(sem_churn, on = feature, how = 'left')
    woe['percent_churn'] = woe['churn']/woe['churn'].sum()
    woe['percent_sem_churn'] = woe['sem_churn']/woe['sem_churn'].sum()
    woe['woe'] = round(np.log(woe['percent_churn']/woe['percent_sem_churn']), 3)
    woe.sort_values(by = 'woe', ascending = True, inplace = True)
    
    weight_of_evidence = woe['woe'].unique()


    x = list(woe[feature])
    y = list(woe['woe'])

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker='o', linestyle='--', linewidth=2, color='#1FB3E5')

    for label, value in zip(x, y):
        plt.text(x=label, y=value, s=str(value), fontsize=10, color='red', ha='left', va='center', rotation=45)

    plt.title(f'Weight of Evidence da variável "{feature}"', fontsize=14)
    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Weight of Evidence', fontsize=14)
    plt.xticks(ha='right', fontsize=10, rotation=45)
    plt.show()


def months_as_a_registered(df):

    df["registration_init_time"] = df["registration_init_time"].apply(lambda x:str(x)[:6])

    # Converter as colunas para objetos datetime
    df['registration_init_time'] = pd.to_datetime(df['registration_init_time'], format='%Y%m')
    df['safra'] = pd.to_datetime(df['safra'], format='%Y%m')

    # Calcular a diferença de meses
    df['months_as_a_registered'] = (df['safra'].dt.year - df['registration_init_time'].dt.year) * 12 + (df['safra'].dt.month - df['registration_init_time'].dt.month)
    df["safra"] = df["safra"].apply(lambda x:str(x)[:7].replace("-", ""))

    return df['months_as_a_registered']


def num_more_than_50(df):

    df["num_less_than_50"] = df["num_25"] + df["num_50"]
    df["num_more_than_50"] = df["num_75"] + df["num_985"] + df["num_100"]
    df["%num_more_than_50"] = round(df["num_more_than_50"]/(df["num_more_than_50"]+df["num_less_than_50"])*100, 2)

    return df["%num_more_than_50"]


def media_movel(df, feature, window):

    df[f"{feature}_mov_avg_m{window}"] = df.groupby('msno')[feature].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
                                           
    return df[f"{feature}_mov_avg_m{window}"]


def max_movel(df, feature, window):

    df[f"{feature}_mov_max_m{window}"] = df.groupby('msno')[feature].rolling(window=window, min_periods=1).max().reset_index(level=0, drop=True)
    
    return df[f"{feature}_mov_max_m{window}"]


def min_movel(df, feature, window):

    df[f"{feature}_mov_min_m{window}"] = df.groupby('msno')[feature].rolling(window=window, min_periods=1).min().reset_index(level=0, drop=True)
    
    return df[f"{feature}_mov_min_m{window}"]


def get_temporal_features(df):

    tabela_de_transacoes_media_movel = pd.read_parquet("../00_DataMaster/data/tabela_de_transacoes_media_movel.parquet")
    tabela_user_logs_media_movel = pd.read_parquet("../00_DataMaster/data/tabela_user_logs_media_movel.parquet")

    tabela_de_transacoes_max_movel = pd.read_parquet("../00_DataMaster/data/tabela_de_transacoes_max_movel.parquet")
    tabela_user_logs_max_movel = pd.read_parquet("../00_DataMaster/data/tabela_user_logs_max_movel.parquet")

    tabela_de_transacoes_min_movel = pd.read_parquet("../00_DataMaster/data/tabela_de_transacoes_min_movel.parquet")
    tabela_user_logs_min_movel = pd.read_parquet("../00_DataMaster/data/tabela_user_logs_min_movel.parquet")

    df = df.merge(tabela_de_transacoes_media_movel, on = ["msno", "safra"], how = "left")
    df = df.merge(tabela_user_logs_media_movel, on = ["msno", "safra"], how = "left")

    df = df.merge(tabela_de_transacoes_max_movel, on = ["msno", "safra"], how = "left")
    df = df.merge(tabela_user_logs_max_movel, on = ["msno", "safra"], how = "left")

    df = df.merge(tabela_de_transacoes_min_movel, on = ["msno", "safra"], how = "left")
    df = df.merge(tabela_user_logs_min_movel, on = ["msno", "safra"], how = "left")

    return df


def transform_to_percentiles(df, n, variavel_continua):
    # Calcula os limites dos percentiles
    percentile_limits = [i / n for i in range(n+1)] 
    
    # Aplica a função qcut para transformar a variável em percentiles
    percentiles = pd.qcut(df[variavel_continua], q=n, labels=False, duplicates='drop')
    
    return percentiles


def agrupa_categorias_cidade_pelo_woe(df):

    df['city'] = (
                np.where(df['city'].isin(['13', '14', '16', '7', '20']), 0, 
                np.where(df['city'].isin(['17', '4', '5', '18']), 1, 
                np.where(df['city'].isin(['19', '3', '15', '10']), 2, 
                np.where(df['city'].isin(['12', '11', '6', '22']), 3, 
                np.where(df['city'].isin(['8', '21', '9', '1']), 4, 
                np.nan)))))
    )

    return df['city']


def agrupa_categorias_metodo_pagamento_pelo_woe(df):

    df['payment_method_id'] = (
                np.where(df['payment_method_id'].isin(['32', '14', '10', '19', '31']), 0, 
                np.where(df['payment_method_id'].isin(['34', '41', '18', '37', '21']), 1, 
                np.where(df['payment_method_id'].isin(['23', '33', '39', '27']), 2, 
                np.where(df['payment_method_id'].isin(['40', '30', '16', '36', '26']), 3, 
                np.where(df['payment_method_id'].isin(['38', '29', '28', '35', '17']), 4, 
                np.nan)))))
    )

    return df['payment_method_id']


def pre_processamento_categoricas_nulas(df):
    variaveis_categoricas = ['is_auto_renew', 'gender', 'registered_via', 'city', 'payment_method_id']

    def aplica_tratamento_categorica_nulas(df, feature):
        df[feature] = np.where(df[feature].isnull(), 999, df[feature])
        return df[feature]

    for categorica in variaveis_categoricas:
        df[categorica] = aplica_tratamento_categorica_nulas(df, categorica)

    return df


def pre_processamento_continuas_nulas(df, calcula_mediana=False):
    variaveis_continuas = [
        'bd',
        'months_as_a_registered',
        'payment_plan_days', 
        'actual_amount_paid', 'actual_amount_paid_mov_avg_m6', 'actual_amount_paid_mov_max_m6', 'actual_amount_paid_mov_min_m6', 
        'num_25', 'num_25_mov_avg_m6', 'num_25_mov_max_m6', 'num_25_mov_min_m6',
        'num_50', 'num_50_mov_avg_m6', 'num_50_mov_max_m6', 'num_50_mov_min_m6', 
        'num_75', 'num_75_mov_avg_m6', 'num_75_mov_max_m6', 'num_75_mov_min_m6', 
        'num_985', 'num_985_mov_avg_m6', 'num_985_mov_max_m6', 'num_985_mov_min_m6', 
        'num_100', 'num_100_mov_avg_m6', 'num_100_mov_max_m6', 'num_100_mov_min_m6',
        'num_unq', 'num_unq_mov_avg_m6','num_unq_mov_max_m6', 'num_unq_mov_min_m6',  
        '%num_more_than_50', '%num_more_than_50_mov_avg_m6', '%num_more_than_50_mov_max_m6', '%num_more_than_50_mov_min_m6'
    ]

    def calcular_e_salvar_mediana_por_cidade(df):
        variaveis_continuas = [
            'bd',
            'months_as_a_registered',
            'payment_plan_days', 
            'actual_amount_paid', 'actual_amount_paid_mov_avg_m6', 'actual_amount_paid_mov_max_m6', 'actual_amount_paid_mov_min_m6',
            'num_25', 'num_25_mov_avg_m6', 'num_25_mov_max_m6', 'num_25_mov_min_m6',
            'num_50', 'num_50_mov_avg_m6', 'num_50_mov_max_m6', 'num_50_mov_min_m6', 
            'num_75', 'num_75_mov_avg_m6', 'num_75_mov_max_m6', 'num_75_mov_min_m6', 
            'num_985', 'num_985_mov_avg_m6', 'num_985_mov_max_m6', 'num_985_mov_min_m6', 
            'num_100', 'num_100_mov_avg_m6', 'num_100_mov_max_m6', 'num_100_mov_min_m6',
            'num_unq', 'num_unq_mov_avg_m6','num_unq_mov_max_m6', 'num_unq_mov_min_m6',  
            '%num_more_than_50', '%num_more_than_50_mov_avg_m6', '%num_more_than_50_mov_max_m6', '%num_more_than_50_mov_min_m6'
        ]
        
        cidades = df['city'].unique()
        for cidade in cidades:
            mediana_df = df[df['city'] == cidade][variaveis_continuas].median().reset_index()
            mediana_df.columns = ['variavel', 'mediana']
            mediana_df.to_excel(f"../00_DataMaster/pre_processing/mediana_city_{cidade}.xlsx", index=False)

    if calcula_mediana:
        calcular_e_salvar_mediana_por_cidade(df)
        print(f'Salvando a Mediana com dados de treinamento...')

    for cidade in df['city'].unique():
        mediana_dict = pd.read_excel(f"../00_DataMaster/pre_processing/mediana_city_{cidade}.xlsx").set_index('variavel')['mediana'].to_dict()
        for feature in variaveis_continuas:
            percentis = df[feature].describe(percentiles=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
            p1 = percentis['1%']
            p99 = percentis['99%']
            df[feature] = np.where((df[feature].isnull() | (df[feature] <= p1) | (df[feature] >= p99)), mediana_dict[feature], df[feature])
            
        return df


def woe_training(df, feature):
    churn = df.dropna().loc[df.dropna()["churn"] == 1].groupby(feature, as_index = False)["churn"].count().rename({"churn":'churn'}, axis = 1)
    sem_churn = df.dropna().loc[df.dropna()["churn"] == 0].groupby(feature, as_index = False)["churn"].count().rename({"churn":'sem_churn'}, axis = 1)

    woe = churn.merge(sem_churn, on = feature, how = 'left')
    woe['percent_churn'] = woe['churn']/woe['churn'].sum()
    woe['percent_sem_churn'] = woe['sem_churn']/woe['sem_churn'].sum()
    woe['woe'] = round(np.log(woe['percent_churn']/woe['percent_sem_churn']), 3)
    woe = woe[[feature, "woe"]]
    woe.rename({f"{feature}":"variavel"}, axis = 1)
    woe.sort_values(by = "woe", ascending = True, inplace = True)
    woe.to_excel(f"../00_DataMaster/pre_processing/target_encoder_{feature}.xlsx", index = False)
    print(f"Excel com o WOE da variável {feature} Salvo!!")


def target_encoder_woe(df):

    woe_is_auto_renew = pd.read_excel(f"../00_DataMaster/pre_processing/target_encoder_is_auto_renew.xlsx").set_index('is_auto_renew')['woe'].to_dict()
    woe_gender = pd.read_excel(f"../00_DataMaster/pre_processing/target_encoder_gender.xlsx").set_index('gender')['woe'].to_dict()
    woe_registered_via = pd.read_excel(f"../00_DataMaster/pre_processing/target_encoder_registered_via.xlsx").set_index('registered_via')['woe'].to_dict()
    woe_city = pd.read_excel(f"../00_DataMaster/pre_processing/target_encoder_city.xlsx").set_index('city')['woe'].to_dict()
    woe_payment_method_id = pd.read_excel(f"../00_DataMaster/pre_processing/target_encoder_payment_method_id.xlsx").set_index('payment_method_id')['woe'].to_dict()

    df["is_auto_renew"] = df['is_auto_renew'].map(woe_is_auto_renew)
    df["gender"] = df['gender'].map(woe_gender)
    df["registered_via"] = df['registered_via'].map(woe_registered_via).fillna(0) # CONTINGÊNCIA POIS UMA DAS CATEGORIAS TINHA 20 REGISTROS E NÃO TINHA CHURN, ENTÃO O WOE SERÁ 0 COMO CONTINGÊNCIA
    df["city"] = df['city'].map(woe_city)
    df["payment_method_id"] = df['payment_method_id'].map(woe_payment_method_id)

    return df


def analisa_correlacao(metodo, df):
    plt.figure(figsize=(30, 15))
    mask = np.triu(np.ones_like(df.corr(method=metodo), dtype=bool))
    heatmap = sns.heatmap(df.corr(method=metodo), vmin=-1, vmax=1, cmap='magma', annot=True, fmt='.1f', cbar_kws={"shrink": .8}, mask=mask)
    heatmap.set_title(f"Analisando Correlação de {metodo}")
    plt.grid(False)
    plt.box(False)
    plt.tight_layout()
    plt.grid(False)
    plt.show()


def aplica_feature_selection(df):
    def separa_feature_target(target, dados):
        x = dados.drop(target, axis = 1)
        y = dados[[target]]

        return x, y

    def remove_features_feature_importance(target, df, class_weight, threshold):
        # Separa entre Features e Target
        x, y = separa_feature_target(target, df)
        
        # Criar o modelo de Random Forest
        model = RandomForestClassifier(random_state=42, criterion='entropy', n_estimators=20, class_weight={0:1, 1:class_weight})
        
        # Treinar o modelo
        model.fit(x, y)
        
        # Obter as importâncias das features
        feature_importances = model.feature_importances_
        
        # Selecionar as features com importância maior que zero
        selected_features = list(x.columns[feature_importances > threshold])
        selected_features.append(target)
        
        feature_importance_df = pd.DataFrame({
            'feature': x.columns,
            'importance': feature_importances
        }).sort_values(by='importance', ascending=False)
        feature_importance_df = feature_importance_df.loc[feature_importance_df['importance'] > 0]
        feature_importance_df['importance'] = feature_importance_df['importance'] * 100
        
        return selected_features, feature_importance_df

    def remove_features_altamente_correlacionadas(df, variaveis_importantes_df, threshold_correlacao=0.9):
        # Filtrar variáveis com alta importância
        alta_importancia_features = variaveis_importantes_df['feature'].tolist()
        
        # Selecionar as colunas do DataFrame com as variáveis de interesse
        df_reduzido = df[alta_importancia_features]
        
        # Calcular a matriz de correlação de Spearman
        correlacoes = df_reduzido.corr(method='spearman')
        
        # Encontrar variáveis altamente correlacionadas
        alta_correlacao = np.abs(correlacoes) > threshold_correlacao
        features_para_remover = set()
        
        for i in range(len(alta_correlacao.columns)):
            for j in range(i):
                if alta_correlacao.iloc[i, j] and correlacoes.columns[j] not in features_para_remover:
                    features_para_remover.add(correlacoes.columns[i])
        
        variaveis_filtradas = [col for col in alta_importancia_features if col not in features_para_remover]
        
        return variaveis_filtradas

    # Aplicando Random Forest e selecionado feature com importância > 0
    features, feature_importances = remove_features_feature_importance('churn', df.drop(['msno', 'safra'], axis=1).copy(), 5, 0)
    feature_importances = feature_importances.loc[feature_importances['importance'] > 0]
    
    # Filtrar variáveis altamente correlacionadas e mantendo a que possui maior importância com a target dentre as correlacionadas
    variaveis_selecionadas = remove_features_altamente_correlacionadas(df, feature_importances)
    feature_importances_final = feature_importances[feature_importances['feature'].isin(variaveis_selecionadas)]

    return feature_importances_final


def aplica_pre_processamento_feature_eng_feature_selection(df):

    # Aplicando Feature Engineering
    df['months_as_a_registered'] = months_as_a_registered(df)
    df['%num_more_than_50'] = num_more_than_50(df)
    df['city'] = agrupa_categorias_cidade_pelo_woe(df)
    df['payment_method_id'] = agrupa_categorias_metodo_pagamento_pelo_woe(df)
    df = get_temporal_features(df)

    # Aplicando Pré-Processamento
    df["is_auto_renew"] = muda_tipagem_variavel(df, "is_auto_renew", "int")
    df["bd"] = muda_tipagem_variavel(df, "bd", "int")
    df["registered_via"] = muda_tipagem_variavel(df, "registered_via", "int")
    df["payment_plan_days"] = muda_tipagem_variavel(df, "payment_plan_days", "float")
    df["actual_amount_paid"] = muda_tipagem_variavel(df, "actual_amount_paid", "float")

    df = pre_processamento_categoricas_nulas(df)
    df = pre_processamento_continuas_nulas(df)

    df = target_encoder_woe(df)

    # Organizando
    variaveis_selecionadas =  [
        'is_auto_renew', 'payment_method_id', 'months_as_a_registered',
        'num_unq_mov_max_m6', 'num_100_mov_max_m6', 'num_unq_mov_min_m6',
        'num_100_mov_min_m6', '%num_more_than_50_mov_max_m6',
        '%num_more_than_50_mov_avg_m6', '%num_more_than_50_mov_min_m6',
        'num_25_mov_max_m6', 'bd', 'num_50_mov_avg_m6',
        'num_985_mov_avg_m6', 'actual_amount_paid_mov_avg_m6',
        'num_25_mov_min_m6', 'num_75_mov_max_m6', 'num_50_mov_min_m6',
        'num_985_mov_min_m6', 'city', 'num_75_mov_min_m6',
        'registered_via', 'payment_plan_days',
        ]
    mensalidade = ['actual_amount_paid']
    safra = ["safra"]
    target = ["churn"]
    user_id = ["msno"]

    df = df[target + user_id + safra + mensalidade + variaveis_selecionadas]

    return df


def separa_feature_target(target, dados):
    x = dados.drop(target, axis = 1)
    y = dados[[target]]

    return x, y


def train_min_max_scaler(df):

    cols = list(df.drop(['churn', 'msno', 'safra', 'actual_amount_paid'], axis = 1).columns)

    df_scaler = df[cols].copy()

    scaler = MinMaxScaler()
    scaler.fit(df_scaler)
    joblib.dump(scaler, "../00_DataMaster/models/scaler.pkl")
    print('Scaler Treinado e Salvo com sucesso!')


def Classificador(classificador, x_train, y_train, x_test, y_test, class_weight):

    # Puxa o Scaler Treinado com os dados de Treino
    scaler = joblib.load("../00_DataMaster/models/scaler.pkl")
    
    cols = list(x_train.drop(['msno', 'safra', 'actual_amount_paid'], axis = 1).columns)

    x_train = x_train[cols]
    x_test = x_test[cols]

    # Define as colunas categóricas e numéricas
    models = {
        'Regressão Logística': make_pipeline(
            ColumnTransformer([
                ('scaler', make_pipeline(scaler), cols)
            ]),
            LogisticRegression(
                random_state=42, # Semente aleatória para reproducibilidade dos resultados
                class_weight={0: 1, 1: class_weight}, # Peso atribuído às classes. Pode ser útil para lidar com conjuntos de dados desbalanceados.
                C=1, # Parâmetro de regularização inversa. Controla a força da regularização.
                penalty='l2', # Tipo de regularização. 'l1', 'l2', 'elasticnet', ou 'none'.
                max_iter=50, # Número máximo de iterações para a convergência do otimizador.
                solver='liblinear' # Algoritmo de otimização. 'newton-cg', 'lbfgs', 'liblinear' (gradiente descendente), 'sag' (Stochastic gradient descent), 'saga' (Stochastic gradient descent que suporta reg L1).
                )
        ),
        'Random Forest': make_pipeline(
        RandomForestClassifier(
            random_state=42,            # Semente aleatória para reproducibilidade dos resultados
            criterion='entropy',       # Critério usado para medir a qualidade de uma divisão
            n_estimators=50,           # Número de árvores na floresta (equivalente ao n_estimators no XGBoost)
            max_depth = 6,                # Profundidade máxima de cada árvore
            class_weight={0:1, 1:class_weight},  # Peso das classes em casos desequilibrados
            bootstrap=True               # Se deve ou não amostrar com substituição ao construir árvores
            )
        ),
        'XGBoost': make_pipeline(
        XGBClassifier(
            random_state=42,            # Semente aleatória para reproducibilidade dos resultados
            tree_method = 'gpu_hist',
            n_estimators=50,           # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
            max_depth = 4,                # Profundidade máxima de cada árvore
            learning_rate = 0.01,         # Taxa de aprendizado - controla a contribuição de cada árvore
            eval_metric='logloss',      # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
            objective='binary:logistic',# Define o objetivo do modelo, 'binary:logistic' para classificação binária
            scale_pos_weight=class_weight,  # Peso das classes positivas em casos desequilibrados
            reg_alpha=1,                # Termo de regularização L1 (penalidade nos pesos)
            reg_lambda=1,               # Termo de regularização L2 (penalidade nos quadrados dos pesos)
            gamma=1,                    # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
            colsample_bytree=0.5,       # Fração de características a serem consideradas ao construir cada árvore --> 0.5 significa que 50% das features (seleção aleatória) será considerada
            subsample=0.5,              # Fração de amostras a serem usadas para treinar cada árvore --> 0.5 significa que 50% da amostra de treino (seleção aleatória) será considerada
            )
        )
    }

    if classificador in models:
        model = models[classificador]
    else:
        print('Utilize Regressão Logística, Random Forest ou XGBoost como opções de Classificadores!')

    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    y_proba_train = model.predict_proba(x_train)
    y_proba_test = model.predict_proba(x_test)

    return model, y_pred_train, y_pred_test, y_proba_train, y_proba_test


def validacao_cruzada_classificacao(classificador, df, target_column, n_splits, class_weight):

    columns_selected = [
       'is_auto_renew', 'payment_method_id', 'months_as_a_registered',
       'num_unq_mov_max_m6', 'num_100_mov_max_m6', 'num_unq_mov_min_m6',
       'num_100_mov_min_m6', '%num_more_than_50_mov_max_m6',
       '%num_more_than_50_mov_avg_m6', '%num_more_than_50_mov_min_m6',
       'num_25_mov_max_m6', 'bd', 'num_50_mov_avg_m6',
       'num_985_mov_avg_m6', 'actual_amount_paid_mov_avg_m6',
       'num_25_mov_min_m6', 'num_75_mov_max_m6', 'num_50_mov_min_m6',
       'num_985_mov_min_m6', 'city', 'num_75_mov_min_m6',
       'registered_via', 'payment_plan_days',
       'churn'
        ]
    

    df_raw = df[columns_selected].copy()

    # Inicializar o KFold para dividir os dados
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Listas para armazenar as métricas para cada fold
    accuracy_scores = [] # Lista para armazenar os valores de ACURÁCIA
    precision_scores = [] # Lista para armazenar os valores de PRECISION
    recall_scores = [] # Lista para armazenar os valores de RECALL
    f1_scores = [] # Lista para armazenar os valores de F1
    auc_scores = []  # Lista para armazenar os valores de AUC
    ks_scores = []   # Lista para armazenar os valores de KS
    logloss_scores = [] # Lista para armazenar os valores de LogLoss
    cv_results = []  # Lista para armazenar os resultados da VALIDAÇÃO CRUZADA

    # Loop pelos folds
    for train_idx, test_idx in kfold.split(df_raw):
        # Criar DataFrames de treino e teste
        df_train = df_raw.iloc[train_idx]
        df_test = df_raw.iloc[test_idx]

        # Filtragem das Features que passaram no Feature Selection
        df_train = df_train[columns_selected]
        df_test = df_test[columns_selected]

        # Separação Feature e Target
        x_train, y_train = separa_feature_target('churn', df_train)
        x_test, y_test = separa_feature_target('churn', df_test)

    # Roda Modelos
        models = {
            'Regressão Logística': make_pipeline(
                LogisticRegression(
                    random_state=42, # Semente aleatória para reproducibilidade dos resultados
                    class_weight={0: 1, 1: class_weight}, # Peso atribuído às classes. Pode ser útil para lidar com conjuntos de dados desbalanceados.
                    C=1, # Parâmetro de regularização inversa. Controla a força da regularização.
                    penalty='l2', # Tipo de regularização. 'l1', 'l2', 'elasticnet', ou 'none'.
                    max_iter=50, # Número máximo de iterações para a convergência do otimizador.
                    solver='liblinear' # Algoritmo de otimização. 'newton-cg', 'lbfgs', 'liblinear' (gradiente descendente), 'sag' (Stochastic gradient descent), 'saga' (Stochastic gradient descent que suporta reg L1).
                    )
            ),
            'Random Forest': make_pipeline(
            RandomForestClassifier(
                random_state=42,            # Semente aleatória para reproducibilidade dos resultados
                criterion='entropy',       # Critério usado para medir a qualidade de uma divisão
                n_estimators=50,           # Número de árvores na floresta (equivalente ao n_estimators no XGBoost)
                max_depth = 4,                # Profundidade máxima de cada árvore
                class_weight={0:1, 1:class_weight},  # Peso das classes em casos desequilibrados
                bootstrap=True               # Se deve ou não amostrar com substituição ao construir árvores
                )
            ),
            'XGBoost': make_pipeline(
            XGBClassifier(
                random_state=42,            # Semente aleatória para reproducibilidade dos resultados
                tree_method = 'gpu_hist',
                n_estimators=50,           # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
                max_depth = 4,                # Profundidade máxima de cada árvore
                learning_rate = 0.005,         # Taxa de aprendizado - controla a contribuição de cada árvore
                eval_metric='logloss',      # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
                objective='binary:logistic',# Define o objetivo do modelo, 'binary:logistic' para classificação binária
                scale_pos_weight=class_weight,  # Peso das classes positivas em casos desequilibrados
                reg_alpha=1,                # Termo de regularização L1 (penalidade nos pesos)
                reg_lambda=1,               # Termo de regularização L2 (penalidade nos quadrados dos pesos)
                gamma=1,                    # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
                colsample_bytree=0.5,       # Fração de características a serem consideradas ao construir cada árvore --> 0.5 significa que 50% das features (seleção aleatória) será considerada
                subsample=0.5,              # Fração de amostras a serem usadas para treinar cada árvore --> 0.5 significa que 50% da amostra de treino (seleção aleatória) será considerada
                )
            )
        }

        if classificador in models:
            model = models[classificador]
        else:
            print('Utilize Regressão Logística, Random Forest ou XGBoost como opções de Classificadores!')

        # Treinar o modelo usando os dados de treinamento
        model.fit(x_train, y_train)

        # Obter as probabilidades previstas para ambas as classes
        y_proba = model.predict_proba(x_test)

        # Fazer as previsões usando o modelo nos dados de teste
        y_pred = model.predict(x_test)

        # Calcular as métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        y_proba = model.predict_proba(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ks = max(tpr - fpr)
        logloss = log_loss(y_test, y_proba[:, 1])

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(roc_auc)
        ks_scores.append(ks)
        logloss_scores.append(logloss)

        # Adicionar resultados de validação cruzada ao DataFrame
        fold_results = pd.DataFrame({
            'churn': y_test['churn'].values,
            'y_predict': y_pred,
            'predict_proba_0': y_proba[:, 0],  # Probabilidade da classe 0
            'predict_proba_1': y_proba[:, 1]  # Probabilidade da classe 1
        })
        cv_results.append(fold_results)


    # Calcular a média das métricas para todos os folds
    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    mean_auc = np.mean(auc_scores),
    mean_ks = np.mean(ks_scores)
    mean_logloss = np.mean(logloss_scores)

    # Criar um DataFrame com as métricas
    metricas_finais = pd.DataFrame({
        'Acuracia': mean_accuracy,
        'Precisao': mean_precision,
        'Recall': mean_recall,
        'F1-Score': mean_f1,
        'AUC':mean_auc,
        'KS': mean_ks,
        'LogLoss': mean_logloss,
        'Etapa': 'validacao_cruzada',
        'Classificador': classificador
    }, index=[1])

    return metricas_finais, cv_results


def metricas_classificacao(classificador, y_train, y_predict_train, y_test, y_predict_test, y_predict_proba_train, y_predict_proba_test, etapa_1, etapa_2):

    predict_proba_train = pd.DataFrame(y_predict_proba_train.tolist(), columns=['predict_proba_0', 'predict_proba_1'])
    predict_proba_test = pd.DataFrame(y_predict_proba_test.tolist(), columns=['predict_proba_0', 'predict_proba_1'])

    # Treino
    accuracy_train = accuracy_score(y_train, y_predict_train)
    precision_train = precision_score(y_train, y_predict_train)
    recall_train = recall_score(y_train, y_predict_train)
    f1_train = f1_score(y_train, y_predict_train)
    roc_auc_train = roc_auc_score(y_train['churn'], predict_proba_train['predict_proba_1'])
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train['churn'], predict_proba_train['predict_proba_1'])
    ks_train = max(tpr_train - fpr_train)
    logloss_train = log_loss(y_train['churn'], predict_proba_train['predict_proba_1'])
    metricas_treino = pd.DataFrame(
        {
            'Acuracia': accuracy_train, 
            'Precisao': precision_train, 
            'Recall': recall_train, 
            'F1-Score': f1_train, 
            'AUC': roc_auc_train, 
            'KS': ks_train, 
            'LogLoss':logloss_train,
            'Etapa': etapa_1, 
            'Classificador': classificador
        }, 
        index=[0]
    )
    
    # Teste
    accuracy_test = accuracy_score(y_test, y_predict_test)
    precision_test = precision_score(y_test, y_predict_test)
    recall_test = recall_score(y_test, y_predict_test)
    f1_test = f1_score(y_test, y_predict_test)
    roc_auc_test = roc_auc_score(y_test['churn'], predict_proba_test['predict_proba_1'])
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test['churn'], predict_proba_test['predict_proba_1'])
    ks_test = max(tpr_test - fpr_test)
    logloss_test = log_loss(y_test['churn'], predict_proba_test['predict_proba_1'])
    metricas_teste = pd.DataFrame(
        {
            'Acuracia': accuracy_test, 
            'Precisao': precision_test, 
            'Recall': recall_test, 
            'F1-Score': f1_test, 
            'AUC': roc_auc_test, 
            'KS': ks_test, 
            'LogLoss':logloss_test,
            'Etapa': etapa_2, 
            'Classificador': classificador
        }, 
        index=[0]
    )
    
    # Consolidando
    metricas_finais = pd.concat([metricas_treino, metricas_teste])

    return metricas_finais


def metricas_classificacao_modelos_juntos(lista_modelos):
    if len(lista_modelos) > 0:
        metricas_modelos = pd.concat(lista_modelos)#.set_index('Classificador')
    else:
        metricas_modelos = lista_modelos[0]
    # Redefina o índice para torná-lo exclusivo
    df = metricas_modelos.reset_index(drop=True)
    df = df.round(2)

    # Função para formatar as células com base na Etapa
    def color_etapa(val):
        color = 'black'
        if val == 'treino':
            color = 'blue'
        elif val == 'teste':
            color = 'red'
        return f'color: {color}; font-weight: bold;'

    # Função para formatar os valores com até duas casas decimais
    def format_values(val):
        if isinstance(val, (int, float)):
            return f'{val:.2f}'
        return val

    # Estilizando o DataFrame
    styled_df = df.style\
        .format(format_values)\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px', subset=pd.IndexSlice[:, :])\
        .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px', subset=pd.IndexSlice[:, 'Acuracia':'F1-Score'])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px', subset=pd.IndexSlice[:, 'Etapa'])\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    # Mostrando o DataFrame estilizado
    styled_df
    return styled_df


def otimizacao(classificador, x_train, y_train, x_test, y_test):
    cols = list(x_train.drop(['msno', 'safra', 'actual_amount_paid'], axis = 1).columns)

    # O FILLNA(0) É UMA CONTINGÊNCIA --> A VARIÁVEL 'REGISTERED_VIA' VEIO COM WOE ZERADO POIS NA AMOSTRA DE TREINO NÃO HAVIAM INADIMPLENTES, 
    # SENDO ASSIM, O REGISTED_VIA = 10 FICOU WOE NULO, MAS FORAM APENAS 2 REGISTROS, ENTÃO SEM PROBLEMAS!!!!!
    # O FILLNA(0) NÃO PREJUDICA MEU TREINAMENTO, POIS UM WOE = 0 SIGNIFICA QUE A VARIÁVEL NÃO TEM NENHUMA ASSOCIAÇÃO ENTRE A CLASSE 0 E 1, OU SEJA, ELA É NEUTRA E NÃO AFETA A DECISÃO DO MODELO
    
    x_train = x_train[cols].copy()
    x_test = x_test[cols].copy()

    # Define o modelo de XGBoost com a otimização de hiperparâmetros via BayesSearch
    model = make_pipeline(
        BayesSearchCV(
            XGBClassifier(random_state=42, tree_method = 'gpu_hist', eval_metric='logloss', objective='binary:logistic'),
            {
                'n_estimators': (50, 75, 100), # Número de Árvores construídas
                'max_depth': (4, 5, 6), # Profundidade Máxima de cada Árvore
                'learning_rate': (0.005, 0.01), # Tamanho do passo utilizado no Método do Gradiente Descendente
                'reg_alpha':(0.5, 1), # Valor do Alpha aplicado durante a Regularização Lasso L1 
                'reg_lambda':(0.5, 1), # Valor do Lambda aplicado durante a Regularização Ridge L2
                'gamma':(0.5, 1), # Valor mínimo permitido para um Nó de Árvore ser aceito. Ajuda a controlar o crescimento das Árvores, evitando divisões insignificantes
                'colsample_bytree':(0.5, 1), # Porcentagem de Colunas utilizada para a amostragem aleatória durante a criação das Árvores
                'subsample':(0.5, 1), # Porcentagem de Linhas utilizada para a amostragem aleatória durante a criação das Árvores
                'scale_pos_weight':(6, 8, 10, 12), # Peso atribuído a classe positiva, aumentando a importância da classe minoritária
            },
            n_iter=10,
            random_state=42,
            n_jobs=-1,
            scoring='roc_auc', #precision, recall, f1, roc_auc, neg_log_loss
            cv=5
        )
    )

    np.int = int # CORREÇÃO POIS O MÉTODO .fit() DA CLASSE SKOPT ESTAVA COM PROBLEMAS DEVIDO A ATUALIZAÇÃO DO NUMPY

    # Treina o modelo
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    y_proba_train = model.predict_proba(x_train)
    y_proba_test = model.predict_proba(x_test)

    melhores_hiperparametros = model.named_steps['bayessearchcv'].best_params_
    hiperparametros = pd.DataFrame([melhores_hiperparametros])

    best_hiperpams = []
    for chave, valor in melhores_hiperparametros.items():
        best_hiperpams.append([chave, valor])

    pivot = pd.DataFrame(best_hiperpams).T
    pivot.columns = pivot.iloc[0]
    pivot = pivot.drop(0)

    # Crie um DataFrame a partir dos hiperparâmetros
    df = hiperparametros.reset_index(drop=True)
    df = df.round(2)

    def color_etapa(val):
        color = 'black'
        if val == 'treino':
            color = 'blue'
        elif val == 'teste':
            color = 'red'
        return f'color: {color}; font-weight: bold;'

    # Função para formatar os valores com até duas casas decimais
    def format_values(val):
        if isinstance(val, (int, float)):
            return f'{val:.2f}'
        return val

    # Estilizando o DataFrame
    styled_df = df.style\
        .format(format_values)\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px')\
        .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    return model, y_pred_train, y_pred_test, y_proba_train, y_proba_test, styled_df, pivot


def validacao_cruzada_classificacao_otimizada(classificador, df, target_column, n_splits, best_hiperpams):

    columns_selected = [
            'is_auto_renew', 'payment_method_id', 'months_as_a_registered',
            'num_unq_mov_max_m6', 'num_100_mov_max_m6', 'num_unq_mov_min_m6',
            'num_100_mov_min_m6', '%num_more_than_50_mov_max_m6',
            '%num_more_than_50_mov_avg_m6', '%num_more_than_50_mov_min_m6',
            'num_25_mov_max_m6', 'bd', 'num_50_mov_avg_m6',
            'num_985_mov_avg_m6', 'actual_amount_paid_mov_avg_m6',
            'num_25_mov_min_m6', 'num_75_mov_max_m6', 'num_50_mov_min_m6',
            'num_985_mov_min_m6', 'city', 'num_75_mov_min_m6',
            'registered_via', 'payment_plan_days',
            'churn'
        ]
    

    df_raw = df[columns_selected].copy()

    # Inicializar o KFold para dividir os dados
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Listas para armazenar as métricas para cada fold
    accuracy_scores = [] # Lista para armazenar os valores de ACURÁCIA
    precision_scores = [] # Lista para armazenar os valores de PRECISION
    recall_scores = [] # Lista para armazenar os valores de RECALL
    f1_scores = [] # Lista para armazenar os valores de F1
    auc_scores = []  # Lista para armazenar os valores de AUC
    ks_scores = []   # Lista para armazenar os valores de KS
    logloss_scores = [] # Lista para armazenar os valores de LogLoss
    cv_results = []  # Lista para armazenar os resultados da VALIDAÇÃO CRUZADA

    # Loop pelos folds
    for train_idx, test_idx in kfold.split(df_raw):
        # Criar DataFrames de treino e teste
        df_train = df_raw.iloc[train_idx]
        df_test = df_raw.iloc[test_idx]

        # Filtragem das Features que passaram no Feature Selection
        df_train = df_train[columns_selected]
        df_test = df_test[columns_selected]

        # Separação Feature e Target
        x_train, y_train = separa_feature_target('churn', df_train)
        x_test, y_test = separa_feature_target('churn', df_test)

        # Melhores Hiperparâmetros
        melhores_hiperparametros = best_hiperpams
        colsample_bytree = round(melhores_hiperparametros['colsample_bytree'][1], 2)
        gamma = round(melhores_hiperparametros['gamma'][1], 2)
        learning_rate = round(melhores_hiperparametros['learning_rate'][1], 2)
        max_depth = int(round(melhores_hiperparametros['max_depth'][1], 2))
        n_estimators = int(round(melhores_hiperparametros['n_estimators'][1], 2))
        reg_alpha = round(melhores_hiperparametros['reg_alpha'][1], 2)
        reg_lambda = round(melhores_hiperparametros['reg_lambda'][1], 2)
        scale_pos_weight = int(round(melhores_hiperparametros['scale_pos_weight'][1], 2))
        subsample = round(melhores_hiperparametros['subsample'][1], 2)

        # Roda Modelo
        model = make_pipeline(
            XGBClassifier(
                random_state=42,            # Semente aleatória para reproducibilidade dos resultados
                tree_method = 'gpu_hist',
                n_estimators=n_estimators,           # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
                max_depth = max_depth,                # Profundidade máxima de cada árvore
                learning_rate = learning_rate,         # Taxa de aprendizado - controla a contribuição de cada árvore
                eval_metric='logloss',      # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
                objective='binary:logistic',# Define o objetivo do modelo, 'binary:logistic' para classificação binária
                scale_pos_weight=scale_pos_weight,  # Peso das classes positivas em casos desequilibrados
                reg_alpha=reg_alpha,                # Termo de regularização L1 (penalidade nos pesos)
                reg_lambda=reg_lambda,               # Termo de regularização L2 (penalidade nos quadrados dos pesos)
                gamma=gamma,                    # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
                colsample_bytree=colsample_bytree,       # Fração de características a serem consideradas ao construir cada árvore
                subsample=subsample,              # Fração de amostras a serem usadas para treinar cada árvore
                )
            )

        # Treinar o modelo usando os dados de treinamento
        model.fit(x_train, y_train)

        # Obter as probabilidades previstas para ambas as classes
        y_proba = model.predict_proba(x_test)

        # Fazer as previsões usando o modelo nos dados de teste
        y_pred = model.predict(x_test)

        # Calcular as métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        y_proba = model.predict_proba(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ks = max(tpr - fpr)
        logloss = log_loss(y_test, y_proba[:, 1])

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(roc_auc)
        ks_scores.append(ks)
        logloss_scores.append(logloss)

        # Adicionar resultados de validação cruzada ao DataFrame
        fold_results = pd.DataFrame({
            'churn': y_test['churn'].values,
            'y_predict': y_pred,
            'predict_proba_0': y_proba[:, 0],  # Probabilidade da classe 0
            'predict_proba_1': y_proba[:, 1]  # Probabilidade da classe 1
        })
        cv_results.append(fold_results)


    # Calcular a média das métricas para todos os folds
    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    mean_auc = np.mean(auc_scores),
    mean_ks = np.mean(ks_scores)
    mean_logloss = np.mean(logloss_scores)

    # Criar um DataFrame com as métricas
    metricas_finais = pd.DataFrame({
        'Acuracia': mean_accuracy,
        'Precisao': mean_precision,
        'Recall': mean_recall,
        'F1-Score': mean_f1,
        'AUC':mean_auc,
        'KS': mean_ks,
        'LogLoss': mean_logloss,
        'Etapa': 'validacao_cruzada',
        'Classificador': classificador
    }, index=[1])

    return metricas_finais, cv_results


def define_ponto_de_corte(x_train, y_train, x_test, y_test, best_hiperpams):

    df_threshold = pd.concat([x_train, y_train], axis=1).copy()
    cols = list(df_threshold.drop(['churn', 'msno', 'safra', 'actual_amount_paid'], axis=1).columns)

    x = df_threshold[cols].copy()
    y = df_threshold['churn'].copy()

    # Melhores Hiperparâmetros
    melhores_hiperparametros = best_hiperpams
    colsample_bytree = round(melhores_hiperparametros['colsample_bytree'][1], 2)
    gamma = round(melhores_hiperparametros['gamma'][1], 2)
    learning_rate = round(melhores_hiperparametros['learning_rate'][1], 2)
    max_depth = int(round(melhores_hiperparametros['max_depth'][1], 2))
    n_estimators = int(round(melhores_hiperparametros['n_estimators'][1], 2))
    reg_alpha = round(melhores_hiperparametros['reg_alpha'][1], 2)
    reg_lambda = round(melhores_hiperparametros['reg_lambda'][1], 2)
    scale_pos_weight = int(round(melhores_hiperparametros['scale_pos_weight'][1], 2))
    subsample = round(melhores_hiperparametros['subsample'][1], 2)


    model = make_pipeline(
        XGBClassifier(
            random_state=42,            # Semente aleatória para reproducibilidade dos resultados
            tree_method = 'gpu_hist',
            n_estimators=n_estimators,           # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
            max_depth = max_depth,                # Profundidade máxima de cada árvore
            learning_rate = learning_rate,         # Taxa de aprendizado - controla a contribuição de cada árvore
            eval_metric='logloss',      # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
            objective='binary:logistic',# Define o objetivo do modelo, 'binary:logistic' para classificação binária
            scale_pos_weight=scale_pos_weight,  # Peso das classes positivas em casos desequilibrados
            reg_alpha=reg_alpha,                # Termo de regularização L1 (penalidade nos pesos)
            reg_lambda=reg_lambda,               # Termo de regularização L2 (penalidade nos quadrados dos pesos)
            gamma=gamma,                    # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
            colsample_bytree=colsample_bytree,       # Fração de características a serem consideradas ao construir cada árvore
            subsample=subsample,              # Fração de amostras a serem usadas para treinar cada árvore
        )
    )

    # Treina o modelo de classificação
    model.fit(x, y)

    def calculate_metrics(x, y, model):

        def retorno_financeiro(df_modelo, y_predict):

            df_aux = df_modelo.loc[df_modelo['safra'].isin(['201603', '201604', '201605', '201606','201607', '201608', '201609'])].copy()
            df_aux['y_predict'] = y_predict

            TN = df_aux.loc[(df_aux['churn'] == 0) & (df_aux['y_predict'] == 0)].shape[0] # O CARA NÃO É CHURN E MEU MODELO FALA QUE ELE NÃO É CHURN
            FN = df_aux.loc[(df_aux['churn'] == 1) & (df_aux['y_predict'] == 0)].shape[0] # O CARA É CHURN E MEU MODELO FALA QUE ELE NÃO É CHURN
            FP = df_aux.loc[(df_aux['churn'] == 0) & (df_aux['y_predict'] == 1)].shape[0] # O CARA NÃO É CHURN E MEU MODELO FALA QUE ELE É CHURN
            TP = df_aux.loc[(df_aux['churn'] == 1) & (df_aux['y_predict'] == 1)].shape[0] # O CARA É CHURN E O MEU MODELO FALA QUE ELE É CHURN
        

            df_aux['retorno_financeiro'] = (
            np.where((df_aux['churn'] == 0) & (df_aux['y_predict'] == 0), 0, # Não sofre nenhuma medida, então não temos retorno nem custo
            np.where((df_aux['churn'] == 1) & (df_aux['y_predict'] == 0), 0, # Embora não tenhamos identificado que era CHURN, não oferecemos nenhum serviço e portanto não houve custo
            np.where((df_aux['churn'] == 0) & (df_aux['y_predict'] == 1), 3*df_aux['actual_amount_paid'], # Implementamos a ação incorretamente, logo, estaremos fornecendo 3 meses de assinatura grátis e tendo custo
            np.where((df_aux['churn'] == 1) & (df_aux['y_predict'] == 1), 9*df_aux['actual_amount_paid'], # Implementamos a ação corretamente, logo, estaremos retendo 50% desses casos e garantindo a assinatura deles por mais 3 meses
            0 # Não ganho nada
            )))))

            quantidade_de_clientes_retidos = 0.5*TP
            taxa_de_clientes_retidos = round((0.5*TP)/(FN+TP)*100, 2) # O RECALL FORNECE A QUANTIDADE DE CLIENTES RETIDOS, MAS PRECISAMOS DIVIDIR O VALOR POR 2 POR CONTA DO ENUNCIADO
            retorno_financeiro_acao_correta = (
                df_aux.loc[
                    (df_aux['churn'] == 1) & (df_aux['y_predict'] == 1)
                ]
                ['retorno_financeiro'].sum()
            )*0.5

            retorno_financeiro_acao_incorreta = (
                df_aux.loc[
                    (df_aux['churn'] == 0) & (df_aux['y_predict'] == 1)
                ]
                ['retorno_financeiro'].sum()
            )

            retorno_financeiro = round(retorno_financeiro_acao_correta - retorno_financeiro_acao_incorreta, 0)


            return quantidade_de_clientes_retidos, taxa_de_clientes_retidos, retorno_financeiro

        df_threshold = pd.concat([x, y], axis=1).copy()
        cols = list(df_threshold.drop(['churn', 'msno', 'safra', 'actual_amount_paid'], axis=1).columns)

        x = df_threshold[cols].copy()
        y = df_threshold['churn'].copy()

        y_pred = model.predict(x)
        y_predict_proba = model.predict_proba(x)[:, 1]

        df_threshold = df_threshold[['churn', 'msno', 'safra', 'actual_amount_paid']]
        df_threshold['Proba Churn'] = y_predict_proba

        list_threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        clientes_retidos_scores = []
        taxa_clientes_retidos_scores = []
        retorno_financeiro_scores = []

        for threshold in list_threshold:
            df_threshold['y_predict_threshold'] = np.where(df_threshold['Proba Churn'] <= threshold, 0, 1)

            quantidade_de_clientes_retidos, taxa_de_clientes_retidos, retorno_financeiro_threshold= retorno_financeiro(df_threshold, df_threshold['y_predict_threshold'])

            clientes_retidos_scores.append(quantidade_de_clientes_retidos)
            taxa_clientes_retidos_scores.append(taxa_de_clientes_retidos)
            retorno_financeiro_scores.append(retorno_financeiro_threshold)

        metrics_df = pd.DataFrame({
            'Threshold': list_threshold,
            'Clientes Retidos': clientes_retidos_scores,
            'Taxa de Clientes Retidos': taxa_clientes_retidos_scores,
            'Retorno Financeiro': retorno_financeiro_scores
        })

        return metrics_df

    metrics_train = calculate_metrics(x_train, y_train, model)
    metrics_test = calculate_metrics(x_test, y_test, model)

    best_threshold_train = metrics_train.loc[metrics_train['Retorno Financeiro'].idxmax(), 'Threshold']
    best_return_train = metrics_train['Retorno Financeiro'].max()
    
    best_threshold_test = metrics_test.loc[metrics_test['Retorno Financeiro'].idxmax(), 'Threshold']
    best_return_test = metrics_test['Retorno Financeiro'].max()

    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(20, 6))

    plt.subplot(1, 2, 1)
    plt.plot(metrics_train['Threshold'], metrics_train['Retorno Financeiro'], marker='o', label='Retorno Financeiro', color='green')
    plt.annotate(f'Melhor Retorno: R${int(best_return_train)}', 
                xy=(best_threshold_train, best_return_train), 
                xytext=(best_threshold_train + 0.1, best_return_train + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12,
                color='black')
    plt.title("Métricas vs Thresholds (Treino)", fontsize=16)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Métricas', fontsize=14)
    plt.xticks(metrics_train['Threshold'], rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics_test['Threshold'], metrics_test['Retorno Financeiro'], marker='o', label='Retorno Financeiro', color='green')
    plt.annotate(f'Melhor Retorno: R${int(best_return_test)}', 
                xy=(best_threshold_test, best_return_test), 
                xytext=(best_threshold_test + 0.1, best_return_test + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12,
                color='black')
    plt.title("Métricas vs Thresholds (Validação)", fontsize=16)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Métricas', fontsize=14)
    plt.xticks(metrics_test['Threshold'], rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()

    plt.tight_layout()
    plt.show()


def modelo_classificador_churn_oficial(df_train, df_test, df_oot, opcao, best_hiperpams):

    base_train = df_train.copy()
    base_test = df_test.copy()
    base_oot = df_oot.copy()

    # Prepara as Amostras
    base_train['tipo_amostra'] = 'train'
    base_test['tipo_amostra'] = 'test'
    base_oot['tipo_amostra'] = 'oot'
    df = pd.concat([base_train, base_test, base_oot])

    # Prepara DataFrame para Treinamento ou Escoragem
    cols = list(df.drop(['churn', 'msno', 'safra', 'actual_amount_paid', 'tipo_amostra'], axis=1).columns)

    # Treina e Salva o Modelo
    if opcao == 'salvar':

        df_model = df.loc[(df['tipo_amostra'] == 'train')].copy()

        x_model = df_model[cols].copy()
        y_model = df_model['churn'].copy()

        # Define o modelo de XGBoost com a otimização de hiperparâmetros via BayesSearch + Calibração de Probabilidade
        # Melhores Hiperparâmetros
        melhores_hiperparametros = best_hiperpams
        colsample_bytree = round(melhores_hiperparametros['colsample_bytree'][1], 2)
        gamma = round(melhores_hiperparametros['gamma'][1], 2)
        learning_rate = round(melhores_hiperparametros['learning_rate'][1], 2)
        max_depth = int(round(melhores_hiperparametros['max_depth'][1], 2))
        n_estimators = int(round(melhores_hiperparametros['n_estimators'][1], 2))
        reg_alpha = round(melhores_hiperparametros['reg_alpha'][1], 2)
        reg_lambda = round(melhores_hiperparametros['reg_lambda'][1], 2)
        scale_pos_weight = int(round(melhores_hiperparametros['scale_pos_weight'][1], 2))
        subsample = round(melhores_hiperparametros['subsample'][1], 2)

        model = make_pipeline(
            XGBClassifier(
                random_state=42,            # Semente aleatória para reproducibilidade dos resultados
                tree_method = 'gpu_hist',
                n_estimators=n_estimators,           # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
                max_depth = max_depth,                # Profundidade máxima de cada árvore
                learning_rate = learning_rate,         # Taxa de aprendizado - controla a contribuição de cada árvore
                eval_metric='logloss',      # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
                objective='binary:logistic',# Define o objetivo do modelo, 'binary:logistic' para classificação binária
                scale_pos_weight=scale_pos_weight,  # Peso das classes positivas em casos desequilibrados
                reg_alpha=reg_alpha,                # Termo de regularização L1 (penalidade nos pesos)
                reg_lambda=reg_lambda,               # Termo de regularização L2 (penalidade nos quadrados dos pesos)
                gamma=gamma,                    # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
                colsample_bytree=colsample_bytree,       # Fração de características a serem consideradas ao construir cada árvore
                subsample=subsample,              # Fração de amostras a serem usadas para treinar cada árvore
            )
        )

        # Treina o modelo de classificação
        model.fit(x_model, y_model)

        joblib.dump(model, "../00_DataMaster/models/classificador_churn.pkl")

        return print('Modelo de Churn Treinado e Salvo com Sucesso!')

    else:
        # Carrega o Classificador e Escora para as bases de Teste e OOT
        classificador_churn = joblib.load("../00_DataMaster/models/classificador_churn.pkl")
        df_scoring = df.loc[df['tipo_amostra'].isin(['test', 'oot'])].copy()
        df_scoring['churn_predict'] = classificador_churn.predict(df_scoring[cols])
        df_scoring['churn_predict_proba_0'] = classificador_churn.predict_proba(df_scoring[cols])[:, 0]
        df_scoring['churn_predict_proba_1'] = classificador_churn.predict_proba(df_scoring[cols])[:, 1]
        df_scoring['churn_predict_calib'] = np.where(df_scoring['churn_predict_proba_1'] <= 0.7, 0, 1)
        df_scoring = df_scoring[['tipo_amostra', 'msno', 'safra', 'actual_amount_paid', 'churn', 'churn_predict', 'churn_predict_calib', 'churn_predict_proba_0', 'churn_predict_proba_1']]

        return df_scoring


def metricas_estabilidade_final(classificador, df):

    def metricas_classificacao_modelos_juntos(lista_modelos):
        if len(lista_modelos) > 0:
            metricas_modelos = pd.concat(lista_modelos)#.set_index('Classificador')
        else:
            metricas_modelos = lista_modelos[0]
        # Redefina o índice para torná-lo exclusivo
        df = metricas_modelos.reset_index(drop=True)
        df = df.round(2)

        # Função para formatar as células com base na Etapa
        def color_etapa(val):
            color = 'black'
            if val == 'treino':
                color = 'blue'
            elif val == 'teste':
                color = 'red'
            return f'color: {color}; font-weight: bold;'

        # Função para formatar os valores com até duas casas decimais
        def format_values(val):
            if isinstance(val, (int, float)):
                return f'{val:.2f}'
            return val

        # Estilizando o DataFrame
        styled_df = df.style\
            .format(format_values)\
            .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px', subset=pd.IndexSlice[:, :])\
            .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
            .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px', subset=pd.IndexSlice[:, 'Acuracia':'F1-Score'])\
            .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px', subset=pd.IndexSlice[:, 'Etapa'])\
            .set_table_styles([
                {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
            ])

        # Mostrando o DataFrame estilizado
        styled_df
        return styled_df

    def retorno_financeiro(df_modelo, y_predict):
        df_aux = df_modelo.loc[df_modelo['safra'].isin(['201603', '201604', '201605', '201606', '201607', '201608', '201609'])].copy()
        df_aux['y_predict'] = y_predict

        TN = df_aux.loc[(df_aux['churn'] == 0) & (df_aux['y_predict'] == 0)].shape[0] # O CARA NÃO É CHURN E MEU MODELO FALA QUE ELE NÃO É CHURN
        FN = df_aux.loc[(df_aux['churn'] == 1) & (df_aux['y_predict'] == 0)].shape[0] # O CARA É CHURN E MEU MODELO FALA QUE ELE NÃO É CHURN
        FP = df_aux.loc[(df_aux['churn'] == 0) & (df_aux['y_predict'] == 1)].shape[0] # O CARA NÃO É CHURN E MEU MODELO FALA QUE ELE É CHURN
        TP = df_aux.loc[(df_aux['churn'] == 1) & (df_aux['y_predict'] == 1)].shape[0] # O CARA É CHURN E O MEU MODELO FALA QUE ELE É CHURN

        df_aux['retorno_financeiro'] = (
            np.where((df_aux['churn'] == 0) & (df_aux['y_predict'] == 0), 0, # Não sofre nenhuma medida, então não temos retorno nem custo
            np.where((df_aux['churn'] == 1) & (df_aux['y_predict'] == 0), 0, # Embora não tenhamos identificado que era CHURN, não oferecemos nenhum serviço e portanto não houve custo
            np.where((df_aux['churn'] == 0) & (df_aux['y_predict'] == 1), 3*df_aux['actual_amount_paid'], # Implementamos a ação incorretamente, logo, estaremos fornecendo 3 meses de assinatura grátis e tendo custo
            np.where((df_aux['churn'] == 1) & (df_aux['y_predict'] == 1), 9*df_aux['actual_amount_paid'], # Implementamos a ação corretamente, logo, estaremos retendo 50% desses casos e garantindo a assinatura deles por mais 3 meses
            0 # Não ganho nada
        )))))

        quantidade_de_clientes_retidos = 0.5*TP
        taxa_de_clientes_retidos = round((0.5*TP)/(FN+TP)*100, 2) # O RECALL FORNECE A QUANTIDADE DE CLIENTES RETIDOS, MAS PRECISAMOS DIVIDIR O VALOR POR 2 POR CONTA DO ENUNCIADO
        retorno_financeiro_acao_correta = (
            df_aux.loc[
                (df_aux['churn'] == 1) & (df_aux['y_predict'] == 1)
            ]
            ['retorno_financeiro'].sum()
        )*0.5

        retorno_financeiro_acao_incorreta = (
            df_aux.loc[
                (df_aux['churn'] == 0) & (df_aux['y_predict'] == 1)
            ]
            ['retorno_financeiro'].sum()
        )

        retorno_financeiro = round(retorno_financeiro_acao_correta - retorno_financeiro_acao_incorreta, 0)

        return quantidade_de_clientes_retidos, taxa_de_clientes_retidos, retorno_financeiro

    df_estabilidade = df.copy()

    # Teste
    df_estabilidade_teste = df_estabilidade.loc[df_estabilidade['safra'] != '201609'].copy()
    accuracy = accuracy_score(df_estabilidade['churn'], df_estabilidade['churn_predict_calib'])
    precision = precision_score(df_estabilidade['churn'], df_estabilidade['churn_predict_calib'])
    recall = recall_score(df_estabilidade['churn'], df_estabilidade['churn_predict_calib'])
    f1 = f1_score(df_estabilidade['churn'], df_estabilidade['churn_predict_calib'])
    roc_auc = roc_auc_score(df_estabilidade['churn'], df_estabilidade['churn_predict_proba_1'])
    fpr, tpr, thresholds = roc_curve(df_estabilidade['churn'], df_estabilidade['churn_predict_proba_1'])
    ks = max(tpr - fpr)
    logloss = log_loss(df_estabilidade['churn'], df_estabilidade['churn_predict_proba_1'])
    metricas_teste = pd.DataFrame(
        {
            'Acuracia': accuracy, 
            'Precisao': precision, 
            'Recall': recall, 
            'F1-Score': f1, 
            'AUC': roc_auc, 
            'KS': ks, 
            'LogLoss':logloss,
            'Etapa': 'Teste', 
            'Classificador': 'Modelo Final'
        }, 
        index=[0]
    )

    # OOT
    df_estabilidade_oot = df_estabilidade.loc[df_estabilidade['safra'] == '201609'].copy()
    accuracy = accuracy_score(df_estabilidade['churn'], df_estabilidade['churn_predict_calib'])
    precision = precision_score(df_estabilidade['churn'], df_estabilidade['churn_predict_calib'])
    recall = recall_score(df_estabilidade['churn'], df_estabilidade['churn_predict_calib'])
    f1 = f1_score(df_estabilidade['churn'], df_estabilidade['churn_predict_calib'])
    roc_auc = roc_auc_score(df_estabilidade['churn'], df_estabilidade['churn_predict_proba_1'])
    fpr, tpr, thresholds = roc_curve(df_estabilidade['churn'], df_estabilidade['churn_predict_proba_1'])
    ks = max(tpr - fpr)
    logloss = log_loss(df_estabilidade['churn'], df_estabilidade['churn_predict_proba_1'])
    metricas_oot = pd.DataFrame(
        {
            'Acuracia': accuracy, 
            'Precisao': precision, 
            'Recall': recall, 
            'F1-Score': f1, 
            'AUC': roc_auc, 
            'KS': ks, 
            'LogLoss':logloss,
            'Etapa': 'OOT', 
            'Classificador': 'Modelo Final'
        }, 
        index=[0]
    )

    metricas_teste_oot = metricas_classificacao_modelos_juntos([metricas_teste, metricas_oot])

    display(metricas_teste_oot)
    
    # Estabilidade
    safras = ['201603', '201604', '201605', '201606', '201607', '201608', '201609']
    metrics = {'safra': [], 'AUC': [], 'Clientes Churn Retidos': []}

    retorno_financeiro_total = 0  # Inicializa o total do retorno financeiro

    for safra in safras:
        df_safras = df.loc[df['safra'] == safra].copy()
        y_true = df_safras['churn']
        y_predict = df_safras['churn_predict_calib']
        y_predict_proba_1 = df_safras['churn_predict_proba_1'].values

        auc = roc_auc_score(y_true, y_predict_proba_1)
        fpr, tpr, _ = roc_curve(y_true, y_predict_proba_1)
        ks = max(tpr - fpr)
        precision = precision_score(y_true, y_predict)

        quantidade_de_clientes_retidos, taxa_clientes_retidos, retorno_financeiro_calculado = retorno_financeiro(df_safras, y_predict)

        metrics['safra'].append(safra)
        metrics['AUC'].append(auc)
        metrics['Clientes Churn Retidos'].append(taxa_clientes_retidos)

        retorno_financeiro_total += retorno_financeiro_calculado

    metrics_df = pd.DataFrame(metrics)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

    # Gráfico das métricas
    ax1.plot(metrics_df['safra'], metrics_df['AUC'] * 100, marker='o', linestyle='-', color='blue', label='AUC')
    ax1.plot(metrics_df['safra'], metrics_df['Clientes Churn Retidos'], marker='o', linestyle='-', color='green', label='Clientes Churn Retidos (%)')

    for i in range(len(metrics_df)):
        ax1.annotate(f'{metrics_df["AUC"].iloc[i] * 100:.2f}', 
                     (metrics_df['safra'].iloc[i], metrics_df['AUC'].iloc[i] * 100), 
                     textcoords="offset points", xytext=(0,5), ha='center', fontsize=9, color='blue')

        ax1.annotate(f'{metrics_df["Clientes Churn Retidos"].iloc[i]:.2f}%', 
                     (metrics_df['safra'].iloc[i], metrics_df['Clientes Churn Retidos'].iloc[i]), 
                     textcoords="offset points", xytext=(0,5), ha='center', fontsize=9, color='green')

    ax1.set_title(f'AUC e Taxa de Clientes Churn Retidos por Safra ({classificador})')
    ax1.set_xlabel('Safra')
    ax1.set_ylabel('Valor')
    ax1.set_ylim(0, 100)
    ax1.legend(loc='lower right', fontsize='small')
    ax1.grid(True)
    ax1.set_xticks(metrics_df['safra'])
    ax1.set_xticklabels(metrics_df['safra'], rotation=45)

    # Gráfico do retorno financeiro total
    retorno_financeiro_total = round(retorno_financeiro_total, 0)
    ax2.text(0.5, 0.5, f'Retorno Financeiro (50% dos VP):\nR${retorno_financeiro_total:.0f}', 
             fontsize=20, ha='center', va='center', color='black', bbox=dict(facecolor='lightgray', alpha=0.5))
    ax2.set_title('Retorno Financeiro Total')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def plot_shap(model, X, titulo):
    # Pega o modelo dentro do pipeline
    model_lgbm = model.named_steps['xgbclassifier']
    
    # Cria o objeto explainer
    explainer = shap.Explainer(model_lgbm, X)
    
    # Calcula os valores SHAP
    shap_values = explainer(X)
    
    # Cria os subplots
    fig, axes = plt.subplots(1, 2, figsize=(16,6))
    
    # Beeswarm
    plt.sca(axes[0])
    shap.plots.beeswarm(shap_values, show=False)
    #axes[0].set_title("SHAP Beeswarm", fontsize=14)
    axes[0].tick_params(axis='y', labelsize=8)
    axes[0].grid(False)
    
    # Bar plot
    plt.sca(axes[1])
    shap.plots.bar(shap_values, show=False)
    #axes[1].set_title("Importância Média Absoluta", fontsize=14)
    axes[1].grid(False)
    axes[1].set_yticklabels([])
    # Título geral
    fig.suptitle(titulo, fontsize=16)
    
    plt.tight_layout()
    plt.show()


def retorno_financeiro(df_modelo, y_predict):
    df_aux = df_modelo.loc[df_modelo['safra'].isin(['201603', '201604', '201605', '201606', '201607', '201608', '201609'])].copy()
    df_aux['y_predict'] = y_predict

    TN = df_aux.loc[(df_aux['churn'] == 0) & (df_aux['y_predict'] == 0)].shape[0] # O CARA NÃO É CHURN E MEU MODELO FALA QUE ELE NÃO É CHURN
    FN = df_aux.loc[(df_aux['churn'] == 1) & (df_aux['y_predict'] == 0)].shape[0] # O CARA É CHURN E MEU MODELO FALA QUE ELE NÃO É CHURN
    FP = df_aux.loc[(df_aux['churn'] == 0) & (df_aux['y_predict'] == 1)].shape[0] # O CARA NÃO É CHURN E MEU MODELO FALA QUE ELE É CHURN
    TP = df_aux.loc[(df_aux['churn'] == 1) & (df_aux['y_predict'] == 1)].shape[0] # O CARA É CHURN E O MEU MODELO FALA QUE ELE É CHURN

    df_aux['retorno_financeiro'] = (
        np.where((df_aux['churn'] == 0) & (df_aux['y_predict'] == 0), 0, # Não sofre nenhuma medida, então não temos retorno nem custo
        np.where((df_aux['churn'] == 1) & (df_aux['y_predict'] == 0), 0, # Embora não tenhamos identificado que era CHURN, não oferecemos nenhum serviço e portanto não houve custo
        np.where((df_aux['churn'] == 0) & (df_aux['y_predict'] == 1), 3*df_aux['actual_amount_paid'], # Implementamos a ação incorretamente, logo, estaremos fornecendo 3 meses de assinatura grátis e tendo custo
        np.where((df_aux['churn'] == 1) & (df_aux['y_predict'] == 1), 12*df_aux['actual_amount_paid'], # Implementamos a ação corretamente, logo, estaremos retendo 50% desses casos e garantindo a assinatura deles por mais 3 meses
        0 # Não ganho nada
    )))))

    quantidade_de_clientes_retidos = 0.5*TP
    taxa_de_clientes_retidos = round((0.5*TP)/(FN+TP)*100, 2) # O RECALL FORNECE A QUANTIDADE DE CLIENTES RETIDOS, MAS PRECISAMOS DIVIDIR O VALOR POR 2 POR CONTA DO ENUNCIADO
    retorno_financeiro_acao_correta = (
        df_aux.loc[
            (df_aux['churn'] == 1) & (df_aux['y_predict'] == 1)
        ]
        ['retorno_financeiro'].sum()
    )*0.5

    retorno_financeiro_acao_incorreta = (
        df_aux.loc[
            (df_aux['churn'] == 0) & (df_aux['y_predict'] == 1)
        ]
        ['retorno_financeiro'].sum()
    )

    retorno_financeiro = round(retorno_financeiro_acao_correta - retorno_financeiro_acao_incorreta, 0)


def escoragem(df):

    # Prepara DataFrame para Treinamento ou Escoragem
    cols = list(df.drop(['churn', 'msno', 'safra', 'actual_amount_paid'], axis=1).columns)

    # Carrega o Classificador e Escora para as bases de Teste e OOT
    classificador_churn = joblib.load("../00_DataMaster/models/classificador_churn.pkl")
    df_scoring = df.copy()
    df_scoring['churn_predict'] = classificador_churn.predict(df_scoring[cols])
    df_scoring['churn_predict_proba_0'] = classificador_churn.predict_proba(df_scoring[cols])[:, 0]
    df_scoring['churn_predict_proba_1'] = classificador_churn.predict_proba(df_scoring[cols])[:, 1]
    df_scoring['churn_predict_calib'] = np.where(df_scoring['churn_predict_proba_1'] <= 0.7, 0, 1)
    df_scoring = df_scoring[['msno', 'safra', 'churn', 'actual_amount_paid', 'churn_predict_calib', 'churn_predict_proba_0', 'churn_predict_proba_1'] + cols]

    return df_scoring


def analise_cluster(df):

    features = [
       'num_unq_mov_max_m6',
       'num_100_mov_max_m6', 'num_unq_mov_min_m6', 'num_100_mov_min_m6',
       '%num_more_than_50_mov_max_m6', '%num_more_than_50_mov_avg_m6',
       '%num_more_than_50_mov_min_m6', 'num_25_mov_max_m6', 'num_50_mov_avg_m6', 'num_985_mov_avg_m6', 'num_25_mov_min_m6',
       'num_75_mov_max_m6', 'num_50_mov_min_m6', 'num_985_mov_min_m6', 'num_75_mov_min_m6', 'months_as_a_registered', 'payment_method_id', 'bd', 'city'
    ]

    # Padronização dos dados com MinMaxScaler
    scaler = MinMaxScaler()
    padronizado = scaler.fit_transform(df[features])
    
    # Aplicando o PCA
    pca = PCA()
    pca.fit(padronizado)
    
    # Variância explicada acumulada
    variancia_explicada_acumulada = np.cumsum(pca.explained_variance_ratio_)
    
    # Aplicando o PCA e selecionando as primeiras componentes principais
    principais_componentes = pca.transform(padronizado)
    
    # Listas para armazenar os scores do Silhouette e WCSS
    silhouette_scores = []
    wcss = []
    
    for n_clusters in np.arange(2, 11):  # Começa em 2 porque não faz sentido calcular para 1 cluster
        kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42, max_iter=100)
        kmeans.fit(principais_componentes)
    
        score = silhouette_score(principais_componentes, kmeans.labels_)
        silhouette_scores.append(score)
        wcss.append(kmeans.inertia_)
    
    # Plotagem dos gráficos
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    # Gráfico de variância acumulada
    ax[0].plot(np.arange(1, len(variancia_explicada_acumulada) + 1), variancia_explicada_acumulada, marker='o', color='orange', label='Variância Explicada Acumulada')
    for i, valor in enumerate(variancia_explicada_acumulada):
        ax[0].text(i + 1, valor - 0.02, f'{valor:.2f}', ha='center', va='bottom', color='orange')
    ax[0].set_title('PCA - Variância Explicada Acumulada')
    ax[0].set_xlabel('Número de Componentes Principais')
    ax[0].set_ylabel('Proporção da Variância Explicada Acumulada')
    ax[0].legend(loc='center right', bbox_to_anchor=(0.9, 0.5), frameon=False)
    ax[0].grid(True)

    # Gráfico do Silhouette Score e WCSS
    ax[1].plot(np.arange(2, 11), silhouette_scores, marker='o', color='blue', label='Silhouette Score')
    ax[1].set_title('Definição do Melhor Número de Clusters')
    ax[1].set_xlabel('Número de Clusters')
    ax[1].set_ylabel('Silhouette Score')
    ax[1].grid(True)

    ax2 = ax[1].twinx()
    ax2.plot(np.arange(2, 11), wcss, marker='x', color='red', linestyle='--', label='WCSS')
    ax2.set_ylabel('WCSS')

    ax2.spines['right'].set_color('none')
    ax2.yaxis.set_ticks([])

    ax[1].legend(loc='center right', bbox_to_anchor=(0.9, 0.8), frameon=False)
    ax2.legend(loc='center right', bbox_to_anchor=(0.9, 0.7), frameon=False)

    plt.tight_layout()
    plt.show()


def train_min_max_scaler_cluster(df, tipo):

    features = [
       'num_unq_mov_max_m6',
       'num_100_mov_max_m6', 'num_unq_mov_min_m6', 'num_100_mov_min_m6',
       '%num_more_than_50_mov_max_m6', '%num_more_than_50_mov_avg_m6',
       '%num_more_than_50_mov_min_m6', 'num_25_mov_max_m6', 'num_50_mov_avg_m6', 'num_985_mov_avg_m6', 'num_25_mov_min_m6',
       'num_75_mov_max_m6', 'num_50_mov_min_m6', 'num_985_mov_min_m6', 'num_75_mov_min_m6', 'months_as_a_registered', 'payment_method_id', 'bd', 'city'
    ]

    df_scaler = df[features].copy()
    scaler = MinMaxScaler()
    scaler.fit(df_scaler)

    joblib.dump(scaler, f"../00_DataMaster/models/scaler_cluster_{tipo}.pkl")

    print('Scaler Treinado e Salvo com sucesso!')


def train_PCA(df, tipo):

    features = [
       'num_unq_mov_max_m6',
       'num_100_mov_max_m6', 'num_unq_mov_min_m6', 'num_100_mov_min_m6',
       '%num_more_than_50_mov_max_m6', '%num_more_than_50_mov_avg_m6',
       '%num_more_than_50_mov_min_m6', 'num_25_mov_max_m6', 'num_50_mov_avg_m6', 'num_985_mov_avg_m6', 'num_25_mov_min_m6',
       'num_75_mov_max_m6', 'num_50_mov_min_m6', 'num_985_mov_min_m6', 'num_75_mov_min_m6', 'months_as_a_registered', 'payment_method_id', 'bd', 'city'
    ]

    # Padronização dos dados
    scaler = joblib.load(f"../00_DataMaster/models/scaler_cluster_{tipo}.pkl")
    padronizado = scaler.transform(df[features])
    
    # Aplicando o PCA
    pca = PCA()
    pca.fit(padronizado)
    
    # Salvando o PCA
    joblib.dump(pca, f"../00_DataMaster/models/pca_cluster_{tipo}.pkl")
    print('PCA Treinado e Salvo com sucesso!')


def Clusterizador(df, tipo):
    scaler = joblib.load(f"../00_DataMaster/models/scaler_cluster_{tipo}.pkl")
    pca = joblib.load(f"../00_DataMaster/models/pca_cluster_{tipo}.pkl")

    features = [
       'num_unq_mov_max_m6',
       'num_100_mov_max_m6', 'num_unq_mov_min_m6', 'num_100_mov_min_m6',
       '%num_more_than_50_mov_max_m6', '%num_more_than_50_mov_avg_m6',
       '%num_more_than_50_mov_min_m6', 'num_25_mov_max_m6', 'num_50_mov_avg_m6', 'num_985_mov_avg_m6', 'num_25_mov_min_m6',
       'num_75_mov_max_m6', 'num_50_mov_min_m6', 'num_985_mov_min_m6', 'num_75_mov_min_m6', 'months_as_a_registered', 'payment_method_id', 'bd', 'city'
    ]

    # Aplicando o Min Max Scaler, PCA e selecionando as primeiras componentes principais
    padronizado = scaler.transform(df[features])
    principais_componentes = pca.transform(padronizado)[:, :8]
    kmeans = KMeans(n_clusters=3, init='random', random_state=42, max_iter=100)
    kmeans.fit(principais_componentes)

    joblib.dump(kmeans, f"../00_DataMaster/models/kmeans_cluster_{tipo}.pkl")

    print('KMeans Treinado e Salvo com sucesso!')


def modelo_clusterizador_churn_oficial(df, tipo):

    scaler = joblib.load(f"../00_DataMaster/models/scaler_cluster_{tipo}.pkl")
    pca = joblib.load(f"../00_DataMaster/models/pca_cluster_{tipo}.pkl")
    kmeans = joblib.load(f"../00_DataMaster/models/kmeans_cluster_{tipo}.pkl")

    features = [
       'num_unq_mov_max_m6',
       'num_100_mov_max_m6', 'num_unq_mov_min_m6', 'num_100_mov_min_m6',
       '%num_more_than_50_mov_max_m6', '%num_more_than_50_mov_avg_m6',
       '%num_more_than_50_mov_min_m6', 'num_25_mov_max_m6', 'num_50_mov_avg_m6', 'num_985_mov_avg_m6', 'num_25_mov_min_m6',
       'num_75_mov_max_m6', 'num_50_mov_min_m6', 'num_985_mov_min_m6', 'num_75_mov_min_m6', 'months_as_a_registered', 'payment_method_id', 'bd', 'city'
    ]

    df_cluster = df[features].copy()
    df_features = scaler.transform(df[features])
    df_features = pca.transform(df_features)[:, :8]
    clusters = kmeans.predict(df_features)

    return clusters

"""Funções reutilizáveis do experimento de forecasting de temperatura."""

from __future__ import annotations

import os
import random
import warnings
import logging
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK", "true")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss


SEED = 42
TARGET = "meantemp"
FEATURES = [
    "temp_lag_1", "temp_lag_2", "temp_lag_3", "temp_lag_7",
    "temp_lag_14", "temp_lag_30", "temp_roll_mean_3",
    "temp_roll_std_3", "temp_roll_mean_7", "temp_roll_std_7",
    "temp_roll_mean_14", "temp_roll_std_14", "temp_roll_mean_30",
    "temp_roll_std_30", "doy_sin", "doy_cos", "trend_days",
]
FOLDS = [
    ("2016-Q1", "2016-01-01", "2016-03-31"),
    ("2016-Q2", "2016-04-01", "2016-06-30"),
    ("2016-Q3", "2016-07-01", "2016-09-30"),
    ("2016-Q4", "2016-10-01", "2016-12-31"),
]


@dataclass
class ExperimentResult:
    data: pd.DataFrame
    cv_detail: pd.DataFrame
    cv_summary: pd.DataFrame
    test_metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    importance: pd.DataFrame
    ablation: pd.DataFrame
    arima_order: tuple[int, int, int]
    sarima_order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_official_splits(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    development = pd.read_csv(data_dir / "DailyDelhiClimateTrain.csv", parse_dates=["date"])
    test = pd.read_csv(data_dir / "DailyDelhiClimateTest.csv", parse_dates=["date"])
    # A data 2017-01-01 aparece com valores conflitantes nos dois arquivos.
    # A versão do teste é preservada; o desenvolvimento termina em 2016-12-31.
    development = development[development["date"] < test["date"].min()].copy()
    development["official_split"] = "development"
    test["official_split"] = "test"
    return development.reset_index(drop=True), test.reset_index(drop=True)


def make_features(development: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    frame = pd.concat([development, test], ignore_index=True).sort_values("date").reset_index(drop=True)
    expected = pd.date_range(frame["date"].min(), frame["date"].max(), freq="D")
    if frame["date"].duplicated().any() or not np.array_equal(frame["date"].to_numpy(), expected.to_numpy()):
        raise ValueError("A série deve ser diária, contínua e sem duplicidades.")
    for lag in (1, 2, 3, 7, 14, 30):
        frame[f"temp_lag_{lag}"] = frame[TARGET].shift(lag)
    past = frame[TARGET].shift(1)
    for window in (3, 7, 14, 30):
        frame[f"temp_roll_mean_{window}"] = past.rolling(window).mean()
        frame[f"temp_roll_std_{window}"] = past.rolling(window).std()
    day = frame["date"].dt.dayofyear
    frame["doy_sin"] = np.sin(2 * np.pi * day / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * day / 365.25)
    frame["trend_days"] = (frame["date"] - frame["date"].min()).dt.days
    return frame.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


def model_catalog() -> dict[str, object]:
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=5,
            max_features=0.8,
            n_jobs=-1,
            random_state=SEED,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            min_child_samples=20,
            reg_alpha=0.5,
            reg_lambda=2.0,
            verbosity=-1,
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def error_metrics(y_true: np.ndarray, pred: np.ndarray, mase_scale: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denominator = (np.abs(y_true) + np.abs(pred)) / 2
    smape = 100 * np.mean(np.divide(np.abs(y_true - pred), denominator, out=np.zeros_like(denominator), where=denominator != 0))
    return {
        "MAE": mean_absolute_error(y_true, pred),
        "RMSE": mean_squared_error(y_true, pred) ** 0.5,
        "MASE": mean_absolute_error(y_true, pred) / mase_scale,
        "sMAPE_%": smape,
    }


def select_statistical_orders(y: pd.Series) -> tuple[tuple, tuple, tuple]:
    """Seleciona ordens por AIC usando somente dados anteriores aos folds."""
    arima_candidates = [(1, 0, 1), (2, 0, 1), (1, 1, 1), (2, 1, 1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scored_arima = []
        for order in arima_candidates:
            try:
                fit = ARIMA(y.to_numpy(), order=order, trend="t" if order[1] == 1 else "ct").fit()
                scored_arima.append((fit.aic, order))
            except Exception:
                continue
        arima_order = min(scored_arima)[1]

        seasonal_candidates = [(1, 0, 0, 7), (0, 0, 1, 7), (1, 0, 1, 7)]
        scored_sarima = []
        for seasonal_order in seasonal_candidates:
            try:
                fit = SARIMAX(
                    y.to_numpy(), order=arima_order, seasonal_order=seasonal_order,
                    trend="t" if arima_order[1] == 1 else "ct",
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False, maxiter=100)
                scored_sarima.append((fit.aic, seasonal_order))
            except Exception:
                continue
        seasonal_order = min(scored_sarima)[1]
    return arima_order, arima_order, seasonal_order


def rolling_statistical_forecast(
    train_y: pd.Series,
    evaluation_y: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Prevê D+1 e atualiza o estado com o realizado, sem reestimar parâmetros."""
    trend = "t" if order[1] == 1 else "ct"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if seasonal_order is None:
            result = ARIMA(train_y.to_numpy(), order=order, trend=trend).fit()
        else:
            result = SARIMAX(
                train_y.to_numpy(), order=order, seasonal_order=seasonal_order,
                trend=trend, enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=150)
        predictions = []
        for actual in evaluation_y.to_numpy():
            predictions.append(float(np.asarray(result.forecast(1))[0]))
            result = result.extend([actual])
    return np.asarray(predictions)


def _build_lstm(lookback: int, seed: int):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    model = Sequential([Input((lookback, 1)), LSTM(32), Dropout(0.20), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    return model


def lstm_forecast(
    history: pd.DataFrame,
    train_mask: pd.Series,
    evaluation_mask: pd.Series,
    lookback: int = 30,
    seed: int = SEED,
) -> tuple[np.ndarray, int]:
    """Early stopping interno e refit no treino externo pelo número de épocas escolhido."""
    from tensorflow.keras.callbacks import EarlyStopping

    values = history[TARGET].to_numpy(dtype=float)
    train_positions = np.flatnonzero(train_mask.to_numpy())
    eval_positions = np.flatnonzero(evaluation_mask.to_numpy())
    train_positions = train_positions[train_positions >= lookback]
    if len(train_positions) < 180:
        raise ValueError("Histórico insuficiente para a LSTM.")

    inner_cut = max(60, int(len(train_positions) * 0.85))
    fit_positions, inner_positions = train_positions[:inner_cut], train_positions[inner_cut:]
    scaler = MinMaxScaler().fit(values[: train_positions[inner_cut - 1] + 1].reshape(-1, 1))
    scaled = scaler.transform(values.reshape(-1, 1)).ravel()

    def sequences(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray([scaled[pos - lookback:pos] for pos in positions])[..., None]
        y = scaled[positions]
        return x, y

    x_fit, y_fit = sequences(fit_positions)
    x_inner, y_inner = sequences(inner_positions)
    probe = _build_lstm(lookback, seed)
    history_fit = probe.fit(
        x_fit, y_fit, validation_data=(x_inner, y_inner), epochs=60,
        batch_size=32, shuffle=False, verbose=0,
        callbacks=[EarlyStopping(patience=6, restore_best_weights=True)],
    )
    best_epoch = int(np.argmin(history_fit.history["val_loss"]) + 1)

    final_scaler = MinMaxScaler().fit(values[train_positions].reshape(-1, 1))
    final_scaled = final_scaler.transform(values.reshape(-1, 1)).ravel()

    def final_sequences(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray([final_scaled[pos - lookback:pos] for pos in positions])[..., None]
        return x, final_scaled[positions]

    x_train, y_train = final_sequences(train_positions)
    x_eval, _ = final_sequences(eval_positions)
    model = _build_lstm(lookback, seed)
    model.fit(x_train, y_train, epochs=best_epoch, batch_size=32, shuffle=False, verbose=0)
    pred_scaled = model.predict(x_eval, verbose=0)
    pred = final_scaler.inverse_transform(pred_scaled).ravel()
    return pred, best_epoch


def _evaluate_period(
    frame: pd.DataFrame,
    train_mask: pd.Series,
    eval_mask: pd.Series,
    arima_order: tuple,
    sarima_order: tuple,
    seasonal_order: tuple,
) -> tuple[list[dict], dict[str, np.ndarray], int]:
    train, evaluation = frame[train_mask], frame[eval_mask]
    scale = train[TARGET].diff().abs().dropna().mean()
    predictions: dict[str, np.ndarray] = {
        "Persistência D-1": evaluation["temp_lag_1"].to_numpy(),
        "Sazonal ingênuo D-7": evaluation["temp_lag_7"].to_numpy(),
        "ARIMA": rolling_statistical_forecast(train[TARGET], evaluation[TARGET], arima_order),
        "SARIMA": rolling_statistical_forecast(train[TARGET], evaluation[TARGET], sarima_order, seasonal_order),
    }
    for name, model in model_catalog().items():
        fitted = clone(model).fit(train[FEATURES], train[TARGET])
        predictions[name] = fitted.predict(evaluation[FEATURES])
    predictions["LSTM"], best_epoch = lstm_forecast(frame, train_mask, eval_mask)
    rows = [{"model": name, **error_metrics(evaluation[TARGET], pred, scale)} for name, pred in predictions.items()]
    return rows, predictions, best_epoch


def interpret_tree_models(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[(frame["official_split"] == "development") & (frame["date"] < "2016-10-01")]
    valid = frame[(frame["date"] >= "2016-10-01") & (frame["date"] <= "2016-12-31")]
    scale = train[TARGET].diff().abs().dropna().mean()
    importance_rows, ablation_rows = [], []
    variants = {
        "todas": FEATURES,
        "sem lag 1": [feature for feature in FEATURES if feature != "temp_lag_1"],
        "somente lag 1": ["temp_lag_1"],
    }
    for name, estimator in model_catalog().items():
        full = clone(estimator).fit(train[FEATURES], train[TARGET])
        permutation = permutation_importance(
            full, valid[FEATURES], valid[TARGET], scoring="neg_mean_absolute_error",
            n_repeats=20, random_state=SEED, n_jobs=-1,
        )
        for feature, mean, std in zip(FEATURES, permutation.importances_mean, permutation.importances_std):
            importance_rows.append({"model": name, "feature": feature, "MAE_increase": mean, "std": std})
        for variant, columns in variants.items():
            fitted = clone(estimator).fit(train[columns], train[TARGET])
            pred = fitted.predict(valid[columns])
            ablation_rows.append({"model": name, "features": variant, **error_metrics(valid[TARGET], pred, scale)})
    importance = pd.DataFrame(importance_rows).sort_values(["model", "MAE_increase"], ascending=[True, False])
    ablation = pd.DataFrame(ablation_rows).sort_values(["model", "MAE"])
    return importance.reset_index(drop=True), ablation.reset_index(drop=True)


def run_experiment(data_dir: str | Path = "data") -> ExperimentResult:
    set_seed()
    development, test = load_official_splits(data_dir)
    frame = make_features(development, test)
    pre_validation = frame[(frame["official_split"] == "development") & (frame["date"] < "2016-01-01")]
    arima_order, sarima_order, seasonal_order = select_statistical_orders(pre_validation[TARGET])

    cv_rows = []
    for fold, start, end in FOLDS:
        train_mask = (frame["official_split"] == "development") & (frame["date"] < start)
        eval_mask = (frame["date"] >= start) & (frame["date"] <= end)
        rows, _, epoch = _evaluate_period(frame, train_mask, eval_mask, arima_order, sarima_order, seasonal_order)
        for row in rows:
            cv_rows.append({"fold": fold, "best_epoch": epoch if row["model"] == "LSTM" else np.nan, **row})
    cv_detail = pd.DataFrame(cv_rows)
    summary = cv_detail.groupby("model")[["MAE", "RMSE", "MASE", "sMAPE_%"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    cv_summary = summary.reset_index().sort_values("MAE_mean").reset_index(drop=True)

    train_mask = frame["official_split"] == "development"
    test_mask = frame["official_split"] == "test"
    test_rows, test_pred, epoch = _evaluate_period(frame, train_mask, test_mask, arima_order, sarima_order, seasonal_order)
    test_metrics = pd.DataFrame(test_rows).sort_values("MAE").reset_index(drop=True)
    baseline_mae = test_metrics.loc[test_metrics["model"] == "Persistência D-1", "MAE"].iloc[0]
    test_metrics["skill_vs_persistence_%"] = 100 * (1 - test_metrics["MAE"] / baseline_mae)
    test_metrics["best_epoch"] = np.where(test_metrics["model"] == "LSTM", epoch, np.nan)
    predictions = frame.loc[test_mask, ["date", TARGET]].copy()
    for name, values in test_pred.items():
        predictions[name] = values

    importance, ablation = interpret_tree_models(frame)
    return ExperimentResult(
        frame, cv_detail, cv_summary, test_metrics, predictions,
        importance, ablation, arima_order, sarima_order, seasonal_order,
    )


def stationarity_row(series, name):
    values = pd.Series(series).dropna()
    adf = adfuller(values, autolag='AIC')
    kpss_result = kpss(values, regression='c', nlags='auto')
    return {
        'série': name,
        'ADF p-valor': adf[1],
        'KPSS p-valor': kpss_result[1],
        'evidência conjunta de estacionaridade': adf[1] < 0.05 and kpss_result[1] > 0.05,
    }


def block_bootstrap_comparison(predictions, baseline='Persistência D-1', block=7, n_boot=5000):
    rng = np.random.default_rng(SEED)
    y = predictions[TARGET].to_numpy()
    base_loss = np.abs(y - predictions[baseline].to_numpy())
    n = len(y)
    rows = []
    for model in [c for c in predictions.columns if c not in {'date', TARGET, baseline}]:
        model_loss = np.abs(y - predictions[model].to_numpy())
        daily_gain = base_loss - model_loss  # positivo = modelo melhora o baseline
        boot = []
        for _ in range(n_boot):
            starts = rng.integers(0, n, size=int(np.ceil(n / block)))
            idx = np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]
            boot.append(daily_gain[idx].mean())
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append({
            'modelo': model,
            'ganho médio de MAE (°C)': daily_gain.mean(),
            'IC 95% inferior': low,
            'IC 95% superior': high,
            'ganho robusto?': low > 0,
        })
    return pd.DataFrame(rows).sort_values('ganho médio de MAE (°C)', ascending=False)


## Bibliotecas Gerais 
import sys
sys.executable
import re

## Bibliotecas de Análise de Dados
import pandas as pd 
import geopandas as gpd
import builtins as builtins
import matplotlib.pyplot as plt
import seaborn as sns 
from IPython.display import display, Image
from tabulate import tabulate
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

# Bibliotecas de Manipulação de Tempo
from datetime import datetime, date, timedelta

## Bibliotecas de Modelagem Matemática e Estatística
import numpy as np
import scipy as sp 
import scipy.stats as stats
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm, normaltest, ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, kruskal, uniform, chi2_contingency
from statsmodels.stats.weightstats import ztest
from numpy import interp
import random


# Bibliotecas de Pré-Processamento e Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import CatBoostEncoder

# Bibliotecas de Modelos de Machine Learning
import joblib
from joblib import Parallel, delayed
import pickle
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping
from sklearn.cluster import KMeans
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import skpro 
import mapie
from skpro.regression.residual import ResidualDouble
from skpro.distributions import Normal, Gamma, LogNormal
from mapie.metrics.regression import regression_coverage_score
from mapie.regression import SplitConformalRegressor
from mapie.utils import train_conformalize_test_split
import networkx as nx
import shap

# Bibliotecas de Métricas de Machine Learning
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error, accuracy_score, roc_auc_score, roc_curve, auc, precision_score, recall_score, precision_recall_curve, average_precision_score, f1_score, log_loss, brier_score_loss, confusion_matrix, cohen_kappa_score, silhouette_score

# Bibliotecas de Spark  

# # Spark Session
# spark = SparkSession.builder.getOrCreate()

def plota_barras(variaveis, df, titulo_base='Distribuição', rotation=0,
                 figsize=(8, 5), top_n=None, limites=None, usar_subplot=False):

    # --- Normalização de entrada ---
    if isinstance(variaveis, str):
        variaveis = [variaveis]
    if limites is None:
        limites = {}

    # ============================================================
    # FUNÇÃO AUXILIAR: ordena apenas se valores forem numéricos
    # ============================================================
    def ordenar_counts_se_numerico(counts):
        try:
            # tenta converter todos os labels para número
            pd.to_numeric(counts.index, errors='raise')
            return counts.sort_index()      # ordena se der certo
        except:
            return counts                   # mantém a ordem original

    # ============================================================
    # CASO 1: Uma variável ou subplot desativado
    # ============================================================
    if len(variaveis) == 1 or not usar_subplot:
        for var in variaveis:

            limite_var = limites.get(var, top_n)
            counts = df[var].value_counts()

            # Ordenação segura
            counts = ordenar_counts_se_numerico(counts)

            if limite_var is not None:
                counts = counts.head(limite_var)

            order = counts.index
            values = counts.values
            total = values.sum()

            plt.figure(figsize=figsize)
            ax = sns.barplot(x=order, y=values, color='#1FB3E5')
            ax.set_title(f'{titulo_base} — {var}', fontsize=14)
            ax.set_xlabel(var, fontsize=12)
            ax.set_ylabel('Quantidade', fontsize=12)

            # Percentuais acima das barras
            for i, v in enumerate(values):
                ax.text(i, v + (max(values) * 0.01), f'{(v/total)*100:.2f}%',
                        ha='center', va='bottom', fontsize=10)

            ax.set_ylim(0, max(values) * 1.15)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right', fontsize=10)
            ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)

            plt.tight_layout()
            plt.show()
        return

    # ============================================================
    # CASO 2: Múltiplas variáveis com subplot
    # ============================================================
    n_vars = len(variaveis)
    n_cols = 2
    n_rows = (n_vars + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize[0]*n_cols, figsize[1]*n_rows))
    axes = axes.flatten()

    for i, var in enumerate(variaveis):

        limite_var = limites.get(var, top_n)
        counts = df[var].value_counts()

        # Ordenação segura
        counts = ordenar_counts_se_numerico(counts)

        if limite_var is not None:
            counts = counts.head(limite_var)

        order = counts.index
        values = counts.values
        total = values.sum()

        ax = axes[i]
        sns.barplot(x=order, y=values, color='#1FB3E5', ax=ax)
        ax.set_title(f'{var}', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('Qtd')

        for j, v in enumerate(values):
            ax.text(j, v + (max(values) * 0.01), f'{(v/total)*100:.1f}%',
                    ha='center', va='bottom', fontsize=9)

        ax.set_ylim(0, max(values) * 1.15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right', fontsize=9)

    # Remove subplots vazios
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(titulo_base, fontsize=16)
    plt.tight_layout()
    plt.show()


def plota_histograma(lista_variaveis, df, linhas, colunas, titulo):
    if (linhas == 1) and (colunas == 1): 
        k = 0
        mediana = df[lista_variaveis[k]].median()
        media = df[lista_variaveis[k]].mean()
        plt.figure(figsize = (14, 5))
        ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', bins = 30)
        ax.set_title(f'{titulo}')
        ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
        ax.set_ylabel(f'Frequência', fontsize = 14)
        ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
        ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
        plt.ticklabel_format(style='plain')
        plt.legend(loc = 'best')
        plt.show()
    elif linhas == 1:
        fig, axis = plt.subplots(linhas, colunas, figsize = (14, 5), sharey = True)
        fig.suptitle(f'{titulo}')
        k = 0
        for i in np.arange(linhas):
            for j in np.arange(colunas):
                mediana = df[lista_variaveis[k]].median()
                media = df[lista_variaveis[k]].mean().round()
                ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[j], bins = 30)
                ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                ax.set_ylabel(f'Frequência', fontsize = 14)
                ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
                ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
                ax.ticklabel_format(style='plain')
                ax.legend(loc = 'best')
                k = k + 1
    elif colunas == 1:
        fig, axis = plt.subplots(linhas, colunas, figsize = (14, 5), sharey = True)
        fig.suptitle(f'{titulo}')
        k = 0
        for i in np.arange(linhas):
            for j in np.arange(colunas):
                mediana = df[lista_variaveis[k]].median()
                media = df[lista_variaveis[k]].mean().round()
                ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[i], bins = 30)
                ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                ax.set_ylabel(f'Frequência', fontsize = 14)
                ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
                ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
                ax.ticklabel_format(style='plain')
                ax.legend(loc = 'best')
                k = k + 1
    else:
        fig, axis = plt.subplots(linhas, colunas, figsize = (14, 5), sharey = True)
        fig.suptitle(f'{titulo}')
        k = 0
        for i in np.arange(linhas):
            for j in np.arange(colunas):
                mediana = df[lista_variaveis[k]].median()
                media = df[lista_variaveis[k]].mean().round()
                ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[i, j], bins = 30)
                ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                ax.set_ylabel(f'Frequência', fontsize = 14)
                ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
                ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
                ax.ticklabel_format(style='plain')
                ax.legend(loc = 'best')
                k = k + 1



def plota_boxplot(df,variaveis,categorias=None,titulo_base='Boxplot',rotation=0,figsize=(8, 5),usar_subplot=False,modo='bivariado'):

    if isinstance(variaveis, str):
        variaveis = [variaveis]

    # --- Caso simples (um gráfico por variável) ---
    if len(variaveis) == 1 or not usar_subplot:
        for var in variaveis:
            plt.figure(figsize=figsize)

            if modo == 'bivariado':
                if categorias is None:
                    raise ValueError("No modo 'bivariado', o parâmetro 'categorias' é obrigatório.")
                sns.boxplot(x=categorias, y=var, data=df, palette=['green', 'yellow', 'red'])
                plt.xlabel(categorias)
            else:  # univariado
                sns.boxplot(y=var, data=df, color='#1FB3E5')

            plt.title(f'{titulo_base} — {var}', fontsize=12)
            plt.ylabel(var)
            plt.xticks(rotation=rotation)
            plt.tight_layout()
            plt.show()
        return

    # --- Caso com subplots (várias variáveis) ---
    n_vars = len(variaveis)
    n_cols = 2
    n_rows = (n_vars + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols, figsize[1]*n_rows))
    axes = axes.flatten()

    for i, var in enumerate(variaveis):
        ax = axes[i]

        if modo == 'bivariado':
            if categorias is None:
                raise ValueError("No modo 'bivariado', o parâmetro 'categorias' é obrigatório.")
            sns.boxplot(x=categorias, y=var, data=df, ax=ax, palette=['green', 'yellow', 'red'])
            ax.set_xlabel(categorias)
        else:
            sns.boxplot(y=var, data=df, ax=ax, color='#1FB3E5')

        ax.set_title(f'{var}', fontsize=11)
        ax.set_ylabel(var)
        ax.tick_params(axis='x', rotation=rotation)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(titulo_base, fontsize=14)
    plt.tight_layout()
    plt.show()



def plota_grafico_linhas(df, x, y, nao_calcula_media, title):

    if nao_calcula_media:
        # Criando o gráfico de linha
        plt.figure(figsize=(20, 8))
        plt.plot(df[x], df[y], marker='o', linestyle='-', color='#1FB3E5')

        # Adicionando títulos e rótulos aos eixos
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)

        for i, txt in enumerate(df[y]):
            plt.annotate(f'{txt:.1f}', (df[x][i], df[y][i]), textcoords="offset points", xytext=(0,1), ha='center')

        # Exibindo o gráfico
        plt.grid(True)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        media = df[y].mean()
        # Criando o gráfico de linha
        plt.figure(figsize=(20, 8))
        plt.plot(df[x], df[y], marker='o', linestyle='-', color='#1FB3E5')

        # Adicionando linha da média
        plt.axhline(y=media, color='r', linestyle='--', linewidth=1, label=f'Média: {media:.2f}')
        plt.legend()

        # Adicionando títulos e rótulos aos eixos
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)

        for i, txt in enumerate(df[y]):
            plt.annotate(f'{txt:.1f}', (df[x][i], df[y][i]), textcoords="offset points", xytext=(0,1), ha='center')

        # Exibindo o gráfico
        plt.grid(True)
        plt.xticks(rotation=90)
        #plt.ylim(0, 50)
        plt.tight_layout()
        plt.show()

def analisa_correlacao(metodo, df):
    plt.figure(figsize=(30, 15))
    mask = np.triu(np.ones_like(df.corr(method=metodo), dtype=bool))
    heatmap = sns.heatmap(df.corr(method=metodo), vmin=-1, vmax=1, cmap='magma', annot=True, fmt='.1f', cbar_kws={"shrink": .8}, mask=mask)
    heatmap.set_title(f"Analisando Correlação de {metodo}")
    plt.grid(False)
    plt.box(False)
    plt.tight_layout()
    plt.grid(False)
    plt.show()

def calcular_psi_temporal(df, coluna_data='data_pedido', coluna_metricas='tempo_entrega', 
                          nome_metrica='Tempo de Entrega', data_base_inicio=None, 
                          data_base_fim=None, data_teste_inicio=None, data_teste_fim=None,
                          tipo_analise='mensal'):
    """
    Função para cálculo e plotagem de PSI temporal com flexibilidade de períodos
    
    Parameters:
    df: DataFrame - DataFrame com os dados
    coluna_data: str - Nome da coluna de data
    coluna_metricas: str - Nome da coluna com a métrica a analisar
    nome_metrica: str - Nome amigável da métrica para exibição
    data_base_inicio: str/datetime - Data de início do período base
    data_base_fim: str/datetime - Data de fim do período base
    data_teste_inicio: str/datetime - Data de início do período de teste
    data_teste_fim: str/datetime - Data de fim do período de teste
    tipo_analise: str - 'mensal' ou 'diária' (para formatação dos labels)
    
    Returns:
    psi_value: float - Valor do PSI calculado
    psi_df: DataFrame - DataFrame com cálculos detalhados
    """
    
    # ============================================================================
    # ETAPA 1: PREPARAÇÃO DOS DADOS TEMPORAIS
    # ============================================================================
    
    # Converter coluna de data para datetime
    df[coluna_data] = pd.to_datetime(df[coluna_data])
    df = df.sort_values(coluna_data)
    
    # Definir datas base e teste
    if data_base_inicio is None:
        data_base_inicio = df[coluna_data].min()
    else:
        data_base_inicio = pd.to_datetime(data_base_inicio)
    
    if data_base_fim is None:
        if tipo_analise == 'mensal':
            data_base_fim = data_base_inicio + pd.DateOffset(months=1)
        else:
            data_base_fim = data_base_inicio + pd.DateOffset(days=1)
    else:
        data_base_fim = pd.to_datetime(data_base_fim)
    
    if data_teste_inicio is None:
        if data_teste_fim is None:
            data_teste_inicio = df[coluna_data].max() - pd.DateOffset(months=1)
        else:
            data_teste_inicio = data_teste_fim - pd.DateOffset(months=1)
    else:
        data_teste_inicio = pd.to_datetime(data_teste_inicio)
    
    if data_teste_fim is None:
        data_teste_fim = df[coluna_data].max()
    else:
        data_teste_fim = pd.to_datetime(data_teste_fim)
    
    # Filtrar dados
    mascara_base = (df[coluna_data] >= data_base_inicio) & (df[coluna_data] < data_base_fim)
    mascara_teste = (df[coluna_data] >= data_teste_inicio) & (df[coluna_data] < data_teste_fim)
    
    dados_base = df[mascara_base].copy()
    dados_teste = df[mascara_teste].copy()
    
    # Verificar se há dados suficientes
    if len(dados_base) == 0:
        raise ValueError(f"Nenhum dado encontrado para o período base: {data_base_inicio.date()} a {data_base_fim.date()}")
    if len(dados_teste) == 0:
        raise ValueError(f"Nenhum dado encontrado para o período de teste: {data_teste_inicio.date()} a {data_teste_fim.date()}")
    
    print("📊 ANÁLISE PSI TEMPORAL")
    print(f"Período base: {data_base_inicio.date()} a {data_base_fim.date()}")
    print(f"Período teste: {data_teste_inicio.date()} a {data_teste_fim.date()}")
    print(f"Registros base: {len(dados_base):,} | Registros teste: {len(dados_teste):,}")
    
    # ============================================================================
    # ETAPA 2: CÁLCULO DO PSI
    # ============================================================================
    
    def calcular_psi(distribuicao_base, distribuicao_teste, num_buckets=10):
        """Calcula o Population Stability Index entre duas distribuições"""
        
        # Remover valores nulos
        base_limpa = np.array(distribuicao_base[~pd.isnull(distribuicao_base)])
        teste_limpa = np.array(distribuicao_teste[~pd.isnull(distribuicao_teste)])
        
        # Definir pontos de corte pelos percentis da distribuição base
        pontos_corte = np.percentile(base_limpa, [i * 100/num_buckets for i in range(num_buckets + 1)])
        pontos_corte = np.unique(pontos_corte)
        
        # Calcular frequências em cada bucket
        freq_base = np.histogram(base_limpa, pontos_corte)[0]
        freq_teste = np.histogram(teste_limpa, pontos_corte)[0]
        
        # Adicionar valor pequeno para evitar divisão por zero
        freq_base = freq_base + 0.0001
        freq_teste = freq_teste + 0.0001
        
        # Calcular proporções
        prop_base = freq_base / len(base_limpa)
        prop_teste = freq_teste / len(teste_limpa)
        
        # Calcular componentes do PSI para cada bucket
        componentes_psi = (prop_teste - prop_base) * np.log(prop_teste / prop_base)
        psi_total = np.sum(componentes_psi)
        
        # Criar DataFrame com resultados detalhados
        psi_detalhado = pd.DataFrame({
            'decil': range(1, len(pontos_corte)),
            'frequencia_base': freq_base,
            'frequencia_teste': freq_teste,
            'proporcao_base': prop_base,
            'proporcao_teste': prop_teste,
            'componente_psi': componentes_psi,
            'limite_inferior': pontos_corte[:-1],
            'limite_superior': pontos_corte[1:]
        })
        
        return psi_total, psi_detalhado
    
    # Calcular PSI
    valor_psi, df_psi = calcular_psi(dados_base[coluna_metricas], dados_teste[coluna_metricas])
    
    # ============================================================================
    # ETAPA 3: PLOTAGEM DOS GRÁFICOS
    # ============================================================================
    
    # Criar figura com dois subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 1, height_ratios=[1, 2])
    
    # --- GRÁFICO SUPERIOR: EVOLUÇÃO DO PSI ---
    ax_superior = plt.subplot(gs[0])
    
    # Plotar linha do PSI acumulado
    ax_superior.plot(df_psi['decil'], df_psi['componente_psi'].cumsum(), 
                     marker='o', linewidth=2, markersize=8, color='blue', 
                     label='PSI Acumulado')
    
    # Linha do valor total do PSI
    ax_superior.axhline(y=valor_psi, color='red', linestyle='--', linewidth=2, 
                        label=f'PSI Total: {valor_psi:.4f}')
    
    # Áreas coloridas para interpretação
    ax_superior.axhspan(0, 0.1, alpha=0.3, color='green', label='PSI ≤ 0.1 (Estável)')
    ax_superior.axhspan(0.1, 0.25, alpha=0.3, color='yellow', label='0.1 < PSI ≤ 0.25 (Atenção)')
    ax_superior.axhspan(0.25, max(valor_psi, 0.5), alpha=0.3, color='red', label='PSI > 0.25 (Instável)')
    
    # Formatar título baseado no tipo de análise
    titulo_temporal = 'Mensal' if tipo_analise == 'mensal' else 'Diária'
    ax_superior.set_title(f'Análise de Estabilidade {titulo_temporal} - {nome_metrica}', 
                          fontsize=14, fontweight='bold', pad=20)
    ax_superior.set_ylabel('Valor do PSI', fontsize=12)
    ax_superior.set_xlabel('Decis', fontsize=12)
    ax_superior.set_xticks(df_psi['decil'])
    ax_superior.set_xticklabels([f'D{i}' for i in df_psi['decil']])
    ax_superior.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_superior.grid(True, alpha=0.3)
    
    # --- GRÁFICO INFERIOR: BARRAS EMPILHADAS POR SAFRA ---
    ax_inferior = plt.subplot(gs[1])
    
    # Preparar dados para o gráfico de barras empilhadas
    num_decis = len(df_psi)
    
    # Criar cores com gradiente de azul (mais escuro para decis maiores)
    cores_decis = plt.cm.Blues(np.linspace(0.3, 0.95, num_decis))
    
    # Formatar nomes das safras
    if tipo_analise == 'diária':
        nome_safra_base = f"Safra Base\n{data_base_inicio.strftime('%d/%m/%Y')}"
        nome_safra_teste = f"Safra Teste\n{data_teste_inicio.strftime('%d/%m/%Y')}"
    else:
        nome_safra_base = f"Safra Base\n{data_base_inicio.strftime('%b/%Y')}"
        nome_safra_teste = f"Safra Teste\n{data_teste_inicio.strftime('%b/%Y')}"
    
    # Posições das barras no eixo X (safras)
    safras = ['Base', 'Teste']
    posicoes_x = np.arange(len(safras))
    largura_barra = 0.6
    
    # Preparar dados empilhados
    # Cada safra tem 10 decis empilhados que somam 100%
    proporcoes_base = df_psi['proporcao_base'].values * 100  # Em porcentagem
    proporcoes_teste = df_psi['proporcao_teste'].values * 100  # Em porcentagem
    
    # Criar barras empilhadas
    acumulado_base = 0
    acumulado_teste = 0
    
    # Plotar cada decil como uma camada empilhada
    for i in range(num_decis-1, -1, -1):  # Do decil 10 ao 1 (para empilhar corretamente)
        decil_num = i + 1
        
        # Barra da safra base para este decil
        altura_base = proporcoes_base[i]
        barra_base = ax_inferior.bar(posicoes_x[0], altura_base, 
                                    width=largura_barra,
                                    bottom=acumulado_base,
                                    color=cores_decis[i],
                                    edgecolor='white',
                                    linewidth=0.5,
                                    alpha=0.9,
                                    label=f'Decil {decil_num}' if i == num_decis-1 else "")
        
        # Barra da safra teste para este decil
        altura_teste = proporcoes_teste[i]
        barra_teste = ax_inferior.bar(posicoes_x[1], altura_teste, 
                                     width=largura_barra,
                                     bottom=acumulado_teste,
                                     color=cores_decis[i],
                                     edgecolor='white',
                                     linewidth=0.5,
                                     alpha=0.9,
                                     hatch='//' if i == num_decis-1 else "//")
        
        # Adicionar texto dentro da barra se espaço suficiente
        if altura_base > 3:
            ax_inferior.text(posicoes_x[0], acumulado_base + altura_base/2, 
                           f'D{decil_num}',
                           ha='center', va='center',
                           fontsize=8, fontweight='bold',
                           color='white')
        
        if altura_teste > 3:
            ax_inferior.text(posicoes_x[1], acumulado_teste + altura_teste/2, 
                           f'D{decil_num}',
                           ha='center', va='center',
                           fontsize=8, fontweight='bold',
                           color='white')
        
        acumulado_base += altura_base
        acumulado_teste += altura_teste
    
    # Configurar eixo X
    ax_inferior.set_xlabel('Safras Analisadas', fontsize=12, fontweight='bold')
    ax_inferior.set_ylabel('Proporção da População (%)', fontsize=12)
    
    # Título do gráfico
    titulo_grafico = f'Distribuição por Decis - Comparação entre Safras'
    ax_inferior.set_title(titulo_grafico, fontsize=13, fontweight='bold', pad=15)
    
    # Configurar ticks do eixo X
    ax_inferior.set_xticks(posicoes_x)
    ax_inferior.set_xticklabels([nome_safra_base, nome_safra_teste], 
                               fontsize=11, fontweight='bold')
    
    # Adicionar linha horizontal em 100% para referência
    ax_inferior.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Adicionar valor total no topo de cada barra
    ax_inferior.text(posicoes_x[0], 102, f'100%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
    
    ax_inferior.text(posicoes_x[1], 102, f'100%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
    
    # Configurar limites do eixo Y
    ax_inferior.set_ylim(0, 110)
    
    # Adicionar grade apenas no eixo Y
    ax_inferior.grid(True, alpha=0.3, axis='y')
    
    # Adicionar legenda dos decis (representativa)
    from matplotlib.patches import Patch
    
    # Criar elementos de legenda para alguns decis representativos
    legend_elements = [
        Patch(facecolor=cores_decis[-1], edgecolor='white', label='Decil 10 (Mais alto)'),
        Patch(facecolor=cores_decis[num_decis//2], edgecolor='white', label=f'Decil {num_decis//2}'),
        Patch(facecolor=cores_decis[0], edgecolor='white', label='Decil 1 (Mais baixo)'),
        Patch(facecolor='white', edgecolor='black', hatch='//', alpha=0.9,
              label='Safra Teste (Hachurado)')
    ]
    
    # ax_inferior.legend(handles=legend_elements, loc='upper left', 
    #                   bbox_to_anchor=(1, 1), title="Legenda dos Decis")
    
    # ============================================================================
    # ETAPA 4: INFORMAÇÕES E INTERPRETAÇÃO
    # ============================================================================
    
    # Determinar interpretação do PSI
    if valor_psi <= 0.1:
        interpretacao = "ESTÁVEL ✅"
        cor_interpretacao = "green"
    elif valor_psi <= 0.25:
        interpretacao = "ATENÇÃO ⚠️"
        cor_interpretacao = "orange"
    else:
        interpretacao = "INSTÁVEL 🚨"
        cor_interpretacao = "red"
    
    # Texto com estatísticas detalhadas
    texto_estatisticas = f'''
    📊 INFORMAÇÕES DAS SAFRAS:
    
    SAFRA BASE:
    • Período: {data_base_inicio.strftime('%d/%m/%Y')} a {data_base_fim.strftime('%d/%m/%Y')}
    • Registros: {len(dados_base):,}
    • Média: {dados_base[coluna_metricas].mean():.2f}
    • Mediana: {dados_base[coluna_metricas].median():.2f}
    
    SAFRA TESTE:
    • Período: {data_teste_inicio.strftime('%d/%m/%Y')} a {data_teste_fim.strftime('%d/%m/%Y')}
    • Registros: {len(dados_teste):,}
    • Média: {dados_teste[coluna_metricas].mean():.2f}
    • Mediana: {dados_teste[coluna_metricas].median():.2f}
    
    📈 RESULTADO DO PSI:
    • Valor PSI: {valor_psi:.4f}
    • Interpretação: {interpretacao}
    • Métrica: {nome_metrica}
    '''
    
    ax_inferior.text(1.02, 0.98, texto_estatisticas, transform=ax_inferior.transAxes, 
                    fontsize=9, verticalalignment='top', color=cor_interpretacao,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print final no console
    print(f"\n🎯 RESULTADO FINAL: PSI = {valor_psi:.4f} - {interpretacao}")
    print("=" * 60)
    
    # Adicionar informação detalhada por decil
    print("\n📋 DETALHAMENTO POR DECIL:")
    print("=" * 60)
    print(f"{'Decil':<6} {'% Base':<10} {'% Teste':<10} {'Diferença':<12} {'PSI Comp.':<10}")
    print("-" * 60)
    
    for _, row in df_psi.iterrows():
        decil = int(row['decil'])
        perc_base = row['proporcao_base'] * 100
        perc_teste = row['proporcao_teste'] * 100
        diferenca = perc_teste - perc_base
        psi_comp = row['componente_psi']
        
        print(f"{f'D{decil}':<6} {perc_base:<10.2f} {perc_teste:<10.2f} {diferenca:<12.2f} {psi_comp:<10.4f}")
    
    return valor_psi, df_psi

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_histograms_comparison(df_train, df_valid, df_test, df_oot, column='tempo_entrega'):
    """
    Plota 4 histogramas lado a lado para comparar as distribuições
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Distribuição de {column} nos Conjuntos de Dados', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    datasets = [
        (df_train[column], 'Treino', '#1f77b4'),
        (df_valid[column], 'Validação', '#2ca02c'),
        (df_test[column], 'Teste', '#ff7f0e'),
        (df_oot[column], 'OOT (Out of Time)', '#d62728')
    ]
    
    for idx, (data, title, color) in enumerate(datasets):
        ax = axes[idx // 2, idx % 2]
        
        # Remover valores NaN
        data_clean = data.dropna()
        
        if len(data_clean) == 0:
            ax.text(0.5, 0.5, 'Sem dados', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{title} (n=0)', fontsize=14, fontweight='bold')
            continue
        
        # Calcular estatísticas
        mean_val = data_clean.mean()
        median_val = data_clean.median()
        std_val = data_clean.std()
        q1 = np.percentile(data_clean, 25)
        q3 = np.percentile(data_clean, 75)
        
        # Determinar número de bins (regra de Sturges)
        n_bins = min(50, int(1 + 3.322 * np.log10(len(data_clean))))
        
        # Plotar histograma
        n, bins, patches = ax.hist(data_clean, bins=n_bins, alpha=0.7, color=color, 
                                   density=True, edgecolor='black', linewidth=0.5)
        
        # Adicionar linhas de média e mediana
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Média: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                  label=f'Mediana: {median_val:.2f}')
        
        # Adicionar KDE (Kernel Density Estimation)
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(data_clean)
            x_range = np.linspace(data_clean.min(), data_clean.max(), 1000)
            ax.plot(x_range, kde(x_range), color='black', linewidth=2, alpha=0.8, label='KDE')
        except:
            pass  # Ignorar se não conseguir calcular KDE
        
        # Configurações do gráfico
        ax.set_title(f'{title} (n={len(data_clean):,})', fontsize=14, fontweight='bold')
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('Densidade', fontsize=12)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Adicionar estatísticas no canto
        stats_text = (f'Média: {mean_val:.2f}\n'
                     f'Mediana: {median_val:.2f}\n'
                     f'Std: {std_val:.2f}\n'
                     f'Q1: {q1:.2f}\n'
                     f'Q3: {q3:.2f}\n'
                     f'IQR: {q3-q1:.2f}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    # Retornar estatísticas para uso posterior
    stats = {}
    for data, title, _ in datasets:
        data_clean = data.dropna()
        if len(data_clean) > 0:
            stats[title] = {
                'n': len(data_clean),
                'mean': data_clean.mean(),
                'median': data_clean.median(),
                'std': data_clean.std(),
                'min': data_clean.min(),
                'max': data_clean.max(),
                'q1': np.percentile(data_clean, 25),
                'q3': np.percentile(data_clean, 75)
            }
    
    return stats

def plot_boxplot_comparison(df_train, df_valid, df_test, df_oot, column='tempo_entrega'):
    """
    Plot boxplot comparativo dos 4 conjuntos
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Preparar dados para boxplot
    train_data = df_train[column].dropna()
    valid_data = df_valid[column].dropna()
    test_data = df_test[column].dropna()
    oot_data = df_oot[column].dropna()
    
    data_to_plot = [train_data, valid_data, test_data, oot_data]
    labels = ['Treino', 'Validação', 'Teste', 'OOT']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    # BOXPLOT PRINCIPAL
    box = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                     showmeans=True, meanline=True, showfliers=False,
                     medianprops=dict(color='yellow', linewidth=2),
                     meanprops=dict(color='red', linewidth=2))
    
    # Colorir as caixas
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Adicionar estatísticas como anotações
    for i, data in enumerate(data_to_plot):
        if len(data) > 0:
            median = np.median(data)
            mean = np.mean(data)
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            
            # Adicionar texto com estatísticas
            ax1.text(i + 1, median, f'Med: {median:.1f}', 
                    ha='center', va='bottom', fontweight='bold', 
                    fontsize=10, color='yellow', 
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
            ax1.text(i + 1, mean, f'Média: {mean:.1f}', 
                    ha='center', va='top', fontsize=9, color='red',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    ax1.set_title(f'Comparação de {column} entre Conjuntos de Dados', 
                 fontsize=16, fontweight='bold')
    ax1.set_ylabel(column, fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # GRÁFICO DE BARRAS COM CONTAGEM E MÉDIA
    x_pos = np.arange(len(labels))
    counts = [len(d) for d in data_to_plot]
    means = [d.mean() if len(d) > 0 else 0 for d in data_to_plot]
    
    # Barras de contagem
    bars1 = ax2.bar(x_pos - 0.2, counts, width=0.4, label='Contagem', 
                   color=colors, alpha=0.7)
    ax2.set_ylabel('Nº Amostras', color='black', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Eixo secundário para médias
    ax2_secondary = ax2.twinx()
    bars2 = ax2_secondary.bar(x_pos + 0.2, means, width=0.4, label='Média', 
                             color=['#8b0000', '#006400', '#8B4513', '#800080'], 
                             alpha=0.7)
    ax2_secondary.set_ylabel('Média', color='black', fontsize=11)
    ax2_secondary.tick_params(axis='y', labelcolor='black')
    
    # Adicionar valores nas barras
    for i, (bar, count) in enumerate(zip(bars1, counts)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for i, (bar, mean_val) in enumerate(zip(bars2, means)):
        ax2_secondary.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.01,
                          f'{mean_val:.1f}', ha='center', va='bottom', 
                          fontsize=9, fontweight='bold', color='darkred')
    
    ax2.set_xlabel('Conjunto de Dados', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_title('Contagem de Amostras e Médias por Conjunto', fontsize=13, fontweight='bold')
    
    # Combinar legendas
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_secondary.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir tabela resumo
    print("\n" + "="*80)
    print("RESUMO ESTATÍSTICO - DISTRIBUIÇÃO DE TEMPO DE ENTREGA")
    print("="*80)
    
    summary_data = []
    for i, (data, label) in enumerate(zip(data_to_plot, labels)):
        if len(data) > 0:
            summary_data.append({
                'Conjunto': label,
                'Amostras': f"{len(data):,}",
                'Média': f"{data.mean():.2f}",
                'Mediana': f"{np.median(data):.2f}",
                'Std': f"{data.std():.2f}",
                'Min': f"{data.min():.2f}",
                'Max': f"{data.max():.2f}",
                'Q1': f"{np.percentile(data, 25):.2f}",
                'Q3': f"{np.percentile(data, 75):.2f}",
                'IQR': f"{np.percentile(data, 75) - np.percentile(data, 25):.2f}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("="*80)

def plot_comparative_density(df_train, df_valid, df_test, df_oot, column='tempo_entrega'):
    """
    Plot de densidade sobreposto para fácil comparação
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Preparar dados
    datasets = [
        (df_train[column].dropna(), 'Treino', '#1f77b4'),
        (df_valid[column].dropna(), 'Validação', '#2ca02c'),
        (df_test[column].dropna(), 'Teste', '#ff7f0e'),
        (df_oot[column].dropna(), 'OOT', '#d62728')
    ]
    
    # Plotar KDE para cada conjunto
    for data, label, color in datasets:
        if len(data) > 0:
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(data)
                x_range = np.linspace(
                    min([d[0].min() for d in datasets if len(d[0]) > 0]),
                    max([d[0].max() for d in datasets if len(d[0]) > 0]),
                    1000
                )
                ax.plot(x_range, kde(x_range), label=label, color=color, linewidth=2.5, alpha=0.8)
                
                # Adicionar linha vertical na média
                mean_val = data.mean()
                ax.axvline(mean_val, color=color, linestyle='--', alpha=0.5, linewidth=1)
                ax.text(mean_val, kde(mean_val)*1.05, f'{mean_val:.1f}', 
                       color=color, fontsize=9, ha='center',
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
            except:
                pass
    
    ax.set_title(f'Comparação de Densidade de {column}', fontsize=16, fontweight='bold')
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Densidade', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Adicionar área sombreada para IQR do Treino (referência)
    if len(datasets[0][0]) > 0:
        train_data = datasets[0][0]
        q1_train = np.percentile(train_data, 25)
        q3_train = np.percentile(train_data, 75)
        ax.axvspan(q1_train, q3_train, alpha=0.1, color='blue', label='IQR Treino')
    
    plt.tight_layout()
    plt.show()

def visualize_all_comparisons(df_train, df_valid, df_test, df_oot, column='tempo_entrega'):
    """
    Função principal que executa todas as visualizações
    """
    print(f"\n📊 ANALISANDO DISTRIBUIÇÃO DE: {column}")
    print("="*60)
    
    # 1. Histogramas individuais
    print("\n1. Gerando histogramas individuais...")
    stats = plot_histograms_comparison(df_train, df_valid, df_test, df_oot, column)
    
    # 2. Boxplot comparativo
    print("\n2. Gerando boxplot comparativo...")
    plot_boxplot_comparison(df_train, df_valid, df_test, df_oot, column)
    
    # 3. Densidade comparativa
    print("\n3. Gerando gráfico de densidade comparativo...")
    plot_comparative_density(df_train, df_valid, df_test, df_oot, column)
    
    return stats

# Versão alternativa para trabalhar com Series diretamente
def visualize_from_series(train_series, valid_series, test_series, oot_series, column_name='tempo_entrega'):
    """
    Versão para trabalhar com Series ao invés de DataFrames completos
    """
    # Converter para DataFrames temporários
    df_train_temp = pd.DataFrame({column_name: train_series})
    df_valid_temp = pd.DataFrame({column_name: valid_series})
    df_test_temp = pd.DataFrame({column_name: test_series})
    df_oot_temp = pd.DataFrame({column_name: oot_series})
    
    return visualize_all_comparisons(df_train_temp, df_valid_temp, df_test_temp, df_oot_temp, column_name)

def plot_kde_comparativo(df_train, df_valid, df_test, df_oot, 
                         figsize=(16, 10), colors=None, alpha=0.7):
    """
    Plota gráficos de KDE comparativos para IC inferior, predição e IC superior
    """
    
    if colors is None:
        colors = ['#1f77b4', '#2ca02c', '#d62728']  # Azul, Verde, Vermelho
    
    # Configurar subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    datasets = [
        ("Treino", df_train),
        ("Validação", df_valid),
        ("Teste", df_test),
        ("Out-of-Time", df_oot)
    ]
    
    for idx, (nome, df) in enumerate(datasets):
        ax = axes[idx]
        
        # Verificar se as colunas existem
        cols_necessarias = ['ic_inferior', 'y_predict', 'ic_superior']
        for col in cols_necessarias:
            if col not in df.columns:
                raise ValueError(f"Coluna '{col}' não encontrada no dataset {nome}")
        
        # Plotar KDEs
        sns.kdeplot(data=df['ic_inferior'], ax=ax, color=colors[0], 
                   label='IC Inferior', linewidth=2, fill=True, alpha=alpha*0.5)
        sns.kdeplot(data=df['y_predict'], ax=ax, color=colors[1], 
                   label='Predição', linewidth=2, fill=True, alpha=alpha*0.5)
        sns.kdeplot(data=df['ic_superior'], ax=ax, color=colors[2], 
                   label='IC Superior', linewidth=2, fill=True, alpha=alpha*0.5)
        
        # Configurações do gráfico
        ax.set_title(f'{nome}\nDistribuição dos Intervalos', fontsize=14, fontweight='bold')
        ax.set_xlabel('Valor', fontsize=11)
        ax.set_ylabel('Densidade', fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Adicionar estatísticas
        mean_inf = df['ic_inferior'].mean()
        mean_pred = df['y_predict'].mean()
        mean_sup = df['ic_superior'].mean()
        
        stats_text = f'Médias:\nInf: {mean_inf:.1f}\nPred: {mean_pred:.1f}\nSup: {mean_sup:.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Comparação de Distribuições: IC Inferior vs Predição vs IC Superior', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def plot_bandas_temporais_unico(df_train, df_valid, df_test, df_oot, variavel_temporal, figsize=(16, 8)):
    """
    Plota um único gráfico com bandas de IC agrupadas pela variável temporal
    """
    
    # Combinar todos os datasets
    df_train['dataset'] = 'Treino'
    df_valid['dataset'] = 'Validação'
    df_test['dataset'] = 'Teste'
    df_oot['dataset'] = 'Out-of-Time'
    
    df_combinado = pd.concat([df_train, df_valid, df_test, df_oot])
    
    # Garantir que a variável temporal existe
    if variavel_temporal not in df_combinado.columns:
        raise ValueError(f"Variável temporal '{variavel_temporal}' não encontrada nos dados")
    
    # Agrupar pela variável temporal
    df_agrupado = df_combinado.groupby(variavel_temporal).agg({
        'ic_inferior': 'mean',
        'y_predict': 'mean',
        'ic_superior': 'mean',
        'tempo_entrega': 'mean' if 'tempo_entrega' in df_combinado.columns else None
    }).reset_index().sort_values(variavel_temporal)
    
    # Criar figura com fundo branco
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    # Dados para plotagem
    x = df_agrupado[variavel_temporal]
    y_inf = df_agrupado['ic_inferior']
    y_pred = df_agrupado['y_predict']
    y_sup = df_agrupado['ic_superior']
    
    # 1. Preencher área entre as margens (cinza claro)
    ax.fill_between(x, y_inf, y_sup, color='lightgray', alpha=0.3, label='Intervalo de Confiança')
    
    # 2. Linha da predição (azul contínua)
    ax.plot(x, y_pred, color='blue', linewidth=2.5, marker='o', 
            markersize=6, label='Predição Média', zorder=5)
    
    # 3. Margens (linhas pontilhadas pretas)
    ax.plot(x, y_inf, color='black', linestyle='--', linewidth=1.5, label='IC Inferior', alpha=0.7)
    ax.plot(x, y_sup, color='black', linestyle='--', linewidth=1.5, label='IC Superior', alpha=0.7)
    
    # Configurações do gráfico
    ax.set_xlabel(variavel_temporal, fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor', fontsize=12, fontweight='bold')
    ax.set_title(f'Bandas de IC do Modelo SKPRO Residual Double ao longo de {variavel_temporal}', fontsize=14, fontweight='bold')
    #ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # REMOVER FUNDO CINZA DO GRID
    ax.set_axisbelow(True)  # Coloca o grid atrás dos dados
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ==============================================
    # AJUSTAR LIMITES DO EIXO Y COM MARGEM DE 10%
    # ==============================================
    
    # Calcular P1 do IC Inferior e P99 do IC Superior
    p1_ic_inf = np.percentile(y_inf, 1)
    p99_ic_sup = np.percentile(y_sup, 99)
    
    # Calcular range atual
    y_range = p99_ic_sup - p1_ic_inf
    
    # Adicionar margem de 10%
    margin = y_range * 0.10
    
    # Definir novos limites
    y_min = p1_ic_inf - margin
    y_max = p99_ic_sup + margin
    
    # Garantir que não temos valores negativos se não for apropriado
    if y_min < 0 and all(y >= 0 for y in [y_inf.min(), y_pred.min(), y_sup.min()]):
        y_min = 0  # Se todos os dados são não-negativos, começar em 0
    
    # Aplicar novos limites
    ax.set_ylim(y_min, y_max)
    
    # Adicionar linhas de referência para os percentis
    ax.axhline(y=p1_ic_inf, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
    ax.axhline(y=p99_ic_sup, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
    
    # Adicionar informação da cobertura se tiver target
    if 'tempo_entrega' in df_combinado.columns:
        cobertura_total = ((df_combinado['tempo_entrega'] >= df_combinado['ic_inferior']) & 
                          (df_combinado['tempo_entrega'] <= df_combinado['ic_superior'])).mean()
        
        # Calcular cobertura por período
        df_combinado['dentro_ic'] = ((df_combinado['tempo_entrega'] >= df_combinado['ic_inferior']) & 
                                     (df_combinado['tempo_entrega'] <= df_combinado['ic_superior'])).astype(int)
        
        cobertura_por_periodo = df_combinado.groupby(variavel_temporal)['dentro_ic'].mean().reset_index()
        
        # # Adicionar texto com cobertura total
        # ax.text(0.98, 0.98, f'Cobertura: {cobertura_total:.1%}', 
        #        transform=ax.transAxes, fontsize=10,
        #        verticalalignment='top', horizontalalignment='right',
        #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # # Adicionar informação do número de amostras
    # contagem_por_periodo = df_combinado.groupby(variavel_temporal).size()
    # ax.text(0.98, 0.90, f'Amostras/período: {contagem_por_periodo.mean():.0f}', 
    #        transform=ax.transAxes, fontsize=9,
    #        verticalalignment='top', horizontalalignment='right',
    #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Rotacionar labels se forem datas ou strings longas
    if (df_agrupado[variavel_temporal].dtype == 'object' or 
        pd.api.types.is_string_dtype(df_agrupado[variavel_temporal]) or
        len(x) > 8):  # Se muitos pontos, rotacionar
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def compute_shap_importance_df(model, X, max_display=30, threshold=0.0):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = (
        pd.DataFrame({
            "feature": X.columns,
            "importance": mean_abs_shap
        })
        .query("importance > @threshold")
        .sort_values("importance", ascending=False)
        .head(max_display)
        .reset_index(drop=True)
    )

    importance_df["importance_pct"] = (importance_df["importance"]/ importance_df["importance"].sum()* 100)

    return importance_df, shap_values

def plot_shap_importance_with_books(
    importance_df,
    books,
    titulo,
    figsize=(16, 8)
):

    # ===============================
    # MAPA FEATURE → BOOK
    # ===============================
    feature_to_book = {
        f: book
        for book, features in books.items()
        for f in features
    }

    importance_df = importance_df.copy()
    importance_df["book"] = (
        importance_df["feature"]
        .map(feature_to_book)
        .fillna("Outros")
    )

    # ===============================
    # AGREGAÇÃO POR BOOK
    # ===============================
    book_importance_df = (
        importance_df
        .groupby("book", as_index=False)["importance_pct"]
        .sum()
        .sort_values("importance_pct", ascending=False)
    )

    # ===============================
    # NORMALIZAÇÃO + COLORMAP SHAP
    # ===============================
    cmap = shap.plots.colors.red_blue
    norm_feat = mcolors.Normalize(
        vmin=importance_df["importance_pct"].min(),
        vmax=importance_df["importance_pct"].max()
    )

    norm_book = mcolors.Normalize(
        vmin=book_importance_df["importance_pct"].min(),
        vmax=book_importance_df["importance_pct"].max()
    )

    colors_feat = cmap(norm_feat(importance_df["importance_pct"].values))
    colors_book = cmap(norm_book(book_importance_df["importance_pct"].values))

    # ===============================
    # PLOT
    # ===============================
    fig, axes = plt.subplots(
        1, 2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.7, 1], "wspace": 0.35}
    )

    # ------------------------------------------------
    # BARRAS POR FEATURE
    # ------------------------------------------------
    axes[0].barh(
        importance_df["feature"][::-1],
        importance_df["importance_pct"][::-1],
        color=colors_feat[::-1]
    )

    for i, v in enumerate(importance_df["importance_pct"][::-1]):
        axes[0].text(
            v + 0.3,
            i,
            f"{v:.1f}%",
            va="center",
            fontsize=9
        )

    axes[0].set_title("Importância SHAP por Feature (%)", fontsize=13)
    axes[0].set_xlabel("Importância (%)")
    axes[0].grid(axis="x", alpha=0.3)

    # ------------------------------------------------
    # BARRAS POR BOOK
    # ------------------------------------------------
    axes[1].barh(
        book_importance_df["book"][::-1],
        book_importance_df["importance_pct"][::-1],
        color=colors_book[::-1]
    )

    for i, v in enumerate(book_importance_df["importance_pct"][::-1]):
        axes[1].text(
            v + 0.3,
            i,
            f"{v:.1f}%",
            va="center",
            fontsize=10
        )

    axes[1].set_title("Importância SHAP por Book (%)", fontsize=13)
    axes[1].set_xlabel("Importância (%)")
    axes[1].grid(axis="x", alpha=0.3)

    # ===============================
    # TÍTULO GERAL
    # ===============================
    fig.suptitle(titulo, fontsize=15)
    plt.subplots_adjust(top=0.9)
    plt.show()

def plot_shap_one_sample(model, X, position_sample, titulo):

    # Pega o modelo dentro do pipeline
    model_lgbm = model
    X_single = X.loc[[position_sample]]
    explainer = shap.Explainer(model_lgbm, X)
    shap_values = explainer(X_single)
    shap.plots.waterfall(shap_values[0])

def plot_shap_one_sample_original_scale(model, X, position_sample, titulo):

    # Seleciona uma observação
    X_single = X.loc[[position_sample]]

    # Explainer
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X_single)

    # Valores no espaço log
    phi = shap_values.values[0]
    base_log = shap_values.base_values[0]

    # Predição final na escala original
    y_pred = np.exp(base_log + phi.sum())

    # Contribuições na escala original (efeito marginal)
    contrib_original = np.exp(base_log + np.cumsum(phi)) - np.exp(
        base_log + np.cumsum(np.r_[0, phi[:-1]])
    )

    # Novo objeto SHAP na escala original
    shap_exp = shap.Explanation(
        values=contrib_original,
        base_values=np.exp(base_log),
        data=X_single.values[0],
        feature_names=X.columns
    )

    shap.plots.waterfall(shap_exp, show=True)


def separa_feature_target(target, dados):
        x = dados.drop(target, axis=1)
        y = dados[target]
        return x, y


def cat_encoder(df = None, categoricas = None, target = None, salvar=False):

    if salvar:
        catboost_encoder = CatBoostEncoder(
            cols=categoricas,
            random_state=42,
            handle_unknown="value",
            handle_missing="value"
        )

        catboost_encoder.fit(df[categoricas],df[target])

        # Salvando o encoder treinado
        joblib.dump(catboost_encoder,"../Modelo_Delivery/models/catboost_encoder.joblib")

    else:
        catboost_encoder = joblib.load("../Modelo_Delivery/models/catboost_encoder.joblib")

        return catboost_encoder
    
def carrega_salva_modelo(opcao, modelo = None):
    # Treina e Salva o Modelo
    if opcao == 'salvar':
        joblib.dump(modelo, "../Modelo_Delivery/models/lgbm_hyperopt.joblib")

    else:
        # Carrega o Classificador e Escora para as bases de Treino, Validação, Teste e OOT
        lgbm_hyperopt = joblib.load("../Modelo_Delivery/models/lgbm_hyperopt.joblib")
        return lgbm_hyperopt
    
def carrega_salva_ic(opcao, modelo = None):
    # Treina e Salva o Modelo
    if opcao == 'salvar':
        joblib.dump(modelo, "../Modelo_Delivery/models/skpro_residual_double.joblib")

    else:
        # Carrega o Classificador e Escora para as bases de Treino, Validação, Teste e OOT
        skpro_residual_double = joblib.load("../Modelo_Delivery/models/skpro_residual_double.joblib")
        return skpro_residual_double
    
def separa_feature_target(target, dados):
    x = dados.drop(target, axis = 1)
    y = dados[[target]]

    return x, y

def aplica_feature_selection_feature_importance(df, target, binarias, categoricas, quantitativas):
    def remove_features_feature_importance(target, df, threshold):
        x, y = separa_feature_target(target, df)
        model = LGBMRegressor(
            random_state=42,
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="regression"
        )
        model.fit(x, y)
        feature_importances = model.feature_importances_
        feature_importance_df = (
            pd.DataFrame({
                "feature": x.columns,
                "importance": feature_importances
            })
            .query("importance > @threshold")
            .sort_values("importance", ascending=False)
        )
        # Normaliza para %
        feature_importance_df["importance"] = (feature_importance_df["importance"]/ feature_importance_df["importance"].sum()* 100)
        return feature_importance_df

    def remove_features_altamente_correlacionadas_quantitativas(df,variaveis_importantes_df,quantitativas,threshold_correlacao=0.9):
        features_quantitativas_importantes = [f for f in variaveis_importantes_df["feature"]if f in quantitativas]

        if len(features_quantitativas_importantes) <= 1:
            print("Nenhuma variável quantitativa removida por correlação.")
            return features_quantitativas_importantes

        df_reduzido = df[features_quantitativas_importantes]
        correlacoes = df_reduzido.corr(method="spearman").abs()

        features_para_remover = set()

        for i in range(len(correlacoes.columns)):
            for j in range(i):
                if correlacoes.iloc[i, j] > threshold_correlacao:
                    col_i = correlacoes.columns[i]
                    col_j = correlacoes.columns[j]

                    imp_i = variaveis_importantes_df.loc[variaveis_importantes_df["feature"] == col_i,"importance"].values[0]
                    imp_j = variaveis_importantes_df.loc[variaveis_importantes_df["feature"] == col_j,"importance"].values[0]

                    # Remove a de menor importância
                    features_para_remover.add(col_i if imp_i < imp_j else col_j)

        if features_para_remover:
            print(f"Variáveis removidas por alta correlação (Spearman > {threshold_correlacao}):")
            for f in sorted(features_para_remover):
                print(f" - {f}")
        else:
            print("Nenhuma variável quantitativa removida por correlação.")

        return [f for f in features_quantitativas_importantes if f not in features_para_remover]

    # 1. Feature importance global (LGBM)
    feature_importances = remove_features_feature_importance(target, df, threshold=0)
    # 2. Correlação apenas nas quantitativas
    quantitativas_filtradas = remove_features_altamente_correlacionadas_quantitativas(df,feature_importances,quantitativas)
    # 3. Binárias e categóricas passam direto
    outras_features = [f for f in feature_importances["feature"]if f not in quantitativas]
    features_finais = set(quantitativas_filtradas + outras_features)
    feature_importances_final = feature_importances[feature_importances["feature"].isin(features_finais)]

    return feature_importances_final

def aplica_feature_selection_shap(df, target, binarias, categoricas, quantitativas):

    def remove_features_shap(target, df, threshold):
        x, y = separa_feature_target(target, df)

        model = LGBMRegressor(
            device='gpu',                         # Usa GPU (se disponível) - substitui tree_method='gpu_hist'
            verbosity = -1,                       # Nível de verbosidade (-1: silencioso, 0: erros, 1: avisos, 2: informações)                
            random_state=42,                      # Semente aleatória para reproducibilidade dos resultados
            boosting_type='gbdt',                 # Tipo de boosting 'gbdt' (Gradient Boosting Decision Tree), 'dart' (Dropouts meet Multiple Additive Regression Trees) ou 'goss' (Gradient-based One-Side Sampling)
            objective='regression',               # Objetivo 'binary' (Classificação Binária) 'regression' (Regressão)
            metric='rmse',                        # Métrica de avaliação 'binary_logloss' (Classificação Binária) 'rmse' (Regressão)
            importance_type='gain',               # Método escolhido para calcular o Feature Importance, podendo ser Gain (ganho médio de informação ao utilizar a Feature), Weight (número de vezes que a Feature foi utilizada) ou Cover (número de amostras impactadas pela Feature)
            #class_weight={0:1, 1:class_weight},  # Pesos para classes (ou 'balanced')
            n_estimators=300,                     # Número de árvores no modelo
            max_depth=10,                         # Profundidade máxima
            learning_rate=0.05,                   # Taxa de aprendizado
            max_bin=255,                          # quantidade de bins que as variáveis numéricas serão divididas
            colsample_bytree=0.5,                 # Fração de features por árvore
            subsample=0.5,                        # Fração de amostras por árvore
            reg_alpha=5,                          # Regularização L1
            reg_lambda=5,                         # Regularização L2
            min_split_gain=5,                     # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
            num_leaves=30,                        # número máximo de folhas por árvore (controle essencial para evitar overfitting no crescimento leaf-wise)
            min_data_in_leaf=300,                 # quantidade de amostras necessárias para que uma Folha seja válida
            min_sum_hessian_in_leaf=0.001,        # A soma das Hessianas em uma folha mede o “peso estatístico” daquela folha, portanto, representa o mínimo na soma das Hessianas em uma folha
            min_child_weight = 0.001,             # A soma das Hessianas em uma folha mede o “peso estatístico” daquela folha, portanto, representa o mínimo na soma das Hessianas em uma folha
            path_smooth = 10                      # Parâmetro de suavização para evitar grandes variações na predição entre nós pai e filho
        )

        model.fit(x, y)

        # SHAP (tree-based)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x)

        # Mean Absolute SHAP
        shap_importance = np.abs(shap_values).mean(axis=0)

        shap_importance_df = (
            pd.DataFrame({
                "feature": x.columns,
                "importance": shap_importance
            })
            .query("importance > @threshold")
            .sort_values("importance", ascending=False)
        )

        shap_importance_df["importance"] = (shap_importance_df["importance"]/ shap_importance_df["importance"].sum()* 100)

        return {
            "importance_df": shap_importance_df,
            "model": model,
            "X": x,
            "shap_values": shap_values
        }

    def remove_features_altamente_correlacionadas_quantitativas(df,variaveis_importantes_df,quantitativas,threshold_correlacao=0.9):
        features_quantitativas_importantes = [f for f in variaveis_importantes_df["feature"]if f in quantitativas]

        if len(features_quantitativas_importantes) <= 1:
            print("Nenhuma variável quantitativa removida por correlação.")
            return features_quantitativas_importantes

        df_reduzido = df[features_quantitativas_importantes]
        correlacoes = df_reduzido.corr(method="spearman").abs()

        features_para_remover = set()

        for i in range(len(correlacoes.columns)):
            for j in range(i):
                if correlacoes.iloc[i, j] > threshold_correlacao:
                    col_i = correlacoes.columns[i]
                    col_j = correlacoes.columns[j]

                    imp_i = variaveis_importantes_df.loc[variaveis_importantes_df["feature"] == col_i,"importance"].values[0]
                    imp_j = variaveis_importantes_df.loc[variaveis_importantes_df["feature"] == col_j,"importance"].values[0]

                    features_para_remover.add(col_i if imp_i < imp_j else col_j)

        return [f for f in features_quantitativas_importantes if f not in features_para_remover]

    # ======================================================
    # 1. SHAP global
    # ======================================================
    shap_output = remove_features_shap(target, df, threshold=0)

    shap_importances = shap_output["importance_df"]

    # ======================================================
    # 2. Correlação apenas nas quantitativas
    # ======================================================
    quantitativas_filtradas = remove_features_altamente_correlacionadas_quantitativas(df,shap_importances,quantitativas)

    # ======================================================
    # 3. Binárias e categóricas passam direto
    # ======================================================
    outras_features = [f for f in shap_importances["feature"]if f not in quantitativas]
    features_finais = set(quantitativas_filtradas + outras_features)
    shap_importances_final = shap_importances[shap_importances["feature"].isin(features_finais)]

    return {
        "feature_importance": shap_importances_final,
        "model": shap_output["model"],
        "X": shap_output["X"],
        "shap_values": shap_output["shap_values"]
    }


def metricas_regressao(model_name, y_train, y_pred_train, y_test, y_pred_test, etapa_1='treino', etapa_2='teste', por_faixa=False):

    # --------------------------
    # Funções auxiliares
    # --------------------------
    def var20(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64).flatten()
        y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
        y_true_safe = np.where(y_true == 0, 1e-10, y_true)
        relative_error = np.abs(y_pred - y_true) / y_true_safe
        within_20_percent = relative_error <= 0.20
        return np.mean(within_20_percent)

    def under20(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64).flatten()
        y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
        y_true_safe = np.where(y_true == 0, 1e-10, y_true)
        relative_error = (y_true - y_pred) / y_true_safe
        return np.mean(relative_error > 0.20)

    def over20(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64).flatten()
        y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
        y_true_safe = np.where(y_true == 0, 1e-10, y_true)
        relative_error = (y_pred - y_true) / y_true_safe
        return np.mean(relative_error > 0.20)

    def rmsle(y_true, y_pred):
        y_true = np.maximum(np.asarray(y_true, dtype=np.float64).flatten(), 0)
        y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64).flatten(), 0)
        return np.sqrt(mean_squared_log_error(y_true, y_pred))

    def cohen_kappa_deciles(y_true, y_pred, n_deciles=10):
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        if len(np.unique(y_true)) < n_deciles:
            n_deciles = len(np.unique(y_true))
        if n_deciles < 2:
            return 0.0
        bins = np.percentile(y_true, np.linspace(0, 100, n_deciles+1))
        y_true_cat = np.digitize(y_true, bins, right=True) - 1
        y_pred_cat = np.digitize(y_pred, bins, right=True) - 1
        return cohen_kappa_score(y_true_cat, y_pred_cat)

    def calcular_mape(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64).flatten()
        y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
        y_true_safe = np.where(y_true == 0, 1e-10, y_true)
        return np.mean(np.abs(y_pred - y_true) / y_true_safe) * 100

    # --------------------------
    # Função para calcular métricas
    # --------------------------
    def calcular_metricas(y_true, y_pred, etapa, pct_amostras=100):
        data = {
            'MAE': [mean_absolute_error(y_true, y_pred)],
            'RMSE': [np.sqrt(mean_squared_error(y_true, y_pred))],
            'RMSLE': [rmsle(y_true, y_pred)],
            'MAPE (%)': [calcular_mape(y_true, y_pred)],
            'Var20 (%)': [var20(y_true, y_pred) * 100],
            'Subestimação (%)': [under20(y_true, y_pred) * 100],
            'Superestimação (%)': [over20(y_true, y_pred) * 100],
            "CohenKappa": [cohen_kappa_deciles(y_true, y_pred)],
            'Etapa': [etapa],
            'Modelo': [model_name]
        }
        if pct_amostras is not None:
            data['Pct_amostras (%)'] = [pct_amostras]
        return pd.DataFrame(data)

    # Métricas globais
    metricas_treino = calcular_metricas(y_train, y_pred_train, etapa_1)
    metricas_teste = calcular_metricas(y_test, y_pred_test, etapa_2)

    # --------------------------
    # Métricas por faixa de tempo (opcional)
    # --------------------------
    if por_faixa:
        bins = [0, 30, 45, 60, 75, 90, 105, np.inf]
        labels = ['Até 30min','Até 45min','Até 60min','Até 75min','Até 90min','Até 105min','Mais que 105min']

        metricas_treino_faixa = []
        metricas_teste_faixa = []

        y_train_arr = np.asarray(y_train).flatten()
        y_pred_train_arr = np.asarray(y_pred_train).flatten()
        y_test_arr = np.asarray(y_test).flatten()
        y_pred_test_arr = np.asarray(y_pred_test).flatten()

        total_train = len(y_train_arr)
        total_test = len(y_test_arr)

        # Treino por faixa
        for i in range(len(bins)-1):
            mask = (y_train_arr > bins[i]) & (y_train_arr <= bins[i+1])
            if np.any(mask):
                pct = np.sum(mask) / total_train * 100
                df_faixa = calcular_metricas(
                    y_train_arr[mask], 
                    y_pred_train_arr[mask], 
                    f"{etapa_1} ({labels[i]})",
                    pct_amostras=pct
                )
                metricas_treino_faixa.append(df_faixa)

        # Teste por faixa
        for i in range(len(bins)-1):
            mask = (y_test_arr > bins[i]) & (y_test_arr <= bins[i+1])
            if np.any(mask):
                pct = np.sum(mask) / total_test * 100
                df_faixa = calcular_metricas(
                    y_test_arr[mask], 
                    y_pred_test_arr[mask], 
                    f"{etapa_2} ({labels[i]})",
                    pct_amostras=pct
                )
                metricas_teste_faixa.append(df_faixa)

        metricas_treino = pd.concat([metricas_treino] + metricas_treino_faixa).reset_index(drop=True)
        metricas_teste = pd.concat([metricas_teste] + metricas_teste_faixa).reset_index(drop=True)

    return pd.concat([metricas_treino, metricas_teste]).reset_index(drop=True)


def metricas_modelos_juntos_regressao(lista_modelos):
    if len(lista_modelos) > 0:
        metricas_modelos = pd.concat(lista_modelos)
    else:
        return pd.DataFrame()

    # Redefinir índice e arredondar
    df = metricas_modelos.reset_index(drop=True)
    df = df.round(2)

    metricas_cols = ['MAE', 'RMSE', 'RMSLE', 'MAPE (%)', 
                     'Var20 (%)', 'Subestimação (%)', 'Superestimação (%)', 
                     'CohenKappa', 'Pct_amostras (%)']

    # Função para colorir por etapa
    def color_etapa(val):
        val = str(val).lower()
        color = 'black'
        if 'treino' in val:
            color = 'blue'
        elif 'teste' in val or 'validacao' in val:
            color = 'red'
        return f'color: {color}; font-weight: bold;'

    # Função para criar borda inferior quando o modelo muda
    def separador_modelos(df):
        estilos = pd.DataFrame('', index=df.index, columns=df.columns)
        modelos = df['Modelo']
        for i in range(len(modelos)-1):
            if modelos[i] != modelos[i+1]:
                estilos.loc[i, :] = 'border-bottom: 3px solid black;'
        return estilos

    # Estilizando o DataFrame
    styled_df = df.style \
        .format({col: "{:.2f}" for col in metricas_cols}) \
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px', subset=pd.IndexSlice[:, :]) \
        .applymap(color_etapa, subset=pd.IndexSlice[:, ['Etapa']]) \
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px', subset=pd.IndexSlice[:, metricas_cols]) \
        .apply(separador_modelos, axis=None) \
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])
    
    return styled_df

def metricas_regressao_diarias(df,coluna_data,y_true_col,y_pred_col):
    # --------------------------
    # Funções auxiliares
    # --------------------------
    def var20(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true_safe = np.where(y_true == 0, 1e-10, y_true)
        return np.mean(np.abs(y_pred - y_true) / y_true_safe <= 0.20) * 100

    def rmsle(y_true, y_pred):
        y_true = np.maximum(np.asarray(y_true), 0)
        y_pred = np.maximum(np.asarray(y_pred), 0)
        return np.sqrt(mean_squared_log_error(y_true, y_pred)) * 100

    # --------------------------
    # Preparação
    # --------------------------
    df = df.copy()
    df[coluna_data] = pd.to_datetime(df[coluna_data])
    df = df.sort_values(coluna_data)

    resultados = []

    # --------------------------
    # Loop diário
    # --------------------------
    for data, grupo in df.groupby(df[coluna_data].dt.date):

        y_true = grupo[y_true_col].values
        y_pred = grupo[y_pred_col].values

        resultados.append({
            "data_pedido": pd.to_datetime(data),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "RMSLE": rmsle(y_true, y_pred),
            "Var20 (%)": var20(y_true, y_pred),
            "Qtd_registros": len(grupo)
        })

    return pd.DataFrame(resultados)


def Regressor(loss_function, x_train, y_train, x_test, y_test):
    
    cols = list(x_train.columns)
    x_train = x_train[cols]
    x_test = x_test[cols]

    base_params = dict(
        device='gpu',                         # Usa GPU (se disponível) - substitui tree_method='gpu_hist'
        verbosity = -1,                       # Nível de verbosidade (-1: silencioso, 0: erros, 1: avisos, 2: informações)                
        random_state=42,                      # Semente aleatória para reproducibilidade dos resultados
        boosting_type='gbdt',                 # Tipo de boosting 'gbdt' (Gradient Boosting Decision Tree), 'dart' (Dropouts meet Multiple Additive Regression Trees) ou 'goss' (Gradient-based One-Side Sampling)
        #objective='regression',               # Função de custo 'binary' (Classificação Binária) 'regression' (Regressão)
        #metric='mae',                         # Métrica de avaliação durante as Logs de Treinamento
        importance_type='gain',               # Método escolhido para calcular o Feature Importance, podendo ser Gain (ganho médio de informação ao utilizar a Feature), Weight (número de vezes que a Feature foi utilizada) ou Cover (número de amostras impactadas pela Feature)
        n_estimators=300,                     # Número de árvores no modelo
        max_depth=7,                         # Profundidade máxima
        learning_rate=0.05,                   # Taxa de aprendizado
        max_bin=255,                          # quantidade de bins que as variáveis numéricas serão divididas
        # colsample_bytree=0.5,                 # Fração de features por árvore
        # subsample=0.5,                        # Fração de amostras por árvore
        # reg_alpha=5,                          # Regularização L1
        # reg_lambda=5,                         # Regularização L2
        # min_split_gain=5,                     # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
        # num_leaves=30,                        # número máximo de folhas por árvore (controle essencial para evitar overfitting no crescimento leaf-wise)
        # min_data_in_leaf=300,                 # quantidade de amostras necessárias para que uma Folha seja válida
        # min_sum_hessian_in_leaf=0.001,        # A soma das Hessianas em uma folha mede o “peso estatístico” daquela folha, portanto, representa o mínimo na soma das Hessianas em uma folha
        # min_child_weight = 0.001,             # A soma das Hessianas em uma folha mede o “peso estatístico” daquela folha, portanto, representa o mínimo na soma das Hessianas em uma folha
        # path_smooth = 10                      # Parâmetro de suavização para evitar grandes variações na predição entre nós pai e filho
        #class_weight={0:1, 1:class_weight},   # Pesos para classes (ou 'balanced')
    )

    models = {
        "MAE": LGBMRegressor(
            objective="regression_l1",
            metric="l1",
            **base_params
        ),

        "RMSE": LGBMRegressor(
            objective="regression",
            metric="l2",
            **base_params
        ),

        "Huber": LGBMRegressor(
            objective="huber",
            metric="l1",
            huber_delta=1.0,
            **base_params
        ),

        "RMSLE": LGBMRegressor(
            objective="regression",
            metric="l2",
            **base_params
        ),

        "Gamma": LGBMRegressor(
            objective="gamma",
            metric="gamma",
            **base_params
        )
    }

    if loss_function not in models:
        raise ValueError(f"Loss function '{loss_function}' não suportada.")

    model = models[loss_function]
    
    # Tratamento especial para RMSLE: aplicar log1p no target e depois expm1 nas predições
    if loss_function == "RMSLE":
        # Aplicar log1p no target (ln(1+y))
        y_train_transformed = np.log1p(y_train)
        
        # Treinar o modelo no target transformado
        model.fit(x_train, y_train_transformed)
        
        # Fazer predições no espaço transformado
        y_pred_train_transformed = model.predict(x_train)
        y_pred_test_transformed = model.predict(x_test)
        
        # Aplicar expm1 para voltar ao espaço original (e^pred - 1)
        y_pred_train = np.expm1(y_pred_train_transformed)
        y_pred_test = np.expm1(y_pred_test_transformed)
        
        # IMPORTANTE: ajustar para garantir valores não-negativos
        y_pred_train = np.maximum(y_pred_train, 0)
        y_pred_test = np.maximum(y_pred_test, 0)
        
    else:
        # Para outras funções de perda, treinar normalmente
        model.fit(x_train, y_train)
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

    return model, y_pred_train, y_pred_test

    
def otimizacao_hyperopt_regression(x_train, y_train, x_test, y_test, max_evals):

    # Espaço de busca dos hiperparâmetros
    search_space = {
        'n_estimators': hp.choice('n_estimators', [700, 800, 900, 1000]),
        'max_depth': hp.choice('max_depth', [10, 11, 12]),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.05),
        'max_bin': hp.choice('max_bin', [64, 128, 255]),
        'reg_alpha': hp.uniform('reg_alpha', 0, 1),
        'reg_lambda': hp.uniform('reg_lambda', 0, 1),
        'min_split_gain': hp.uniform('min_split_gain', 0, 10),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'num_leaves': hp.choice('num_leaves', [30, 35, 40, 45, 50]),
        'min_data_in_leaf': hp.choice('min_data_in_leaf', [300, 400, 500]),
        'min_sum_hessian_in_leaf': hp.uniform('min_sum_hessian_in_leaf', 0.001, 0.005),
        #'path_smooth': hp.uniform('path_smooth', 0, 20)
    }


    # Função de custo do Hyperopt
    def objective(params):
        # Split interno para validação
        X_tr, X_val, y_tr, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )

        model = LGBMRegressor(
            device='gpu',                         # Usa GPU (se disponível) - substitui tree_method='gpu_hist'
            verbosity = -1,                       # Nível de verbosidade (-1: silencioso, 0: erros, 1: avisos, 2: informações)                
            random_state=42,                      # Semente aleatória para reproducibilidade dos resultados
            boosting_type='gbdt',                 # Tipo de boosting 'gbdt' (Gradient Boosting Decision Tree), 'dart' (Dropouts meet Multiple Additive Regression Trees) ou 'goss' (Gradient-based One-Side Sampling)
            importance_type='gain',               # Método escolhido para calcular o Feature Importance, podendo ser Gain (ganho médio de informação ao utilizar a Feature), Weight (número de vezes que a Feature foi utilizada) ou Cover (número de amostras impactadas pela Feature)
            objective='gamma',  # Função de Custo
            metric='rmse',  # usado apenas para log/monitoramento
            **params
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[early_stopping(stopping_rounds=20, verbose=False)]
        )


        # Previsões na validação
        preds = model.predict(X_val)
        preds = np.maximum(preds, 1e-6)

        # RMSLE real
        score = np.sqrt(mean_squared_log_error(y_val, preds))

        return {'loss': score, 'status': STATUS_OK}

    # Rodando a otimização
    trials = Trials()
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(42)
    )

    # Reconstruir hiperparâmetros escolhidos
    best['n_estimators'] = [700, 800, 900, 1000][best['n_estimators']]
    best['max_depth'] = [10, 11, 12][best['max_depth']]
    best['max_bin'] = [64, 128, 255][best['max_bin']]
    best['num_leaves'] = [30, 35, 40, 45, 50][best['num_leaves']]
    best['min_data_in_leaf'] = [300, 400, 500][best['min_data_in_leaf']]


    # Treinar modelo final com os melhores hiperparâmetros
    final_model = LGBMRegressor(
        device='gpu',                         # Usa GPU (se disponível) - substitui tree_method='gpu_hist'
        verbosity = -1,                       # Nível de verbosidade (-1: silencioso, 0: erros, 1: avisos, 2: informações)                
        random_state=42,                      # Semente aleatória para reproducibilidade dos resultados
        boosting_type='gbdt',                 # Tipo de boosting 'gbdt' (Gradient Boosting Decision Tree), 'dart' (Dropouts meet Multiple Additive Regression Trees) ou 'goss' (Gradient-based One-Side Sampling)
        importance_type='gain',               # Método escolhido para calcular o Feature Importance, podendo ser Gain (ganho médio de informação ao utilizar a Feature), Weight (número de vezes que a Feature foi utilizada) ou Cover (número de amostras impactadas pela Feature)
        objective='gamma',  # Função de Custo
        metric='rmse',  # usado apenas para log/monitoramento
        **best
    )

    final_model.fit(
        x_train, y_train,
        eval_set=[(x_test, y_test)],
        eval_metric='rmse',
        callbacks=[early_stopping(stopping_rounds=20, verbose=False)]
    )

    # Previsões finais
    y_pred_train = final_model.predict(x_train)
    y_pred_test = final_model.predict(x_test)

    # Organiza melhores hiperparâmetros em DataFrame
    hiperparametros = pd.DataFrame([best])

    return final_model, y_pred_train, y_pred_test, hiperparametros, trials


def skpro_artesanal(target, x_train, y_train,x_valid, y_valid, x_test, y_test, x_oot, y_oot,modelo_otimizado, alpha = 0.1):
    # ===============================
    # MODELO DA MÉDIA
    # ===============================
    y_pred_train = modelo_otimizado.predict(x_train)

    y_train_array = y_train[target].values.ravel()
    erro_abs_train = np.abs(y_train_array - y_pred_train)

    # ===============================
    # MODELO DA INCERTEZA
    # ===============================
    modelo_sigma = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    modelo_sigma.fit(x_train, erro_abs_train)

    # ===============================
    # FUNÇÃO DE PREDIÇÃO
    # ===============================
    def predict_interval(X):
        mu = modelo_otimizado.predict(X)
        sigma = modelo_sigma.predict(X)
        sigma = np.clip(sigma, 1e-6, None)

        z = norm.ppf(1 - alpha / 2)

        lower = mu - z * sigma
        upper = mu + z * sigma

        return mu, lower, upper

    # ===============================
    # PREDIÇÕES (VETORES)
    # ===============================
    mu_train, lower_train, upper_train = predict_interval(x_train)
    mu_valid, lower_valid, upper_valid = predict_interval(x_valid)
    mu_test,  lower_test,  upper_test  = predict_interval(x_test)
    mu_oot,   lower_oot,   upper_oot   = predict_interval(x_oot)

    # ===============================
    # COBERTURA
    # ===============================
    def coverage(y_true, lower, upper):
        return np.mean((y_true >= lower) & (y_true <= upper))

    cov_train = coverage(y_train[target].values.ravel(), lower_train, upper_train)
    cov_valid = coverage(y_valid[target].values.ravel(), lower_valid, upper_valid)
    cov_test  = coverage(y_test[target].values.ravel(),  lower_test,  upper_test)
    cov_oot   = coverage(y_oot[target].values.ravel(),   lower_oot,   upper_oot)

    print(f"Coverage Treino: {cov_train:.3f}")
    print(f"Coverage Validação: {cov_valid:.3f}")
    print(f"Coverage Teste: {cov_test:.3f}")
    print(f"Coverage OOT: {cov_oot:.3f}")

    # ===============================
    # RETORNO FINAL (LINHA A LINHA)
    # ===============================
    return {
        "train": {
            "y_pred": mu_train,
            "ic_inferior": lower_train,
            "ic_superior": upper_train
        },
        "valid": {
            "y_pred": mu_valid,
            "ic_inferior": lower_valid,
            "ic_superior": upper_valid
        },
        "test": {
            "y_pred": mu_test,
            "ic_inferior": lower_test,
            "ic_superior": upper_test
        },
        "oot": {
            "y_pred": mu_oot,
            "ic_inferior": lower_oot,
            "ic_superior": upper_oot
        }
    }

def skpro_residual_double(target,x_train, y_train, x_valid, y_valid, x_test, y_test, x_oot, y_oot, modelo_otimizado, alpha=0.1, salvar = None):

    if salvar:
        # ===============================
        # RESIDUAL DOUBLE (API ANTIGA)
        # ===============================
        skpro_residual_double = ResidualDouble(modelo_otimizado)
        skpro_residual_double.fit(X=x_train,y=y_train[target].values)
        carrega_salva_ic('salvar', skpro_residual_double)
    else:

        skpro_residual_double = carrega_salva_ic('carregar')
        # ===============================
        # FUNÇÃO DE IC
        # ==============================n=
        def predict_interval(X):
            # Retorna um objeto de distribuição
            dist = skpro_residual_double.predict_proba(X)

            mu = dist.mean()
            lower = dist.ppf(alpha / 2)
            upper = dist.ppf(1 - alpha / 2)

            return mu, lower, upper

        # ===============================
        # PREDIÇÕES
        # ===============================
        mu_train, lower_train, upper_train = predict_interval(x_train)
        mu_valid, lower_valid, upper_valid = predict_interval(x_valid)
        mu_test,  lower_test,  upper_test  = predict_interval(x_test)
        mu_oot,   lower_oot,   upper_oot   = predict_interval(x_oot)

        # ===============================
        # COVERAGE
        # ===============================
        def coverage(y_true, lower, upper):
            # .ravel() funciona tanto para DataFrames quanto para arrays numpy
            return np.mean((np.array(y_true).ravel() >= np.array(lower).ravel()) & (np.array(y_true).ravel() <= np.array(upper).ravel()))

        cov_train = coverage(y_train[target].values, lower_train, upper_train)
        cov_valid = coverage(y_valid[target].values, lower_valid, upper_valid)
        cov_test  = coverage(y_test[target].values,  lower_test,  upper_test)
        cov_oot   = coverage(y_oot[target].values,   lower_oot,   upper_oot)

        print(f"Coverage Treino: {cov_train:.3f}")
        print(f"Coverage Validação: {cov_valid:.3f}")
        print(f"Coverage Teste: {cov_test:.3f}")
        print(f"Coverage OOT: {cov_oot:.3f}")

        # ===============================
        # RETORNO LINHA A LINHA
        # ===============================
        return {
        "train": {
                "y_pred": mu_train,
                "ic_inferior": lower_train,
                "ic_superior": upper_train
            },
            "valid": {
                "y_pred": mu_valid,
                "ic_inferior": lower_valid,
                "ic_superior": upper_valid
            },
            "test": {
                "y_pred": mu_test,
                "ic_inferior": lower_test,
                "ic_superior": upper_test
            },
            "oot": {
                "y_pred": mu_oot,
                "ic_inferior": lower_oot,
                "ic_superior": upper_oot
            }
        }


"""Funções reutilizáveis do projeto de detecção de fraude."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


RANDOM_STATE = 42


def plota_barras(variavel, df, titulo, top_n=None, rotation=0):
    contagem = df[variavel].value_counts(dropna=False).head(top_n)
    ax = contagem.plot.bar(color="#1FB3E5", title=titulo)
    ax.set_ylabel("Quantidade")
    ax.tick_params(axis="x", rotation=rotation)
    for barra, valor in zip(ax.patches, contagem.values):
        ax.annotate(
            f"{valor / contagem.sum():.1%}",
            (barra.get_x() + barra.get_width() / 2, barra.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.show()


def analisa_distribuicao_via_percentis(df, variaveis):
    return df[variaveis].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T


def taxa_por_grupo(df, coluna, target="is_fraud", top_n=20):
    tabela = (
        df.groupby(coluna, observed=True)[target]
        .agg(qtd="size", fraudes="sum", taxa_fraude="mean")
        .sort_values(["taxa_fraude", "qtd"], ascending=False)
    )
    return tabela.head(top_n)


def haversine_km(lat1, lon1, lat2, lon2):
    """Calcula a distância do arco entre dois pontos da Terra."""
    raio_terra = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * raio_terra * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def media_std_historica(df, grupo, valor, prefixo):
    """Calcula estatísticas acumuladas excluindo a observação atual."""
    g = df.groupby(grupo, sort=False, observed=True)[valor]
    n = df.groupby(grupo, sort=False, observed=True).cumcount().astype("float32")
    soma = g.cumsum() - df[valor]
    soma2 = (
        df.assign(_quadrado=df[valor] ** 2)
        .groupby(grupo, sort=False, observed=True)["_quadrado"]
        .cumsum()
        - df[valor] ** 2
    )
    media = soma / n.replace(0, np.nan)
    variancia = (soma2 / n.replace(0, np.nan) - media**2).clip(lower=0)
    df[f"media_{prefixo}"] = media.fillna(0).astype("float32")
    df[f"std_{prefixo}"] = np.sqrt(variancia).fillna(0).astype("float32")
    return df


def cria_features(df):
    d = df.sort_values(["trans_date_trans_time", "trans_num"]).reset_index(drop=True).copy()

    d["log_amt"] = np.log1p(d.amt).astype("float32")
    d["hora"] = d.trans_date_trans_time.dt.hour.astype("int8")
    d["dia_semana"] = d.trans_date_trans_time.dt.dayofweek.astype("int8")
    d["fim_semana"] = (d.dia_semana >= 5).astype("int8")
    d["hora_sin"] = np.sin(2 * np.pi * d.hora / 24).astype("float32")
    d["hora_cos"] = np.cos(2 * np.pi * d.hora / 24).astype("float32")
    d["idade"] = ((d.trans_date_trans_time.dt.normalize() - d.dob).dt.days / 365.25).astype("float32")
    d["distancia_cliente_lojista_km"] = haversine_km(
        d.lat, d.long, d.merch_lat, d.merch_long
    ).astype("float32")

    g_cartao = d.groupby("cc_num", sort=False, observed=True)
    d["qtd_transacoes_cartao_hist"] = g_cartao.cumcount().astype("int32")
    d["segundos_ultima_transacao"] = (
        g_cartao.trans_date_trans_time.diff()
        .dt.total_seconds()
        .fillna(-1)
        .clip(-1, 30 * 86400)
        .astype("float32")
    )
    d = media_std_historica(d, "cc_num", "amt", "amt_cartao_hist")
    d["zscore_amt_cartao"] = (
        ((d.amt - d.media_amt_cartao_hist) / d.std_amt_cartao_hist.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(-20, 20)
        .astype("float32")
    )
    primeiro_cartao = g_cartao.trans_date_trans_time.transform("min")
    dias_relacionamento = (d.trans_date_trans_time - primeiro_cartao).dt.total_seconds() / 86400
    d["freq_transacoes_cartao_dia"] = (
        d.qtd_transacoes_cartao_hist / np.maximum(dias_relacionamento, 1)
    ).astype("float32")

    g_lojista = d.groupby("merchant", sort=False, observed=True)
    d["qtd_transacoes_lojista_hist"] = g_lojista.cumcount().astype("int32")
    d = media_std_historica(d, "merchant", "amt", "amt_lojista_hist")

    d["qtd_transacoes_par_hist"] = d.groupby(
        ["cc_num", "merchant"], sort=False, observed=True
    ).cumcount().astype("int16")
    primeira_aresta = d.qtd_transacoes_par_hist.eq(0).astype("int8")
    d["grau_cartao_hist"] = (
        primeira_aresta.groupby(d.cc_num, sort=False).cumsum() - primeira_aresta
    ).astype("int16")
    d["grau_lojista_hist"] = (
        primeira_aresta.groupby(d.merchant, sort=False).cumsum() - primeira_aresta
    ).astype("int16")
    d["novo_par_cartao_lojista"] = primeira_aresta
    d["participacao_lojista_cartao"] = (
        d.qtd_transacoes_par_hist / d.qtd_transacoes_cartao_hist.replace(0, np.nan)
    ).fillna(0).astype("float32")
    return d


def metricas_modelo(nome, etapa, y_true, proba, cutoff=None):
    auc = roc_auc_score(y_true, proba)
    resultado = {
        "Modelo": nome,
        "Etapa": etapa,
        "Gini": 2 * auc - 1,
        "PR_AUC": average_precision_score(y_true, proba),
        "Taxa_Fraude": np.mean(y_true),
    }
    if cutoff is not None:
        pred = (np.asarray(proba) >= cutoff).astype(int)
        resultado.update(
            {
                "Cutoff": cutoff,
                "Precisao": precision_score(y_true, pred, zero_division=0),
                "Recall": recall_score(y_true, pred, zero_division=0),
                "F1": f1_score(y_true, pred, zero_division=0),
                "Alert_Rate": pred.mean(),
            }
        )
    return pd.DataFrame([resultado])


def tabela_capacidade(y_true, proba, capacidades=(0.001, 0.0025, 0.005, 0.01, 0.02)):
    aux = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)}).sort_values(
        "p", ascending=False
    )
    total_transacoes, total_fraudes = len(aux), int(aux.y.sum())
    prevalencia = total_fraudes / max(total_transacoes, 1)
    linhas = []
    for cap in capacidades:
        n = max(1, int(np.ceil(total_transacoes * cap)))
        fila = aux.head(n)
        fraudes_capturadas = int(fila.y.sum())
        linhas.append(
            {
                "Capacidade_maxima": f"{cap:.2%}",
                "Total_transacoes": total_transacoes,
                "Alertas_qtd": n,
                "Taxa_alertas": f"{n / total_transacoes:.2%}",
                "Fraudes_capturadas": fraudes_capturadas,
                "Precisao": f"{fraudes_capturadas / n:.2%}",
                "Fraudes_totais": total_fraudes,
                "Recall": f"{fraudes_capturadas / max(total_fraudes, 1):.2%}",
                "Prevalencia_base": f"{prevalencia:.2%}",
                "Cutoff_score": round(float(fila.p.min()), 4),
            }
        )
    return pd.DataFrame(linhas)


def modelo_lightgbm(parametros=None):
    base = dict(
        objective="binary",
        n_estimators=225,
        learning_rate=0.035,
        num_leaves=12,
        max_depth=4,
        min_child_samples=250,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=2,
        reg_lambda=5,
        subsample_freq=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    if parametros:
        base.update(parametros)
    return LGBMClassifier(**base)


def otimizacao_hyperopt(x_train, y_train, x_valid, y_valid, max_evals=20):
    espaco = {
        "n_estimators": hp.quniform("n_estimators", 100, 350, 25),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.015), np.log(0.08)),
        "num_leaves": hp.quniform("num_leaves", 7, 21, 2),
        "max_depth": hp.quniform("max_depth", 3, 5, 1),
        "min_child_samples": hp.quniform("min_child_samples", 150, 600, 50),
        "subsample": hp.uniform("subsample", 0.65, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(0.1), np.log(20)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(0.5), np.log(30)),
    }

    inteiros = ["n_estimators", "num_leaves", "max_depth", "min_child_samples"]

    def converte(parametros):
        parametros = parametros.copy()
        for coluna in inteiros:
            parametros[coluna] = int(parametros[coluna])
        return parametros

    def objetivo(parametros):
        parametros = converte(parametros)
        modelo = modelo_lightgbm(parametros).fit(x_train, y_train, categorical_feature="auto")
        proba_train = modelo.predict_proba(x_train)[:, 1]
        proba_valid = modelo.predict_proba(x_valid)[:, 1]
        ap_valid = average_precision_score(y_valid, proba_valid)
        ap_train = average_precision_score(y_train, proba_train)
        gini_train = 2 * roc_auc_score(y_train, proba_train) - 1
        gini_valid = 2 * roc_auc_score(y_valid, proba_valid) - 1
        gap_excessivo = max(0, gini_train - gini_valid - 0.05)
        gap_ap_excessivo = max(0, ap_train - ap_valid - 0.08)
        perda = -ap_valid + 0.25 * gap_excessivo + 0.75 * gap_ap_excessivo
        return {
            "loss": perda,
            "status": STATUS_OK,
            "ap_valid": ap_valid,
            "ap_train": ap_train,
            "gap_ap": ap_train - ap_valid,
            "gini_train": gini_train,
            "gini_valid": gini_valid,
        }

    trials = Trials()
    best = fmin(
        objetivo,
        espaco,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(RANDOM_STATE),
        show_progressbar=True,
    )
    best = converte(best)
    modelo = modelo_lightgbm(best).fit(x_train, y_train, categorical_feature="auto")
    return modelo, best, trials


def cria_rating(score, cortes):
    return pd.cut(
        score,
        bins=[-np.inf] + list(cortes) + [np.inf],
        labels=["A", "B", "C", "D", "E"],
        include_lowest=True,
    )


def ajusta_calibrador_score(score, y, metodo):
    if metodo == "isotonic":
        modelo = IsotonicRegression(out_of_bounds="clip").fit(score, y)
    else:
        modelo = LogisticRegression(C=1, random_state=RANDOM_STATE).fit(
            np.asarray(score).reshape(-1, 1), y
        )
    return modelo


def prediz_calibrador(modelo, score, metodo):
    if metodo == "isotonic":
        return modelo.predict(score)
    return modelo.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1]


def retorno_financeiro_incremental(
    df, pred, taxa_recuperacao=0.75, custo_revisao=2, custo_atrito_fp=5
):
    pred = np.asarray(pred).astype(bool)
    fraude = df.is_fraud.to_numpy().astype(bool)
    valores = df.amt.to_numpy()
    tp, fp = pred & fraude, pred & ~fraude
    perda_sem_modelo = valores[fraude].sum()
    perda_residual = valores[fraude & ~pred].sum() + (1 - taxa_recuperacao) * valores[tp].sum()
    custo_operacional = pred.sum() * custo_revisao + fp.sum() * custo_atrito_fp
    retorno_incremental = perda_sem_modelo - perda_residual - custo_operacional
    return {
        "Perda_sem_modelo": perda_sem_modelo,
        "Fraude_evitada": taxa_recuperacao * valores[tp].sum(),
        "Custo_revisao_atrito": custo_operacional,
        "Retorno_incremental": retorno_incremental,
    }


def escolhe_cutoff_politica(
    df_politica,
    score,
    capacidade_max=0.01,
    taxa_recuperacao=0.75,
    custo_revisao=2,
    custo_atrito_fp=5,
):
    aux = df_politica[["is_fraud", "amt"]].copy()
    aux["score"] = np.asarray(score)
    candidatos = np.unique(np.quantile(score, np.linspace(0.90, 0.9999, 600)))
    linhas = []
    for cutoff in candidatos:
        pred = aux.score >= cutoff
        financeiro = retorno_financeiro_incremental(
            aux, pred, taxa_recuperacao, custo_revisao, custo_atrito_fp
        )
        linhas.append(
            {
                "cutoff": cutoff,
                "alert_rate": pred.mean(),
                "precision": precision_score(aux.is_fraud, pred, zero_division=0),
                "recall": recall_score(aux.is_fraud, pred, zero_division=0),
                "retorno_incremental": financeiro["Retorno_incremental"],
            }
        )
    curva = pd.DataFrame(linhas)
    elegiveis = curva[curva.alert_rate <= capacidade_max]
    melhor = elegiveis.sort_values(
        ["retorno_incremental", "precision"], ascending=False
    ).iloc[0]
    return float(melhor.cutoff), curva


def metricas_mensais(df, score, cutoff):
    aux = df[["trans_date_trans_time", "is_fraud"]].copy()
    aux["score"] = np.asarray(score)
    aux["mes"] = aux.trans_date_trans_time.dt.to_period("M").astype(str)
    linhas = []
    for mes, dados_mes in aux.groupby("mes"):
        pred = dados_mes.score >= cutoff
        auc = roc_auc_score(dados_mes.is_fraud, dados_mes.score)
        linhas.append(
            {
                "Mes": mes,
                "N": len(dados_mes),
                "Taxa_Fraude": dados_mes.is_fraud.mean(),
                "Gini": 2 * auc - 1,
                "PR_AUC": average_precision_score(dados_mes.is_fraud, dados_mes.score),
                "Precisao": precision_score(dados_mes.is_fraud, pred, zero_division=0),
                "Recall": recall_score(dados_mes.is_fraud, pred, zero_division=0),
                "Alert_Rate": pred.mean(),
            }
        )
    return pd.DataFrame(linhas)


def plot_shap(modelo, x_amostra, titulo):
    explainer = shap.TreeExplainer(modelo)
    valores = explainer(x_amostra)
    shap.plots.beeswarm(valores, max_display=20, show=False)
    plt.title(titulo)
    plt.tight_layout()
    plt.show()
    return explainer, valores


def define_amostra(data):
    return np.select(
        [
            data < pd.Timestamp("2020-01-01"),
            data < pd.Timestamp("2020-04-01"),
            data < pd.Timestamp("2020-06-01"),
            data < pd.Timestamp("2020-06-21 12:14:00"),
            data < pd.Timestamp("2020-10-01"),
        ],
        ["Treino", "Validacao", "Calibracao", "Politica", "Teste"],
        default="OOT",
    )


