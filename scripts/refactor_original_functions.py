"""Gera módulos conservadores a partir do apanhado original de funções."""
from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "exemplo_funcoes.md"
SRC = ROOT / "src"

EXCLUDE = {"_build_lstm", "lstm_forecast"}  # Deep Learning foi explicitamente adiado.

# Seleção deliberadamente enxuta. Para nomes repetidos, o número indica qual
# ocorrência original será mantida (1 = primeira). Nomes ausentes são removidos.
KEEP = {
    # Visualização
    "plota_barras_agrupadas": 1, "plota_barras": 3, "plota_histograma": 2,
    "plota_boxplot": 2, "plota_dispersao": 1, "plota_grafico_linhas": 2,
    "auc_ks_juntos": 1, "auc_ks_final": 1, "plot_linear_separability": 1,
    "plot_histograms_comparison": 1, "plot_boxplot_comparison": 1,
    "plot_comparative_density": 1,
    "plot_bandas_temporais_unico": 1, "plot_matriz_migracao_rating": 1,
    "plot_rating_risco": 1, "plot_shap_beeswarm": 1,
    "plot_shap_one_sample": 1, "plot_shap_one_sample_original_scale": 1,
    # Análise
    "analisa_correlacao": 3, "analisa_normalidade": 1,
    "analisa_outliers": 1, "teste_hipotese_duas_amostras_independentes": 1,
    "teste_hipotese_muitas_amostras_independentes": 1,
    "teste_hipotese_duas_variaveis_categoricas": 1,
    "analisa_distribuicao_via_percentis": 2,
    "taxa_por_grupo": 1,
    # Pré-processamento
    "separa_feature_target": 4, "separa_treino_teste": 1, "discretiza_variavel": 1,
    "transform_to_percentiles": 2, "simple_imputer": 1, "target_encoder_bad_rate": 1,
    "muda_tipagem_variavel": 1,
    "formatar_valor_milhoes": 1, "abbreviate_number": 1, "haversine_km": 1,
    # Seleção e interpretação
    "remove_features_baixa_variancia": 1, "remove_features_mutual_information": 1,
    "remove_features_feature_importance": 1, "compute_shap_importance_df": 1,
    # Modelagem, otimização e persistência
    "Regressor": 1, "modelo_lightgbm": 1,
    "otimizacao_hyperopt_regression": 1, "otimizacao_hyperopt": 1,
    "salvar_modelo_pickle": 1, "carregar_modelo_pickle": 1,
    # Métricas
    "ks_test": 1, "error_metrics": 1, "metricas_regressao": 1,
    "metricas_modelos_juntos_regressao": 1, "metricas_regressao_diarias": 1,
    "metricas_modelo": 1,
    # Risco geral
    "woe": 2, "iv": 1, "calcular_rating": 1, "criar_base_rating": 1,
    "comparar_ratings": 1, "matriz_migracao_rating": 1, "resumo_migracao_rating": 1,
    "tabela_capacidade": 1, "cria_rating": 1, "ajusta_calibrador_score": 1,
    "prediz_calibrador": 1,
    # Temporal e incerteza
    "select_statistical_orders": 1, "rolling_statistical_forecast": 1,
    "stationarity_row": 1, "calcular_psi_temporal": 1,
}

BOILERPLATE = '''"""Implementações preservadas do arquivo exemplo_funcoes.md.

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

try:
    import shap
except ImportError: shap = None
try:
    from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
except ImportError: LGBMClassifier = LGBMRegressor = early_stopping = None
try:
    from xgboost import XGBClassifier
except ImportError: XGBClassifier = None
try:
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
except ImportError: STATUS_OK = Trials = fmin = hp = tpe = None
try:
    from skopt import BayesSearchCV
except ImportError: BayesSearchCV = None
try:
    from category_encoders import BinaryEncoder, CatBoostEncoder
except ImportError: BinaryEncoder = CatBoostEncoder = None

SEED = RANDOM_STATE = 42
TARGET = "meantemp"
FEATURES = ["temp_lag_1", "temp_lag_2", "temp_lag_3", "temp_lag_7", "temp_lag_14", "temp_lag_30",
            "temp_roll_mean_3", "temp_roll_std_3", "temp_roll_mean_7", "temp_roll_std_7",
            "temp_roll_mean_14", "temp_roll_std_14", "temp_roll_mean_30", "temp_roll_std_30",
            "doy_sin", "doy_cos", "trend_days"]
FOLDS = [("2016-Q1", "2016-01-01", "2016-03-31"), ("2016-Q2", "2016-04-01", "2016-06-30"),
         ("2016-Q3", "2016-07-01", "2016-09-30"), ("2016-Q4", "2016-10-01", "2016-12-31")]
'''

CUSTOM_METHODS = {
"modeling": r'''
    @staticmethod
    def Classificador(classificador, x_train, y_train, x_test, y_test,
                      class_weight=None, parametros=None, preprocessador=None):
        """Treina um classificador geral e devolve modelo, classes e probabilidades.

        Modelos suportados: Regressão Logística, Random Forest, LightGBM e XGBoost.
        ``preprocessador`` pode ser qualquer transformer compatível com scikit-learn.
        """
        nome = str(classificador).strip().lower().replace("_", " ")
        params = dict(parametros or {})
        if nome in {"regressão logística", "regressao logistica", "logistic regression", "logistica"}:
            base = dict(max_iter=2000, random_state=42,
                        class_weight="balanced" if class_weight is None else {0: 1, 1: class_weight})
            base.update(params); modelo = LogisticRegression(**base)
        elif nome in {"random forest", "randomforest"}:
            base = dict(n_estimators=300, random_state=42, n_jobs=-1,
                        class_weight="balanced" if class_weight is None else {0: 1, 1: class_weight})
            base.update(params); modelo = RandomForestClassifier(**base)
        elif nome in {"lightgbm", "lgbm"}:
            if LGBMClassifier is None: raise ImportError("Instale lightgbm para usar este classificador.")
            base = dict(objective="binary", n_estimators=300, learning_rate=0.03,
                        random_state=42, n_jobs=-1, verbosity=-1)
            if class_weight is not None: base["scale_pos_weight"] = class_weight
            base.update(params); modelo = LGBMClassifier(**base)
        elif nome in {"xgboost", "xgb"}:
            if XGBClassifier is None: raise ImportError("Instale xgboost para usar este classificador.")
            base = dict(n_estimators=300, learning_rate=0.03, max_depth=6,
                        objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=-1)
            if class_weight is not None: base["scale_pos_weight"] = class_weight
            base.update(params); modelo = XGBClassifier(**base)
        else:
            raise ValueError("Use Regressão Logística, Random Forest, LightGBM ou XGBoost.")
        if preprocessador is not None:
            modelo = Pipeline([("preprocessamento", clone(preprocessador)), ("modelo", modelo)])
        y_train_array = np.asarray(y_train).ravel(); y_test_array = np.asarray(y_test).ravel()
        modelo.fit(x_train, y_train_array)
        return (modelo, modelo.predict(x_train), modelo.predict(x_test),
                modelo.predict_proba(x_train), modelo.predict_proba(x_test))

    @staticmethod
    def calibracao_probabilidade(modelo, x_calibracao, y_calibracao,
                                 metodo="isotonic", cv="prefit", n_jobs=-1):
        """Calibra probabilidades por isotonic ou sigmoid (Platt scaling)."""
        if metodo not in {"isotonic", "sigmoid"}:
            raise ValueError("metodo deve ser 'isotonic' ou 'sigmoid'.")
        calibrador = CalibratedClassifierCV(
            estimator=modelo, method=metodo, cv=cv, n_jobs=n_jobs
        )
        calibrador.fit(x_calibracao, np.asarray(y_calibracao).ravel())
        return calibrador

    @staticmethod
    def predizer_probabilidade(modelo, x, classe_positiva=1):
        """Retorna somente a probabilidade da classe positiva."""
        if not hasattr(modelo, "predict_proba"):
            raise TypeError("O modelo não implementa predict_proba.")
        probabilidades = np.asarray(modelo.predict_proba(x))
        return probabilidades[:, classe_positiva]

    @staticmethod
    def validacao_cruzada_classificacao(modelo, x, y, n_splits=5, scoring=None,
                                        shuffle=True, random_state=42, n_jobs=-1):
        """Validação cruzada estratificada com detalhe por fold e resumo."""
        metricas = scoring or {
            "AUC_ROC": "roc_auc", "PR_AUC": "average_precision",
            "Precisao": "precision", "Recall": "recall", "F1": "f1",
        }
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle,
                            random_state=random_state if shuffle else None)
        resultado = cross_validate(modelo, x, np.asarray(y).ravel(), cv=cv,
                                   scoring=metricas, n_jobs=n_jobs,
                                   return_train_score=True)
        detalhe = pd.DataFrame(resultado)
        resumo = pd.DataFrame({
            "Metrica": list(metricas),
            "Treino_media": [detalhe[f"train_{m}"].mean() for m in metricas],
            "Treino_std": [detalhe[f"train_{m}"].std() for m in metricas],
            "Validacao_media": [detalhe[f"test_{m}"].mean() for m in metricas],
            "Validacao_std": [detalhe[f"test_{m}"].std() for m in metricas],
        })
        return detalhe, resumo
''',
"metrics": r'''
    @staticmethod
    def metricas_classificacao(classificador, y_true, y_predict=None,
                               y_predict_proba=None, etapa="teste", cutoff=0.5):
        """Consolida métricas preditivas, discriminatórias e operacionais."""
        y = np.asarray(y_true).ravel().astype(int)
        if y_predict_proba is None:
            raise ValueError("Informe y_predict_proba para calcular AUC, KS e calibração.")
        proba = np.asarray(y_predict_proba)
        if proba.ndim == 2: proba = proba[:, 1]
        proba = proba.ravel()
        pred = (proba >= cutoff).astype(int) if y_predict is None else np.asarray(y_predict).ravel().astype(int)
        if not (len(y) == len(pred) == len(proba)):
            raise ValueError("y_true, y_predict e y_predict_proba devem ter o mesmo tamanho.")
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        if np.unique(y).size > 1:
            auc_roc = roc_auc_score(y, proba); fpr, tpr, _ = roc_curve(y, proba)
            ks = float(np.max(tpr - fpr)); pr_auc = average_precision_score(y, proba)
        else:
            auc_roc = ks = pr_auc = np.nan
        linha = {
            "Modelo": classificador, "Etapa": etapa, "N": len(y),
            "Prevalencia": y.mean(), "Cutoff": cutoff,
            "Acuracia": accuracy_score(y, pred),
            "Precisao": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "AUC_ROC": auc_roc, "Gini": 2 * auc_roc - 1 if np.isfinite(auc_roc) else np.nan,
            "PR_AUC": pr_auc, "KS": ks,
            "LogLoss": log_loss(y, np.clip(proba, 1e-15, 1-1e-15), labels=[0, 1]),
            "Brier": brier_score_loss(y, proba), "Alert_Rate": pred.mean(),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        }
        return pd.DataFrame([linha])

    @staticmethod
    def metricas_classificacao_treino_teste(classificador, y_train, proba_train,
                                            y_test, proba_test, cutoff=0.5,
                                            etapa_treino="treino", etapa_teste="teste"):
        treino = metricas_classificacao(classificador, y_train, None, proba_train, etapa_treino, cutoff)
        teste = metricas_classificacao(classificador, y_test, None, proba_test, etapa_teste, cutoff)
        return pd.concat([treino, teste], ignore_index=True)

    @staticmethod
    def consolidar_metricas_classificacao(lista_metricas):
        """Concatena resultados de modelos/amostras em uma tabela única."""
        if not lista_metricas: return pd.DataFrame()
        return pd.concat(lista_metricas, ignore_index=True)

    @staticmethod
    def metricas_calibracao(y_true, y_predict_proba, n_bins=10):
        """Retorna Brier/Log Loss e tabela observada versus prevista por faixa."""
        y = np.asarray(y_true).ravel().astype(int)
        proba = np.asarray(y_predict_proba)
        if proba.ndim == 2: proba = proba[:, 1]
        proba = proba.ravel()
        observado, previsto = calibration_curve(y, proba, n_bins=n_bins,
                                                strategy="quantile")
        tabela = pd.DataFrame({"Probabilidade_prevista": previsto,
                               "Frequencia_observada": observado})
        tabela["Erro_absoluto"] = abs(tabela.Frequencia_observada - tabela.Probabilidade_prevista)
        resumo = pd.DataFrame([{
            "Brier": brier_score_loss(y, proba),
            "LogLoss": log_loss(y, np.clip(proba, 1e-15, 1-1e-15), labels=[0, 1]),
            "Erro_calibracao_medio": tabela.Erro_absoluto.mean(),
        }])
        return resumo, tabela
''',
"visualization": r'''
    @staticmethod
    def plot_calibracao(y_true, y_predict_proba, n_bins=10,
                        titulo="Curva de Calibração", nome_modelo="Modelo"):
        """Plota calibração observada versus prevista no estilo visual do projeto."""
        y = np.asarray(y_true).ravel().astype(int)
        proba = np.asarray(y_predict_proba)
        if proba.ndim == 2: proba = proba[:, 1]
        observado, previsto = calibration_curve(y, proba.ravel(), n_bins=n_bins,
                                                strategy="quantile")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(previsto, observado, marker="o", linewidth=2.5,
                color="#1FB3E5", label=nome_modelo)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
                linewidth=1.5, label="Calibração perfeita")
        for x, y_obs in zip(previsto, observado):
            ax.annotate(f"{y_obs:.1%}", (x, y_obs), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(titulo, fontsize=15, fontweight="bold")
        ax.set_xlabel("Probabilidade prevista"); ax.set_ylabel("Frequência observada")
        ax.grid(alpha=0.2, linestyle="--"); ax.legend(frameon=False)
        plt.tight_layout()
        return fig, ax
'''
}


def extract_blocks(text: str):
    lines = text.splitlines(True)
    starts = [i for i, line in enumerate(lines) if re.match(r"^(def|class)\s+", line)]
    blocks = []
    for pos, start in enumerate(starts):
        stop = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        candidate = lines[start:stop]
        chosen = None
        for end in range(len(candidate), 0, -1):
            code = "".join(candidate[:end]).rstrip() + "\n"
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            if len(tree.body) == 1 and isinstance(tree.body[0], (ast.FunctionDef, ast.ClassDef)):
                chosen = code
                break
        if chosen is None:
            raise SyntaxError(f"Não foi possível extrair definição na linha {start + 1}")
        node = ast.parse(chosen).body[0]
        blocks.append({"line": start + 1, "name": node.name, "kind": type(node).__name__, "code": chosen})
    return blocks


def category(name: str, line: int) -> str:
    n = name.lower()
    if n in {"formatar_valor_milhoes", "abbreviate_number", "haversine_km"}: return "preprocessing"
    if n == "iv": return "risk"
    if n.startswith("skpro_"): return "temporal"
    if n.startswith("aplica_feature_selection") or n in {"compute_shap_importance_df", "interpret_tree_models"}: return "feature_selection"
    if n in {"experimentresult", "set_seed", "load_official_splits", "make_features", "_evaluate_period", "run_experiment"}: return "temporal"
    if n in {"modelo_lightgbm", "modelo_classificador_churn_oficial", "calibracao_probabilidade"}: return "modeling"
    if any(k in n for k in ("plota", "plot_", "visualize")) or n.startswith("auc_ks"): return "visualization"
    if "shap" in n: return "feature_selection"
    if any(k in n for k in ("feature_selection", "remove_features", "importance_df")): return "feature_selection"
    if any(k in n for k in ("metrica", "error_metrics", "ks_test", "bootstrap")): return "metrics"
    if any(k in n for k in ("woe", "rating", "migracao", "retorno_financeiro", "politica", "cutoff", "corte_probabilidade", "capacidade", "calibrador")): return "risk"
    if any(k in n for k in ("cluster", "pca")): return "segmentation"
    if any(k in n for k in ("salvar", "carregar", "carrega_salva")): return "modeling"
    if any(k in n for k in ("temporal", "movel", "forecast", "stationarity", "statistical_orders")): return "temporal"
    if any(k in n for k in ("separa_", "imputer", "discretiza", "transform_to_", "encoder", "pre_processamento", "scaler", "muda_tipagem", "define_amostra")): return "preprocessing"
    if any(k in n for k in ("otimizacao", "hyperopt")): return "modeling"
    if any(k in n for k in ("classificador", "regressor", "validacao_cruzada", "modelo_oficial", "escoragem", "calibracao_probabilidade", "model_catalog")): return "modeling"
    if any(k in n for k in ("analisa", "verifica", "teste_hipotese", "compara_medias", "taxa_por_grupo")): return "analysis"
    if line >= 5200 and name in {"set_seed", "load_official_splits", "make_features", "_evaluate_period", "run_experiment"}: return "temporal"
    return "project_specific"


def main():
    extracted = extract_blocks(SOURCE.read_text(encoding="utf-8"))
    occurrence = Counter()
    blocks = []
    for block in extracted:
        occurrence[block["name"]] += 1
        if block["name"] in EXCLUDE: continue
        if KEEP.get(block["name"]) == occurrence[block["name"]]: blocks.append(block)
    totals = Counter(b["name"] for b in blocks)
    seen = Counter()
    groups = defaultdict(list)
    emitted = []
    aliases = []
    for block in blocks:
        original = block["name"]
        seen[original] += 1
        public = f"{original}_v{seen[original]}" if totals[original] > 1 else original
        code = block["code"]
        code = re.sub(rf"^(def|class)\s+{re.escape(original)}\b", rf"\1 {public}", code, count=1)
        if category(original, block["line"]) == "modeling":
            code = re.sub(r"^\s*device\s*=\s*['\"]gpu['\"].*\n", "", code, flags=re.MULTILINE)
        if original == "woe":
            for old, new in (("percent_sem_churn", "percent_nao_evento"),
                             ("percent_churn", "percent_evento"),
                             ("sem_churn", "nao_evento"), ("churn", "evento")):
                code = code.replace(old, new)
        if block["kind"] == "ClassDef" and original == "ExperimentResult": code = "@dataclass\n" + code
        emitted.append(f"# Fonte original: linha {block['line']}\n{code}")
        groups[category(original, block["line"])].append({"name": public, "code": code, "kind": block["kind"]})
        if totals[original] > 1 and seen[original] == totals[original]: aliases.append(f"{original} = {public}\n")
    (SRC / "_shared.py").write_text(BOILERPLATE, encoding="utf-8")

    class_names = {
        "visualization": "Visualizacao", "interpretability": "Interpretabilidade",
        "feature_selection": "SelecaoFeatures", "metrics": "Metricas", "risk": "RiscoCredito",
        "segmentation": "Segmentacao", "persistence": "Persistencia", "temporal": "SeriesTemporais",
        "preprocessing": "PreProcessamento", "optimization": "Otimizacao", "modeling": "Modelagem",
        "analysis": "AnaliseExploratoria", "project_specific": "FuncoesEspecificas",
        "formatting": "Formatacao", "geospatial": "Geoespacial", "uncertainty": "Incerteza",
    }
    for module, items in groups.items():
        cls = class_names[module]
        rows = ['"""Métodos generalistas selecionados do arquivo original."""',
                "from ._shared import *",]
        if module == "feature_selection": rows.append("from .preprocessing import separa_feature_target")
        rows += ["", f"class {cls}:"]
        for item in items:
            name, code, kind = item["name"], item["code"], item["kind"]
            if kind == "FunctionDef":
                rows.append("    @staticmethod")
                rows.extend("    " + line if line else "" for line in code.rstrip().splitlines())
                rows.append("")
            else:
                raise RuntimeError(f"Classe original inesperada no núcleo enxuto: {name}")
        custom = CUSTOM_METHODS.get(module, "").strip("\n")
        if custom:
            rows.extend(custom.splitlines())
            rows.append("")
        rows += ["", "# Aliases funcionais para compatibilidade com notebooks antigos."]
        for item in items: rows.append(f"{item['name']} = {cls}.{item['name']}")
        custom_names = re.findall(r"^    def\s+([A-Za-z_]\w*)", CUSTOM_METHODS.get(module, ""), flags=re.MULTILINE)
        for name in custom_names: rows.append(f"{name} = {cls}.{name}")
        (SRC / f"{module}.py").write_text("\n".join(rows) + "\n", encoding="utf-8")

    exports = [class_names[m] for m in groups]
    init = ['"""Funções originais organizadas por classes e módulos."""']
    for module in groups: init.append(f"from .{module} import {class_names[module]}")
    init += ["from .causal import InferenciaCausal"]
    exports += ["InferenciaCausal"]
    init += ["", "__all__ = ["] + [f'    "{name}",' for name in exports] + ["]", ""]
    (SRC / "__init__.py").write_text("\n".join(init), encoding="utf-8")

    custom_total = sum(len(re.findall(r"^    def\s+", code, flags=re.MULTILINE)) for code in CUSTOM_METHODS.values())
    report = ["# Mapeamento da biblioteca enxuta", "", f"Definições originais generalistas mantidas: {len(blocks)}.",
              f"Funções gerais complementares desenvolvidas: {custom_total}.",
              "Deep Learning permanece fora deste escopo.", "",
              "Para cada família duplicada foi escolhida uma única implementação canônica, preservando seu corpo original.",
              "Foram removidos pipelines completos e funções dependentes de colunas, datas, arquivos ou regras de um projeto específico.", ""]
    for module, items in groups.items():
        names = [i["name"] for i in items]
        names += re.findall(r"^    def\s+([A-Za-z_]\w*)", CUSTOM_METHODS.get(module, ""), flags=re.MULTILINE)
        report += [f"## {module}.py — {class_names[module]}", "", ", ".join(f"`{n}`" for n in names), ""]
    (ROOT / "MAPEAMENTO_FUNCOES.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__": main()
