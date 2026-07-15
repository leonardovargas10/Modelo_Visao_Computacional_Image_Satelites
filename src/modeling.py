"""Métodos generalistas selecionados do arquivo original."""
from ._shared import *

@dataclass
class CausalResult:
    """Resultado padronizado das estimativas causais."""
    estimate: float
    std_error: float
    p_value: float
    confidence_interval: tuple
    model: object
    diagnostics: dict

class Modelagem:

    # ------------------------------------------------------------------ #
    # Persistência
    # ------------------------------------------------------------------ #
    @staticmethod
    def salvar_modelo_pickle(modelo, caminho_arquivo):
        with open(caminho_arquivo, 'wb') as arquivo:
            pickle.dump(modelo, arquivo)
        print(f"Modelo salvo em {caminho_arquivo}")

    @staticmethod
    def carregar_modelo_pickle(caminho_arquivo):
        with open(caminho_arquivo, 'rb') as arquivo:
            modelo_carregado = pickle.load(arquivo)
        print(f"Modelo carregado de {caminho_arquivo}")
        return modelo_carregado

    # ------------------------------------------------------------------ #
    # Treino multi-algoritmo — CLASSIFICADOR
    # ------------------------------------------------------------------ #
    @staticmethod
    def Classificador(classificador, x_train, y_train, x_test, y_test,
                       class_weight=None, parametros=None, preprocessador=None):
        """Treina Regressão Logística, Random Forest ou LightGBM."""
        nome = str(classificador).strip().lower().replace("_", " ")
        params = dict(parametros or {})
        if nome in {"regressão logística", "regressao logistica", "logistic regression", "logistica"}:
            base = dict(C=1.0, penalty='l2', solver='lbfgs', tol=1e-4,
                        max_iter=2000, random_state=42,
                        class_weight="balanced" if class_weight is None else {0: 1, 1: class_weight})
            base.update(params); modelo = LogisticRegression(**base)
        elif nome in {"random forest", "randomforest"}:
            base = dict(n_estimators=300, criterion='gini', max_depth=None,
                        min_samples_split=2, min_samples_leaf=1,
                        max_features='sqrt', bootstrap=True,
                        random_state=42, n_jobs=-1,
                        class_weight="balanced" if class_weight is None else {0: 1, 1: class_weight})
            base.update(params); modelo = RandomForestClassifier(**base)
        elif nome in {"lightgbm", "lgbm"}:
            if LGBMClassifier is None: raise ImportError("Instale lightgbm para usar este classificador.")
            base = dict(objective="binary", boosting_type='gbdt',
                        n_estimators=300, learning_rate=0.03,
                        num_leaves=31, max_depth=-1, min_child_samples=20,
                        min_child_weight=1e-3, min_split_gain=0.0,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                        feature_fraction_bynode=1.0,
                        reg_alpha=0.0, reg_lambda=0.0, max_bin=255,
                        max_delta_step=0.0, path_smooth=0.0, extra_trees=False,
                        deterministic=True, force_col_wise=True,
                        importance_type='gain', random_state=42, n_jobs=-1, verbosity=-1)
            if class_weight is not None: base["scale_pos_weight"] = class_weight
            base.update(params); modelo = LGBMClassifier(**base)
        else:
            raise ValueError("Use Regressão Logística, Random Forest ou LightGBM.")
        if preprocessador is not None:
            modelo = Pipeline([("preprocessamento", clone(preprocessador)), ("modelo", modelo)])
        y_train_array = np.asarray(y_train).ravel(); y_test_array = np.asarray(y_test).ravel()
        modelo.fit(x_train, y_train_array)
        return (modelo, modelo.predict(x_train), modelo.predict(x_test),
                modelo.predict_proba(x_train), modelo.predict_proba(x_test))

    # ------------------------------------------------------------------ #
    # Treino multi-loss — REGRESSOR (espelha o Classificador)
    # ------------------------------------------------------------------ #
    @staticmethod
    def Regressor(regressor, x_train, y_train, x_test, y_test, parametros=None,
                  loss_function='RMSE', preprocessador=None):
        """Treina Regressão Linear, Random Forest ou LightGBM.

        Por compatibilidade, ``Regressor('RMSE', ...)`` e as demais losses
        históricas continuam sendo interpretadas como LightGBM.
        """
        cols = list(x_train.columns)
        x_train, x_test = x_train[cols], x_test[cols]
        nome = str(regressor).strip().lower().replace('_', ' ')
        losses = {'mae': 'MAE', 'rmse': 'RMSE', 'huber': 'Huber',
                  'rmsle': 'RMSLE', 'gamma': 'Gamma'}
        if nome in losses:
            loss_function = losses[nome]
            nome = 'lightgbm'
        loss_function = str(loss_function).strip().upper()

        if nome in {'linear', 'regressao linear', 'regressão linear', 'linear regression'}:
            base = dict(fit_intercept=True, copy_X=True, n_jobs=-1, positive=False)
            base.update(parametros or {})
            model = LinearRegression(**base)
        elif nome in {'random forest', 'randomforest', 'random forest regressor'}:
            base = dict(
                n_estimators=500, criterion='squared_error', max_depth=None,
                min_samples_split=2, min_samples_leaf=1, max_features=1.0,
                max_leaf_nodes=None, min_impurity_decrease=0.0,
                bootstrap=True, max_samples=None, random_state=42, n_jobs=-1,
            )
            base.update(parametros or {})
            model = RandomForestRegressor(**base)
        elif nome not in {'lightgbm', 'lgbm'}:
            raise ValueError("Use Regressão Linear, Random Forest ou LightGBM.")
        else:
            if LGBMRegressor is None:
                raise ImportError('Instale lightgbm para usar este regressor.')

            base_params = dict(
                verbosity=-1, random_state=42, boosting_type='gbdt',
                importance_type='gain', n_estimators=300, learning_rate=0.05,
                num_leaves=31, max_depth=-1, min_child_samples=20,
                min_child_weight=1e-3, min_split_gain=0.0,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                feature_fraction_bynode=1.0,
                reg_alpha=0.0, reg_lambda=0.0, max_bin=255,
                max_delta_step=0.0, path_smooth=0.0, extra_trees=False,
                deterministic=True, force_col_wise=True, n_jobs=-1,
            )
            base_params.update(parametros or {})
            configuracoes = {
                'MAE': dict(objective='regression_l1', metric='l1'),
                'RMSE': dict(objective='regression', metric='l2'),
                'HUBER': dict(objective='huber', metric='l1'),
                'RMSLE': dict(objective='regression', metric='l2'),
                'GAMMA': dict(objective='gamma', metric='gamma'),
            }
            if loss_function not in configuracoes:
                raise ValueError(f"Loss function '{loss_function}' não suportada.")
            config = configuracoes[loss_function]
            model = LGBMRegressor(**config, **base_params)

        y_train_array = np.asarray(y_train).ravel()
        if nome == 'lightgbm' and loss_function == "RMSLE" and np.any(y_train_array < 0):
            raise ValueError("RMSLE exige target não negativo.")
        if nome == 'lightgbm' and loss_function == "GAMMA" and np.any(y_train_array <= 0):
            raise ValueError("Regressão Gamma exige target estritamente positivo.")
        if preprocessador is not None:
            model = Pipeline([('preprocessamento', clone(preprocessador)), ('modelo', model)])
        if nome == 'lightgbm' and loss_function == "RMSLE":
            y_train_transformed = np.log1p(y_train)
            model.fit(x_train, y_train_transformed)
            y_pred_train = np.maximum(np.expm1(model.predict(x_train)), 0)
            y_pred_test = np.maximum(np.expm1(model.predict(x_test)), 0)
        else:
            model.fit(x_train, y_train)
            y_pred_train = model.predict(x_train)
            y_pred_test = model.predict(x_test)

        return model, y_pred_train, y_pred_test

    # ------------------------------------------------------------------ #
    # Otimização Hyperopt — CLASSIFICADOR
    # (agora com retorno padronizado, igual ao da regressão)
    # ------------------------------------------------------------------ #
    @staticmethod
    def otimizacao_hyperopt_classificacao(x_train, y_train, x_valid, y_valid,
                                           max_evals=30, class_weight=None,
                                           parametros_fixos=None):
        """Otimiza um LGBMClassifier via Hyperopt maximizando PR-AUC de
        validação com penalização de overfitting (gap treino-validação).

        Returns
        -------
        modelo, y_pred_train, y_pred_test, y_proba_train, y_proba_test,
        hiperparametros (DataFrame), trials
        — mesmo formato de `otimizacao_hyperopt_regressao`.
        """
        espaco = {
            "n_estimators": hp.quniform("n_estimators", 150, 1000, 25),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.15)),
            "num_leaves": hp.quniform("num_leaves", 15, 127, 2),
            "max_depth": hp.quniform("max_depth", 3, 12, 1),
            "min_child_samples": hp.quniform("min_child_samples", 10, 200, 5),
            "min_child_weight": hp.loguniform("min_child_weight", np.log(1e-3), np.log(10)),
            "min_split_gain": hp.uniform("min_split_gain", 0, 2),
            "subsample": hp.uniform("subsample", 0.6, 1),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
            "feature_fraction_bynode": hp.uniform("feature_fraction_bynode", 0.5, 1),
            "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-4), np.log(20)),
            "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-4), np.log(30)),
            "max_bin": hp.quniform("max_bin", 63, 511, 16),
            "max_delta_step": hp.uniform("max_delta_step", 0, 10),
            "path_smooth": hp.uniform("path_smooth", 0, 10),
            "extra_trees": hp.quniform("extra_trees", 0, 1, 1),
        }
        inteiros = ["n_estimators", "num_leaves", "max_depth", "min_child_samples", "max_bin"]

        def converte(parametros):
            parametros = parametros.copy()
            for coluna in inteiros:
                parametros[coluna] = int(parametros[coluna])
            if 'extra_trees' in parametros:
                parametros['extra_trees'] = bool(int(parametros['extra_trees']))
            return parametros

        def constroi_modelo(parametros):
            base = dict(objective="binary", boosting_type='gbdt', subsample_freq=1,
                        importance_type='gain', random_state=42, n_jobs=-1, verbosity=-1)
            if class_weight is not None:
                base['scale_pos_weight'] = class_weight
            base.update(parametros_fixos or {})
            base.update(parametros)
            return LGBMClassifier(**base)

        def objetivo(parametros):
            parametros = converte(parametros)
            modelo = constroi_modelo(parametros).fit(x_train, y_train, categorical_feature="auto")
            proba_train = modelo.predict_proba(x_train)[:, 1]
            proba_valid = modelo.predict_proba(x_valid)[:, 1]
            ap_valid = average_precision_score(y_valid, proba_valid)
            ap_train = average_precision_score(y_train, proba_train)
            gini_train = 2 * roc_auc_score(y_train, proba_train) - 1
            gini_valid = 2 * roc_auc_score(y_valid, proba_valid) - 1
            gap_excessivo = max(0, gini_train - gini_valid - 0.05)
            gap_ap_excessivo = max(0, ap_train - ap_valid - 0.08)
            perda = -ap_valid + 0.25 * gap_excessivo + 0.75 * gap_ap_excessivo
            return {"loss": perda, "status": STATUS_OK}

        trials = Trials()
        best = fmin(objetivo, espaco, algo=tpe.suggest, max_evals=max_evals,
                    trials=trials, rstate=np.random.default_rng(42))
        best = converte(best)

        modelo_final = constroi_modelo(best).fit(x_train, y_train, categorical_feature="auto")
        y_pred_train, y_pred_test = modelo_final.predict(x_train), modelo_final.predict(x_valid)
        y_proba_train, y_proba_test = modelo_final.predict_proba(x_train), modelo_final.predict_proba(x_valid)
        hiperparametros = pd.DataFrame([best])

        return modelo_final, y_pred_train, y_pred_test, y_proba_train, y_proba_test, hiperparametros, trials

    # ------------------------------------------------------------------ #
    # Otimização Hyperopt — REGRESSOR
    # ------------------------------------------------------------------ #
    @staticmethod
    def otimizacao_hyperopt_regressao(x_train, y_train, x_valid, y_valid,
                                      max_evals=30, metrica='rmse',
                                      objective='regression', parametros_fixos=None):
        """Otimiza no treino e usa exclusivamente uma amostra de validação externa.

        Não passe o teste final em ``x_valid``/``y_valid``; ele deve permanecer
        intocado até a avaliação definitiva.
        """
        metrica = str(metrica).lower()
        if metrica not in {'rmse', 'mae', 'rmsle'}:
            raise ValueError("metrica deve ser 'rmse', 'mae' ou 'rmsle'.")
        y_otimizacao = np.asarray(y_train)
        if objective == 'gamma' and np.any(y_otimizacao <= 0):
            raise ValueError('Objetivo Gamma exige target estritamente positivo.')
        if metrica == 'rmsle' and np.any(y_otimizacao < 0):
            raise ValueError('RMSLE exige target não negativo.')
        search_space = {
            'n_estimators': hp.quniform('reg_n_estimators', 150, 1200, 25),
            'max_depth': hp.quniform('reg_max_depth', 3, 15, 1),
            'learning_rate': hp.loguniform('reg_learning_rate', np.log(0.01), np.log(0.15)),
            'max_bin': hp.quniform('reg_max_bin', 63, 511, 16),
            'reg_alpha': hp.loguniform('reg_alpha_search', np.log(1e-4), np.log(20)),
            'reg_lambda': hp.loguniform('reg_lambda_search', np.log(1e-4), np.log(30)),
            'min_split_gain': hp.uniform('reg_min_split_gain', 0, 2),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'feature_fraction_bynode': hp.uniform('reg_feature_fraction_bynode', 0.5, 1),
            'subsample': hp.uniform('reg_subsample', 0.6, 1),
            'num_leaves': hp.quniform('reg_num_leaves', 15, 127, 2),
            'min_child_samples': hp.quniform('reg_min_child_samples', 10, 200, 5),
            'min_child_weight': hp.loguniform('reg_min_child_weight', np.log(1e-3), np.log(10)),
            'max_delta_step': hp.uniform('reg_max_delta_step', 0, 10),
            'path_smooth': hp.uniform('reg_path_smooth', 0, 10),
            'extra_trees': hp.quniform('reg_extra_trees', 0, 1, 1),
        }

        inteiros = {'n_estimators', 'max_depth', 'max_bin', 'num_leaves', 'min_child_samples'}

        def converte(params):
            convertido = params.copy()
            for nome in inteiros:
                convertido[nome] = int(convertido[nome])
            if 'extra_trees' in convertido:
                convertido['extra_trees'] = bool(int(convertido['extra_trees']))
            return convertido

        def calcula_erro(y_real, pred):
            if metrica == 'mae':
                return mean_absolute_error(y_real, pred)
            if metrica == 'rmsle':
                return np.sqrt(mean_squared_log_error(y_real, np.maximum(pred, 0)))
            return np.sqrt(mean_squared_error(y_real, pred))

        def funcao_objetivo(params):
            params = converte(params)
            X_tr, X_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
            base = dict(
                verbosity=-1, random_state=42, boosting_type='gbdt', importance_type='gain',
                objective=objective, subsample_freq=1, n_jobs=-1,
            )
            base.update(parametros_fixos or {}); base.update(params)
            model = LGBMRegressor(**base)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="rmse",
                      callbacks=[early_stopping(stopping_rounds=20, verbose=False)])
            preds = model.predict(X_val)
            score = calcula_erro(y_val, preds)
            return {'loss': score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=funcao_objetivo, space=search_space, algo=tpe.suggest,
                    max_evals=max_evals, trials=trials, rstate=np.random.default_rng(42))
        best = converte({
            'n_estimators': best['reg_n_estimators'], 'max_depth': best['reg_max_depth'],
            'learning_rate': best['reg_learning_rate'], 'max_bin': best['reg_max_bin'],
            'reg_alpha': best['reg_alpha_search'], 'reg_lambda': best['reg_lambda_search'],
            'min_split_gain': best['reg_min_split_gain'], 'colsample_bytree': best['colsample_bytree'],
            'feature_fraction_bynode': best['reg_feature_fraction_bynode'],
            'subsample': best['reg_subsample'], 'num_leaves': best['reg_num_leaves'],
            'min_child_samples': best['reg_min_child_samples'],
            'min_child_weight': best['reg_min_child_weight'],
            'max_delta_step': best['reg_max_delta_step'], 'path_smooth': best['reg_path_smooth'],
            'extra_trees': best['reg_extra_trees'],
        })
        base_final = dict(
            verbosity=-1, random_state=42, boosting_type='gbdt', importance_type='gain',
            objective=objective, subsample_freq=1, n_jobs=-1,
        )
        base_final.update(parametros_fixos or {}); base_final.update(best)
        final_model = LGBMRegressor(**base_final)
        final_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric='rmse',
                         callbacks=[early_stopping(stopping_rounds=20, verbose=False)])

        y_pred_train, y_pred_test = final_model.predict(x_train), final_model.predict(x_valid)
        hiperparametros = pd.DataFrame([best])

        return final_model, y_pred_train, y_pred_test, hiperparametros, trials

    # ------------------------------------------------------------------ #
    # Calibração — CLASSIFICADOR (probabilidade) x REGRESSOR (intervalo)
    # ------------------------------------------------------------------ #
    @staticmethod
    def calibracao_probabilidade(modelo, x_calibracao, y_calibracao,
                                  metodo="isotonic", cv="prefit", n_jobs=-1):
        """Calibra probabilidades por isotonic ou sigmoid (Platt scaling)."""
        if metodo not in {"isotonic", "sigmoid"}:
            raise ValueError("metodo deve ser 'isotonic' ou 'sigmoid'.")
        calibrador = CalibratedClassifierCV(estimator=modelo, method=metodo, cv=cv, n_jobs=n_jobs)
        calibrador.fit(x_calibracao, np.asarray(y_calibracao).ravel())
        return calibrador

    @staticmethod
    def calibra_intervalo_predicao_regressao(modelo, x_calibracao, y_calibracao, alpha=0.1):
        """Equivalente regressivo de `calibracao_probabilidade`: em vez de
        calibrar uma probabilidade, calibra um INTERVALO de predição via
        conformal prediction split (residual quantile) — o mesmo conceito
        de "confiar no que o modelo diz", adaptado a saída contínua.

        Returns
        -------
        dict com:
          - 'margem': valor fixo somado/subtraído da predição pontual para
            formar o intervalo com cobertura ~ (1 - alpha).
          - 'predict_interval': função(x) -> (y_pred, limite_inferior, limite_superior)
        """
        y_calibracao = np.asarray(y_calibracao).ravel()
        y_pred_calibracao = np.asarray(modelo.predict(x_calibracao)).ravel()
        residuos_absolutos = np.abs(y_calibracao - y_pred_calibracao)
        if not 0 < alpha < 1:
            raise ValueError('alpha deve estar entre 0 e 1.')
        n = len(residuos_absolutos)
        if n == 0:
            raise ValueError('A amostra de calibração está vazia.')
        quantil = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        margem = float(np.quantile(residuos_absolutos, quantil, method='higher'))

        def predict_interval(x):
            y_pred = np.asarray(modelo.predict(x)).ravel()
            return y_pred, y_pred - margem, y_pred + margem

        return {"margem": margem, "predict_interval": predict_interval}

    @staticmethod
    def treinar_skpro_residual_double(modelo_base, x_train, y_train,
                                      modelo_residuos=None, distr_type='Normal',
                                      cv=None, **parametros):
        """Treina regressão probabilística SKPRO pelo método ResidualDouble."""
        if ResidualDouble is None:
            raise ImportError('Instale skpro para usar ResidualDouble: pip install skpro')
        modelo = ResidualDouble(
            estimator=clone(modelo_base),
            estimator_resid=clone(modelo_residuos) if modelo_residuos is not None else None,
            distr_type=distr_type,
            cv=cv,
            **parametros,
        )
        modelo.fit(X=x_train, y=np.asarray(y_train).ravel())
        return modelo

    @staticmethod
    def predizer_intervalos_skpro(modelo, x, cobertura=0.90):
        """Retorna predição e intervalo de uma distribuição preditiva SKPRO."""
        if not 0 < cobertura < 1:
            raise ValueError('cobertura deve estar entre 0 e 1.')
        distribuicao = modelo.predict_proba(x)
        alpha = 1 - cobertura
        predicao = np.asarray(distribuicao.mean()).ravel()
        inferior = np.asarray(distribuicao.ppf(alpha / 2)).ravel()
        superior = np.asarray(distribuicao.ppf(1 - alpha / 2)).ravel()
        return pd.DataFrame({
            'y_pred': predicao,
            'ic_inferior': inferior,
            'ic_superior': superior,
            'cobertura_nominal': cobertura,
        }, index=getattr(x, 'index', None))

    @staticmethod
    def treinar_mapie_split_conformal(modelo_base, x_train, y_train,
                                      x_calibracao, y_calibracao,
                                      cobertura=0.90, **parametros):
        """Treina e calibra um SplitConformalRegressor do MAPIE.

        A separação treino/calibração deve ser feita antes da chamada para
        impedir vazamento. A função aceita as APIs recentes do MAPIE.
        """
        if SplitConformalRegressor is None:
            raise ImportError('Instale MAPIE para usar regressão conformal: pip install mapie')
        if not 0 < cobertura < 1:
            raise ValueError('cobertura deve estar entre 0 e 1.')
        y_treino = np.asarray(y_train).ravel()
        y_cal = np.asarray(y_calibracao).ravel()
        try:
            conformal = SplitConformalRegressor(
                estimator=clone(modelo_base), confidence_level=cobertura,
                prefit=False, **parametros,
            )
            conformal.fit(x_train, y_treino)
            conformal.conformalize(x_calibracao, y_cal)
        except TypeError:
            # Compatibilidade com versões que não expõem ``prefit``.
            conformal = SplitConformalRegressor(
                estimator=clone(modelo_base), confidence_level=cobertura,
                **parametros,
            )
            if hasattr(conformal, 'fit'):
                conformal.fit(x_train, y_treino)
            conformal.conformalize(x_calibracao, y_cal)
        return conformal

    @staticmethod
    def predizer_intervalos_mapie(modelo, x, cobertura=0.90):
        """Normaliza a saída do MAPIE para y_pred/ic_inferior/ic_superior."""
        resultado = modelo.predict_interval(x)
        if not isinstance(resultado, tuple) or len(resultado) != 2:
            raise TypeError('A versão do MAPIE retornou um formato de intervalo não reconhecido.')
        predicao, intervalos = resultado
        intervalos = np.asarray(intervalos)
        if intervalos.ndim == 3:
            intervalos = intervalos[:, :, 0]
        if intervalos.ndim != 2 or intervalos.shape[1] != 2:
            raise ValueError('Esperava intervalos com formato (n_amostras, 2).')
        return pd.DataFrame({
            'y_pred': np.asarray(predicao).ravel(),
            'ic_inferior': intervalos[:, 0],
            'ic_superior': intervalos[:, 1],
            'cobertura_nominal': cobertura,
        }, index=getattr(x, 'index', None))

    # ------------------------------------------------------------------ #
    # Predição utilitária — CLASSIFICADOR x REGRESSOR
    # ------------------------------------------------------------------ #
    @staticmethod
    def predizer_probabilidade(modelo, x, classe_positiva=1):
        """Retorna somente a probabilidade da classe positiva."""
        if not hasattr(modelo, "predict_proba"):
            raise TypeError("O modelo não implementa predict_proba.")
        return np.asarray(modelo.predict_proba(x))[:, classe_positiva]

    @staticmethod
    def predizer(modelo, x):
        """Equivalente regressivo de `predizer_probabilidade`: retorna a
        predição pontual do modelo."""
        return np.asarray(modelo.predict(x)).ravel()

    # ------------------------------------------------------------------ #
    # Validação cruzada — CLASSIFICADOR x REGRESSOR
    # ------------------------------------------------------------------ #
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
                                    scoring=metricas, n_jobs=n_jobs, return_train_score=True)
        detalhe = pd.DataFrame(resultado)
        resumo = pd.DataFrame({
            "Metrica": list(metricas),
            "Treino_media": [detalhe[f"train_{m}"].mean() for m in metricas],
            "Treino_std": [detalhe[f"train_{m}"].std() for m in metricas],
            "Validacao_media": [detalhe[f"test_{m}"].mean() for m in metricas],
            "Validacao_std": [detalhe[f"test_{m}"].std() for m in metricas],
        })
        return detalhe, resumo

    @staticmethod
    def validacao_cruzada_regressao(modelo, x, y, n_splits=5, scoring=None,
                                     shuffle=True, random_state=42, n_jobs=-1):
        """Equivalente regressivo de `validacao_cruzada_classificacao`
        — mesma estrutura de saída (detalhe por fold + resumo)."""
        metricas = scoring or {
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2",
        }
        cv = KFold(n_splits=n_splits, shuffle=shuffle,
                   random_state=random_state if shuffle else None)
        resultado = cross_validate(modelo, x, np.asarray(y).ravel(), cv=cv,
                                    scoring=metricas, n_jobs=n_jobs, return_train_score=True)
        detalhe = pd.DataFrame(resultado)
        # Scorers de erro do sklearn são negativos por convenção; expomos
        # MAE/RMSE positivos para evitar relatórios contraintuitivos.
        for nome, scorer in metricas.items():
            if isinstance(scorer, str) and scorer.startswith('neg_'):
                detalhe[f'train_{nome}'] = -detalhe[f'train_{nome}']
                detalhe[f'test_{nome}'] = -detalhe[f'test_{nome}']
        resumo = pd.DataFrame({
            "Metrica": list(metricas),
            "Treino_media": [detalhe[f"train_{m}"].mean() for m in metricas],
            "Treino_std": [detalhe[f"train_{m}"].std() for m in metricas],
            "Validacao_media": [detalhe[f"test_{m}"].mean() for m in metricas],
            "Validacao_std": [detalhe[f"test_{m}"].std() for m in metricas],
        })
        return detalhe, resumo
    
# ------------------------------------------------------------------ #
    # Calibração de intervalo — REGRESSOR (skpro Residual Double)
    # ------------------------------------------------------------------ #
    @staticmethod
    def skpro_residual_double(target, x_train, y_train, x_valid, y_valid, x_test, y_test,
                               x_oot, y_oot, modelo_otimizado, caminho_modelo,
                               alpha=0.1, salvar=None):

        if salvar:
            # ===============================
            # RESIDUAL DOUBLE (API ANTIGA)
            # ===============================
            calibrador_skpro = ResidualDouble(modelo_otimizado)
            calibrador_skpro.fit(X=x_train, y=y_train[target].values)
            Modelagem.salvar_modelo_pickle(calibrador_skpro, caminho_modelo)

        else:
            calibrador_skpro = Modelagem.carregar_modelo_pickle(caminho_modelo)

            # ===============================
            # FUNÇÃO DE IC
            # ===============================
            def predict_interval(X):
                # Retorna um objeto de distribuição
                dist = calibrador_skpro.predict_proba(X)

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
                "train": {"y_pred": mu_train, "ic_inferior": lower_train, "ic_superior": upper_train},
                "valid": {"y_pred": mu_valid, "ic_inferior": lower_valid, "ic_superior": upper_valid},
                "test":  {"y_pred": mu_test,  "ic_inferior": lower_test,  "ic_superior": upper_test},
                "oot":   {"y_pred": mu_oot,   "ic_inferior": lower_oot,   "ic_superior": upper_oot},
            }

    # ------------------------------------------------------------------ #
    # Inferência Causal (agora dentro de Modelagem, não em classe à parte)
    # ------------------------------------------------------------------ #
    @staticmethod
    def propensity_score_matching(df, treatment, outcome, covariates,
                                   caliper=.2, replace=False):
        """Pareamento 1:1 no logit do propensity score; retorna pares e balanço SMD."""
        base = df[[treatment, outcome, *covariates]].dropna().copy()
        t = base[treatment].astype(int)
        modelo = LogisticRegression(max_iter=2000).fit(base[covariates], t)
        ps = np.clip(modelo.predict_proba(base[covariates])[:, 1], 1e-6, 1 - 1e-6)
        base["propensity_score"] = ps
        base["propensity_logit"] = np.log(ps / (1 - ps))
        tratados, controles = base[t.eq(1)], base[t.eq(0)]
        if tratados.empty or controles.empty:
            raise ValueError('A base deve conter tratados (1) e controles (0).')
        n_vizinhos = 1 if replace else len(controles)
        nn = NearestNeighbors(n_neighbors=n_vizinhos).fit(controles[["propensity_logit"]])
        dist, idx = nn.kneighbors(tratados[["propensity_logit"]])
        usados, pares = set(), []
        limite = None if caliper is None else caliper * base.propensity_logit.std()
        for posicao, (i, row) in enumerate(tratados.iterrows()):
            candidato = next(((d, j) for d, j in zip(dist[posicao], idx[posicao])
                              if replace or controles.index[j] not in usados), None)
            if candidato is None:
                continue
            d, j = candidato
            controle_idx = controles.index[j]
            if limite is not None and d > limite:
                continue
            usados.add(controle_idx)
            pares.append({
                "treated_index": i, "control_index": controle_idx, "distance": d,
                "outcome_treated": row[outcome], "outcome_control": controles.loc[controle_idx, outcome],
            })
        matched = pd.DataFrame(pares)
        if matched.empty:
            raise ValueError("Nenhum par encontrado; revise caliper e covariáveis.")
        ti, ci = matched.treated_index, matched.control_index

        def smd_arrays(a, b):
            pooled = np.sqrt((a.var() + b.var()) / 2)
            return (a.mean() - b.mean()) / pooled if pooled else 0.

        diagnostico = {
            "att": float((matched.outcome_treated - matched.outcome_control).mean()),
            "n_pairs": len(matched),
            "smd_before": {c: float(smd_arrays(base.loc[t.eq(1), c], base.loc[t.eq(0), c])) for c in covariates},
            "smd_after": {c: float(smd_arrays(base.loc[ti, c], base.loc[ci, c])) for c in covariates},
            "model": modelo,
            "propensity_data": base[[treatment, "propensity_score"]].copy(),
        }
        return matched, diagnostico

    @staticmethod
    def diferencas_em_diferencas(df, outcome, treatment, post, covariates=None):
        controles = " + " + " + ".join(covariates) if covariates else ""
        modelo = smf.ols(f"{outcome} ~ {treatment} + {post} + {treatment}:{post}{controles}", data=df).fit(cov_type="HC1")
        termo = f"{treatment}:{post}"
        ci = modelo.conf_int().loc[termo]
        return CausalResult(modelo.params[termo], modelo.bse[termo], modelo.pvalues[termo], tuple(ci), modelo,
                             {"assumption": "tendências paralelas antes do tratamento"})

    @staticmethod
    def regressao_descontinua(df, outcome, running, cutoff, bandwidth=None, polynomial_order=1):
        base = df[[outcome, running]].dropna().copy()
        base["running_centered"] = base[running] - cutoff
        if bandwidth is not None:
            base = base[base.running_centered.abs() <= bandwidth]
        base["treated"] = (base.running_centered >= 0).astype(int)
        termos = [f"I(running_centered ** {p}) + treated:I(running_centered ** {p})" for p in range(1, polynomial_order + 1)]
        modelo = smf.ols(f"{outcome} ~ treated + " + " + ".join(termos), data=base).fit(cov_type="HC1")
        ci = modelo.conf_int().loc["treated"]
        return CausalResult(modelo.params.treated, modelo.bse.treated, modelo.pvalues.treated, tuple(ci), modelo,
                             {"n": len(base), "cutoff": cutoff, "bandwidth": bandwidth,
                              "assumption": "não manipulação e continuidade ao redor do cutoff"})

    @staticmethod
    def variavel_instrumental(df, outcome, treatment, instrument, covariates=None):
        controles = covariates or []
        base = df[[outcome, treatment, instrument, *controles]].dropna().copy()
        exog = sm.add_constant(base[[treatment, *controles]], has_constant="add")
        instrumentos = sm.add_constant(base[[instrument, *controles]], has_constant="add")
        second = IV2SLS(base[outcome], exog, instrumentos).fit()
        first = sm.OLS(base[treatment], instrumentos).fit()
        ci = second.conf_int().loc[treatment]
        return CausalResult(second.params[treatment], second.bse[treatment], second.pvalues[treatment], tuple(ci), second,
                             {"first_stage_f": float(first.f_test(f"{instrument} = 0").fvalue),
                              "assumption": "instrumento relevante, independente e sem efeito direto no desfecho"})


# Aliases funcionais para compatibilidade com notebooks antigos.
salvar_modelo_pickle = Modelagem.salvar_modelo_pickle
carregar_modelo_pickle = Modelagem.carregar_modelo_pickle
Regressor = Modelagem.Regressor
Classificador = Modelagem.Classificador
otimizacao_hyperopt_regressao = Modelagem.otimizacao_hyperopt_regressao
otimizacao_hyperopt_classificacao = Modelagem.otimizacao_hyperopt_classificacao
calibracao_probabilidade = Modelagem.calibracao_probabilidade
calibra_intervalo_predicao_regressao = Modelagem.calibra_intervalo_predicao_regressao
treinar_skpro_residual_double = Modelagem.treinar_skpro_residual_double
predizer_intervalos_skpro = Modelagem.predizer_intervalos_skpro
treinar_mapie_split_conformal = Modelagem.treinar_mapie_split_conformal
predizer_intervalos_mapie = Modelagem.predizer_intervalos_mapie
predizer_probabilidade = Modelagem.predizer_probabilidade
predizer = Modelagem.predizer
validacao_cruzada_classificacao = Modelagem.validacao_cruzada_classificacao
validacao_cruzada_regressao = Modelagem.validacao_cruzada_regressao
skpro_residual_double = Modelagem.skpro_residual_double
propensity_score_matching = Modelagem.propensity_score_matching
diferencas_em_diferencas = Modelagem.diferencas_em_diferencas
regressao_descontinua = Modelagem.regressao_descontinua
variavel_instrumental = Modelagem.variavel_instrumental
