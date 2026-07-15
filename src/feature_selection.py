"""Métodos generalistas selecionados do arquivo original."""
from ._shared import *

class SelecaoFeatures:
    @staticmethod
    def remove_features_baixa_variancia(target, df, threshold):
        target_column = df[target]
        features = df.drop(target, axis=1)

        selector = VarianceThreshold(threshold=threshold)
        features_filtered = selector.fit_transform(features)

        feature_indices = selector.get_support(indices=True)
        selected_features = features.columns[feature_indices]
        selected_features = selected_features.append(pd.Index([target]))

        return selected_features.tolist()

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def compute_shap_importance_df(model, X, max_display=30, threshold=0.0):

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        valores = shap_values
        if isinstance(valores, list):
            valores_importancia = np.stack([np.asarray(v) for v in valores], axis=0)
            mean_abs_shap = np.abs(valores_importancia).mean(axis=(0, 1))
        else:
            valores_importancia = np.asarray(valores)
            if valores_importancia.ndim == 3:
                # Suporta tanto (amostra, feature, classe) quanto
                # (classe, amostra, feature).
                eixo_feature = 1 if valores_importancia.shape[1] == X.shape[1] else 2
                eixos_media = tuple(i for i in range(3) if i != eixo_feature)
                mean_abs_shap = np.abs(valores_importancia).mean(axis=eixos_media)
            else:
                mean_abs_shap = np.abs(valores_importancia).mean(axis=0)

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

        total = importance_df["importance"].sum()
        importance_df["importance_pct"] = (importance_df["importance"] / total * 100) if total else 0.0

        return importance_df, shap_values

    @staticmethod
    def seleciona_shap_lgbm_dummy(x, y, tipo='classificacao', n_dummies=1, semente=42, **lgbm_params):
        """Combina Feature Importance nativa do LGBM (gain) com SHAP (média
        absoluta), ambas normalizadas em percentual, e filtra apenas as
        variáveis cuja importância combinada supera a de uma (ou mais)
        variável dummy puramente aleatória — um corte mais defensável do
        que um threshold arbitrário, pois a dummy representa o "ruído" que
        o próprio modelo capta por acaso.

        Parameters
        ----------
        x, y : features e target de treino.
        tipo : 'classificacao' ou 'regressao' -> escolhe LGBMClassifier/Regressor.
        n_dummies : quantas colunas dummy aleatórias adicionar (mais de uma
            deixa o corte mais robusto a variação amostral da dummy).
        lgbm_params : hiperparâmetros extras repassados ao LGBM.

        Returns
        -------
        features_selecionadas : list[str]
        importancia : DataFrame com a importância de cada feature (e das
            dummies, para auditoria) ordenado de forma decrescente.
        """
        rng = np.random.default_rng(semente)
        x_aux = x.copy()
        colunas_dummy = []
        for i in range(n_dummies):
            nome_dummy = f"_dummy_aleatoria_{i + 1}"
            x_aux[nome_dummy] = rng.normal(size=len(x_aux))
            colunas_dummy.append(nome_dummy)

        if tipo == 'classificacao':
            modelo = LGBMClassifier(random_state=semente, verbosity=-1, **lgbm_params)
        elif tipo == 'regressao':
            modelo = LGBMRegressor(random_state=semente, verbosity=-1, **lgbm_params)
        else:
            raise ValueError("tipo deve ser 'classificacao' ou 'regressao'.")

        modelo.fit(x_aux, y)

        # Feature Importance nativa (gain), normalizada em %
        fi = pd.Series(modelo.feature_importances_, index=x_aux.columns)
        fi_pct = (fi / fi.sum() * 100).rename('lgbm_importance_pct')

        # SHAP (média absoluta), normalizada em %
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(x_aux)
        if isinstance(shap_values, list):  # saída multi-classe do TreeExplainer
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        shap_abs = np.abs(shap_values).mean(axis=0)
        shap_pct = pd.Series(shap_abs, index=x_aux.columns)
        shap_pct = (shap_pct / shap_pct.sum() * 100).rename('shap_importance_pct')

        importancia = pd.concat([fi_pct, shap_pct], axis=1)
        importancia['importancia_combinada'] = importancia[
            ['lgbm_importance_pct', 'shap_importance_pct']
        ].mean(axis=1)
        importancia = (
            importancia
            .sort_values('importancia_combinada', ascending=False)
            .reset_index()
            .rename(columns={'index': 'feature'})
        )
        importancia['eh_dummy'] = importancia['feature'].isin(colunas_dummy)

        limite_dummy = importancia.loc[importancia['eh_dummy'], 'importancia_combinada'].max()

        features_selecionadas = importancia.loc[
            (~importancia['eh_dummy']) & (importancia['importancia_combinada'] > limite_dummy),
            'feature'
        ].tolist()

        return features_selecionadas, importancia

    @staticmethod
    def rfe_por_metrica(
        model_factory,
        x_train, y_train, x_valid, y_valid,
        metric_func,
        maior_e_melhor=True,
        tolerancia=0.0,
        min_features=1,
        usa_proba=False,
        verbose=True,
    ):
        """RFE (Recursive Feature Elimination) guiado por uma métrica de
        interesse: a cada rodada remove a feature menos importante do
        modelo atual e só efetiva a remoção se a métrica na base de
        validação não piorar além de `tolerancia`. Para assim que remover
        a próxima feature deixaria de valer a pena.

        Parameters
        ----------
        model_factory : callable
            Função sem argumentos que retorna uma NOVA instância do modelo
            a cada chamada (ex.: `lambda: LGBMClassifier(...)`). O modelo
            precisa expor `feature_importances_` após o `.fit`.
        x_train, y_train, x_valid, y_valid : bases de treino/validação.
        metric_func : callable(y_true, y_pred) -> float
            Métrica de interesse (ex.: roc_auc_score, mean_absolute_error,
            uma função de KS ou de retorno financeiro).
        maior_e_melhor : bool
            True para métricas tipo AUC/KS/retorno (quanto maior, melhor);
            False para métricas tipo erro (RMSE/MAE, quanto menor, melhor).
        tolerancia : float
            Margem de piora aceitável na métrica para ainda assim manter a
            remoção (0.0 = só remove se a métrica não piorar nada).
        min_features : int
            Número mínimo de features a manter, independente da métrica.
        usa_proba : bool
            True se `metric_func` espera probabilidade (`predict_proba`,
            classe positiva) em vez de classe/predição pontual.

        Returns
        -------
        features_finais : list[str]
        historico : DataFrame com o histórico de score a cada remoção
            (útil para plotar a curva nº de features x métrica).
        """
        features_atuais = list(x_train.columns)
        historico = []

        def _avalia(features):
            modelo = model_factory()
            modelo.fit(x_train[features], y_train)
            if usa_proba:
                pred = modelo.predict_proba(x_valid[features])[:, 1]
            else:
                pred = modelo.predict(x_valid[features])
            return metric_func(y_valid, pred), modelo

        score_atual, modelo_atual = _avalia(features_atuais)
        historico.append({
            'n_features': len(features_atuais),
            'score': score_atual,
            'feature_removida': None,
        })

        continuar = True
        while continuar and len(features_atuais) > min_features:
            if not hasattr(modelo_atual, 'feature_importances_'):
                raise ValueError(
                    "O modelo retornado por model_factory precisa expor "
                    "'feature_importances_' após o fit."
                )
            importancias = pd.Series(modelo_atual.feature_importances_, index=features_atuais)
            feature_menos_importante = importancias.sort_values(ascending=True).index[0]

            features_candidatas = [f for f in features_atuais if f != feature_menos_importante]
            score_candidato, modelo_candidato = _avalia(features_candidatas)

            if maior_e_melhor:
                manter_remocao = score_candidato >= (score_atual - tolerancia)
            else:
                manter_remocao = score_candidato <= (score_atual + tolerancia)

            historico.append({
                'n_features': len(features_candidatas),
                'score': score_candidato,
                'feature_removida': feature_menos_importante,
            })

            if manter_remocao:
                features_atuais = features_candidatas
                score_atual = score_candidato
                modelo_atual = modelo_candidato
                if verbose:
                    print(
                        f"Removida '{feature_menos_importante}' | "
                        f"features restantes: {len(features_atuais)} | "
                        f"score: {score_atual:.5f}"
                    )
            else:
                continuar = False
                if verbose:
                    print(
                        f"Parando: remover '{feature_menos_importante}' pioraria "
                        f"o score além da tolerância ({tolerancia})."
                    )

        historico_df = pd.DataFrame(historico)
        return features_atuais, historico_df

    @staticmethod
    def seleciona_boruta(x, y, tipo='classificacao', max_iter=100, percentil=100,
                          alpha=0.05, semente=42, **model_params):
        """Seleção via Boruta: cria cópias "sombra" (shadow features)
        embaralhadas de cada variável e só mantém as que superam
        consistentemente a importância da melhor sombra ao longo de várias
        iterações — é o mesmo espírito do corte por dummy aleatória
        (`seleciona_shap_lgbm_dummy`), só que estatisticamente mais rigoroso
        (teste binomial a cada rodada em vez de um único corte).

        Requer o pacote `boruta` (`pip install Boruta`).

        Parameters
        ----------
        x, y : features e target de treino.
        tipo : 'classificacao' ou 'regressao' -> escolhe RandomForestClassifier/Regressor
            como estimador base do Boruta (mais estável que LGBM para esse
            algoritmo, que já foi validado majoritariamente com RF).
        max_iter : número máximo de iterações do Boruta.
        percentil : percentil da importância das sombras usado como referência
            (100 = a sombra mais importante; valores menores tornam o corte
            mais permissivo).
        alpha : nível de significância do teste estatístico interno do Boruta.
        model_params : hiperparâmetros extras repassados ao RandomForest.

        Returns
        -------
        features_confirmadas : list[str] — aprovadas com confiança estatística.
        features_tentativas : list[str] — zona cinzenta ("tentative"), nem
            confirmadas nem rejeitadas; útil para decisão manual.
        ranking : DataFrame com o status final (confirmada/tentativa/rejeitada)
            e o ranking de cada feature.
        """
        from boruta import BorutaPy

        if tipo == 'classificacao':
            base = dict(n_estimators=200, max_depth=7, class_weight='balanced', n_jobs=-1)
            base.update(model_params)
            estimador = RandomForestClassifier(random_state=semente, **base)
        elif tipo == 'regressao':
            base = dict(n_estimators=200, max_depth=7, n_jobs=-1)
            base.update(model_params)
            estimador = RandomForestRegressor(random_state=semente, **base)
        else:
            raise ValueError("tipo deve ser 'classificacao' ou 'regressao'.")

        colunas = list(x.columns)
        selecionador = BorutaPy(
            estimador, n_estimators='auto', max_iter=max_iter,
            perc=percentil, alpha=alpha, random_state=semente, verbose=0,
        )
        selecionador.fit(x.values, np.asarray(y).ravel())

        ranking = pd.DataFrame({
            'feature': colunas,
            'confirmada': selecionador.support_,
            'tentativa': selecionador.support_weak_,
            'ranking': selecionador.ranking_,
        }).sort_values('ranking').reset_index(drop=True)

        ranking['status'] = np.select(
            [ranking['confirmada'], ranking['tentativa']],
            ['confirmada', 'tentativa'],
            default='rejeitada',
        )

        features_confirmadas = ranking.loc[ranking['status'] == 'confirmada', 'feature'].tolist()
        features_tentativas = ranking.loc[ranking['status'] == 'tentativa', 'feature'].tolist()

        return features_confirmadas, features_tentativas, ranking

    @staticmethod
    def seleciona_estabilidade_temporal_psi(df, colunas, coluna_tempo, n_bins=10,
                                             limite_psi=0.25, periodo_referencia=None,
                                             criterio='psi_maximo'):
        """Filtra variáveis pela estabilidade da distribuição ao longo do
        tempo — uma feature preditiva no desenvolvimento mas instável nas
        safras seguintes é um risco de drift silencioso em produção, então
        aqui o corte é por estabilidade, não por poder discriminante.

        Reaproveita `Analytics.calcula_psi_temporal` por trás: para cada
        variável, calcula o PSI de cada período contra o período de
        referência e resume num único critério de corte.

        Parameters
        ----------
        df : DataFrame contendo as variáveis e a coluna de tempo.
        colunas : lista de variáveis candidatas a avaliar.
        coluna_tempo : coluna de período (safra/data/mês).
        limite_psi : acima disso a variável é considerada instável.
        periodo_referencia : ver `Analytics.calcula_psi_temporal`; se None,
            usa o primeiro período.
        criterio : {'psi_maximo', 'psi_medio'}
            'psi_maximo' -> reprova a variável se QUALQUER período isolado
                estourar o limite (mais conservador — pega instabilidade
                pontual de uma safra específica).
            'psi_medio'  -> reprova pela média do PSI entre os períodos
                (mais tolerante a um período fora da curva isolado).

        Returns
        -------
        features_estaveis : list[str]
        resumo : DataFrame com o PSI por variável (máximo, médio e status)
            — ordenado da mais estável para a menos estável.
        """
        from .analytics import Analytics

        if criterio not in {'psi_maximo', 'psi_medio'}:
            raise ValueError("criterio deve ser 'psi_maximo' ou 'psi_medio'.")
        linhas = []
        for coluna in colunas:
            try:
                resumo_psi, _ = Analytics.calcula_psi_temporal(
                    df, coluna_variavel=coluna, coluna_tempo=coluna_tempo,
                    tipo='numerica', n_bins=n_bins, periodo_referencia=periodo_referencia,
                )
            except (ValueError, TypeError):
                # Variável não numérica ou incompatível com discretização -> trata como categórica
                resumo_psi, _ = Analytics.calcula_psi_temporal(
                    df, coluna_variavel=coluna, coluna_tempo=coluna_tempo,
                    tipo='categorica', periodo_referencia=periodo_referencia,
                )

            linhas.append({
                'feature': coluna,
                'psi_maximo': resumo_psi['psi'].max(),
                'psi_medio': resumo_psi['psi'].mean(),
                'n_periodos': len(resumo_psi),
            })

        resumo = pd.DataFrame(linhas)
        criterio_coluna = 'psi_maximo' if criterio == 'psi_maximo' else 'psi_medio'
        resumo['estavel'] = resumo[criterio_coluna] <= limite_psi
        resumo = resumo.sort_values(criterio_coluna).reset_index(drop=True)

        features_estaveis = resumo.loc[resumo['estavel'], 'feature'].tolist()

        return features_estaveis, resumo

    @staticmethod
    def remove_altamente_correlacionadas_spearman(df, colunas, threshold=0.9, importancia=None):
        """Remove variáveis quantitativas redundantes por correlação de
        Spearman (monotônica, mais robusta a outliers e não-linearidades
        que a de Pearson — mais adequada para variáveis de crédito como
        renda, valor de empréstimo, idade).

        Para cada par com |correlação| > threshold, mantém a de MAIOR
        importância e descarta a outra. Se `importancia` não for informada,
        a decisão de qual manter é pela maior variância (proxy simples de
        conteúdo informativo).

        Parameters
        ----------
        df : DataFrame contendo as variáveis.
        colunas : lista de variáveis quantitativas candidatas.
        threshold : limite absoluto de correlação acima do qual uma das
            duas variáveis é removida.
        importancia : DataFrame opcional com colunas ['feature', 'importance']
            (ex.: saída de `remove_features_feature_importance` ou
            `seleciona_shap_lgbm_dummy`) usado para decidir qual variável do
            par correlacionado é descartada.

        Returns
        -------
        features_mantidas : list[str]
        features_removidas : dict {feature_removida: feature_que_a_substituiu}
            — auditoria de qual par gerou cada remoção.
        """
        if importancia is not None:
            mapa_importancia = dict(zip(importancia['feature'], importancia['importance']))
        else:
            mapa_importancia = df[colunas].var().to_dict()

        correlacoes = df[colunas].corr(method='spearman').abs()

        features_removidas = {}
        for i in range(len(correlacoes.columns)):
            for j in range(i):
                col_i, col_j = correlacoes.columns[i], correlacoes.columns[j]
                if col_i in features_removidas or col_j in features_removidas:
                    continue
                if correlacoes.iloc[i, j] > threshold:
                    imp_i = mapa_importancia.get(col_i, 0)
                    imp_j = mapa_importancia.get(col_j, 0)
                    if imp_i >= imp_j:
                        features_removidas[col_j] = col_i
                    else:
                        features_removidas[col_i] = col_j

        features_mantidas = [c for c in colunas if c not in features_removidas]

        return features_mantidas, features_removidas

# Aliases funcionais para compatibilidade com notebooks antigos.
remove_features_baixa_variancia = SelecaoFeatures.remove_features_baixa_variancia
remove_features_mutual_information = SelecaoFeatures.remove_features_mutual_information
remove_features_feature_importance = SelecaoFeatures.remove_features_feature_importance
compute_shap_importance_df = SelecaoFeatures.compute_shap_importance_df
seleciona_shap_lgbm_dummy = SelecaoFeatures.seleciona_shap_lgbm_dummy
rfe_por_metrica = SelecaoFeatures.rfe_por_metrica
seleciona_boruta = SelecaoFeatures.seleciona_boruta
seleciona_estabilidade_temporal_psi = SelecaoFeatures.seleciona_estabilidade_temporal_psi
remove_altamente_correlacionadas_spearman = SelecaoFeatures.remove_altamente_correlacionadas_spearman
