"""Análises, métricas, tabelas e visualizações generalistas."""
from ._shared import *

class Analytics:
    """Centraliza análises numéricas e visuais em uma única interface.

    Reúne métricas, tabelas, diagnósticos e gráficos em uma única interface.
    """

    # ------------------------------------------------------------------ #
    # Testes de hipótese / distribuição
    # ------------------------------------------------------------------ #
    @staticmethod
    def ks_test(y_proba_0, y_proba_1):
        KS, p_value = stats.ks_2samp(y_proba_0, y_proba_1)
        if p_value > 0.05:
            ks_message = 'Pelo Teste de KS, não há diferença significativa entre as amostras'
        else:
            ks_message = 'Pelo Teste de KS, há diferença significativa entre as amostras'
        return KS, ks_message

    @staticmethod
    def error_metrics(y_true: np.ndarray, pred: np.ndarray, mase_scale: float) -> dict:
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

    # ------------------------------------------------------------------ #
    # Métricas de regressão (SEM CohenKappa — removido a pedido)
    # ------------------------------------------------------------------ #
    @staticmethod
    def metricas_regressao(model_name, y_train, y_pred_train, y_test, y_pred_test,
                            etapa_1='treino', etapa_2='teste', por_faixa=False):

        def var20(y_true, y_pred):
            y_true = np.asarray(y_true, dtype=np.float64).flatten()
            y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
            y_true_safe = np.where(y_true == 0, 1e-10, y_true)
            relative_error = np.abs(y_pred - y_true) / y_true_safe
            return np.mean(relative_error <= 0.20)

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

        def calcular_mape(y_true, y_pred):
            y_true = np.asarray(y_true, dtype=np.float64).flatten()
            y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
            y_true_safe = np.where(y_true == 0, 1e-10, y_true)
            return np.mean(np.abs(y_pred - y_true) / y_true_safe) * 100

        def calcular_metricas(y_true, y_pred, etapa, pct_amostras=100):
            data = {
                'MAE': [mean_absolute_error(y_true, y_pred)],
                'RMSE': [np.sqrt(mean_squared_error(y_true, y_pred))],
                'RMSLE': [rmsle(y_true, y_pred)],
                'MAPE (%)': [calcular_mape(y_true, y_pred)],
                'Var20 (%)': [var20(y_true, y_pred) * 100],
                'Subestimação (%)': [under20(y_true, y_pred) * 100],
                'Superestimação (%)': [over20(y_true, y_pred) * 100],
                'Etapa': [etapa],
                'Modelo': [model_name]
            }
            if pct_amostras is not None:
                data['Pct_amostras (%)'] = [pct_amostras]
            return pd.DataFrame(data)

        metricas_treino = calcular_metricas(y_train, y_pred_train, etapa_1)
        metricas_teste = calcular_metricas(y_test, y_pred_test, etapa_2)

        if por_faixa:
            bins = [0, 30, 45, 60, 75, 90, 105, np.inf]
            labels = ['Até 30min', 'Até 45min', 'Até 60min', 'Até 75min', 'Até 90min', 'Até 105min', 'Mais que 105min']

            metricas_treino_faixa, metricas_teste_faixa = [], []

            y_train_arr = np.asarray(y_train).flatten()
            y_pred_train_arr = np.asarray(y_pred_train).flatten()
            y_test_arr = np.asarray(y_test).flatten()
            y_pred_test_arr = np.asarray(y_pred_test).flatten()

            total_train, total_test = len(y_train_arr), len(y_test_arr)

            for i in range(len(bins) - 1):
                mask = (y_train_arr > bins[i]) & (y_train_arr <= bins[i + 1])
                if np.any(mask):
                    pct = np.sum(mask) / total_train * 100
                    metricas_treino_faixa.append(
                        calcular_metricas(y_train_arr[mask], y_pred_train_arr[mask],
                                           f"{etapa_1} ({labels[i]})", pct_amostras=pct)
                    )

            for i in range(len(bins) - 1):
                mask = (y_test_arr > bins[i]) & (y_test_arr <= bins[i + 1])
                if np.any(mask):
                    pct = np.sum(mask) / total_test * 100
                    metricas_teste_faixa.append(
                        calcular_metricas(y_test_arr[mask], y_pred_test_arr[mask],
                                           f"{etapa_2} ({labels[i]})", pct_amostras=pct)
                    )

            metricas_treino = pd.concat([metricas_treino] + metricas_treino_faixa).reset_index(drop=True)
            metricas_teste = pd.concat([metricas_teste] + metricas_teste_faixa).reset_index(drop=True)

        return pd.concat([metricas_treino, metricas_teste]).reset_index(drop=True)

    @staticmethod
    def metricas_modelos_juntos_regressao(lista_modelos):
        if len(lista_modelos) == 0:
            return pd.DataFrame()

        df = pd.concat(lista_modelos).reset_index(drop=True).round(2)

        metricas_cols = ['MAE', 'RMSE', 'RMSLE', 'MAPE (%)',
                          'Var20 (%)', 'Subestimação (%)', 'Superestimação (%)',
                          'Pct_amostras (%)']

        def color_etapa(val):
            val = str(val).lower()
            color = 'black'
            if 'treino' in val:
                color = 'blue'
            elif 'teste' in val or 'validacao' in val:
                color = 'red'
            return f'color: {color}; font-weight: bold;'

        def separador_modelos(df):
            estilos = pd.DataFrame('', index=df.index, columns=df.columns)
            modelos = df['Modelo']
            for i in range(len(modelos) - 1):
                if modelos[i] != modelos[i + 1]:
                    estilos.loc[i, :] = 'border-bottom: 3px solid black;'
            return estilos

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

    # ------------------------------------------------------------------ #
    # Métricas de classificação — conjunto fixo:
    # Acuracia, Precisao, Recall, F1, AUC, PR_AUC, Gini, KS, LogLoss
    # ------------------------------------------------------------------ #
    @staticmethod
    def metricas_classificacao(classificador, y_true, y_predict=None,
                                y_predict_proba=None, etapa="teste", cutoff=0.5):
        """Consolida as métricas discriminatórias e preditivas padrão de um
        classificador binário."""
        y = np.asarray(y_true).ravel().astype(int)
        if y_predict_proba is None:
            raise ValueError("Informe y_predict_proba para calcular AUC, PR-AUC, Gini, KS e LogLoss.")

        proba = np.asarray(y_predict_proba)
        if proba.ndim == 2:
            proba = proba[:, 1]
        proba = proba.ravel()

        pred = (proba >= cutoff).astype(int) if y_predict is None else np.asarray(y_predict).ravel().astype(int)

        if not (len(y) == len(pred) == len(proba)):
            raise ValueError("y_true, y_predict e y_predict_proba devem ter o mesmo tamanho.")

        if np.unique(y).size > 1:
            auc = roc_auc_score(y, proba)
            fpr, tpr, _ = roc_curve(y, proba)
            ks = float(np.max(tpr - fpr))
            pr_auc = average_precision_score(y, proba)
            gini = 2 * auc - 1
        else:
            auc = ks = pr_auc = gini = np.nan

        linha = {
            "Modelo": classificador,
            "Etapa": etapa,
            "Acuracia": accuracy_score(y, pred),
            "Precisao": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "AUC": auc,
            "PR_AUC": pr_auc,
            "Gini": gini,
            "KS": ks,
            "LogLoss": log_loss(y, np.clip(proba, 1e-15, 1 - 1e-15), labels=[0, 1]),
        }
        return pd.DataFrame([linha])

    @staticmethod
    def metricas_classificacao_juntos(lista_metricas):
        """Equivalente classificatório de `metricas_modelos_juntos_regressao`:
        concatena as saídas de `metricas_classificacao` / `metricas_classificacao_treino_teste`
        de múltiplos modelos/etapas numa única tabela estilizada, com separador
        visual a cada troca de modelo e cor por etapa (treino/teste/validação/OOT)."""
        if not lista_metricas:
            return pd.DataFrame()

        df = pd.concat(lista_metricas, ignore_index=True).round(4)

        metricas_cols = ['Acuracia', 'Precisao', 'Recall', 'F1', 'AUC', 'PR_AUC', 'Gini', 'KS', 'LogLoss']

        def color_etapa(val):
            val = str(val).lower()
            color = 'black'
            if 'treino' in val:
                color = 'blue'
            elif 'teste' in val or 'validacao' in val or 'oot' in val:
                color = 'red'
            return f'color: {color}; font-weight: bold;'

        def separador_modelos(df):
            estilos = pd.DataFrame('', index=df.index, columns=df.columns)
            modelos = df['Modelo']
            for i in range(len(modelos) - 1):
                if modelos[i] != modelos[i + 1]:
                    estilos.loc[i, :] = 'border-bottom: 3px solid black;'
            return estilos

        styled_df = df.style \
            .format({col: "{:.4f}" for col in metricas_cols}) \
            .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px', subset=pd.IndexSlice[:, :]) \
            .applymap(color_etapa, subset=pd.IndexSlice[:, ['Etapa']]) \
            .apply(separador_modelos, axis=None) \
            .set_table_styles([
                {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
            ])
        return styled_df

    @staticmethod
    def metricas_calibracao(y_true, y_predict_proba, n_bins=10):
        """Retorna Brier/Log Loss e tabela observada versus prevista por faixa
        (a calibração fica separada das métricas discriminatórias por natureza —
        não entra no conjunto fixo de `metricas_classificacao`)."""
        y = np.asarray(y_true).ravel().astype(int)
        proba = np.asarray(y_predict_proba)
        if proba.ndim == 2:
            proba = proba[:, 1]
        proba = proba.ravel()
        observado, previsto = calibration_curve(y, proba, n_bins=n_bins, strategy="quantile")
        tabela = pd.DataFrame({"Probabilidade_prevista": previsto, "Frequencia_observada": observado})
        tabela["Erro_absoluto"] = abs(tabela.Frequencia_observada - tabela.Probabilidade_prevista)
        resumo = pd.DataFrame([{
            "Brier": brier_score_loss(y, proba),
            "LogLoss": log_loss(y, np.clip(proba, 1e-15, 1 - 1e-15), labels=[0, 1]),
            "Erro_calibracao_medio": tabela.Erro_absoluto.mean(),
        }])
        return resumo, tabela

    # ------------------------------------------------------------------ #
    # Discretização genérica (usada pelo PSI numérico e por qualquer projeto)
    # ------------------------------------------------------------------ #
    @staticmethod
    def discretiza_variavel(serie, n_bins=10, pontos_corte=None, labels=None, metodo='quantil'):
        """Discretiza uma variável contínua de forma genérica.

        Parameters
        ----------
        serie : array-like
            Variável contínua a discretizar (score, renda, idade, etc.).
        n_bins : int
            Número de faixas, usado quando `pontos_corte` não é informado.
        pontos_corte : list[float] | None
            Se informado, usa esses cortes fixos em vez de recalcular por
            quantil — essencial para reaplicar a MESMA discretização
            aprendida numa safra base em safras futuras (ex.: PSI temporal,
            escoragem em produção).
        labels : list | None
            Rótulos das faixas. Se None, mantém os intervalos como rótulo.
            Se `len(labels) == n_bins`, aceita rótulos customizados (ex.:
            ['A','B','C','D','E'] para rating, ou [1..10] para decil).
        metodo : {'quantil', 'largura_igual'}
            'quantil' -> pd.qcut (faixas com frequência semelhante, ex.: decil).
            'largura_igual' -> pd.cut com faixas de mesmo tamanho.

        Returns
        -------
        faixas : pd.Series
        pontos_corte : list[float]
            Cortes efetivamente usados — guarde-os para reaplicar a mesma
            discretização em outra amostra/período.
        """
        serie = pd.Series(serie)

        if pontos_corte is None:
            if metodo == 'quantil':
                cortes = np.unique(np.percentile(serie.dropna(), np.linspace(0, 100, n_bins + 1)))
            elif metodo == 'largura_igual':
                cortes = np.unique(np.linspace(serie.min(), serie.max(), n_bins + 1))
            else:
                raise ValueError("metodo deve ser 'quantil' ou 'largura_igual'.")
            cortes = cortes.astype(float)
            cortes[0], cortes[-1] = -np.inf, np.inf
        else:
            cortes = list(pontos_corte)

        n_faixas_efetivas = len(cortes) - 1
        rotulos = labels if (labels is not None and len(labels) == n_faixas_efetivas) else None

        faixas = pd.cut(serie, bins=cortes, labels=rotulos, include_lowest=True)
        return faixas, list(cortes)

    # ------------------------------------------------------------------ #
    # PSI (Population Stability Index) — numérico ou categórico
    # ------------------------------------------------------------------ #
    @staticmethod
    def _psi_componentes(perc_base, perc_atual):
        eps = 1e-4
        perc_base = np.where(np.asarray(perc_base) == 0, eps, perc_base)
        perc_atual = np.where(np.asarray(perc_atual) == 0, eps, perc_atual)
        componentes = (perc_atual - perc_base) * np.log(perc_atual / perc_base)
        return componentes, float(np.sum(componentes))

    @staticmethod
    def _interpreta_psi(psi):
        if psi <= 0.10:
            return 'Estável'
        elif psi <= 0.25:
            return 'Atenção'
        return 'Instável'

    @staticmethod
    def calcula_psi_amostras(amostra_base, amostra_comparacao, tipo='numerica', n_bins=10,
                              nome_base='Amostra Base', nome_comparacao='Amostra Comparação'):
        """PSI entre DUAS AMOSTRAS quaisquer — não precisam ser períodos de
        tempo. Serve para checar, por exemplo, se treino e teste vêm de
        populações parecidas, se um cluster A é similar a um cluster B, ou
        se uma base própria é comparável a um benchmark externo.

        tipo='numerica'   -> discretiza pelos quantis da amostra_base (n_bins).
        tipo='categorica' -> usa as categorias originais, sem discretizar
                              (ex.: PSI de uma variável categórica ou de um
                              decil de score já calculado).

        Returns
        -------
        resumo : DataFrame de 1 linha com o PSI total e a interpretação.
        detalhe : DataFrame por faixa/categoria com os componentes do PSI
                  (útil para identificar qual faixa mais contribuiu pra
                  instabilidade).
        """
        base = pd.Series(amostra_base).dropna()
        atual = pd.Series(amostra_comparacao).dropna()

        if tipo == 'numerica':
            faixa_base, cortes = Analytics.discretiza_variavel(base, n_bins=n_bins)
            faixa_atual = pd.cut(atual, bins=cortes, include_lowest=True)
        elif tipo == 'categorica':
            faixa_base, faixa_atual = base, atual
        else:
            raise ValueError("tipo deve ser 'numerica' ou 'categorica'.")

        contagem_base = faixa_base.value_counts(normalize=True, sort=False).sort_index()
        contagem_atual = faixa_atual.value_counts(normalize=True, sort=False).reindex(contagem_base.index).fillna(0)

        componentes, psi_total = Analytics._psi_componentes(contagem_base.values, contagem_atual.values)

        detalhe = pd.DataFrame({
            'faixa': contagem_base.index.astype(str),
            'perc_base': contagem_base.values,
            'perc_comparacao': contagem_atual.values,
            'psi_componente': componentes,
        })

        resumo = pd.DataFrame([{
            'amostra_base': nome_base,
            'amostra_comparacao': nome_comparacao,
            'psi': round(psi_total, 4),
            'interpretacao': Analytics._interpreta_psi(psi_total),
        }])

        return resumo, detalhe

    @staticmethod
    def calcula_psi_temporal(df, coluna_variavel, coluna_tempo, tipo='numerica',
                              n_bins=10, periodo_referencia=None):
        """PSI de uma variável (score contínuo, decil já calculado, ou
        categoria) AO LONGO DO TEMPO, sempre comparando cada período contra
        um período de referência fixo (a "safra base"). Ideal para
        monitoramento pós-deploy: PSI do score por safra, PSI da distribuição
        de decil de rating por mês, PSI de uma variável categórica (produto,
        canal, UF) por safra, etc.

        Parameters
        ----------
        df : DataFrame contendo a variável e a coluna de tempo (safra/mês/data).
        coluna_variavel : nome da coluna a monitorar.
        coluna_tempo : nome da coluna de período (safra, data, mês — qualquer
            tipo ordenável).
        tipo : 'numerica' (discretiza pelos quantis da safra base, cortes
            fixos reaplicados em todos os períodos) ou 'categorica' (usa as
            categorias originais, ex.: decil já calculado, UF, produto).
        periodo_referencia : período usado como base. Se None, usa o
            primeiro período disponível (ordenado).

        Returns
        -------
        resumo : DataFrame com uma linha por período (psi, n_amostras,
            interpretação) — pronto para plotar a evolução do PSI no tempo.
        detalhes : dict {período: DataFrame de componentes do PSI naquele
            período} — para auditar qual faixa puxou a instabilidade.
        """
        dados = df[[coluna_variavel, coluna_tempo]].dropna().copy()
        periodos = sorted(dados[coluna_tempo].unique())
        if not periodos:
            raise ValueError("Não há períodos válidos em coluna_tempo.")
        if periodo_referencia is None:
            periodo_referencia = periodos[0]

        base = dados.loc[dados[coluna_tempo] == periodo_referencia, coluna_variavel]

        cortes = None
        if tipo == 'numerica':
            _, cortes = Analytics.discretiza_variavel(base, n_bins=n_bins)

        linhas, detalhes = [], {}

        for periodo in periodos:
            atual = dados.loc[dados[coluna_tempo] == periodo, coluna_variavel]

            if tipo == 'numerica':
                faixa_base = pd.cut(base, bins=cortes, include_lowest=True)
                faixa_atual = pd.cut(atual, bins=cortes, include_lowest=True)
            elif tipo == 'categorica':
                faixa_base, faixa_atual = base, atual
            else:
                raise ValueError("tipo deve ser 'numerica' ou 'categorica'.")

            contagem_base = faixa_base.value_counts(normalize=True, sort=False).sort_index()
            contagem_atual = faixa_atual.value_counts(normalize=True, sort=False).reindex(contagem_base.index).fillna(0)

            componentes, psi_total = Analytics._psi_componentes(contagem_base.values, contagem_atual.values)

            linhas.append({
                'periodo': periodo,
                'psi': round(psi_total, 4),
                'n_amostras': len(atual),
                'interpretacao': Analytics._interpreta_psi(psi_total),
            })

            detalhes[periodo] = pd.DataFrame({
                'faixa': contagem_base.index.astype(str),
                'perc_base': contagem_base.values,
                'perc_periodo': contagem_atual.values,
                'psi_componente': componentes,
            })

        resumo = pd.DataFrame(linhas)
        return resumo, detalhes

    # ------------------------------------------------------------------ #
    # Testes exploratórios (mantidos, sem alteração de escopo)
    # ------------------------------------------------------------------ #
    @staticmethod
    def analisa_normalidade(amostra, variavel):
        normaltest_amostra = normaltest(amostra[variavel])
        if normaltest_amostra[1] < 0.05:
            print(f'Pelo Teste de Hipótese, a Hipótese Nula de que "{variavel}" segue Distribuição Normal é REJEITADA!')
        else:
            print(f'Pelo Teste de Hipótese, a Hipótese Nula de que "{variavel}" segue Distribuição Normal é ACEITA')
        plt.figure(figsize=(5, 3))
        stats.probplot(amostra[variavel], dist='norm', plot=plt)
        plt.title('Amostra 1', fontsize=14)
        plt.grid(False)
        plt.box(False)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def analisa_outliers(df):
        Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
        IIQ = Q3 - Q1
        return Q1 - 1.5 * IIQ, Q3 + 1.5 * IIQ

    @staticmethod
    def teste_hipotese_duas_amostras_independentes(parametrico, amostra1, amostra2, variavel):
        if parametrico:
            print(f'Média Amostra 1: {amostra1[variavel].mean()}')
            print(f'Média Amostra 2: {amostra2[variavel].mean()}')
            stat, p_value = ztest(amostra1[variavel], amostra2[variavel])
            msg = 'não há' if p_value > 0.05 else 'há'
            print(f'Pelo Teste Z, {msg} diferença significativa entre as médias.')
        else:
            print(f'Mediana Amostra 1: {amostra1[variavel].median()}')
            print(f'Mediana Amostra 2: {amostra2[variavel].median()}')
            stat, p_value = stats.mannwhitneyu(amostra1[variavel], amostra2[variavel])
            msg = 'não há' if p_value > 0.05 else 'há'
            print(f'Pelo Teste de Mann-Whitney, {msg} diferença significativa entre as medianas.')

    @staticmethod
    def teste_hipotese_muitas_amostras_independentes(amostras, variavel):
        for i, amostra in enumerate(amostras):
            print(f'Mediana Amostra {i+1}: {amostra[variavel].median()}')
        stat, p_value = kruskal(*[amostra[variavel] for amostra in amostras])
        msg = 'não há' if p_value > 0.05 else 'há'
        print(f'Pelo teste de Kruskal-Wallis, {msg} diferença significativa entre as medianas.')

    @staticmethod
    def teste_hipotese_duas_variaveis_categoricas(df, variavel1, variavel2):
        crosstab = pd.crosstab(df[variavel1], df[variavel2])
        chi2, p, _, _ = chi2_contingency(crosstab)
        msg = 'não há' if p > 0.05 else 'há'
        print(f'Pelo Teste Qui-Quadrado, {msg} associação significativa entre {variavel1} e {variavel2}.')

    @staticmethod
    def analisa_correlacao(metodo, df):
        plt.figure(figsize=(30, 15))
        mask = np.triu(np.ones_like(df.corr(method=metodo), dtype=bool))
        heatmap = sns.heatmap(df.corr(method=metodo), vmin=-1, vmax=1, cmap='magma', annot=True, fmt='.1f',
                               cbar_kws={"shrink": .8}, mask=mask)
        heatmap.set_title(f"Analisando Correlação de {metodo}")
        plt.grid(False)
        plt.box(False)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def analisa_distribuicao_via_percentis(df, variaveis):
        return df[variaveis].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T

    @staticmethod
    def taxa_por_grupo(df, coluna, target, top_n=20):
        tabela = (
            df.groupby(coluna, observed=True)[target]
            .agg(qtd="size", eventos="sum", taxa_evento="mean")
            .sort_values(["taxa_evento", "qtd"], ascending=False)
        )
        return tabela.head(top_n)

    @staticmethod
    def calcula_woe_iv(df, feature, target, plotar=True, suavizacao=0.5):
        """Calcula WOE e IV de uma variável em relação a um target binário
        (0 = bom/não-evento, 1 = mau/evento). Substitui as antigas `woe()` e
        `iv()` duplicadas — agora uma única função devolve a tabela completa
        (com woe e iv) e, opcionalmente, plota."""
        evento = df.loc[df[target] == 1].groupby(feature, as_index=False)[target].count().rename({target: 'evento'}, axis=1)
        nao_evento = df.loc[df[target] == 0].groupby(feature, as_index=False)[target].count().rename({target: 'nao_evento'}, axis=1)

        tabela = evento.merge(nao_evento, on=feature, how='outer').fillna(0)
        if suavizacao < 0:
            raise ValueError('suavizacao deve ser maior ou igual a zero.')
        n_grupos = max(len(tabela), 1)
        tabela['percent_evento'] = ((tabela['evento'] + suavizacao) /
                                     (tabela['evento'].sum() + suavizacao * n_grupos))
        tabela['percent_nao_evento'] = ((tabela['nao_evento'] + suavizacao) /
                                         (tabela['nao_evento'].sum() + suavizacao * n_grupos))
        tabela['woe'] = np.log(tabela['percent_evento'] / tabela['percent_nao_evento']).round(4)
        contribuicao_iv = ((tabela['percent_evento'] - tabela['percent_nao_evento']) * tabela['woe'])
        tabela['iv_grupo'] = contribuicao_iv
        tabela['iv'] = contribuicao_iv.sum()
        tabela.sort_values(by='woe', ascending=True, inplace=True)

        if plotar:
            plt.figure(figsize=(10, 4))
            plt.plot(tabela[feature].astype(str), tabela['woe'], marker='o', linestyle='--', linewidth=2, color='#1FB3E5')
            for x, y in zip(tabela[feature].astype(str), tabela['woe']):
                plt.text(x=x, y=y, s=str(y), fontsize=10, color='red', ha='left', va='center', rotation=45)
            plt.title(f'Weight of Evidence da variável "{feature}" (IV = {tabela["iv"].iloc[0]:.3f})', fontsize=14)
            plt.xlabel('Classes', fontsize=14)
            plt.ylabel('Weight of Evidence', fontsize=14)
            plt.xticks(ha='right', fontsize=10, rotation=45)
            plt.show()

        return tabela.reset_index(drop=True)

    @staticmethod
    def calcular_rating(probabilidade, n_ratings=10, maior_rating_menor_risco=True):
        probabilidades = pd.Series(probabilidade).astype(float)
        if n_ratings < 2:
            raise ValueError('n_ratings deve ser maior ou igual a 2.')
        if probabilidades.notna().sum() < n_ratings:
            raise ValueError('Não há observações suficientes para formar os ratings.')
        # Valores iguais permanecem no mesmo rating; por isso o número efetivo
        # de ratings pode ser menor quando há muitos empates.
        decis = pd.qcut(probabilidades.rank(method='average', pct=True), q=n_ratings,
                        labels=False, duplicates='drop').astype('Int64')
        n_efetivo = int(decis.max()) + 1
        rating = n_efetivo - 1 - decis if maior_rating_menor_risco else decis
        return rating.rename('rating')

    @staticmethod
    def criar_base_rating(df, probabilidade, nome_rating='rating_model',
                           target=None, coluna_exposicao=None, coluna_lgd=None,
                           lgd=1.0, n_ratings=10):
        resultado = df.copy()
        resultado['Probability of Default'] = np.asarray(probabilidade)
        resultado[nome_rating] = Analytics.calcular_rating(resultado['Probability of Default'], n_ratings=n_ratings).to_numpy()
        resultado['qt_pessoas_rating'] = resultado.groupby(nome_rating)[nome_rating].transform('size')
        if coluna_exposicao is not None:
            lgd_valor = resultado[coluna_lgd] if coluna_lgd is not None else lgd
            resultado['expected_loss'] = (resultado['Probability of Default'] *
                                           lgd_valor * resultado[coluna_exposicao])
        if target is not None:
            resultado['taxa_evento_rating'] = resultado.groupby(nome_rating)[target].transform('mean')
        return resultado

    @staticmethod
    def comparar_grupos(df, coluna_grupo_a, coluna_grupo_b, nome_a='grupo_a', nome_b='grupo_b'):
        """Compara duas classificações categóricas quaisquer sobre a mesma
        população (ex.: rating do modelo vs. rating da política, cluster
        antigo vs. cluster novo) e sinaliza upgrade/downgrade/manutenção.
        Requer que as duas colunas sejam ordinais e comparáveis (ex.: ambas
        geradas por `calcular_rating`)."""
        resultado = df.copy()
        resultado[f'{nome_a}_valor'] = resultado[coluna_grupo_a]
        resultado[f'{nome_b}_valor'] = resultado[coluna_grupo_b]
        resultado['mesmo_resultado'] = resultado[coluna_grupo_a].eq(resultado[coluna_grupo_b])
        resultado['migracao'] = np.select(
            [resultado[coluna_grupo_a] < resultado[coluna_grupo_b],
             resultado[coluna_grupo_a] > resultado[coluna_grupo_b]],
            ['Upgrade', 'Downgrade'],
            default='Manutencao',
        )
        return resultado

    @staticmethod
    def matriz_migracao(df, coluna_origem, coluna_destino, normalizar='total',
                         segmento=None, coluna_segmento=None):
        """Matriz de migração entre duas classificações ordinais quaisquer
        (rating, decil de risco, cluster, faixa de renda) — função exclusiva
        e genérica, desacoplada de nomes de rating específicos.

        normalizar: None (contagens) | 'total' (% da carteira) | 'linha'
        (% dentro de cada categoria de origem — leitura de "para onde foi
        quem estava em X").
        """
        if segmento is not None and coluna_segmento is None:
            raise ValueError('Informe coluna_segmento ao filtrar um segmento.')
        dados = df if segmento is None else df.loc[df[coluna_segmento].eq(segmento)]

        matriz = pd.crosstab(dados[coluna_origem], dados[coluna_destino], dropna=False)
        categorias = sorted(set(dados[coluna_origem].dropna()) | set(dados[coluna_destino].dropna()))
        matriz = matriz.reindex(index=categorias, columns=categorias, fill_value=0)

        if normalizar is None:
            return matriz
        if normalizar == 'total':
            total = matriz.to_numpy().sum()
            return (matriz / total * 100).round(2) if total else matriz.astype(float)
        if normalizar == 'linha':
            return matriz.div(matriz.sum(axis=1).replace(0, np.nan), axis=0).mul(100).fillna(0).round(2)
        raise ValueError("normalizar deve ser None, 'total' ou 'linha'.")

    @staticmethod
    def resumo_migracao(df, coluna='migracao'):
        resumo = df[coluna].value_counts().rename('quantidade').to_frame()
        resumo['percentual'] = (resumo['quantidade'] / len(df) * 100).round(2)
        return resumo.rename_axis('migracao').reset_index()

    @staticmethod
    def ajusta_calibrador_score(score, y, metodo='isotonic', random_state=42):
        if metodo not in {'isotonic', 'sigmoid'}:
            raise ValueError("metodo deve ser 'isotonic' ou 'sigmoid'.")
        if metodo == 'isotonic':
            modelo = IsotonicRegression(out_of_bounds='clip').fit(score, y)
        else:
            modelo = LogisticRegression(C=1, random_state=random_state).fit(np.asarray(score).reshape(-1, 1), y)
        return modelo

    @staticmethod
    def prediz_calibrador(modelo, score, metodo='isotonic'):
        if metodo not in {'isotonic', 'sigmoid'}:
            raise ValueError("metodo deve ser 'isotonic' ou 'sigmoid'.")
        if metodo == 'isotonic':
            return modelo.predict(score)
        return modelo.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1]

    @staticmethod
    def calcula_perda_esperada(pd_score, ead, lgd=1.0):
        """Calcula perda esperada linha a linha: ``PD × LGD × EAD``.

        ``lgd`` pode ser escalar ou array-like. Os três componentes devem
        estar em escala decimal/monetária compatível.
        """
        pd_array = np.asarray(pd_score, dtype=float)
        ead_array = np.asarray(ead, dtype=float)
        lgd_array = np.asarray(lgd, dtype=float)
        if np.any((pd_array < 0) | (pd_array > 1)):
            raise ValueError('pd_score deve estar entre 0 e 1.')
        if np.any((lgd_array < 0) | (lgd_array > 1)):
            raise ValueError('lgd deve estar entre 0 e 1.')
        if np.any(ead_array < 0):
            raise ValueError('ead não pode conter valores negativos.')
        return pd_array * lgd_array * ead_array

    @staticmethod
    def metricas_intervalo_regressao(y_true, limite_inferior, limite_superior,
                                     alpha=0.10, y_pred=None):
        """Avalia intervalos de predição de SKPRO, MAPIE ou outro método.

        Retorna cobertura empírica, largura, falhas por lado e Winkler score.
        Quanto menor a largura e o Winkler, melhor, desde que a cobertura
        empírica permaneça próxima da cobertura nominal ``1 - alpha``.
        """
        if not 0 < alpha < 1:
            raise ValueError('alpha deve estar entre 0 e 1.')
        y = np.asarray(y_true, dtype=float).ravel()
        inferior = np.asarray(limite_inferior, dtype=float).ravel()
        superior = np.asarray(limite_superior, dtype=float).ravel()
        if not (len(y) == len(inferior) == len(superior)):
            raise ValueError('y_true e limites devem ter o mesmo tamanho.')
        if np.any(superior < inferior):
            raise ValueError('limite_superior não pode ser menor que limite_inferior.')
        validos = np.isfinite(y) & np.isfinite(inferior) & np.isfinite(superior)
        y, inferior, superior = y[validos], inferior[validos], superior[validos]
        if not len(y):
            raise ValueError('Não há observações válidas para avaliar.')

        abaixo = y < inferior
        acima = y > superior
        dentro = ~(abaixo | acima)
        largura = superior - inferior
        winkler = largura.copy()
        winkler[abaixo] += (2 / alpha) * (inferior[abaixo] - y[abaixo])
        winkler[acima] += (2 / alpha) * (y[acima] - superior[acima])
        resultado = {
            'n_observacoes': len(y),
            'cobertura_nominal': 1 - alpha,
            'cobertura_empirica': dentro.mean(),
            'erro_cobertura': dentro.mean() - (1 - alpha),
            'largura_media': largura.mean(),
            'largura_mediana': np.median(largura),
            'taxa_abaixo_intervalo': abaixo.mean(),
            'taxa_acima_intervalo': acima.mean(),
            'winkler_score': winkler.mean(),
        }
        if y_pred is not None:
            pred = np.asarray(y_pred, dtype=float).ravel()[validos]
            resultado['mae_predicao_pontual'] = mean_absolute_error(y, pred)
            resultado['rmse_predicao_pontual'] = np.sqrt(mean_squared_error(y, pred))
        return pd.DataFrame([resultado])

    @staticmethod
    def diagnostico_intervalos_regressao(y_true, limite_inferior, limite_superior,
                                         y_pred=None):
        """Cria diagnóstico linha a linha para gráficos e análises por grupo."""
        y = np.asarray(y_true, dtype=float).ravel()
        inferior = np.asarray(limite_inferior, dtype=float).ravel()
        superior = np.asarray(limite_superior, dtype=float).ravel()
        if not (len(y) == len(inferior) == len(superior)):
            raise ValueError('y_true e limites devem ter o mesmo tamanho.')
        resultado = pd.DataFrame({
            'y_true': y,
            'ic_inferior': inferior,
            'ic_superior': superior,
        })
        resultado['dentro_intervalo'] = y >= inferior
        resultado['dentro_intervalo'] &= y <= superior
        resultado['largura_intervalo'] = superior - inferior
        resultado['posicao_intervalo'] = np.select(
            [y < inferior, y > superior], ['abaixo', 'acima'], default='dentro'
        )
        if y_pred is not None:
            resultado['y_pred'] = np.asarray(y_pred, dtype=float).ravel()
            resultado['erro'] = resultado['y_true'] - resultado['y_pred']
            resultado['erro_absoluto'] = resultado['erro'].abs()
        return resultado

    @staticmethod
    def tabela_cutoffs(y_true, y_predict_proba, cutoffs=None,
                       funcao_objetivo=None, nome_objetivo='valor_objetivo',
                       parametros_objetivo=None):
        """Avalia cutoffs de um classificador binário de forma neutra.

        A classe prevista é 1 quando ``probabilidade >= cutoff``. Para usar
        lucro, retorno, custo ou qualquer regra específica do projeto, passe
        uma função em ``funcao_objetivo`` com a assinatura::

            funcao(y_true, y_pred, y_proba, cutoff, **parametros) -> número | dict

        Se ela devolver um dicionário, cada item vira uma coluna. Isso mantém
        toda a regra financeira fora da biblioteca e permite otimizar qualquer
        objetivo definido pelo projeto.
        """
        y = np.asarray(y_true).ravel().astype(int)
        proba = np.asarray(y_predict_proba)
        if proba.ndim == 2:
            proba = proba[:, 1]
        proba = proba.ravel().astype(float)
        if len(y) != len(proba):
            raise ValueError('y_true e y_predict_proba devem ter o mesmo tamanho.')
        if set(np.unique(y)) - {0, 1}:
            raise ValueError('y_true deve ser binário, codificado como 0/1.')
        if cutoffs is None:
            cutoffs = np.linspace(0.01, 0.99, 99)
        parametros_objetivo = dict(parametros_objetivo or {})

        linhas = []
        for cutoff in np.asarray(cutoffs, dtype=float):
            pred = (proba >= cutoff).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            positivos_preditos = int(pred.sum())
            negativos_preditos = int((pred == 0).sum())
            linha = {
                'cutoff': cutoff,
                'positivos_preditos': positivos_preditos,
                'taxa_predicao_positiva': positivos_preditos / len(y) if len(y) else np.nan,
                'negativos_preditos': negativos_preditos,
                'taxa_predicao_negativa': negativos_preditos / len(y) if len(y) else np.nan,
                'precision': precision_score(y, pred, zero_division=0),
                'recall': recall_score(y, pred, zero_division=0),
                'f1': f1_score(y, pred, zero_division=0),
                'especificidade': tn / (tn + fp) if tn + fp else np.nan,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            }
            if funcao_objetivo is not None:
                valor = funcao_objetivo(
                    y_true=y.copy(), y_pred=pred.copy(), y_proba=proba.copy(),
                    cutoff=float(cutoff), **parametros_objetivo,
                )
                if isinstance(valor, dict):
                    linha.update(valor)
                else:
                    linha[nome_objetivo] = valor
            linhas.append(linha)
        return pd.DataFrame(linhas)

    @staticmethod
    def seleciona_cutoff(tabela_cutoffs, criterio='f1', maximizar=True,
                         filtro=None):
        """Seleciona o cutoff ótimo por uma métrica técnica ou de negócio.

        ``filtro`` pode ser uma função que recebe a tabela e devolve uma
        máscara booleana. Assim, restrições específicas do projeto continuam
        fora desta função generalista.
        """
        tabela = tabela_cutoffs.copy()
        if criterio not in tabela.columns:
            raise ValueError(f"Critério '{criterio}' não encontrado na tabela.")
        if filtro is not None:
            mascara = filtro(tabela.copy())
            tabela = tabela.loc[np.asarray(mascara, dtype=bool)]
        tabela = tabela.dropna(subset=[criterio])
        if tabela.empty:
            raise ValueError('Nenhum cutoff atende aos critérios informados.')
        indice = tabela[criterio].idxmax() if maximizar else tabela[criterio].idxmin()
        return tabela.loc[indice].copy()

    @staticmethod
    def cutoff_curva_precision_recall(y_true, y_predict_proba, criterio='f1',
                                      precision_minima=None, recall_minimo=None,
                                      beta=1.0):
        """Seleciona cutoff diretamente pela curva Precision–Recall.

        Parameters
        ----------
        criterio : {'f_beta', 'precision', 'recall'}
            Métrica maximizada após a aplicação das restrições opcionais.
        precision_minima, recall_minimo : float, opcional
            Restrições operacionais entre 0 e 1.
        beta : float
            Peso do recall no F-beta. ``beta=1`` equivale ao F1;
            ``beta>1`` prioriza recall e ``beta<1`` prioriza precision.

        Returns
        -------
        melhor : Series com cutoff e métricas selecionadas.
        curva : DataFrame completo para análise e visualização.
        """
        y = np.asarray(y_true).ravel().astype(int)
        proba = np.asarray(y_predict_proba)
        if proba.ndim == 2:
            proba = proba[:, 1]
        proba = proba.ravel().astype(float)
        if len(y) != len(proba):
            raise ValueError('y_true e y_predict_proba devem ter o mesmo tamanho.')
        if set(np.unique(y)) - {0, 1} or np.unique(y).size < 2:
            raise ValueError('y_true deve conter as duas classes binárias 0 e 1.')
        if beta <= 0:
            raise ValueError('beta deve ser maior que zero.')
        for nome, valor in {'precision_minima': precision_minima,
                            'recall_minimo': recall_minimo}.items():
            if valor is not None and not 0 <= valor <= 1:
                raise ValueError(f'{nome} deve estar entre 0 e 1.')

        precision, recall, thresholds = precision_recall_curve(y, proba)
        # precision/recall possuem um ponto adicional sem threshold.
        precision, recall = precision[:-1], recall[:-1]
        beta2 = beta ** 2
        denominador = beta2 * precision + recall
        f_beta = np.divide(
            (1 + beta2) * precision * recall,
            denominador,
            out=np.zeros_like(denominador, dtype=float),
            where=denominador != 0,
        )
        curva = pd.DataFrame({
            'cutoff': thresholds,
            'precision': precision,
            'recall': recall,
            'f_beta': f_beta,
        })
        elegivel = pd.Series(True, index=curva.index)
        if precision_minima is not None:
            elegivel &= curva['precision'] >= precision_minima
        if recall_minimo is not None:
            elegivel &= curva['recall'] >= recall_minimo
        candidatos = curva.loc[elegivel]
        if candidatos.empty:
            raise ValueError('Nenhum cutoff atende às restrições de precision/recall.')
        if criterio not in {'f_beta', 'precision', 'recall'}:
            raise ValueError("criterio deve ser 'f_beta', 'precision' ou 'recall'.")
        melhor = candidatos.loc[candidatos[criterio].idxmax()].copy()
        return melhor, curva

    # ------------------------------------------------------------------ #
    # Diagnóstico de resíduos — regressão
    # ------------------------------------------------------------------ #
    @staticmethod
    def metricas_residuos_regressao(y_true, y_pred):
        """Resume magnitude, viés, normalidade e autocorrelação dos resíduos."""
        y = np.asarray(y_true, dtype=float).ravel()
        pred = np.asarray(y_pred, dtype=float).ravel()
        if len(y) != len(pred):
            raise ValueError('y_true e y_pred devem ter o mesmo tamanho.')
        validos = np.isfinite(y) & np.isfinite(pred)
        y, pred = y[validos], pred[validos]
        if len(y) < 3:
            raise ValueError('São necessárias ao menos três observações válidas.')
        residuos = y - pred
        normal_stat, normal_p = normaltest(residuos) if len(residuos) >= 8 else (np.nan, np.nan)
        return pd.DataFrame([{
            'n_observacoes': len(y),
            'media_residuo': residuos.mean(),
            'mediana_residuo': np.median(residuos),
            'desvio_residuo': residuos.std(ddof=1),
            'mae': mean_absolute_error(y, pred),
            'rmse': np.sqrt(mean_squared_error(y, pred)),
            'mape': mean_absolute_percentage_error(y, pred),
            'r2': r2_score(y, pred),
            'normaltest_estatistica': normal_stat,
            'normaltest_pvalor': normal_p,
            'durbin_watson': durbin_watson(residuos),
        }])

    @staticmethod
    def diagnostico_residuos_regressao(y_true, y_pred, indice=None):
        """Retorna a base linha a linha usada nas análises de resíduos."""
        y = np.asarray(y_true, dtype=float).ravel()
        pred = np.asarray(y_pred, dtype=float).ravel()
        if len(y) != len(pred):
            raise ValueError('y_true e y_pred devem ter o mesmo tamanho.')
        resultado = pd.DataFrame({'y_true': y, 'y_pred': pred}, index=indice)
        resultado['residuo'] = resultado['y_true'] - resultado['y_pred']
        resultado['residuo_absoluto'] = resultado['residuo'].abs()
        resultado['residuo_quadrado'] = resultado['residuo'] ** 2
        return resultado

    @staticmethod
    def plot_diagnostico_residuos(y_true, y_pred, titulo='Diagnóstico de Resíduos'):
        """Plota observado×previsto, resíduos, distribuição e QQ-plot."""
        base = Analytics.diagnostico_residuos_regressao(y_true, y_pred).dropna()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        sns.scatterplot(data=base, x='y_pred', y='y_true', alpha=.55, ax=axes[0, 0])
        minimo = min(base.y_true.min(), base.y_pred.min())
        maximo = max(base.y_true.max(), base.y_pred.max())
        axes[0, 0].plot([minimo, maximo], [minimo, maximo], '--', color='red')
        axes[0, 0].set(title='Observado × Previsto', xlabel='Previsto', ylabel='Observado')
        sns.scatterplot(data=base, x='y_pred', y='residuo', alpha=.55, ax=axes[0, 1])
        axes[0, 1].axhline(0, linestyle='--', color='red')
        axes[0, 1].set(title='Resíduos × Previsto', xlabel='Previsto', ylabel='Resíduo')
        sns.histplot(base.residuo, kde=True, ax=axes[1, 0], color='#1FB3E5')
        axes[1, 0].axvline(0, linestyle='--', color='red')
        axes[1, 0].set_title('Distribuição dos resíduos')
        stats.probplot(base.residuo, dist='norm', plot=axes[1, 1])
        axes[1, 1].set_title('QQ-plot dos resíduos')
        fig.suptitle(titulo, fontsize=16, fontweight='bold')
        for ax in axes.ravel():
            ax.grid(alpha=.2, linestyle='--')
        fig.tight_layout()
        return fig, axes, base

    # ------------------------------------------------------------------ #
    # Classificação multiclasse
    # ------------------------------------------------------------------ #
    @staticmethod
    def metricas_classificacao_multiclasse(y_true, y_predict=None,
                                           y_predict_proba=None, average='weighted',
                                           labels=None, nome_modelo='Modelo', etapa='teste'):
        """Consolida métricas para classificação com duas ou mais classes."""
        y = np.asarray(y_true).ravel()
        proba = None if y_predict_proba is None else np.asarray(y_predict_proba)
        if y_predict is None:
            if proba is None or proba.ndim != 2:
                raise ValueError('Informe y_predict ou probabilidades com formato (n, classes).')
            pred = np.argmax(proba, axis=1)
            if labels is not None:
                pred = np.asarray(labels)[pred]
        else:
            pred = np.asarray(y_predict).ravel()
        if len(y) != len(pred):
            raise ValueError('y_true e y_predict devem ter o mesmo tamanho.')
        classes = np.asarray(labels) if labels is not None else np.unique(np.concatenate([y, pred]))
        linha = {
            'Modelo': nome_modelo,
            'Etapa': etapa,
            'Acuracia': accuracy_score(y, pred),
            'Acuracia_balanceada': balanced_accuracy_score(y, pred),
            'Precisao': precision_score(y, pred, average=average, zero_division=0),
            'Recall': recall_score(y, pred, average=average, zero_division=0),
            'F1': f1_score(y, pred, average=average, zero_division=0),
            'Kappa': cohen_kappa_score(y, pred),
        }
        if proba is not None:
            if proba.ndim != 2 or proba.shape[0] != len(y):
                raise ValueError('y_predict_proba deve ter formato (n_amostras, n_classes).')
            linha['LogLoss'] = log_loss(y, proba, labels=classes)
            try:
                linha['AUC_OVR'] = roc_auc_score(y, proba, labels=classes,
                                                 multi_class='ovr', average=average)
                linha['AUC_OVO'] = roc_auc_score(y, proba, labels=classes,
                                                 multi_class='ovo', average=average)
            except ValueError:
                linha['AUC_OVR'] = linha['AUC_OVO'] = np.nan
        return pd.DataFrame([linha])

    @staticmethod
    def matriz_confusao_multiclasse(y_true, y_predict, labels=None, normalizar=None):
        """Retorna matriz de confusão como DataFrame, absoluta ou normalizada."""
        y = np.asarray(y_true).ravel()
        pred = np.asarray(y_predict).ravel()
        classes = np.asarray(labels) if labels is not None else np.unique(np.concatenate([y, pred]))
        normalizacoes = {None: None, 'true': 'true', 'pred': 'pred', 'all': 'all'}
        if normalizar not in normalizacoes:
            raise ValueError("normalizar deve ser None, 'true', 'pred' ou 'all'.")
        matriz = confusion_matrix(y, pred, labels=classes, normalize=normalizacoes[normalizar])
        return pd.DataFrame(matriz, index=pd.Index(classes, name='Real'),
                            columns=pd.Index(classes, name='Predito'))

    @staticmethod
    def plot_matriz_confusao_multiclasse(y_true, y_predict, labels=None,
                                         normalizar='true', titulo='Matriz de Confusão'):
        matriz = Analytics.matriz_confusao_multiclasse(y_true, y_predict, labels, normalizar)
        fig, ax = plt.subplots(figsize=(9, 7))
        formato = '.1%' if normalizar is not None else 'g'
        sns.heatmap(matriz, annot=True, fmt=formato, cmap='Blues', ax=ax,
                    linewidths=.5, cbar=True)
        ax.set_title(titulo, fontsize=15, fontweight='bold')
        fig.tight_layout()
        return fig, ax, matriz

    @staticmethod
    def plot_curvas_roc_multiclasse(y_true, y_predict_proba, labels=None,
                                    titulo='Curvas ROC Multiclasse'):
        """Plota curvas ROC one-vs-rest para cada classe."""
        y = np.asarray(y_true).ravel()
        proba = np.asarray(y_predict_proba)
        classes = np.asarray(labels) if labels is not None else np.unique(y)
        if proba.ndim != 2 or proba.shape[1] != len(classes):
            raise ValueError('Probabilidades devem ter uma coluna por classe, na ordem de labels.')
        fig, ax = plt.subplots(figsize=(10, 7))
        registros = []
        for indice, classe in enumerate(classes):
            binario = (y == classe).astype(int)
            fpr, tpr, _ = roc_curve(binario, proba[:, indice])
            auc_classe = roc_auc_score(binario, proba[:, indice])
            ax.plot(fpr, tpr, linewidth=2, label=f'{classe} — AUC {auc_classe:.3f}')
            registros.append({'classe': classe, 'auc_ovr': auc_classe})
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set(title=titulo, xlabel='Taxa de falsos positivos', ylabel='Taxa de verdadeiros positivos',
               xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=.2, linestyle='--'); ax.legend(frameon=False)
        fig.tight_layout()
        return fig, ax, pd.DataFrame(registros)

    # ------------------------------------------------------------------ #
    # Séries temporais
    # ------------------------------------------------------------------ #
    @staticmethod
    def testes_estacionariedade(serie, alpha=0.05, regression='c', nlags='auto'):
        """Executa ADF e KPSS em conjunto e resume suas hipóteses distintas."""
        valores = pd.Series(serie).dropna().astype(float)
        if len(valores) < 10:
            raise ValueError('A série precisa de ao menos dez observações válidas.')
        adf_resultado = adfuller(valores, regression=regression, autolag='AIC')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            kpss_resultado = kpss(valores, regression=regression, nlags=nlags)
        return pd.DataFrame([
            {'teste': 'ADF', 'estatistica': adf_resultado[0], 'p_valor': adf_resultado[1],
             'lags': adf_resultado[2], 'hipotese_nula': 'possui raiz unitária',
             'estacionaria': adf_resultado[1] < alpha},
            {'teste': 'KPSS', 'estatistica': kpss_resultado[0], 'p_valor': kpss_resultado[1],
             'lags': kpss_resultado[2], 'hipotese_nula': 'é estacionária',
             'estacionaria': kpss_resultado[1] >= alpha},
        ])

    @staticmethod
    def plot_autocorrelacao(serie, lags=40, titulo='Autocorrelação da Série'):
        """Plota ACF e PACF com quantidade de lags validada."""
        valores = pd.Series(serie).dropna().astype(float)
        if len(valores) < 4:
            raise ValueError('A série precisa de ao menos quatro observações válidas.')
        lags_validos = min(int(lags), max(1, len(valores) // 2 - 1))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_acf(valores, lags=lags_validos, ax=axes[0], zero=False)
        plot_pacf(valores, lags=lags_validos, ax=axes[1], zero=False, method='ywm')
        axes[0].set_title('ACF — Autocorrelação')
        axes[1].set_title('PACF — Autocorrelação Parcial')
        fig.suptitle(titulo, fontsize=15, fontweight='bold')
        fig.tight_layout()
        return fig, axes

    @staticmethod
    def decomposicao_serie_temporal(serie, periodo, modelo='additive', extrapolate_trend='freq'):
        """Decompõe uma série em tendência, sazonalidade e resíduo."""
        if modelo not in {'additive', 'multiplicative'}:
            raise ValueError("modelo deve ser 'additive' ou 'multiplicative'.")
        valores = pd.Series(serie).dropna().astype(float)
        if periodo < 2 or len(valores) < 2 * periodo:
            raise ValueError('São necessários ao menos dois ciclos completos e periodo >= 2.')
        if modelo == 'multiplicative' and np.any(valores <= 0):
            raise ValueError('Decomposição multiplicativa exige valores positivos.')
        resultado = seasonal_decompose(valores, model=modelo, period=periodo,
                                       extrapolate_trend=extrapolate_trend)
        fig = resultado.plot()
        fig.set_size_inches(14, 10)
        fig.suptitle(f'Decomposição {modelo} — período {periodo}', fontsize=15, fontweight='bold')
        fig.tight_layout()
        componentes = pd.DataFrame({
            'observado': resultado.observed,
            'tendencia': resultado.trend,
            'sazonalidade': resultado.seasonal,
            'residuo': resultado.resid,
        })
        return resultado, componentes, fig

    @staticmethod
    def analisa_tendencia(serie, janela=None):
        """Estima tendência linear e média móvel sem impor sazonalidade."""
        valores = pd.Series(serie).dropna().astype(float)
        if len(valores) < 3:
            raise ValueError('A série precisa de ao menos três observações.')
        x = np.arange(len(valores), dtype=float)
        coeficiente, intercepto = np.polyfit(x, valores.to_numpy(), 1)
        tendencia = intercepto + coeficiente * x
        janela = janela or max(2, min(len(valores) // 5, 12))
        media_movel = valores.rolling(janela, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(valores.index, valores, alpha=.55, label='Série')
        ax.plot(valores.index, tendencia, linewidth=2.2, label='Tendência linear')
        ax.plot(valores.index, media_movel, linewidth=2, label=f'Média móvel ({janela})')
        ax.set(title='Análise de Tendência', xlabel='Tempo', ylabel=valores.name or 'Valor')
        ax.grid(alpha=.2, linestyle='--'); ax.legend(frameon=False)
        fig.tight_layout()
        resumo = pd.DataFrame([{
            'coeficiente_tendencia': coeficiente,
            'intercepto': intercepto,
            'direcao': 'crescente' if coeficiente > 0 else ('decrescente' if coeficiente < 0 else 'estável'),
            'janela_media_movel': janela,
        }])
        return resumo, pd.Series(tendencia, index=valores.index, name='tendencia'), fig, ax


    @staticmethod
    def plota_barras_agrupadas(df, x, y, titulo, ylabel=None, figsize=(10, 5)):
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=df, x=x, y=y, ax=ax)
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                        textcoords='offset points')
        ax.set_title(f'{titulo}')
        ax.set_xlabel(f'{x}', fontsize = 14)
        ax.set_ylabel(ylabel or str(y), fontsize=14)
        ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
        fig.tight_layout()
        return fig, ax

    @staticmethod
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

    @staticmethod
    def plot_linear_separability(feature_1, feature_2, x_train, y_train, target):
        # Crie um DataFrame para facilitar a visualização com seaborn
        df = pd.DataFrame(x_train, columns=[feature_1, feature_2])
        y = y_train[target] if isinstance(y_train, pd.DataFrame) else np.asarray(y_train).ravel()
        df['Target'] = np.asarray(y)

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

    @staticmethod
    def plot_shap_beeswarm(modelo_pipeline, X, titulo="SHAP Beeswarm",
                           etapa_modelo=None, etapa_preprocessamento=None, max_display=30):
        """
        Plota apenas o gráfico Beeswarm de valores SHAP para um modelo XGBoost dentro de um Pipeline sklearn.
        """
        if hasattr(modelo_pipeline, 'named_steps'):
            etapas = modelo_pipeline.named_steps
            nome_modelo = etapa_modelo or list(etapas)[-1]
            model = etapas[nome_modelo]
            if etapa_preprocessamento is None and len(etapas) > 1:
                etapa_preprocessamento = list(etapas)[-2]
            if etapa_preprocessamento is not None:
                transformer = etapas[etapa_preprocessamento]
                X_transformado = transformer.transform(X)
                feature_names = (transformer.get_feature_names_out()
                                 if hasattr(transformer, 'get_feature_names_out') else None)
            else:
                X_transformado, feature_names = X, getattr(X, 'columns', None)
        else:
            model, X_transformado = modelo_pipeline, X
            feature_names = getattr(X, 'columns', None)

        # --- 3) Cria o SHAP Explainer e calcula valores ---
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(X_transformado)

        # --- 4) Plota apenas o Beeswarm ---
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, X_transformado, feature_names=feature_names,
                          max_display=max_display, show=False, plot_size=(12, 6))
        plt.title(titulo, fontsize=14)
        plt.tight_layout()
        return plt.gcf(), plt.gca()


    # -----------------------------------------------------------------------------
    # Funções gerais de rating, migração e análise de risco
    # Rotinas extraídas e generalizadas a partir do notebook de modelagem.
    # -----------------------------------------------------------------------------

    @staticmethod
    def plot_matriz_migracao(matriz, titulo='Matriz de Migração', fmt='.2f',
                             xlabel='Categoria de destino', ylabel='Categoria de origem'):
        """Plota uma matriz de migração entre classificações quaisquer."""
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(matriz, annot=True, cmap='Blues', fmt=fmt, linewidths=.5, ax=ax)
        ax.set_title(titulo)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def plot_resumo_grupos(df, grupo, coluna_evento=None, coluna_valor=None,
                           titulo='Resumo por Grupo'):
        """Plota volumetria e até duas medidas agregadas por grupo."""
        agregacoes = {'quantidade': (grupo, 'size')}
        if coluna_evento is not None:
            agregacoes['eventos'] = (coluna_evento, 'sum')
        if coluna_valor is not None:
            agregacoes['valor_total'] = (coluna_valor, 'sum')
        resumo = df.groupby(grupo, observed=True).agg(
            **agregacoes,
        ).reset_index()
        fig, ax1 = plt.subplots(figsize=(12, 6))
        sns.barplot(data=resumo, x=grupo, y='quantidade', color='#1FB3E5', ax=ax1)
        ax1.set(title=titulo, xlabel=grupo, ylabel='Quantidade')
        ax2 = ax1.twinx()
        if coluna_evento is not None:
            sns.lineplot(data=resumo, x=grupo, y='eventos', marker='o', color='blue', ax=ax2,
                         label='Eventos')
        if coluna_valor is not None:
            sns.lineplot(data=resumo, x=grupo, y='valor_total', marker='o', color='red', ax=ax2,
                         label='Valor total')
        ax2.set_ylabel('Medidas agregadas')
        fig.tight_layout()
        return fig, (ax1, ax2), resumo

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def plot_histograms_comparison(df_train, df_valid, df_test, df_oot, column):
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
        
        return fig, axes, pd.DataFrame(stats).T.rename_axis('conjunto').reset_index()

    @staticmethod
    def plot_boxplot_comparison(df_train, df_valid, df_test, df_oot, column):
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
        return fig, (ax1, ax2, ax2_secondary), summary_df

    @staticmethod
    def plot_comparative_density(df_train, df_valid, df_test, df_oot, column):
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
                except (ValueError, np.linalg.LinAlgError):
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
        return fig, ax

    @staticmethod
    def plot_bandas_temporais_unico(df_train, df_valid, df_test, df_oot, variavel_temporal,
                                    coluna_predicao='y_predict', coluna_inferior='ic_inferior',
                                    coluna_superior='ic_superior', coluna_target=None,
                                    titulo=None, figsize=(16, 8)):
        """
        Plota um único gráfico com bandas de IC agrupadas pela variável temporal
        """
        
        # Combinar todos os datasets
        bases = []
        for base, nome in ((df_train, 'Treino'), (df_valid, 'Validação'),
                           (df_test, 'Teste'), (df_oot, 'Out-of-Time')):
            aux = base.copy()
            aux['dataset'] = nome
            bases.append(aux)
        df_combinado = pd.concat(bases, ignore_index=True)
        
        # Garantir que a variável temporal existe
        if variavel_temporal not in df_combinado.columns:
            raise ValueError(f"Variável temporal '{variavel_temporal}' não encontrada nos dados")
        
        # Agrupar pela variável temporal
        colunas_obrigatorias = {coluna_inferior, coluna_predicao, coluna_superior}
        ausentes = colunas_obrigatorias - set(df_combinado.columns)
        if ausentes:
            raise ValueError(f"Colunas obrigatórias ausentes: {sorted(ausentes)}")
        agregacoes = {coluna_inferior: 'mean', coluna_predicao: 'mean', coluna_superior: 'mean'}
        if coluna_target is not None:
            agregacoes[coluna_target] = 'mean'
        df_agrupado = (df_combinado.groupby(variavel_temporal).agg(agregacoes)
                       .reset_index().sort_values(variavel_temporal))
        
        # Criar figura com fundo branco
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('white')
        
        # Dados para plotagem
        x = df_agrupado[variavel_temporal]
        y_inf = df_agrupado[coluna_inferior]
        y_pred = df_agrupado[coluna_predicao]
        y_sup = df_agrupado[coluna_superior]
        
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
        ax.set_title(titulo or f'Intervalo de predição ao longo de {variavel_temporal}', fontsize=14, fontweight='bold')
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
        if coluna_target is not None and coluna_target in df_combinado.columns:
            cobertura_total = ((df_combinado[coluna_target] >= df_combinado[coluna_inferior]) &
                              (df_combinado[coluna_target] <= df_combinado[coluna_superior])).mean()
            
            # Calcular cobertura por período
            df_combinado['dentro_ic'] = ((df_combinado[coluna_target] >= df_combinado[coluna_inferior]) &
                                         (df_combinado[coluna_target] <= df_combinado[coluna_superior])).astype(int)
            
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

    @staticmethod
    def plot_shap_one_sample(model, X, position_sample, titulo):

        # Pega o modelo dentro do pipeline
        model_lgbm = model
        X_single = X.loc[[position_sample]]
        explainer = shap.Explainer(model_lgbm, X)
        shap_values = explainer(X_single)
        shap.plots.waterfall(shap_values[0])

    @staticmethod
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

    @staticmethod
    def plot_curva_precision_recall(y_true, y_predict_proba, cutoff=None,
                                    titulo='Curva Precision–Recall',
                                    nome_modelo='Modelo'):
        """Plota a curva Precision–Recall e destaca um cutoff opcional."""
        y = np.asarray(y_true).ravel().astype(int)
        proba = np.asarray(y_predict_proba)
        if proba.ndim == 2:
            proba = proba[:, 1]
        proba = proba.ravel().astype(float)
        precision, recall, thresholds = precision_recall_curve(y, proba)
        pr_auc = average_precision_score(y, proba)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(recall, precision, color='#1FB3E5', linewidth=2.5,
                label=f'{nome_modelo} — PR-AUC: {pr_auc:.3f}')
        ax.axhline(y.mean(), color='gray', linestyle='--', linewidth=1.3,
                   label=f'Referência aleatória: {y.mean():.3f}')
        if cutoff is not None and len(thresholds):
            indice = int(np.argmin(np.abs(thresholds - cutoff)))
            ax.scatter(recall[indice], precision[indice], s=90, color='#E74C3C',
                       zorder=5, label=f'Cutoff: {thresholds[indice]:.3f}')
            ax.annotate(
                f'Precision: {precision[indice]:.3f}\nRecall: {recall[indice]:.3f}',
                (recall[indice], precision[indice]), xytext=(12, -35),
                textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=.9),
            )
        ax.set(title=titulo, xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=.2, linestyle='--')
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig, ax

plota_barras_agrupadas = Analytics.plota_barras_agrupadas
plota_dispersao = Analytics.plota_dispersao
plot_linear_separability = Analytics.plot_linear_separability
plot_shap_beeswarm = Analytics.plot_shap_beeswarm
plot_matriz_migracao = Analytics.plot_matriz_migracao
plot_resumo_grupos = Analytics.plot_resumo_grupos
plota_barras = Analytics.plota_barras
plota_histograma = Analytics.plota_histograma
plota_boxplot = Analytics.plota_boxplot
plota_grafico_linhas = Analytics.plota_grafico_linhas
plot_histograms_comparison = Analytics.plot_histograms_comparison
plot_boxplot_comparison = Analytics.plot_boxplot_comparison
plot_comparative_density = Analytics.plot_comparative_density
plot_bandas_temporais_unico = Analytics.plot_bandas_temporais_unico
plot_shap_one_sample = Analytics.plot_shap_one_sample
plot_shap_one_sample_original_scale = Analytics.plot_shap_one_sample_original_scale
plot_calibracao = Analytics.plot_calibracao
plot_curva_precision_recall = Analytics.plot_curva_precision_recall


ks_test = Analytics.ks_test
metricas_regressao = Analytics.metricas_regressao
metricas_modelos_juntos_regressao = Analytics.metricas_modelos_juntos_regressao
metricas_classificacao = Analytics.metricas_classificacao
metricas_classificacao_juntos = Analytics.metricas_classificacao_juntos
metricas_calibracao = Analytics.metricas_calibracao
discretiza_variavel = Analytics.discretiza_variavel
calcula_psi_amostras = Analytics.calcula_psi_amostras
calcula_psi_temporal = Analytics.calcula_psi_temporal
analisa_normalidade = Analytics.analisa_normalidade
analisa_outliers = Analytics.analisa_outliers
teste_hipotese_duas_amostras_independentes = Analytics.teste_hipotese_duas_amostras_independentes
teste_hipotese_muitas_amostras_independentes = Analytics.teste_hipotese_muitas_amostras_independentes
teste_hipotese_duas_variaveis_categoricas = Analytics.teste_hipotese_duas_variaveis_categoricas
analisa_correlacao = Analytics.analisa_correlacao
analisa_distribuicao_via_percentis = Analytics.analisa_distribuicao_via_percentis
taxa_por_grupo = Analytics.taxa_por_grupo
calcula_woe_iv = Analytics.calcula_woe_iv
calcular_rating = Analytics.calcular_rating
criar_base_rating = Analytics.criar_base_rating
comparar_grupos = Analytics.comparar_grupos
matriz_migracao = Analytics.matriz_migracao
resumo_migracao = Analytics.resumo_migracao
ajusta_calibrador_score = Analytics.ajusta_calibrador_score
prediz_calibrador = Analytics.prediz_calibrador
calcula_perda_esperada = Analytics.calcula_perda_esperada
metricas_intervalo_regressao = Analytics.metricas_intervalo_regressao
diagnostico_intervalos_regressao = Analytics.diagnostico_intervalos_regressao
tabela_cutoffs = Analytics.tabela_cutoffs
seleciona_cutoff = Analytics.seleciona_cutoff
cutoff_curva_precision_recall = Analytics.cutoff_curva_precision_recall
metricas_residuos_regressao = Analytics.metricas_residuos_regressao
diagnostico_residuos_regressao = Analytics.diagnostico_residuos_regressao
plot_diagnostico_residuos = Analytics.plot_diagnostico_residuos
metricas_classificacao_multiclasse = Analytics.metricas_classificacao_multiclasse
matriz_confusao_multiclasse = Analytics.matriz_confusao_multiclasse
plot_matriz_confusao_multiclasse = Analytics.plot_matriz_confusao_multiclasse
plot_curvas_roc_multiclasse = Analytics.plot_curvas_roc_multiclasse
testes_estacionariedade = Analytics.testes_estacionariedade
plot_autocorrelacao = Analytics.plot_autocorrelacao
decomposicao_serie_temporal = Analytics.decomposicao_serie_temporal
analisa_tendencia = Analytics.analisa_tendencia
