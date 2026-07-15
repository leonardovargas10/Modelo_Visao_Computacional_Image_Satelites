"""Métodos generalistas selecionados do arquivo original."""
from ._shared import *

class Analytics:

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
    def calcula_woe_iv(df, feature, target, plotar=True):
        """Calcula WOE e IV de uma variável em relação a um target binário
        (0 = bom/não-evento, 1 = mau/evento). Substitui as antigas `woe()` e
        `iv()` duplicadas — agora uma única função devolve a tabela completa
        (com woe e iv) e, opcionalmente, plota."""
        evento = df.loc[df[target] == 1].groupby(feature, as_index=False)[target].count().rename({target: 'evento'}, axis=1)
        nao_evento = df.loc[df[target] == 0].groupby(feature, as_index=False)[target].count().rename({target: 'nao_evento'}, axis=1)

        tabela = evento.merge(nao_evento, on=feature, how='outer').fillna(0)
        tabela['percent_evento'] = tabela['evento'] / max(tabela['evento'].sum(), 1)
        tabela['percent_nao_evento'] = tabela['nao_evento'] / max(tabela['nao_evento'].sum(), 1)

        eps = 1e-6
        tabela['woe'] = np.log((tabela['percent_evento'] + eps) / (tabela['percent_nao_evento'] + eps)).round(4)
        tabela['iv'] = ((tabela['percent_evento'] - tabela['percent_nao_evento']) * tabela['woe']).sum()
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
        decis = pd.qcut(probabilidades.rank(method='first'), q=n_ratings, labels=False).astype('Int64')
        rating = n_ratings - 1 - decis if maior_rating_menor_risco else decis
        return rating.rename('rating')

    @staticmethod
    def criar_base_rating(df, probabilidade, nome_rating='rating_model',
                           target=None, coluna_exposicao=None, n_ratings=10):
        resultado = df.copy()
        resultado['Probability of Default'] = np.asarray(probabilidade)
        resultado[nome_rating] = Analytics.calcular_rating(resultado['Probability of Default'], n_ratings=n_ratings).to_numpy()
        resultado['qt_pessoas_rating'] = resultado.groupby(nome_rating)[nome_rating].transform('size')
        if coluna_exposicao is not None:
            resultado['expected_loss'] = resultado['Probability of Default'] * resultado[coluna_exposicao]
        if target is not None:
            resultado['bad_rate_rating'] = resultado.groupby(nome_rating)[target].transform('mean')
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
            ['Downgrade', 'Upgrade'],
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
        if metodo == 'isotonic':
            modelo = IsotonicRegression(out_of_bounds='clip').fit(score, y)
        else:
            modelo = LogisticRegression(C=1, random_state=random_state).fit(np.asarray(score).reshape(-1, 1), y)
        return modelo

    @staticmethod
    def prediz_calibrador(modelo, score, metodo='isotonic'):
        if metodo == 'isotonic':
            return modelo.predict(score)
        return modelo.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1]


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
discretiza_variavel = Analytics.discretiza_variavel  # a discretização genérica mora em Metricas
ajusta_calibrador_score = Analytics.ajusta_calibrador_score
prediz_calibrador = Analytics.prediz_calibrador