"""Métodos generalistas selecionados do arquivo original."""
from ._shared import *

class Visualizacao:
    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def plot_matriz_migracao_rating(matriz, titulo='Matriz de Migração de Rating', fmt='.2f'):
        """Plota uma matriz de migração já calculada."""
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(matriz, annot=True, cmap='Blues', fmt=fmt, linewidths=.5, ax=ax)
        ax.set_title(titulo)
        ax.set_xlabel('Rating de destino')
        ax.set_ylabel('Rating de origem')
        fig.tight_layout()
        return fig, ax

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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


# Aliases funcionais para compatibilidade com notebooks antigos.
plota_barras_agrupadas = Visualizacao.plota_barras_agrupadas
plota_dispersao = Visualizacao.plota_dispersao
auc_ks_juntos = Visualizacao.auc_ks_juntos
auc_ks_final = Visualizacao.auc_ks_final
plot_linear_separability = Visualizacao.plot_linear_separability
plot_shap_beeswarm = Visualizacao.plot_shap_beeswarm
plot_matriz_migracao_rating = Visualizacao.plot_matriz_migracao_rating
plot_rating_risco = Visualizacao.plot_rating_risco
plota_barras = Visualizacao.plota_barras
plota_histograma = Visualizacao.plota_histograma
plota_boxplot = Visualizacao.plota_boxplot
plota_grafico_linhas = Visualizacao.plota_grafico_linhas
plot_histograms_comparison = Visualizacao.plot_histograms_comparison
plot_boxplot_comparison = Visualizacao.plot_boxplot_comparison
plot_comparative_density = Visualizacao.plot_comparative_density
plot_bandas_temporais_unico = Visualizacao.plot_bandas_temporais_unico
plot_shap_one_sample = Visualizacao.plot_shap_one_sample
plot_shap_one_sample_original_scale = Visualizacao.plot_shap_one_sample_original_scale
plot_calibracao = Visualizacao.plot_calibracao
