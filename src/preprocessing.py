"""Métodos generalistas selecionados do arquivo original."""
from ._shared import *

class PreProcessamento:
    @staticmethod
    def separa_treino_teste(target, dados, size, estratificar=False, random_state=42):
        x = dados.drop(target, axis = 1)
        y = dados[target]
        stratify = y if estratificar else None
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=size, random_state=random_state, stratify=stratify
        )

        df_train = pd.concat([y_train, x_train], axis = 1)
        df_test = pd.concat([y_test, x_test], axis = 1)

        return df_train, df_test

    @staticmethod
    def discretiza_variavel(df, variavel_quant, variavel_qualit, bins, labels, right=True, inplace=False):
        resultado = df if inplace else df.copy()
        resultado[variavel_qualit] = pd.cut(
            resultado[variavel_quant],
            bins= bins, 
            labels= labels, 
            right = right
        )
        resultado.drop(variavel_quant, axis=1, inplace=True)
        return resultado

    @staticmethod
    def simple_imputer(df, strategy='median', fill_value=None):

        df_aux = df.copy()
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        imputer.fit(df_aux)

        return imputer

    @staticmethod
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

    @staticmethod
    def formatar_valor_milhoes(valor, pos=None):
        """Formata um valor monetário em milhões de reais."""
        return f'R$ {valor / 1e6:.2f} MM'

    @staticmethod
    def abbreviate_number(number):
        """Abrevia números com os sufixos K, MM e B."""
        for fator, sufixo in ((1e9, 'B'), (1e6, 'MM'), (1e3, 'K')):
            if abs(number) >= fator:
                return f'{number / fator:.1f}'.rstrip('0').rstrip('.') + sufixo
        return str(int(number))

    @staticmethod
    def muda_tipagem_variavel(df, feature, type, inplace=False):
        """Converte uma coluna sem usar sentinelas nem alterar outras colunas."""
        if type not in {"int", "float"}:
            raise ValueError("type deve ser 'int' ou 'float'.")
        serie = pd.to_numeric(df[feature], errors='coerce')
        serie = serie.astype('Int64') if type == "int" else serie.astype(float)
        if inplace:
            df[feature] = serie
        return serie

    @staticmethod
    def transform_to_percentiles(df, n, variavel_continua):
        # Calcula os limites dos percentiles
        # Aplica a função qcut para transformar a variável em percentiles
        percentiles = pd.qcut(df[variavel_continua], q=n, labels=False, duplicates='drop')
        
        return percentiles

    @staticmethod
    def separa_feature_target(target, dados):
        x = dados.drop(target, axis = 1)
        y = dados[target]

        return x, y

# ------------------------------------------------------------------ #
    # Winsorização (cap por percentil)
    # ------------------------------------------------------------------ #
    @staticmethod
    def calcula_limites_winsorizacao(df, colunas, percentil_inferior=0.01, percentil_superior=0.99):
        """Aprende os limites de winsorização (percentis) em uma base
        (treino) para reaplicar depois em validação/teste/OOT via
        `aplica_winsorizacao` — nunca recalcule os limites em cima do teste,
        isso é vazamento.

        Por padrão capa em p1/p99; ajuste `percentil_superior=0.99` para
        capar só a cauda superior (ex.: renda, valor de empréstimo) mantendo
        `percentil_inferior=0.0` se não quiser tratar a cauda inferior.

        Returns
        -------
        dict {coluna: (limite_inferior, limite_superior)}
        """
        limites = {}
        for coluna in colunas:
            serie = df[coluna].dropna()
            limites[coluna] = (
                serie.quantile(percentil_inferior),
                serie.quantile(percentil_superior),
            )
        return limites

    @staticmethod
    def aplica_winsorizacao(df, limites):
        """Capa (clip) os valores de cada coluna aos limites informados —
        reduz a influência de outliers extremos no modelo sem descartar a
        observação inteira.

        Parameters
        ----------
        limites : dict {coluna: (limite_inferior, limite_superior)},
            tipicamente a saída de `calcula_limites_winsorizacao`.
        """
        resultado = df.copy()
        for coluna, (limite_inferior, limite_superior) in limites.items():
            resultado[coluna] = resultado[coluna].clip(lower=limite_inferior, upper=limite_superior)
        return resultado

# ------------------------------------------------------------------ #
    # Balanceamento de classes
    # ------------------------------------------------------------------ #
    @staticmethod
    def calcula_class_weight(y, metodo='balanced', formato='scale_pos_weight'):
        """Calcula o peso ideal da classe minoritária a partir do
        desbalanceamento observado em `y` — evita ficar chutando
        `class_weight`/`scale_pos_weight` manualmente a cada projeto.

        Parameters
        ----------
        y : array-like binário (0/1).
        metodo : {'balanced', 'sqrt_balanced'}
            'balanced'      -> peso = n_negativos / n_positivos (o mesmo
                                critério do `class_weight='balanced'` do
                                sklearn, mas explícito e reaproveitável fora
                                dele, por exemplo em LightGBM).
            'sqrt_balanced' -> raiz do peso balanceado, para suavizar casos
                                de desbalanceamento extremo (ex.: fraude,
                                onde o peso 'balanced' puro tende a
                                super-corrigir e gerar excesso de falsos
                                positivos).
        formato : {'scale_pos_weight', 'dict', 'sklearn'}
            'scale_pos_weight' -> retorna só o float (uso direto em
                                   LightGBM: `scale_pos_weight=peso`).
            'dict'             -> retorna {0: 1, 1: peso} (uso em
                                   `class_weight` de sklearn/LightGBM
                                   no formato dict).
            'sklearn'          -> retorna {0: peso_0, 1: peso_1} calculado
                                   via `sklearn.utils.class_weight.compute_class_weight`,
                                   útil quando o problema não é 0/1 estrito
                                   ou quando se quer o peso oficial do sklearn.

        Returns
        -------
        float ou dict, conforme `formato`.
        """
        y = np.asarray(y).ravel()
        classes, contagens = np.unique(y, return_counts=True)

        if formato == 'sklearn':
            pesos = compute_class_weight(class_weight='balanced', classes=classes, y=y)
            return dict(zip(classes, pesos))

        if len(classes) != 2 or set(classes) != {0, 1}:
            raise ValueError("calcula_class_weight espera um target binário (0/1) para 'scale_pos_weight'/'dict'.")

        n_negativos = contagens[classes == 0][0] if 0 in classes else contagens[0]
        n_positivos = contagens[classes == 1][0] if 1 in classes else contagens[1]

        peso = n_negativos / max(n_positivos, 1)

        if metodo == 'sqrt_balanced':
            peso = np.sqrt(peso)
        elif metodo != 'balanced':
            raise ValueError("metodo deve ser 'balanced' ou 'sqrt_balanced'.")

        if formato == 'scale_pos_weight':
            return float(peso)
        elif formato == 'dict':
            return {0: 1, 1: float(peso)}
        else:
            raise ValueError("formato deve ser 'scale_pos_weight', 'dict' ou 'sklearn'.")

    # ------------------------------------------------------------------ #
    # Qualidade de dados
    # ------------------------------------------------------------------ #
    @staticmethod
    def calcula_percentual_nulos(df, colunas=None, ordenar_decrescente=True):
        """Calcula o percentual (e a contagem) de valores nulos por variável
        — primeiro passo de qualquer EDA de qualidade de dados, e insumo
        direto para decidir o que imputar, o que descartar ou o que virar
        `missing flag`.

        Parameters
        ----------
        colunas : lista opcional de colunas a analisar; se None, usa todas
            as colunas do df.

        Returns
        -------
        DataFrame com colunas: variavel, qtd_nulos, percentual_nulos (%),
        ordenado do maior para o menor percentual de nulos por padrão.
        """
        base = df[colunas] if colunas is not None else df
        qtd_nulos = base.isnull().sum()
        percentual_nulos = (qtd_nulos / len(base) * 100).round(2) if len(base) else qtd_nulos.astype(float)

        resultado = pd.DataFrame({
            'variavel': qtd_nulos.index,
            'qtd_nulos': qtd_nulos.values,
            'percentual_nulos': percentual_nulos.values,
        })

        if ordenar_decrescente:
            resultado = resultado.sort_values('percentual_nulos', ascending=False)

        return resultado.reset_index(drop=True)


# Aliases funcionais para compatibilidade com notebooks antigos.
separa_treino_teste = PreProcessamento.separa_treino_teste
discretiza_variavel = PreProcessamento.discretiza_variavel
simple_imputer = PreProcessamento.simple_imputer
target_encoder_bad_rate = PreProcessamento.target_encoder_bad_rate
formatar_valor_milhoes = PreProcessamento.formatar_valor_milhoes
abbreviate_number = PreProcessamento.abbreviate_number
muda_tipagem_variavel = PreProcessamento.muda_tipagem_variavel
transform_to_percentiles = PreProcessamento.transform_to_percentiles
separa_feature_target = PreProcessamento.separa_feature_target
calcula_limites_winsorizacao = PreProcessamento.calcula_limites_winsorizacao
aplica_winsorizacao = PreProcessamento.aplica_winsorizacao
calcula_class_weight = PreProcessamento.calcula_class_weight
calcula_percentual_nulos = PreProcessamento.calcula_percentual_nulos
