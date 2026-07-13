# Projeto de Benchmark em Visão Computacional

---

# Objetivo

O objetivo deste projeto é realizar um benchmark entre diferentes estratégias de aprendizado de representações em Visão Computacional, comparando desde modelos treinados do zero até arquiteturas baseadas em Transformers.

O foco do projeto não será apenas obter a maior acurácia, mas compreender o **trade-off entre desempenho, custo computacional, interpretabilidade, robustez e qualidade das representações aprendidas**.

Além da tarefa de classificação, o projeto também explorará:

- Transfer Learning
- Fine-Tuning
- LoRA
- Extração de Embeddings
- Modelos clássicos de Machine Learning utilizando embeddings
- Engenharia de atributos baseada em embeddings
- Técnicas de interpretabilidade
- Otimização de hiperparâmetros

O notebook deverá servir tanto como um benchmark quanto como um material completo de estudo em Deep Learning aplicado à Visão Computacional.

---

# Estrutura do Notebook

O notebook deverá ser desenvolvido integralmente em um único arquivo **Jupyter Notebook (.ipynb)**.

A estrutura deverá seguir exatamente o mesmo padrão visual utilizado no notebook de Forecasting, utilizando títulos HTML semelhantes aos exemplos abaixo.

```html
# <font color='red' style='font-size: 40px;'> Título Principal </font>
<hr style='border: 2px solid red;'>
```

e

```html
# <font color='green' style='font-size: 30px;'> Subtítulo </font>
<hr style='border: 2px solid green;'>
```

Cada etapa do projeto deverá possuir sua própria seção contendo:

- Explicação teórica
- Fundamentação matemática (quando pertinente)
- Código comentado
- Interpretação dos resultados
- Conclusões da etapa

Todo o notebook deverá ser escrito de forma didática, funcionando como um guia completo de estudo.

---

# Estrutura do Projeto

# <font color='red' style='font-size: 40px;'> Problema de Negócio </font>

- Objetivo do projeto
- Dataset utilizado
- Tipo de problema
- Classes existentes
- Motivação do benchmark
- Critérios de comparação entre os modelos

---

# <font color='red' style='font-size: 40px;'> Bibliotecas Utilizadas </font>

Importação de todas as bibliotecas utilizadas no projeto.

---

# <font color='red' style='font-size: 40px;'> Funções Auxiliares </font>

Criar funções reutilizáveis para todo o projeto.

Exemplos:

- cálculo de métricas
- matriz de confusão
- curvas ROC
- visualização de imagens
- visualização dos filtros
- visualização dos feature maps
- comparação entre modelos
- extração de embeddings
- medição de tempo
- contagem de parâmetros
- cálculo do tamanho do modelo
- geração de gráficos

---

# <font color='red' style='font-size: 40px;'> Leitura dos Dados </font>

## Estrutura das Pastas

## Carregamento do Dataset

## Separação

- treino
- validação
- teste

---

# <font color='red' style='font-size: 40px;'> Análise Exploratória das Imagens </font>

Realizar uma EDA completa.

Exemplos:

- quantidade de imagens por classe
- distribuição das classes
- resolução das imagens
- largura × altura
- estatísticas gerais
- imagens por classe
- visualização de amostras
- distribuição das resoluções
- análise de possíveis problemas no dataset

---

# <font color='red' style='font-size: 40px;'> Pré-processamento </font>

Desenvolver toda a etapa de preparação dos dados.

Incluindo:

- Resize
- Normalização
- Data Augmentation
- DataLoaders

Explicar detalhadamente:

- Resize
- Normalização
- Cada técnica de Data Augmentation
- Como cada transformação influencia overfitting e underfitting
- Pipeline de transformação das imagens

---

# <font color='red' style='font-size: 40px;'> Baseline — CNN Desenvolvida do Zero </font>

## Construção da Arquitetura

Explicar detalhadamente cada decisão arquitetural.

### Número de camadas convolucionais

### Número de filtros

### Kernel Size

### Padding

### Stride

### Campo Receptivo (Receptive Field)

### Função de Ativação

Comparar:

- ReLU
- LeakyReLU
- GELU
- SiLU

Justificar a escolha.

### Pooling

Comparar:

- Max Pooling
- Average Pooling
- Global Average Pooling

### Batch Normalization

Explicar:

- normalização das ativações
- estabilidade do treinamento
- aceleração da convergência

### Dropout

Explicar:

- regularização
- redução do overfitting

### Flatten

Explicar a transformação dos Feature Maps em vetor.

### Camadas Fully Connected

### Camada de Saída

### Contagem de Parâmetros

### Resumo da Arquitetura

Apresentar um Model Summary completo.

### Fluxo da Informação

Mostrar como a imagem percorre toda a rede.

---

## Inspeção da Arquitetura

Visualizar:

- arquitetura da rede
- quantidade de parâmetros
- tamanho das ativações
- dimensão dos Feature Maps em cada camada

---

## Função de Perda

Explicar matematicamente a CrossEntropy Loss.

---

## Otimizador

Apresentar e comparar:

- SGD
- Momentum
- Adam
- AdamW

Justificar a escolha do Adam.

---

## Learning Rate

Explicar:

- conceito
- influência no treinamento
- escolha do valor inicial

---

## Scheduler

Comparar:

- StepLR
- Cosine Annealing
- ReduceLROnPlateau

Justificar a escolha.

---

## Weight Decay

Explicar a regularização L2.

---

## Early Stopping

Explicar quando interromper o treinamento.

---

## Loop de Treinamento

Explicar detalhadamente:

- Forward
- Cálculo da Loss
- Backpropagation
- Atualização dos Pesos
- Zero Grad

Mostrar um fluxograma do processo.

---

## Acompanhamento do Treinamento

Gerar gráficos contendo:

- Loss de treino
- Loss de validação
- Accuracy
- Learning Rate
- Tempo por época

Interpretar cada gráfico.

---

## Avaliação

Apresentar:

- Accuracy
- Precision
- Recall
- F1-score
- AUC
- Matriz de Confusão
- Curvas ROC (quando aplicável)

---

## O que a CNN Aprendeu?

Visualizar:

- primeiros filtros
- Feature Maps das primeiras camadas
- Feature Maps das últimas camadas

Explicar a evolução das representações:

Imagem

↓

Bordas

↓

Linhas

↓

Texturas

↓

Partes do objeto

↓

Objeto completo

---

## Otimização de Hiperparâmetros com Optuna

Explicar:

- diferença entre parâmetros e hiperparâmetros
- Grid Search
- Random Search
- Bayesian Optimization
- TPE
- Pruning

Definir o espaço de busca para:

- Learning Rate
- Batch Size
- Weight Decay
- Número de filtros
- Número de camadas
- Kernel Size
- Dropout
- Scheduler

Executar a otimização.

Apresentar:

- melhores hiperparâmetros
- importância dos hiperparâmetros
- evolução dos trials

---

## Avaliação Final

Comparar:

- CNN Base
- CNN Otimizada

Apresentar todas as métricas.

---

# <font color='red' style='font-size: 40px;'> Fine-Tuning de uma ResNet </font>

Selecionar uma arquitetura robusta.

Exemplo:

- ResNet50

Explicar:

- Transfer Learning
- congelamento de camadas
- substituição da camada final
- treinamento
- avaliação

Comparar com a CNN desenvolvida do zero.

---

# <font color='red' style='font-size: 40px;'> Fine-Tuning de um Vision Transformer </font>

Selecionar um ViT pré-treinado.

Explicar:

- funcionamento do Vision Transformer
- Patch Embeddings
- Self-Attention
- diferenças em relação às CNNs

Realizar:

- carregamento
- adaptação da cabeça de classificação
- treinamento
- avaliação

---

# <font color='red' style='font-size: 40px;'> Fine-Tuning utilizando LoRA </font>

Aplicar LoRA ao Vision Transformer.

Explicar:

- Low Rank Adaptation
- Matrizes A e B
- Rank
- Alpha
- Camadas adaptadas
- Quantidade de parâmetros treináveis

Comparar:

- parâmetros totais
- parâmetros treináveis
- tempo de treinamento
- memória utilizada

---

# <font color='red' style='font-size: 40px;'> Extração de Embeddings </font>

Utilizar o backbone do melhor modelo.

Explicar:

- conceito de embedding
- dimensão
- representação vetorial
- reutilização dos embeddings

Salvar os embeddings.

---

# <font color='red' style='font-size: 40px;'> Classificação utilizando Machine Learning </font>

Treinar um modelo utilizando apenas os embeddings.

Implementar:

- LightGBM

Comparar os resultados com as redes neurais.

---

# <font color='red' style='font-size: 40px;'> Visualização dos Embeddings </font>

Utilizar:

- t-SNE
- UMAP

Gerar gráficos mostrando a separação das classes.

Discutir a qualidade das representações aprendidas.

---

# <font color='red' style='font-size: 40px;'> Interpretabilidade </font>

Para CNN:

- Grad-CAM

Para ViT:

- Attention Maps

Comparar visualmente quais regiões das imagens foram utilizadas durante a classificação.

---

# <font color='red' style='font-size: 40px;'> Robustez dos Modelos </font>

Criar versões modificadas das imagens.

Exemplos:

- blur
- ruído gaussiano
- redução de resolução
- diferentes níveis de iluminação

Avaliar a degradação das métricas.

---

# <font color='red' style='font-size: 40px;'> Comparação dos Modelos </font>

Construir uma tabela consolidada contendo:

- Accuracy
- Precision
- Recall
- F1-score
- AUC
- Tempo de treinamento
- Tempo de inferência
- Número de parâmetros
- Número de parâmetros treináveis
- Uso aproximado de memória
- Tempo por época
- Tamanho do modelo salvo

Gerar gráficos comparativos.

---

# <font color='red' style='font-size: 40px;'> Trade-offs </font>

Discutir:

- desempenho
- custo computacional
- velocidade de treinamento
- velocidade de inferência
- interpretabilidade
- robustez
- escalabilidade
- facilidade de treinamento
- capacidade de generalização

Responder quando cada estratégia é mais indicada.

---

# <font color='red' style='font-size: 40px;'> Engenharia de Atributos baseada em Embeddings </font>

Construir indicadores derivados dos embeddings.

Exemplos:

- norma L2
- similaridade entre imagens
- distância ao centróide
- score de confiança
- entropia das probabilidades
- margem entre primeira e segunda classe

Discutir aplicações práticas.

---

# <font color='red' style='font-size: 40px;'> Conclusões </font>

Responder, entre outras, às seguintes perguntas:

- Qual estratégia apresentou a melhor performance?
- Qual apresentou o melhor custo-benefício?
- Quando vale a pena treinar uma CNN do zero?
- Quando utilizar Transfer Learning?
- Quando utilizar Fine-Tuning?
- Quando utilizar LoRA?
- Em quais cenários embeddings são suficientes?
- Quanto a otimização de hiperparâmetros melhorou a CNN?
- Quais limitações foram encontradas?
- Possíveis trabalhos futuros.

---

# Requisitos Gerais

Durante todo o desenvolvimento:

- Utilizar PyTorch.
- Manter código limpo, modular e reutilizável.
- Comentar detalhadamente todos os blocos de código.
- Explicar matematicamente os principais conceitos.
- Interpretar todos os resultados obtidos.
- Utilizar boas práticas de engenharia de software.
- Encapsular trechos repetitivos em funções.
- Gerar gráficos claros e bem apresentados.
- Garantir reprodutibilidade utilizando sementes aleatórias (seed).
- Utilizar boas práticas de documentação.
- Organizar o notebook como um material completo de estudo em Visão Computacional.
- Comparar todas as abordagens sob os mesmos critérios experimentais.
- Justificar tecnicamente todas as decisões de projeto.

# Exemplo de Organização de Código

Sempre deixar códigos comentados, identados e didáticos
Se necessário, pode criar células de markdown para representar o que está sendo feito
Sempre explique a matemática, mesmo que de forma resumida. Isso ajuda no entendimento

Exemplo:

# =========================================================
# DEFINIÇÃO DA ARQUITETURA DA CNN
# =========================================================

# Define uma nova classe chamada DSANet
# Toda rede neural no PyTorch herda de nn.Module
class DSANet(nn.Module):

    # =====================================================
    # CONSTRUTOR DA REDE
    # Aqui definimos TODAS as camadas da arquitetura
    # =====================================================
    def __init__(self):

        # Inicializa a classe pai (nn.Module)
        super(DSANet, self).__init__()

        # =================================================
        # BLOCO CONVOLUCIONAL
        # Responsável por extrair padrões da imagem
        # =================================================

        # Primeira convolução
        # Entrada: 3 canais RGB
        # Saída: 32 feature maps
        # Kernel: 3x3
        # Stride: 1. O stride define quantos pixels o kernel "anda" a cada convolução. Stride = 1 → o filtro anda pixel por pixel.
        # Padding: 1. O padding adiciona pixels extras (preenchidos com zeros) nas bordas da imagem para preservar as dimensões após a convolução. Com kernel 3x3 e padding 1, a saída terá a mesma altura e largura da entrada.
        # Intuição: aprende padrões simples como bordas, linhas e contrastes
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding = 1)

        # Segunda convolução
        # Entrada: 32 feature maps
        # Saída: 64 feature maps
        # Intuição: combina padrões anteriores para aprender texturas e formas
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding = 1)

        # Terceira convolução
        # Entrada: 64 feature maps
        # Saída: 128 feature maps
        # Intuição: aprende estruturas mais abstratas e regiões relevantes
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding = 1)

        # =================================================
        # REGULARIZAÇÃO
        # Dropout ajuda a evitar overfitting
        # =================================================

        # Dropout de 25%
        # Durante o treino: desliga aleatoriamente 25% dos neurônios
        self.dropout1 = nn.Dropout(0.25)

        # Dropout de 50%
        # Regularização mais forte
        self.dropout2 = nn.Dropout(0.5)

        # =================================================
        # CAMADAS DENSAS (MLP)
        # Responsáveis pela classificação final
        # =================================================

        # Primeira camada totalmente conectada
        # A entrada será muito menor após os poolings
        self.fc1 = nn.Linear(8192, 512)

        # Segunda camada densa
        self.fc2 = nn.Linear(512, 128)

        # Camada de saída
        # 10 neurônios → 10 classes
        self.fc3 = nn.Linear(128, 10)

    # =====================================================
    # FORWARD PASS
    # Define como os dados percorrem a rede
    # =====================================================
    def forward(self, x):

        # =================================================
        # ETAPA 1 — PRIMEIRA CONVOLUÇÃO
        # =================================================

        # Convolução + ReLU
        x = F.relu(self.conv1(x))

        # Max Pooling
        # Reduz altura e largura pela metade
        x = F.max_pool2d(x, 2)

        # =================================================
        # ETAPA 2 — SEGUNDA CONVOLUÇÃO
        # =================================================

        # Convolução + ReLU
        x = F.relu(self.conv2(x))

        # Max Pooling
        x = F.max_pool2d(x, 2)

        # =================================================
        # ETAPA 3 — TERCEIRA CONVOLUÇÃO
        # =================================================

        # Convolução + ReLU
        x = F.relu(self.conv3(x))

        # Max Pooling
        x = F.max_pool2d(x, 2)

        # =================================================
        # ETAPA 4 — DROPOUT
        # =================================================

        # Desliga neurônios aleatoriamente
        x = self.dropout1(x)

        # =================================================
        # ETAPA 5 — FLATTEN
        # =================================================

        # CNN → vetor 1D
        x = torch.flatten(x, 1)

        # =================================================
        # ETAPA 6 — PRIMEIRA CAMADA DENSA
        # =================================================

        # Camada densa + ReLU
        x = F.relu(self.fc1(x))

        # Dropout
        x = self.dropout2(x)

        # =================================================
        # ETAPA 7 — SEGUNDA CAMADA DENSA
        # =================================================

        # Camada densa + ReLU
        x = F.relu(self.fc2(x))

        # =================================================
        # ETAPA 8 — CAMADA DE SAÍDA
        # =================================================

        # Produz logits para as classes
        x = self.fc3(x)

        # Converte logits em probabilidades
        return F.log_softmax(x, dim = 1)