# Modelo_Visao_Computacional_Image_Satelites

---

# Guia de Estudos e Benchmark em Visão Computacional — EuroSAT

<p align="center">
  <img src="./img_01.jpeg" width="50%">
</p>

> **Autor:** Leonardo Aderaldo Vargas · T789785<br>
> **Dataset:** EuroSAT RGB

> **Status:**

<p align="center">
<img src="http://img.shields.io/static/v1?label=STATUS&message=BENCHMARK%20DOS%204%20MODELOS%20CONCLUIDO&color=GREEN&style=for-the-badge"/>
</p>

---

# Sumário

1. Contexto de Negócio
2. Objetivos e Problemas
3. Fundamentação Teórica
4. Fontes de Dados
5. Arquitetura da Solução
6. Análise Exploratória das Imagens
7. Pré-Processamento
8. CNN Desenvolvida do Zero
9. Otimização da CNN com Optuna
10. Transfer Learning com ResNet50
11. Vision Transformer com LoRA
12. Extração de Embeddings com ViT Congelado
13. LightGBM sobre Embeddings
14. Comparação Consolidada
15. Resultados Consolidados
16. Artefatos Gerados

---

# Como Usar Este Guia

Este README é um **guia evolutivo dos estudos e dos experimentos** realizados no projeto. Seu objetivo é registrar não apenas o código ou a pontuação final, mas a ligação entre:

```text
Problema estudado
        ↓
Conceito e técnica
        ↓
Aplicação prática no notebook
        ↓
Resultado, interpretação e decisão
```

O arquivo [`benchmark_visao_computacional_eurosat.ipynb`](./benchmark_visao_computacional_eurosat.ipynb) contém as demonstrações e implementações completas. Este README funciona como material de revisão: resume o que aprendemos, as decisões tomadas, os resultados obtidos e as próximas perguntas do benchmark.

## Trilha Cronológica dos Estudos

| Etapa | Pergunta de aprendizagem | Aplicação prática | Estado |
|------:|--------------------------|------------------|--------|
| 1 | Como uma imagem é representada computacionalmente? | Pixels, RGB, `uint8`, NumPy e tensores PyTorch | Concluído |
| 2 | O que significam `N`, `C`, `H` e `W`? | Conversão HWC → CHW e formação de batches NCHW | Concluído |
| 3 | Como preparar imagens para uma rede neural? | Resize, interpolação, normalização e augmentation | Concluído |
| 4 | Como uma CNN encontra padrões locais? | Kernels, convolução, feature maps, ReLU e pooling | Concluído |
| 5 | Como avaliar sem vazamento de informação? | Splits estratificados de treino, validação e teste | Concluído |
| 6 | Como construir e treinar uma CNN do zero? | `CNNDoZero`, forward, loss, backpropagation e checkpoint | Concluído |
| 7 | Como escolher hiperparâmetros? | Optuna em modo rápido e treino final com todos os dados | Concluído |
| 8 | Quanto ganhamos reutilizando filtros pré-treinados? | Transfer learning com ResNet50 | Concluído |
| 9 | Como representar imagens com Transformers? | ViT congelado, embeddings `[CLS]` e LightGBM | Concluído |
| 10 | Como adaptar um ViT treinando poucos parâmetros? | ViT com LoRA | Concluído |

## Resultado Atual do Estudo

A primeira referência completa foi a **CNN do zero + Optuna**, com **F1 Macro de 0,8379 no teste**. O pipeline **ViT congelado + LightGBM** alcançou **0,9174** e o **ViT adaptado com LoRA** chegou a **0,9360**. A **ResNet50 pré-treinada** obteve o melhor resultado do benchmark, com **0,9576**.

---

# 1. Contexto de Negócio

Modelos de Visão Computacional vêm sendo utilizados em diversas aplicações envolvendo sensoriamento remoto, monitoramento ambiental, agricultura de precisão, planejamento urbano, defesa civil, logística e geointeligência.

Imagens de satélite permitem identificar automaticamente diferentes tipos de cobertura e uso do solo, fornecendo informações relevantes para tomada de decisão em setores públicos e privados.

Nos últimos anos surgiram diversas estratégias para classificação de imagens. Desde Redes Neurais Convolucionais treinadas do zero até grandes modelos pré-treinados baseados em Transformers, cada abordagem apresenta vantagens e limitações relacionadas ao desempenho, custo computacional e facilidade de implantação.

Dessa forma, a escolha da arquitetura ideal não deve considerar apenas a acurácia obtida sobre um conjunto de testes.

Também devem ser avaliados fatores como:

- desempenho preditivo;
- custo de treinamento;
- tempo de inferência;
- quantidade de parâmetros;
- consumo de memória;
- facilidade de implementação;
- escalabilidade para produção.

Este projeto propõe um benchmark controlado comparando diferentes estratégias modernas de aprendizado de representação para classificação de imagens de satélite.

---

# 2. Objetivos e Problemas

## Objetivo Central

Comparar, utilizando exatamente os mesmos conjuntos de treino, validação e teste, diferentes estratégias de aprendizado de representação para classificação multiclasse de imagens do EuroSAT.

O objetivo não consiste em encontrar a arquitetura perfeita, mas compreender os principais trade-offs entre desempenho, custo computacional e complexidade de implementação.

## Estratégias Comparadas

- CNN desenvolvida do zero e otimizada com Optuna;
- Transfer Learning utilizando ResNet50;
- Embeddings extraídos de um Vision Transformer combinados com LightGBM;
- Vision Transformer adaptado utilizando LoRA.

## Entregas Mínimas

- Análise exploratória das imagens;
- Pipeline completo de pré-processamento;
- Implementação dos quatro modelos;
- Otimização da CNN utilizando Optuna;
- Extração de embeddings;
- LightGBM utilizando embeddings;
- Comparação consolidada entre todos os modelos.

---

# 3. Fundamentação Teórica

O desenvolvimento deste projeto envolve conceitos fundamentais de Deep Learning e Visão Computacional.

Entre os principais tópicos estudados destacam-se:

- Python;
- PyTorch;
- Redes Neurais Convolucionais;
- Convolução;
- Pooling;
- Batch Normalization;
- Dropout;
- Transfer Learning;
- Fine-Tuning;
- ResNet;
- Vision Transformers;
- Self-Attention;
- Patch Embeddings;
- LoRA;
- Embeddings Visuais;
- LightGBM;
- Otimização Bayesiana;
- Optuna;

---

# 4. Fontes de Dados

**Dataset:** EuroSAT RGB

O conjunto de dados é composto por imagens de satélite Sentinel-2 distribuídas em dez classes de uso e ocupação do solo.

Cada imagem pertence a apenas uma classe.

As categorias são:

- AnnualCrop;
- Forest;
- HerbaceousVegetation;
- Highway;
- Industrial;
- Pasture;
- PermanentCrop;
- Residential;
- River;
- SeaLake.

Durante a preparação dos dados foram verificados:

- quantidade de imagens;
- distribuição das classes;
- imagens duplicadas;
- imagens corrompidas;
- resolução das imagens.

## Inventário Validado

- **27.000 imagens RGB**;
- **10 classes**;
- resolução original de **64 × 64 pixels**;
- nenhuma cópia exata encontrada na verificação por MD5;
- split estratificado e reproduzível com semente 42.

| Split | Imagens | Proporção aproximada |
|-------|--------:|---------------------:|
| Treino | 18.899 | 70% |
| Validação | 4.051 | 15% |
| Teste | 4.050 | 15% |
| **Total** | **27.000** | **100%** |

---

# 5. Arquitetura da Solução

```text
Imagens Brutas

↓

Leitura e Validação

↓

Treino / Validação / Teste

↓

Análise Exploratória

↓

Pré-processamento

↓

┌──────────────────────────────────────────┐
│ CNN do Zero + Optuna                     │
├──────────────────────────────────────────┤
│ Transfer Learning (ResNet50)             │
├──────────────────────────────────────────┤
│ Vision Transformer + LoRA                │
├──────────────────────────────────────────┤
│ Embeddings + LightGBM                    │
└──────────────────────────────────────────┘

↓

Comparação Consolidada
```

---

# 6. Análise Exploratória das Imagens

Antes da modelagem foi realizada uma análise exploratória do conjunto de imagens.

Foram avaliados:

- quantidade de imagens por classe;
- distribuição percentual das classes;
- resolução das imagens;
- estatísticas dos canais RGB;
- visualização de amostras;
- possíveis inconsistências.

Essa etapa confirmou a estrutura RGB das imagens, a resolução de 64 × 64 pixels, a distribuição levemente desigual entre as classes e a ausência de duplicatas exatas com rótulos conflitantes.

## Distribuição das Classes

| Classe | Imagens | Proporção |
|--------|--------:|----------:|
| AnnualCrop | 3.000 | 11,11% |
| Forest | 3.000 | 11,11% |
| HerbaceousVegetation | 3.000 | 11,11% |
| Highway | 2.500 | 9,26% |
| Industrial | 2.500 | 9,26% |
| Pasture | 2.000 | 7,41% |
| PermanentCrop | 2.500 | 9,26% |
| Residential | 3.000 | 11,11% |
| River | 2.500 | 9,26% |
| SeaLake | 3.000 | 11,11% |

A classe menos frequente, `Pasture`, possui 2.000 imagens, enquanto as maiores possuem 3.000. A razão entre a maior e a menor classe é de `1,5:1`, indicando um **desbalanceamento leve**, sem classes extremamente raras. Por esse motivo, o treinamento foi mantido sem pesos de classe, enquanto o **F1 Macro** foi adotado para garantir que as dez classes tenham a mesma importância na avaliação.

## Qualidade e Padronização

- Uma amostra aleatória e reproduzível de **1.500 imagens** foi verificada com PIL;
- todas as imagens da amostra estavam legíveis;
- todas apresentaram resolução de **64 × 64 pixels**;
- todas foram decodificadas no modo **RGB**, com três canais;
- a busca por hash MD5 nas **27.000 imagens** não encontrou cópias exatas;
- não foram encontrados arquivos idênticos associados a rótulos diferentes.

A resolução uniforme simplifica a construção dos lotes. Para as arquiteturas do benchmark, as imagens são redimensionadas de `64 × 64` para `224 × 224` por interpolação bilinear. Esse aumento padroniza a entrada, mas não cria novos detalhes espaciais na imagem original.

## Descobertas Visuais

A inspeção de exemplos das dez classes mostrou que a classificação não depende apenas da cor. As categorias também apresentam diferenças de textura, organização espacial, presença de linhas, densidade de construções, vegetação e corpos d'água. Ao mesmo tempo, algumas classes possuem padrões visualmente próximos, especialmente categorias agrícolas, vegetação herbácea, pastagem e rodovias cercadas por cobertura vegetal.

Essa observação é consistente com o resultado posterior da CNN: `Highway`, `PermanentCrop` e `HerbaceousVegetation` ficaram entre as classes com menor F1, enquanto `SeaLake` e `Forest`, visualmente mais características, apresentaram os melhores resultados.

## Implicações para o Pipeline

- o split foi estratificado para preservar a proporção das classes;
- os mesmos splits são reutilizados por todos os modelos;
- data augmentation é aplicado somente no treino;
- validação e teste utilizam transformações determinísticas;
- o teste não participa da escolha de hiperparâmetros ou da melhor época;
- o fingerprint `c3b4fc18906f34dc` identifica a versão oficial do split.

---

# 7. Pré-Processamento

## Problema Estudado

Redes neurais precisam receber tensores com dimensões e escalas previsíveis. Além disso, treino, validação e teste não podem receber transformações aleatórias da mesma maneira, pois isso tornaria a avaliação instável.

## Técnica Aplicada

Foi definido um pipeline único de preparação das imagens.

As principais etapas incluem:

- Resize;
- Conversão para Tensor;
- Normalização utilizando estatísticas do ImageNet;
- Data Augmentation apenas no conjunto de treinamento;
- Construção dos DataLoaders.

As transformações de Data Augmentation incluíram técnicas consolidadas:

- Flip Horizontal;
- Rotação;
- Pequenas variações de brilho e contraste.

O objetivo é aumentar a capacidade de generalização da CNN sem tornar o pipeline excessivamente complexo.

## O que Aprendemos

- o resize de `64 × 64` para `224 × 224` é um upsampling por interpolação, mas não cria novos detalhes;
- `ToTensor` converte pixels `0–255` em `float32` e reorganiza HWC para CHW;
- a normalização coloca os canais na escala esperada pelos modelos;
- augmentation cria variações das imagens de treino, mas não substitui novas imagens reais;
- validação e teste devem utilizar transformações determinísticas;
- um batch de 32 imagens possui formato `(32, 3, 224, 224)` e atravessa uma única rede com pesos compartilhados.

---

# 8. CNN Desenvolvida do Zero

## Problema Estudado

Compreender como uma rede aprende representações visuais sem reutilizar pesos externos e como as dimensões do tensor se transformam entre convolução, pooling e classificação.

## Técnica Aplicada

A primeira estratégia consiste na construção manual de uma Rede Neural Convolucional utilizando PyTorch.

Fluxo da arquitetura:

```text
Imagem

↓

Convoluções

↓

Batch Normalization

↓

ReLU

↓

Max Pooling

↓

Dropout

↓

Camadas Fully Connected

↓

Logits
```

A arquitetura utiliza:

- camadas convolucionais;
- Batch Normalization;
- ReLU;
- Max Pooling;
- Dropout;
- Camadas totalmente conectadas.

Durante o treinamento foram utilizados:

- CrossEntropyLoss;
- Adam;
- Weight Decay;
- Gradient Clipping;
- ReduceLROnPlateau;
- Early Stopping.

O objetivo desta etapa é compreender como construir uma CNN moderna e bem regularizada, sem comparar diferentes funções de ativação, otimizadores ou métodos de pooling.

## Fluxo dos Tensores

```text
(N, 3, 224, 224)
        ↓ Conv2d + BatchNorm + ReLU + MaxPool
(N, filtros, 112, 112)
        ↓ segundo bloco
(N, filtros × 2, 56, 56)
        ↓ terceiro bloco
(N, filtros × 4, 28, 28)
        ↓ AdaptiveAvgPool + Flatten + camadas lineares
(N, 10 logits)
```

## O que Aprendemos

- o número de imagens `N` é preservado durante o forward;
- a convolução altera os canais conforme a quantidade de filtros;
- o pooling reduz altura e largura;
- os logits são escores brutos, um conjunto de dez valores para cada imagem;
- a classe `CNNDoZero` define a arquitetura e o forward;
- loss, backpropagation, otimizador, validação e early stopping pertencem ao loop de treinamento;
- Dropout, weight decay, augmentation e early stopping ajudam a reduzir overfitting.

O resultado quantitativo desta etapa está documentado na seção [Resultados Consolidados](#15-resultados-consolidados).

---

# 9. Otimização da CNN com Optuna

Após a definição da arquitetura da CNN, foi executada uma busca de hiperparâmetros utilizando **Optuna**.

O objetivo desta etapa não é modificar a arquitetura da rede, mas encontrar uma configuração de treinamento mais eficiente para o modelo desenvolvido.

## Objetivo

Maximizar o **F1-score Macro** no conjunto de validação.

## Hiperparâmetros Otimizados

O espaço de busca foi restrito aos parâmetros de maior impacto:

- Learning Rate;
- Batch Size;
- Número de filtros;
- Dimensão da camada totalmente conectada;
- Dropout;
- Weight Decay.

Os seguintes componentes permaneceram fixos:

- ReLU;
- Max Pooling;
- Batch Normalization;
- Adam;
- CrossEntropyLoss;
- ReduceLROnPlateau;
- Early Stopping.

Para reduzir o custo computacional, a busca foi executada em modo rápido com cinco trials, usando 1.200 imagens de treino e 1.200 de validação. A configuração vencedora foi então utilizada para treinar uma nova CNN com todos os dados disponíveis no split de desenvolvimento.

## Melhor Configuração Encontrada

| Hiperparâmetro | Valor |
|----------------|------:|
| Learning rate | 0,0002701053 |
| Batch size | 64 |
| Filtros iniciais | 24 |
| Dimensão da camada FC | 384 |
| Dropout | 0,251157 |
| Weight decay | 0,0000015673 |

O melhor trial alcançou **F1 Macro de validação de 0,5687** no subconjunto rápido. Em seguida, a CNN definitiva foi treinada do zero com 18.899 imagens, validada em 4.051 imagens e avaliada uma única vez nas 4.050 imagens de teste.

## O que Aprendemos

- hiperparâmetros são escolhas feitas antes do treinamento; pesos são aprendidos pelo backpropagation;
- cada trial cria e treina uma nova CNN com uma combinação diferente;
- o Optuna compara os trials somente pela validação;
- `best_params` escolhe a configuração da CNN;
- `melhor_estado_cnn` guarda os pesos da melhor época do treinamento final;
- o conjunto de teste não participa da busca nem da escolha do checkpoint;
- é possível otimizar em uma amostra rápida e depois treinar a configuração escolhida com todos os dados, aceitando o trade-off entre velocidade e precisão da busca.

---

# 10. Transfer Learning com ResNet50

## Problema Estudado

Avaliar quanto podemos ganhar ao reutilizar uma representação visual aprendida previamente, em vez de começar o treinamento com filtros aleatórios.

## Técnica Aplicada

A segunda estratégia utiliza uma **ResNet50 pré-treinada** no ImageNet.

Ao contrário da CNN desenvolvida do zero, esta abordagem parte de um modelo que já aprendeu representações visuais gerais em milhões de imagens.

Fluxo:

```text
Imagem

↓

ResNet50 Pré-Treinada

↓

Nova Camada de Classificação

↓

10 Classes
```

Durante esta etapa foram estudados:

- Transfer Learning;
- reutilização de conhecimento prévio;
- congelamento das camadas convolucionais;
- substituição da cabeça de classificação;
- possibilidade de fine-tuning parcial das últimas camadas.

O objetivo é compreender como modelos pré-treinados podem reduzir tempo de treinamento e melhorar a capacidade de generalização em datasets menores.

Neste primeiro experimento, `DESCONGELAR_LAYER4=False`: o backbone permaneceu congelado e somente a nova cabeça `Dropout + Linear(2048, 10)` foi treinada. Portanto, trata-se de **transfer learning com extração fixa de features**, e não de fine-tuning completo do backbone.

## Configuração Executada

| Configuração | Valor |
|--------------|------:|
| Pesos iniciais | ImageNet-1K V2 |
| Imagens de treino | 18.899 |
| Imagens de validação | 4.051 |
| Imagens de teste | 4.050 |
| Batch size | 32 |
| Épocas máximas | 15 |
| Learning rate | 0,0003 |
| Weight decay | 0,0001 |
| Dropout da cabeça | 0,25 |
| Layer4 descongelada | Não |
| Parâmetros treináveis | 20.490 de 23.528.522 (0,087%) |

## O que Aprendemos

- as features pré-treinadas transferiram-se muito bem para o EuroSAT RGB;
- a ResNet50 começou com F1 de validação superior a `0,90` já na primeira época;
- somente a cabeça precisou aprender a separar as dez classes;
- congelar o backbone reduziu o custo do backward e o número de parâmetros atualizados;
- o modelo completo ainda precisa executar o forward, por isso sua inferência é mais lenta e seu arquivo é maior que o da CNN;
- descongelar a `layer4` permanece como um experimento futuro de fine-tuning parcial.

O melhor checkpoint ocorreu na **época 13**, com **F1 Macro de validação de 0,9526**. No teste, a ResNet50 alcançou **F1 Macro de 0,9576**.

---

# 11. Vision Transformer com LoRA

A terceira estratégia adaptou o checkpoint **`google/vit-base-patch16-224-in21k`** utilizando **LoRA (Low-Rank Adaptation)**. Em vez de atualizar os mais de 86 milhões de parâmetros do ViT, o experimento manteve os pesos originais congelados e treinou pequenas matrizes adicionais nas projeções de atenção, além da nova cabeça classificadora.

Fluxo executado:

Fluxo:

```text
Imagem

↓

Patch Embeddings

↓

Vision Transformer

↓

LoRA

↓

Cabeça de Classificação

↓

10 Classes
```

Durante esta etapa foram estudados:

- Patch Embeddings;
- Self-Attention;
- Query;
- Key;
- Value;
- Multi-Head Attention;
- Feed-Forward Networks;
- LoRA.

## Configuração Executada

| Configuração | Valor |
|--------------|------:|
| Checkpoint | `google/vit-base-patch16-224-in21k` |
| Imagens de treino | 18.899 |
| Imagens de validação | 4.051 |
| Imagens de teste | 4.050 |
| Módulos adaptados | `q_proj` e `v_proj` |
| Rank LoRA | 8 |
| Alpha | 16 |
| Dropout LoRA | 0,10 |
| Batch size | 8 |
| Épocas máximas | 20 |
| Learning rate máxima | 0,0002 |
| Weight decay | 0,0001 |
| Otimizador | AdamW |
| Scheduler | Warmup + decaimento linear |
| Parâmetros treináveis | 302.602 de 86.108.948 (0,351%) |

O treinamento percorreu as 20 épocas planejadas. O melhor checkpoint foi obtido na **época 19**, com **F1 Macro de validação de 0,9363**. No teste, o modelo alcançou **F1 Macro de 0,9360**, diferença de apenas `0,0003`, indicando excelente generalização.

O experimento demonstrou que o LoRA consegue adaptar a representação do Transformer ao EuroSAT treinando apenas uma pequena fração do modelo. O adapter salvo ocupa somente **1,17 MB**, embora a inferência ainda dependa do backbone ViT completo.

---

# 12. Extração de Embeddings com ViT Congelado

Neste experimento, o checkpoint `google/vit-base-patch16-224-in21k` foi usado como **extrator de características totalmente congelado**. O ViT não foi treinado com os rótulos do EuroSAT: cada imagem foi apenas convertida em um vetor numérico por uma passagem de inferência sem gradientes.

Os embeddings foram gerados para todo o conjunto:

- treino;
- validação;
- teste.

Fluxo:

```text
Imagem

↓

Vision Transformer

↓

Embedding

↓

Arquivo .npy
```

Cada imagem de `224 × 224` foi dividida em 196 patches de `16 × 16`. Depois dos blocos Transformer, o token `[CLS]` da saída `(N, 197, 768)` foi selecionado, produzindo uma matriz `(N, 768)` por split.

Durante esta etapa foram estudados:

- conceito de embedding;
- dimensão do vetor;
- representação semântica das imagens;
- diferença entre embeddings e classificação fim-a-fim.

### Artefatos e Tempo de Extração

| Split | Shape dos embeddings | Arquivo |
|-------|----------------------|---------|
| Treino | `(18.899, 768)` | `vit_embeddings_train_completo.npy` |
| Validação | `(4.051, 768)` | `vit_embeddings_val_completo.npy` |
| Teste | `(4.050, 768)` | `vit_embeddings_test_completo.npy` |

A extração dos três conjuntos levou **257,6 segundos**, aproximadamente **4,3 minutos**, na NVIDIA GeForce RTX 2060 SUPER. Os rótulos e os índices dos arquivos também foram salvos separadamente, preservando a correspondência `imagem ↔ embedding ↔ classe`.

Os embeddings foram reutilizados pelo LightGBM sem necessidade de executar novamente o ViT durante o treinamento do classificador.

---

# 13. LightGBM sobre Embeddings

A quarta estratégia utiliza somente os embeddings extraídos pelo ViT congelado como variáveis de entrada de um LightGBM multiclasse. O modelo recebe uma tabela com 768 características por imagem; ele não recebe pixels e não envia gradientes de volta ao Transformer.

Fluxo:

```text
Imagem

↓

Vision Transformer

↓

Embedding

↓

LightGBM

↓

Classe
```

Foram comparadas três configurações, ajustando:

- Learning Rate;
- Número de Árvores;
- Número de Folhas;
- Profundidade Máxima;
- Regularização L1;
- Regularização L2.

As três configurações foram treinadas somente no conjunto de treino e comparadas pelo F1 Macro de validação. A melhor foi a configuração 3:

| Hiperparâmetro | Valor |
|----------------|------:|
| Learning rate | 0,08 |
| Número de folhas | 31 |
| Profundidade máxima | 8 |
| Número de árvores | 250 |
| Regularização L1 | 0,50 |
| Regularização L2 | 0,50 |

Ela alcançou **F1 Macro de validação de 0,9247**. Somente depois dessa escolha o conjunto de teste foi consultado.

O objetivo foi avaliar até que ponto uma representação aprendida previamente por um Transformer pode ser aproveitada por um algoritmo clássico de Machine Learning. O resultado confirmou que os embeddings genéricos já separam bem as classes do EuroSAT, embora tenham ficado abaixo da ResNet50 transferida.

---

# 14. Comparação Consolidada

O benchmark utiliza uma tabela única para comparar as quatro estratégias avaliadas.

| Modelo | Representação |
|---------|---------------|
| CNN do Zero + Optuna | Features aprendidas do zero |
| ResNet50 | Transfer Learning |
| Vision Transformer + LoRA | Transformer adaptado |
| Embeddings + LightGBM | Embeddings visuais |

Para todos os modelos foram registrados:

- Accuracy;
- Precision Macro;
- Recall Macro;
- F1-score Macro;
- F1 por classe;
- AUC;
- Tempo de treinamento;
- Tempo de inferência;
- Número de parâmetros treináveis;
- Tamanho do modelo.

Também foi analisado o equilíbrio entre:

- desempenho preditivo;
- custo computacional;
- velocidade de treinamento;
- velocidade de inferência;
- facilidade de implementação;

---

# 15. Resultados Consolidados

## Progresso do Benchmark

| Etapa | Estado |
|-------|--------|
| Inventário, qualidade e splits | Concluído |
| Fundamentos de imagens, tensores e convolução | Concluído |
| CNN do zero | Concluído |
| Busca de hiperparâmetros com Optuna | Concluído |
| Treinamento final da CNN com todos os dados | Concluído |
| ResNet50 com transfer learning | Concluído |
| Embeddings do ViT + LightGBM | Concluído |
| ViT adaptado com LoRA | Concluído |
| Comparação consolidada | Concluída para os quatro modelos |

## CNN do Zero + Optuna

A CNN final percorreu as 30 épocas configuradas. O melhor checkpoint foi obtido na **época 27**, com **F1 Macro de validação de 0,8420**. Os pesos dessa época foram restaurados antes da avaliação no teste.

### Métricas no Teste

| Métrica | Resultado |
|---------|----------:|
| Accuracy | 0,8430 |
| Precision Macro | 0,8434 |
| Recall Macro | 0,8393 |
| **F1 Macro** | **0,8379** |
| AUC OVR Macro | 0,9870 |
| Log Loss | 0,4544 |

### F1 por Classe

| Classe | F1 |
|--------|---:|
| SeaLake | 0,9657 |
| Forest | 0,9447 |
| Industrial | 0,9096 |
| Residential | 0,8649 |
| AnnualCrop | 0,8520 |
| Pasture | 0,8115 |
| River | 0,7961 |
| HerbaceousVegetation | 0,7581 |
| PermanentCrop | 0,7503 |
| Highway | 0,7259 |

### Custo Computacional Observado

| Indicador | Resultado |
|-----------|----------:|
| Tempo de treinamento | 3.276,2 s (aprox. 54,6 min) |
| Tempo de inferência no teste | 7,2 s |
| Parâmetros treináveis | 646.882 |
| Hardware | NVIDIA GeForce RTX 2060 SUPER, 8 GB |

O resultado de teste ficou próximo do melhor resultado de validação: `0,8379` contra `0,8420`, uma diferença de aproximadamente **0,0041**. Isso indica boa generalização neste experimento. As classes mais difíceis foram `Highway`, `PermanentCrop` e `HerbaceousVegetation`; `SeaLake` e `Forest` apresentaram os maiores F1.

## ResNet50 com Transfer Learning

A ResNet50 percorreu as 15 épocas configuradas. O melhor checkpoint foi obtido na **época 13**, com **F1 Macro de validação de 0,9526**. O backbone permaneceu congelado e apenas a nova cabeça classificadora foi otimizada.

### Métricas no Teste

| Métrica | Resultado |
|---------|----------:|
| Accuracy | 0,9585 |
| Precision Macro | 0,9576 |
| Recall Macro | 0,9582 |
| **F1 Macro** | **0,9576** |
| AUC OVR Macro | 0,9987 |
| Log Loss | 0,1375 |

### F1 por Classe

| Classe | F1 |
|--------|---:|
| Forest | 0,9845 |
| Residential | 0,9834 |
| SeaLake | 0,9831 |
| Industrial | 0,9800 |
| Pasture | 0,9581 |
| AnnualCrop | 0,9526 |
| HerbaceousVegetation | 0,9397 |
| Highway | 0,9379 |
| River | 0,9376 |
| PermanentCrop | 0,9193 |

### Custo Computacional Observado

| Indicador | Resultado |
|-----------|----------:|
| Tempo de treinamento | 2.308,5 s (aprox. 38,5 min) |
| Tempo de inferência no teste | 20,4 s |
| Parâmetros totais | 23.528.522 |
| Parâmetros treináveis | 20.490 (0,087%) |
| Tamanho do checkpoint | 90,06 MB |
| Hardware | NVIDIA GeForce RTX 2060 SUPER, 8 GB |

## ViT Congelado + LightGBM

O pipeline completo utilizou as 18.899 imagens de treino, 4.051 de validação e 4.050 de teste. O backbone ViT permaneceu totalmente congelado; o aprendizado supervisionado ocorreu apenas no LightGBM.

### Métricas no Teste

| Métrica | Resultado |
|---------|----------:|
| Accuracy | 0,9210 |
| Precision Macro | 0,9181 |
| Recall Macro | 0,9172 |
| **F1 Macro** | **0,9174** |
| AUC OVR Macro | 0,9950 |
| Log Loss | 0,2561 |

### F1 por Classe

| Classe | F1 |
|--------|---:|
| Forest | 0,9779 |
| SeaLake | 0,9618 |
| Residential | 0,9532 |
| HerbaceousVegetation | 0,9464 |
| Industrial | 0,9401 |
| AnnualCrop | 0,9023 |
| Pasture | 0,8995 |
| PermanentCrop | 0,8743 |
| River | 0,8630 |
| Highway | 0,8556 |

### Custo Computacional Observado

| Indicador | Resultado |
|-----------|----------:|
| Extração dos embeddings | 257,6 s (aprox. 4,3 min) |
| Busca de três configurações LightGBM | 194,3 s (aprox. 3,2 min) |
| Preparação total do pipeline | 451,9 s (aprox. 7,5 min) |
| Inferência apenas do LightGBM | 0,247 s |
| Inferência ponta a ponta no teste | 38,2 s |
| Parâmetros do backbone ViT | 86.389.248, todos congelados |
| Tamanho do classificador LightGBM | 6,07 MB |
| Tamanho estimado do pipeline completo | 335,62 MB |
| Hardware da extração | NVIDIA GeForce RTX 2060 SUPER, 8 GB |

O F1 Macro de teste ficou **0,0073 abaixo** do melhor F1 de validação (`0,9174` contra `0,9247`), uma diferença pequena e compatível com boa generalização. As classes mais fáceis foram `Forest`, `SeaLake` e `Residential`; `Highway`, `River` e `PermanentCrop` concentraram os menores F1.

## ViT + LoRA

O ViT com LoRA foi treinado com todos os dados do benchmark. Os pesos originais do Transformer permaneceram congelados; somente os adapters inseridos em `q_proj` e `v_proj` e a cabeça classificadora receberam gradientes.

O melhor checkpoint ocorreu na **época 19**, com **F1 Macro de validação de 0,9363**. Embora a loss de treino tenha ficado abaixo da loss de validação nas últimas épocas, a F1 de validação continuou melhorando e o resultado de teste permaneceu praticamente igual ao melhor resultado de validação.

### Métricas no Teste

| Métrica | Resultado |
|---------|----------:|
| Accuracy | 0,9385 |
| Precision Macro | 0,9361 |
| Recall Macro | 0,9366 |
| **F1 Macro** | **0,9360** |
| AUC OVR Macro | 0,9971 |
| Log Loss | 0,2428 |

### F1 por Classe

| Classe | F1 |
|--------|---:|
| SeaLake | 0,9798 |
| Residential | 0,9780 |
| Forest | 0,9756 |
| Industrial | 0,9606 |
| AnnualCrop | 0,9314 |
| HerbaceousVegetation | 0,9262 |
| Pasture | 0,9173 |
| Highway | 0,9048 |
| PermanentCrop | 0,8954 |
| River | 0,8910 |

### Custo Computacional Observado

| Indicador | Resultado |
|-----------|----------:|
| Tempo de treinamento | 4.773,5 s (aprox. 79,6 min) |
| Tempo de inferência no teste | 41,0 s |
| Parâmetros totais | 86.108.948 |
| Parâmetros treináveis | 302.602 (0,351%) |
| Tamanho do adapter LoRA | 1,17 MB |
| Tamanho estimado do pipeline completo | 328,48 MB |
| Hardware | NVIDIA GeForce RTX 2060 SUPER, 8 GB |

A diferença entre validação e teste foi de aproximadamente **0,0003** (`0,9363` contra `0,9360`). Portanto, apesar do aumento do gap entre as losses de treino e validação nas últimas épocas, não houve evidência relevante de perda de generalização no checkpoint selecionado.

## Comparação Consolidada dos Quatro Modelos

| Indicador | CNN do Zero + Optuna | ResNet50 | ViT + LightGBM | ViT + LoRA |
|-----------|---------------------:|---------:|----------------:|-----------:|
| Accuracy | 0,8430 | **0,9585** | 0,9210 | 0,9385 |
| F1 Macro | 0,8379 | **0,9576** | 0,9174 | 0,9360 |
| AUC OVR Macro | 0,9870 | **0,9987** | 0,9950 | 0,9971 |
| Log Loss | 0,4544 | **0,1375** | 0,2561 | 0,2428 |
| Preparação/treinamento | 54,6 min | 38,5 min | **7,5 min** | 79,6 min |
| Inferência ponta a ponta no teste | **7,2 s** | 20,4 s | 38,2 s | 41,0 s |
| Artefato adicional treinado | 2,48 MB | 90,06 MB | 6,07 MB | **1,17 MB** |
| Pipeline completo estimado | 2,48 MB | 90,06 MB | 335,62 MB | 328,48 MB |

Neste estudo, a expressão **pipeline mais rápido** refere-se ao ciclo medido de preparação e treinamento: embeddings + LightGBM concluiu a extração e a busca em aproximadamente **7,5 minutos**, o menor tempo entre as quatro estratégias. Com os embeddings já disponíveis, o LightGBM também classificou o teste em apenas **0,247 segundo**. A inferência ponta a ponta é uma medição diferente, pois inclui executar novamente o ViT; nessa medição específica, a CNN registrou 7,2 segundos.

## Conclusão do Benchmark

### 1º lugar — ResNet50: F1 Macro 0,9576

A ResNet50 apresentou o melhor desempenho, a maior AUC e a menor Log Loss. O resultado indica que as representações convolucionais aprendidas no ImageNet foram especialmente adequadas ao EuroSAT. Imagens de uso e cobertura do solo dependem fortemente de padrões locais, como texturas, bordas, vegetação, estradas e agrupamentos de construções; esse tipo de estrutura combina diretamente com o viés indutivo das convoluções.

Além disso, somente a cabeça classificadora precisou ser treinada. Isso preservou uma representação visual madura, reduziu o risco de modificar excessivamente os pesos pré-treinados e permitiu alcançar desempenho elevado com apenas 20.490 parâmetros atualizados. O modelo mais sofisticado ou mais recente não é necessariamente o melhor para todo problema: neste dataset, a arquitetura convolucional mostrou uma correspondência particularmente forte com o tipo de padrão visual existente.

### 2º lugar — ViT + LoRA: F1 Macro 0,9360

O LoRA ficou **0,0216** abaixo da ResNet50, mas superou o ViT congelado + LightGBM em **0,0186** e a CNN em **0,0981**. A adaptação supervisionada de `q_proj` e `v_proj` permitiu que a atenção do ViT se ajustasse ao EuroSAT, explicando o ganho sobre os embeddings congelados.

O segundo lugar não significa que a metodologia seja inferior. O LoRA foi criado para adaptar modelos grandes com poucos parâmetros e cumpriu exatamente esse objetivo: somente **0,351%** do modelo foi treinado e o adapter resultante ocupa **1,17 MB**. Entretanto, eficiência de parâmetros não significa menor tempo de execução. O forward e o backward ainda atravessam o backbone ViT completo, tornando esta a estratégia mais demorada do benchmark.

Alguns fatores ajudam a explicar por que o LoRA não superou a ResNet50:

- o ViT possui menor viés indutivo para padrões locais do que uma CNN;
- as imagens originais têm apenas `64 × 64` pixels e o resize para `224 × 224` não cria novos detalhes;
- somente `q_proj` e `v_proj` receberam adapters de rank 8, limitando deliberadamente a capacidade de adaptação;
- o backbone continuou congelado, portanto não houve fine-tuning completo das representações;
- o EuroSAT é relativamente bem resolvido por texturas e estruturas locais, sem exigir necessariamente toda a flexibilidade de uma arquitetura Transformer.

Assim, o ViT + LoRA não foi “avançado demais” no sentido de ser inadequado, mas ofereceu uma capacidade maior e mais geral do que a necessária para vencer uma ResNet50 muito bem alinhada ao problema. Sua principal vantagem aparece quando armazenamento dos ajustes, reutilização de um mesmo backbone e adaptação eficiente para várias tarefas são requisitos importantes.

### 3º lugar — ViT congelado + LightGBM: F1 Macro 0,9174

O pipeline mostrou que os embeddings genéricos do ViT já contêm informação suficiente para separar bem as classes. Seu treinamento supervisionado foi o mais rápido porque o LightGBM trabalhou sobre vetores previamente extraídos. Entretanto, o ViT nunca recebeu gradientes dos rótulos do EuroSAT: o classificador pôde reorganizar as features existentes, mas não adaptar a própria representação visual. Essa limitação explica o resultado inferior ao LoRA.

A abordagem continua muito útil quando embeddings podem ser calculados uma vez e reutilizados em diferentes análises. Depois da extração, o LightGBM isolado classificou todo o teste em apenas 0,247 segundo. Na inferência ponta a ponta, porém, ainda é necessário executar o ViT, elevando tempo e tamanho do pipeline.

### 4º lugar — CNN do Zero + Optuna: F1 Macro 0,8379

A CNN começou com pesos aleatórios e precisou aprender no próprio EuroSAT desde filtros básicos até representações de alto nível. Os outros três métodos partiram de conhecimento visual adquirido em bases muito maiores. Essa diferença de ponto de partida, somada à arquitetura didática e relativamente compacta, explica a menor pontuação.

Mesmo em quarto lugar, a CNN não representa um experimento ruim: apresentou boa generalização e o menor pipeline completo. Ela cumpriu o papel de demonstrar todo o aprendizado de features do zero e permanece atraente quando simplicidade e tamanho do artefato são mais importantes do que obter a maior F1 possível.

### Decisão Final

Para este problema e este protocolo experimental, a **ResNet50 é a melhor escolha preditiva e também o melhor equilíbrio geral**. O **ViT + LoRA** ocupa o segundo lugar e é a melhor demonstração de adaptação eficiente em número de parâmetros. O **ViT + LightGBM foi o pipeline mais rápido de preparar e treinar**, concluindo extração e busca do classificador em aproximadamente 7,5 minutos, além de favorecer o reuso dos embeddings. A **CNN do zero** oferece o pipeline mais compacto e didático.

O ranking não deve ser interpretado como uma hierarquia universal entre arquiteturas. Ele descreve o comportamento das configurações executadas no EuroSAT RGB, com os mesmos splits e o hardware disponível.

---

# 16. Artefatos Gerados

| Artefato | Localização | Descrição |
|-----------|-------------|-----------|
| `splits_eurosat_seed42.csv` | `data/` | Split estratificado e reproduzível |
| `cnn_eurosat_best.pt` | `models/` | Pesos da melhor época da CNN final |
| `resnet50_eurosat_best.pt` | `models/` | ResNet50 com backbone congelado e cabeça ajustada |
| `lightgbm_embeddings_completo.pkl` | `models/` | Melhor LightGBM treinado sobre os embeddings completos |
| `vit_embeddings_train_completo.npy` | `data/` | Embeddings `[CLS]` das 18.899 imagens de treino |
| `vit_embeddings_val_completo.npy` | `data/` | Embeddings `[CLS]` das 4.051 imagens de validação |
| `vit_embeddings_test_completo.npy` | `data/` | Embeddings `[CLS]` das 4.050 imagens de teste |
| `vit_labels_*_completo.npy` | `data/` | Rótulos alinhados aos embeddings de cada split |
| `vit_index_*_completo.csv` | `data/` | Índices e caminhos na mesma ordem das matrizes de embeddings |
| `busca_lightgbm_completo.csv` | `reports/` | Histórico das três configurações avaliadas na validação |
| `vit_lora_adapter_completo/` | `models/` | Adapter LoRA e cabeça classificadora do melhor checkpoint |
| `historico_lora_completo.csv` | `reports/` | Loss, F1 de validação e learning rate por época do LoRA |
| `CNN_DO_ZERO.png` | Projeto | Infográfico da arquitetura da CNN |
| `OPTUNA_CNN.png` | Projeto | Infográfico do fluxo Optuna + treinamento final |
| `TRANSFER_LEARNING_RESNET50.png` | Projeto | Infográfico do transfer learning com ResNet50 |
| `VIT_EMBEDDINGS_LIGHTGBM.png` | Projeto | Infográfico do pipeline ViT congelado + LightGBM |
| `README.md` | Projeto | Documentação completa do benchmark |

Os quatro experimentos principais do benchmark possuem resultados registrados e comparados sob o mesmo protocolo de splits.

Os arquivos não serão submetidos ao repositório por questões de limite de memória do GitHub

---

# Considerações Finais

Este projeto desenvolve um benchmark de estratégias modernas de Visão Computacional aplicadas à classificação de imagens de satélite.

Ao comparar uma CNN desenvolvida do zero, uma arquitetura baseada em Transfer Learning, um Vision Transformer adaptado com LoRA e uma abordagem híbrida baseada em embeddings e LightGBM, torna-se possível compreender os principais trade-offs entre desempenho, custo computacional e facilidade de implantação.

O benchmark final apresentou a seguinte ordem por F1 Macro de teste: **ResNet50, 0,9576**; **ViT + LoRA, 0,9360**; **ViT congelado + LightGBM, 0,9174**; e **CNN do zero + Optuna, 0,8379**. As três estratégias baseadas em representações pré-treinadas superaram a CNN do zero, confirmando o valor da transferência de conhecimento visual.

A ResNet50 ofereceu o melhor desempenho preditivo e a melhor adequação aos padrões locais do EuroSAT. O LoRA mostrou que é possível adaptar um Transformer treinando somente 0,351% dos parâmetros e mantendo validação e teste praticamente iguais. O pipeline de embeddings + LightGBM foi o mais rápido de preparar e treinar, além de mostrar o valor do reuso de representações. A CNN permaneceu como a solução mais compacta e cumpriu o objetivo didático de construir todo o aprendizado visual desde os pixels.

O objetivo não é identificar uma arquitetura universalmente superior, mas construir um guia prático que auxilie na escolha da abordagem mais adequada para diferentes cenários de classificação de imagens, utilizando um protocolo experimental único, reproduzível e didático.
