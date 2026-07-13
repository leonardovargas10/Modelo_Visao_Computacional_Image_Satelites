# INSTRUÇÕES PARA GERAÇÃO DE NOTEBOOK — BENCHMARK DE VISÃO COMPUTACIONAL (ESCOPO REDUZIDO)

Gere um único arquivo Jupyter Notebook chamado:

```text
benchmark_visao_computacional_eurosat.ipynb
```

O notebook deverá implementar um benchmark de classificação de imagens utilizando o dataset:

```text
EuroSAT RGB
```

Utilize a estrutura de diretórios:

```text
data/
```

O dataset possui 10 classes de uso e ocupação do solo.

---

# Regras Gerais

1. Gerar diretamente um arquivo `.ipynb` válido.
2. Utilizar PyTorch como framework principal.
3. Utilizar LightGBM para classificação utilizando embeddings.
4. Utilizar Hugging Face Transformers para o Vision Transformer.
5. Utilizar PEFT para LoRA.
6. Utilizar Optuna apenas para otimizar a CNN desenvolvida do zero.
7. Definir `SEED = 42`.
8. Utilizar exatamente os mesmos splits em todos os experimentos.
9. Não utilizar o conjunto de teste durante treinamento ou seleção de modelos.
10. Evitar data leakage entre treino, validação e teste.
11. Utilizar F1-score Macro como principal métrica de avaliação.
12. Manter código modular, comentado e didático.
13. Priorizar simplicidade em vez de testar dezenas de arquiteturas e hiperparâmetros.

---

# Modelos do Benchmark

Implementar apenas:

1. CNN desenvolvida do zero e otimizada com Optuna;
2. Transfer Learning utilizando ResNet50;
3. Extração de Embeddings (ViT) + LightGBM;
4. Vision Transformer adaptado utilizando LoRA.

Não implementar:

- Fine-Tuning completo do Vision Transformer;
- Comparação entre diferentes CNNs;
- Comparação entre diferentes Vision Transformers;
- Comparação entre diferentes funções de ativação;
- Comparação entre diferentes métodos de pooling;
- Comparação entre diferentes otimizadores;
- Comparação entre diferentes schedulers;
- Comparação entre diferentes funções de perda.

---

# Estrutura do Notebook

1. Problema de Negócio
2. Bibliotecas e Ambiente
3. Funções Auxiliares
4. Leitura dos Dados
5. Separação dos Dados
6. Análise Exploratória das Imagens
7. Pré-processamento
8. CNN Desenvolvida do Zero
9. Otimização da CNN com Optuna
10. Transfer Learning com ResNet50
11. Vision Transformer com LoRA
12. Extração de Embeddings
13. LightGBM sobre Embeddings
14. Interpretabilidade
15. Comparação Consolidada
16. Trade-offs
17. Conclusões

---

# Diretrizes por Modelo

## CNN Desenvolvida do Zero

Pipeline:

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

Utilizar obrigatoriamente:

- ReLU;
- Batch Normalization;
- Max Pooling;
- Dropout;
- CrossEntropyLoss;
- Adam;
- Weight Decay;
- Gradient Clipping;
- ReduceLROnPlateau;
- Early Stopping.

A arquitetura deverá ser construída manualmente utilizando PyTorch.

Não comparar diferentes funções de ativação, diferentes métodos de pooling ou diferentes otimizadores.

---

## Otimização com Optuna

O Optuna deverá otimizar apenas a CNN.

Ajustar apenas:

- learning_rate;
- batch_size;
- número de filtros;
- dimensão da camada totalmente conectada;
- dropout;
- weight_decay.

Manter fixos:

- ReLU;
- Max Pooling;
- Batch Normalization;
- Adam;
- CrossEntropyLoss;
- ReduceLROnPlateau.

Após a otimização:

- recuperar os melhores hiperparâmetros;
- treinar a CNN definitiva;
- avaliar no conjunto de teste;
- salvar o melhor checkpoint.

---

## Transfer Learning com ResNet50

Utilizar apenas:

```text
ResNet50
```

Pipeline:

```text
Imagem

↓

ResNet50 Pré-Treinada

↓

Nova Cabeça de Classificação

↓

10 Classes
```

Explicar:

- Transfer Learning;
- congelamento das camadas convolucionais;
- substituição da camada de classificação;
- Fine-Tuning parcial das últimas camadas, quando necessário.

Não comparar diferentes arquiteturas pré-treinadas.

---

## Vision Transformer + LoRA

Utilizar um Vision Transformer pré-treinado.

Aplicar apenas LoRA.

Não realizar Fine-Tuning completo.

Configuração sugerida:

- Rank = 8;
- Alpha = 16;
- Dropout = 0.10;
- AdamW;
- Warmup;
- Early Stopping.

Treinar apenas:

- matrizes LoRA;
- cabeça de classificação.

Explicar resumidamente:

- Patch Embeddings;
- Self-Attention;
- Query;
- Key;
- Value;
- Multi-Head Attention;
- Feed-Forward;
- LoRA.

---

## Extração de Embeddings

Utilizar o mesmo Vision Transformer empregado no experimento com LoRA.

Congelar completamente o backbone.

Extrair os embeddings para:

- treino;
- validação;
- teste.

Salvar os embeddings em:

```text
.npy
```

Explicar:

- conceito de embedding;
- dimensão do vetor;
- diferença entre embeddings e classificação fim-a-fim.

---

## LightGBM sobre Embeddings

Treinar um LightGBM utilizando apenas os embeddings como variáveis de entrada.

Pipeline:

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

Realizar apenas pequenos ajustes em:

- learning_rate;
- num_leaves;
- max_depth;
- n_estimators;
- reg_alpha;
- reg_lambda.

Não realizar uma busca extensa de hiperparâmetros.

---

# Avaliação

Todos os modelos deverão utilizar exatamente as mesmas métricas:

- Accuracy;
- Precision Macro;
- Recall Macro;
- F1-score Macro;
- F1 por classe;
- AUC;
- Matriz de Confusão;
- Tempo de treinamento;
- Tempo de inferência;
- Quantidade de parâmetros treináveis;
- Tamanho do modelo.

---

# Interpretabilidade

Comparar os quatro modelos utilizando técnicas apropriadas.

Para a CNN:

- visualização dos filtros;
- feature maps;
- Grad-CAM.

Para o Vision Transformer:

- mapas de atenção;
- visualização das regiões mais relevantes.

Discutir qualitativamente:

- padrões aprendidos;
- regiões utilizadas para classificação;
- diferenças entre CNNs e Transformers.

---

# Comparação Final

Comparar:

- CNN Desenvolvida do Zero + Optuna;
- ResNet50;
- Vision Transformer + LoRA;
- Embeddings + LightGBM.

Responder:

- Qual apresentou melhor desempenho?
- Qual apresentou melhor custo-benefício?
- Quando utilizar cada abordagem?
- Quais limitações foram observadas?

---

# Filosofia do Projeto

Este projeto tem como objetivo aprender os principais pipelines modernos de Visão Computacional.

Cada abordagem deverá utilizar uma metodologia consolidada e suficientemente representativa, evitando uma quantidade excessiva de experimentos.

O foco será compreender as diferenças entre as estratégias de aprendizado de representação e os trade-offs entre desempenho, custo computacional, interpretabilidade e facilidade de implementação.

Ao final, o benchmark deverá comparar quatro metodologias distintas:

- Aprendizado do zero utilizando uma CNN desenvolvida manualmente;
- Transfer Learning utilizando uma ResNet50 pré-treinada;
- Extração de embeddings utilizando um Vision Transformer seguida de classificação com LightGBM;
- Adaptação eficiente de um Vision Transformer utilizando LoRA.

O objetivo não é encontrar a arquitetura perfeita, mas compreender quando cada estratégia é mais adequada para diferentes cenários de classificação de imagens, produzindo um benchmark limpo, didático, reproduzível e alinhado às práticas modernas de Deep Learning.