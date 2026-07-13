# INSTRUÇÕES PARA GERAÇÃO DE NOTEBOOK — LEIA ANTES DE EXECUTAR

Você deve gerar **um único arquivo Jupyter Notebook (.ipynb)**, chamado `benchmark_visao_computacional_eurosat.ipynb`, implementando o projeto de benchmark de Visão Computacional descrito abaixo. Este documento é a especificação completa e autoritativa. Não pule seções. Não resuma seções em uma única célula. Siga a ordem exata.

## Regras Gerais de Execução

1. Gere o notebook diretamente no formato `.ipynb` (JSON válido de nbformat), não um script `.py`.
2. Cada seção listada abaixo deve virar, no mínimo: uma célula Markdown de título + explicação teórica, uma ou mais células de código comentadas, e uma célula Markdown de interpretação/conclusão dos resultados daquela etapa.
3. Use **PyTorch** como framework principal de Deep Learning. Use **LightGBM** para o modelo clássico sobre embeddings.
4. Todo código deve ser modular: encapsule lógica repetida em funções reutilizáveis definidas na seção "Funções Auxiliares" e reutilize essas funções nas seções seguintes (não duplique código).
5. Fixe seed aleatória (`SEED = 42`) em todas as bibliotecas relevantes (`random`, `numpy`, `torch`, `torch.cuda`) logo na seção de bibliotecas, para garantir reprodutibilidade.
6. Toda decisão de projeto (arquitetura, hiperparâmetro, técnica) deve ser acompanhada de uma justificativa técnica em texto, não apenas do código.
7. Sempre que houver fundamentação matemática relevante (ex.: Cross-Entropy, Self-Attention, LoRA, PCA/t-SNE/UMAP), inclua a formulação em LaTeX dentro da célula Markdown (usando `$$...$$` ou `$...$`).
8. Gere gráficos com `matplotlib`/`seaborn` sempre que houver comparação de métricas, distribuições ou visualizações de imagens/embeddings. Todo gráfico deve ter título, labels de eixo e, quando aplicável, legenda.
9. Ao final de cada bloco de treinamento, imprima/exiba uma tabela de métricas (Accuracy, Precision, Recall, F1-score, AUC) para treino, validação e teste.
10. Não invente resultados numéricos fixos — o código deve calcular os valores reais a partir da execução; onde a execução completa não for viável (ex.: sem GPU disponível), deixe o código pronto para rodar e sinalize isso explicitamente em uma célula Markdown, sem simular métricas falsas.

## Formatação de Títulos (usar exatamente este padrão HTML em células Markdown)

Título principal de seção (nível 1):
```html
# <font color='red' style='font-size: 40px;'> Título da Seção </font>
<hr style='border: 2px solid red;'>
```

Subtítulo (nível 2, dentro de uma seção):
```html
# <font color='green' style='font-size: 30px;'> Subtítulo </font>
<hr style='border: 2px solid green;'>
```

Use exclusivamente esses dois padrões para títulos. Não use `#`, `##`, `###` puros do Markdown para os títulos de seção — apenas para sub-sub-tópicos internos, se necessário.

---

## ESTRUTURA COMPLETA DO NOTEBOOK (seguir esta ordem)

### 1. Problema de Negócio
- Objetivo do projeto: comparar estratégias de aprendizado de representação em Visão Computacional (CNN do zero, Transfer Learning, ViT Fine-Tuning, LoRA, Embeddings + ML clássico).
- Dataset: EuroSAT (https://github.com/phelber/EuroSAT), classificação multiclasse de imagens de satélite, 10 classes de uso e ocupação do solo.
- Tipo de problema: classificação de imagens multiclasse.
- Listar as 10 classes.
- Motivação do benchmark: decisão de arquitetura não deve ser guiada só por acurácia — trade-off entre performance, custo computacional, interpretabilidade e robustez.
- Critérios de comparação entre modelos (listar explicitamente): Accuracy, Precision, Recall, F1-score, AUC, tempo de treino, tempo de inferência, nº de parâmetros (totais e treináveis), uso de memória, tamanho do modelo salvo.

### 2. Bibliotecas Utilizadas
Importar (e justificar brevemente cada grupo): `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `lightgbm`, `optuna`, `shap` (se aplicável a embeddings), `umap-learn`, `opencv-python` ou `PIL`, `tqdm`, `time`, `os`. Definir `SEED = 42` e função `set_seed(seed)`.

### 3. Funções Auxiliares
Implementar e documentar (docstring) as seguintes funções, a serem reutilizadas em todo o notebook:
- `calcular_metricas(y_true, y_pred, y_proba)` → retorna dict com Accuracy, Precision, Recall, F1, AUC.
- `plot_matriz_confusao(y_true, y_pred, classes)`
- `plot_curva_roc(y_true, y_proba, classes)`
- `visualizar_amostras(dataset, classes, n=9)`
- `visualizar_filtros(modelo, camada)`
- `visualizar_feature_maps(modelo, camada, imagem)`
- `comparar_modelos(lista_resultados)` → gera tabela/gráfico comparativo
- `extrair_embeddings(modelo, dataloader, device)`
- `medir_tempo(func)` → decorator para medir tempo de execução
- `contar_parametros(modelo)` → total e treináveis
- `tamanho_modelo_mb(modelo)`
- `treinar_modelo(modelo, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, early_stopping_patience)` → loop de treino genérico reutilizado por CNN, ResNet e ViT.

### 4. Leitura dos Dados
- Estrutura de pastas esperada do EuroSAT (por classe).
- Carregamento via `torchvision.datasets.ImageFolder` ou equivalente.
- Separação treino/validação/teste (proporção sugerida: 70/15/15), com `random_state=SEED`.

### 5. Análise Exploratória das Imagens
- Contagem e distribuição de imagens por classe (gráfico de barras).
- Resolução das imagens (largura × altura) — estatísticas e histograma.
- Estatísticas gerais de pixel/canal.
- Visualização de amostras por classe (grid de imagens).
- Discussão textual de eventuais problemas encontrados (classes desbalanceadas, resolução inconsistente etc.).

### 6. Pré-processamento
Subtítulos (nível 2) para cada técnica, cada um com explicação teórica + código:
- Resize
- Normalização (média/desvio do ImageNet quando for usar modelos pré-treinados)
- Data Augmentation (explicar cada transformação: flip, rotação, color jitter, crop — e o efeito de cada uma sobre overfitting/underfitting)
- Pipeline de transformação (`torchvision.transforms.Compose`) e `DataLoader`s de treino/validação/teste

### 7. Baseline — CNN Desenvolvida do Zero
Subtítulos obrigatórios (nível 2), cada um com explicação teórica:
- Construção da Arquitetura (justificar nº de camadas conv., nº de filtros, kernel size, padding, stride, campo receptivo)
- Função de Ativação (comparar ReLU, LeakyReLU, GELU, SiLU — justificar escolha)
- Pooling (comparar Max/Average/Global Average Pooling)
- Batch Normalization (explicar efeito na estabilidade/convergência)
- Dropout (explicar regularização)
- Flatten e Camadas Fully Connected
- Camada de Saída
- Contagem de Parâmetros e Model Summary (usar `torchinfo.summary` ou equivalente)
- Fluxo da Informação (diagrama textual ou imagem mostrando dimensões em cada camada)
- Inspeção da Arquitetura (visualizar dimensões dos feature maps por camada)
- Função de Perda (derivar Cross-Entropy Loss matematicamente em LaTeX)
- Otimizador (comparar SGD, Momentum, Adam, AdamW — justificar Adam/AdamW)
- Learning Rate (conceito e escolha do valor inicial)
- Scheduler (comparar StepLR, Cosine Annealing, ReduceLROnPlateau — justificar escolha)
- Weight Decay (explicar regularização L2)
- Early Stopping (explicar critério de parada)
- Loop de Treinamento (explicar forward, cálculo da loss, backpropagation, atualização dos pesos, zero_grad; usar a função `treinar_modelo`)
- Acompanhamento do Treinamento (gráficos de loss treino/validação, accuracy, learning rate, tempo por época)
- Avaliação (Accuracy, Precision, Recall, F1, AUC, matriz de confusão, curva ROC)
- O que a CNN Aprendeu (visualizar filtros e feature maps das primeiras e últimas camadas; discutir evolução bordas → texturas → partes → objeto)

### 8. Otimização de Hiperparâmetros com Optuna
- Explicar Grid Search vs Random Search vs Bayesian Optimization vs TPE vs Pruning.
- Definir espaço de busca: learning rate, batch size, weight decay, nº de filtros, nº de camadas, kernel size, dropout, scheduler.
- Executar `optuna.create_study` com pruning.
- Reportar: melhores hiperparâmetros, importância dos hiperparâmetros (`optuna.visualization`), evolução dos trials.
- Avaliação Final: comparar CNN Base vs CNN Otimizada com todas as métricas lado a lado.

### 9. Fine-Tuning de uma ResNet
- Carregar ResNet50 pré-treinada (`torchvision.models.resnet50(weights=...)`).
- Explicar Transfer Learning, estratégia de congelamento de camadas, substituição da camada final para 10 classes.
- Treinar (reutilizar `treinar_modelo`) e avaliar.
- Comparar com a CNN do zero.

### 10. Fine-Tuning de um Vision Transformer
- Carregar ViT pré-treinado (ex.: `timm` ou `transformers`, `google/vit-base-patch16-224`).
- Explicar Patch Embeddings, Self-Attention (formulação matemática em LaTeX), diferenças estruturais frente a CNNs.
- Adaptar cabeça de classificação para 10 classes, treinar e avaliar.

### 11. Fine-Tuning utilizando LoRA
- Aplicar LoRA ao ViT (ex.: biblioteca `peft`).
- Explicar Low-Rank Adaptation matematicamente (matrizes A e B, rank r, alpha), quais camadas são adaptadas.
- Comparar: parâmetros totais vs treináveis, tempo de treinamento, memória utilizada — frente ao fine-tuning completo do ViT.

### 12. Extração de Embeddings
- Usar o backbone do melhor modelo (dentre CNN, ResNet, ViT, ViT+LoRA) para extrair embeddings do conjunto de teste.
- Explicar o conceito de embedding e a dimensão do vetor resultante.
- Salvar os embeddings (`.parquet` ou `.npy`).

### 13. Classificação utilizando Machine Learning
- Treinar um `LightGBM` classificador usando apenas os embeddings extraídos como features.
- Comparar métricas com os modelos neurais fim-a-fim.

### 14. Visualização dos Embeddings
- Aplicar t-SNE e UMAP sobre os embeddings.
- Gerar gráficos de dispersão coloridos por classe.
- Discutir textualmente a qualidade/separabilidade das representações aprendidas.

### 15. Interpretabilidade
- CNN: Grad-CAM (explicar o método e implementar).
- ViT: Attention Maps (explicar e implementar).
- Comparar visualmente (grid de imagens) quais regiões cada arquitetura usa para decidir.

### 16. Robustez dos Modelos
- Gerar versões perturbadas do conjunto de teste: blur, ruído gaussiano, redução de resolução, variação de iluminação.
- Avaliar a degradação das métricas de cada modelo sob cada perturbação.
- Gráfico comparativo de degradação por modelo/perturbação.

### 17. Comparação dos Modelos
- Tabela consolidada (pandas DataFrame) com todos os modelos treinados e todas as métricas: Accuracy, Precision, Recall, F1, AUC, tempo de treino, tempo de inferência, nº parâmetros totais, nº parâmetros treináveis, uso de memória, tempo por época, tamanho do modelo salvo (MB).
- Gráficos comparativos (barras) para as principais métricas.

### 18. Trade-offs
Discussão textual estruturada cobrindo: desempenho, custo computacional, velocidade de treino/inferência, interpretabilidade, robustez, escalabilidade, facilidade de treinamento, capacidade de generalização. Responder explicitamente quando cada estratégia é mais indicada.

### 19. Engenharia de Atributos baseada em Embeddings
Construir e explicar os seguintes indicadores derivados dos embeddings: norma L2, similaridade entre imagens (cosseno), distância ao centróide da classe, score de confiança da predição, entropia das probabilidades, margem entre a 1ª e 2ª classe mais prováveis. Discutir aplicações práticas (ex.: detecção de casos de baixa confiança / drift).

### 20. Conclusões
Responder explicitamente, em texto corrido, às seguintes perguntas:
- Qual estratégia apresentou a melhor performance?
- Qual apresentou o melhor custo-benefício?
- Quando vale a pena treinar uma CNN do zero?
- Quando utilizar Transfer Learning?
- Quando utilizar Fine-Tuning completo?
- Quando utilizar LoRA?
- Em quais cenários embeddings + ML clássico são suficientes?
- Quanto a otimização de hiperparâmetros melhorou a CNN (comparação quantitativa)?
- Quais limitações foram encontradas no projeto?
- Quais são os próximos passos / trabalhos futuros?

---

## Checklist Final (validar antes de considerar o notebook completo)

- [ ] Todas as 20 seções presentes, na ordem, com o padrão HTML de título correto
- [ ] Todas as funções auxiliares implementadas e reutilizadas (sem duplicação de código de treino/avaliação)
- [ ] Seed fixada e reprodutibilidade garantida
- [ ] Todas as fundamentações matemáticas presentes em LaTeX (Cross-Entropy, Self-Attention, LoRA)
- [ ] Todos os 6 "modelos" comparáveis na tabela final: CNN Base, CNN Otimizada, ResNet50, ViT, ViT+LoRA, Embeddings+LightGBM
- [ ] Gráficos com título e labels em todas as seções que geram visualização
- [ ] Seção de Conclusões responde a todas as perguntas listadas explicitamente
- [ ] Nenhuma métrica numérica inventada — apenas resultado de execução real ou placeholder explicitamente sinalizado
