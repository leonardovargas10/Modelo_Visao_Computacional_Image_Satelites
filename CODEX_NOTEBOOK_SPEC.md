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

# Propósito Didático e Perfil do Leitor

Este notebook não deverá ser apenas um script de treinamento. Ele deverá funcionar como uma aula prática, progressiva e autocontida para uma pessoa iniciante em Visão Computacional.

Antes de apresentar modelos prontos, o notebook deverá construir a intuição necessária para responder:

- como um arquivo de imagem é lido pelo computador;
- como uma imagem se transforma em uma matriz e depois em um tensor;
- o que representam altura, largura, canais, pixels, dtype e faixa de valores;
- como os canais RGB se combinam para formar a imagem;
- como redimensionamento, normalização e data augmentation alteram o tensor;
- como um kernel percorre uma imagem e produz um mapa de características;
- como filtros detectam bordas, texturas e outros padrões locais;
- como uma CNN aprende seus próprios filtros;
- como uma CNN, uma ResNet pré-treinada e um ViT representam uma imagem de formas diferentes;
- por que o benchmark é justo e o que cada métrica permite concluir.

O fluxo pedagógico obrigatório será:

```text
Arquivo de imagem
→ pixels e canais RGB
→ array NumPy
→ tensor PyTorch
→ transformações e filtros manuais
→ convolução e feature maps
→ Dataset e DataLoader
→ CNN do zero
→ modelos pré-treinados e embeddings
→ benchmark e interpretação dos resultados
```

Cada grande seção deverá começar com uma pergunta ou objetivo de aprendizagem e terminar com uma pequena conclusão baseada no que foi observado, evitando células que apenas executem código sem explicar sua finalidade.

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
14. Introduzir conceitos antes de usá-los nos modelos.
15. Mostrar formas, tipos e faixas de valores dos principais arrays e tensores.
16. Utilizar células pequenas, com uma responsabilidade clara, evitando células monolíticas.
17. Toda visualização deverá ter título, rótulos, legenda ou colorbar quando aplicável.
18. Todo resultado importante deverá ser seguido por uma célula Markdown com interpretação em linguagem simples.
19. Não transformar a exploração didática em novos experimentos do benchmark; filtros manuais e augmentations servem para aprendizagem, não como modelos concorrentes.
20. Preservar a organização didática observada nos notebooks `exemplo_CNN_1.ipynb`, `exemplo_CNN_2.ipynb` e `exemplo_ViT.ipynb`, adaptando-a ao EuroSAT e às funções reutilizáveis do projeto.

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

## Parte I — Fundamentos e exploração guiada

1. Apresentação, objetivos de aprendizagem e mapa do notebook
2. Problema de negócio e descrição do EuroSAT RGB
3. Bibliotecas, ambiente, dispositivo e reprodutibilidade
4. Configurações e funções auxiliares
5. Leitura e inventário dos arquivos
6. Da imagem ao array e ao tensor
7. Canais RGB, escala de cinza e visualização numérica
8. Transformações geométricas, interpolação e normalização
9. Filtros clássicos e convolução passo a passo
10. Data augmentation e diferença entre treino e avaliação
11. Análise exploratória e qualidade dos dados
12. Separação estratificada e DataLoaders

## Parte II — Modelagem e benchmark

13. Baseline e protocolo experimental comum
14. CNN desenvolvida do zero
15. Otimização da CNN com Optuna
16. Transfer Learning com ResNet50
17. ViT congelado: extração de embeddings e LightGBM
18. Vision Transformer adaptado com LoRA
19. Interpretabilidade
20. Comparação consolidada e análise de erros
21. Trade-offs, conclusões e próximos passos

---

# Laboratório Guiado Antes dos Modelos

## 1. Leitura e inspeção das imagens

Selecionar exemplos reais do EuroSAT e demonstrar, passo a passo:

- caminho do arquivo, classe e formato;
- leitura com PIL e conversão para NumPy;
- `shape`, `dtype`, valores mínimo e máximo;
- resolução, modo de cor e quantidade de canais;
- exibição de uma imagem e de uma grade com exemplos das 10 classes;
- ampliação de uma pequena região para tornar os pixels visíveis;
- impressão de um recorte pequeno da matriz, sem despejar a imagem inteira como números.

Explicar explicitamente as convenções:

```text
PIL / NumPy: altura × largura × canais (H, W, C)
PyTorch:     canais × altura × largura (C, H, W)
Batch:       amostras × canais × altura × largura (N, C, H, W)
```

Mostrar a conversão `uint8 [0, 255] → float32 [0, 1]` e esclarecer que o arquivo JPEG/PNG não entra diretamente na rede: ele é decodificado em números.

## 2. Decomposição dos canais RGB

Para a mesma imagem:

- plotar a imagem original;
- plotar separadamente as matrizes R, G e B em escala de cinza;
- plotar cada canal mantendo apenas sua cor correspondente;
- comparar histogramas de intensidade dos três canais;
- inspecionar o vetor RGB de alguns pixels escolhidos;
- produzir uma versão em escala de cinza e explicar a perda de informação.

Relacionar as intensidades observadas às características de imagens de satélite, sem afirmar que RGB contém bandas multiespectrais. Deixar claro que o escopo utiliza apenas o EuroSAT RGB.

## 3. Redimensionamento, interpolação e normalização

Demonstrar visualmente:

- imagem antes e depois de `Resize`;
- pelo menos nearest neighbor e bilinear em uma região ampliada;
- alteração do `shape`, mas não da classe da imagem;
- normalização por média e desvio padrão;
- imagem/tensor antes e depois da normalização;
- procedimento correto de desfazer a normalização apenas para visualização.

Explicar por que os modelos exigem dimensões padronizadas e por que modelos pré-treinados devem usar o tamanho e a normalização esperados por seus pesos/processadores.

## 4. Filtros e convolução passo a passo

Antes de usar `nn.Conv2d`, implementar uma demonstração pequena e legível da operação:

1. converter uma imagem para escala de cinza;
2. selecionar uma pequena janela da matriz;
3. definir um kernel 3 × 3 simples;
4. mostrar multiplicação elemento a elemento e soma em uma posição;
5. deslizar o kernel e formar o mapa de saída;
6. comparar a entrada e o mapa resultante.

Aplicar poucos filtros clássicos, suficientes para criar intuição:

- suavização/média;
- detecção de bordas horizontal e vertical (Sobel);
- realce de contornos ou nitidez.

Mostrar e explicar os efeitos de:

- tamanho do kernel;
- `stride`;
- `padding`;
- dimensões da saída;
- um canal de entrada versus três canais;
- múltiplos filtros gerando múltiplos feature maps.

Depois, repetir um exemplo equivalente com `torch.nn.Conv2d`, deixando explícita a diferença entre filtros manuais fixos e kernels aprendidos por backpropagation.

## 5. Pooling, ativações e feature maps

Demonstrar em exemplos pequenos:

- ReLU removendo ativações negativas;
- Max Pooling reduzindo as dimensões espaciais;
- efeito dessas operações sobre o tensor;
- visualização de feature maps de uma camada convolucional;
- progressão conceitual de bordas e texturas para representações mais abstratas.

Sempre imprimir o `shape` antes e depois de cada operação e relacionar a mudança ao custo computacional e à perda controlada de resolução espacial.

## 6. Data augmentation

Construir duas pipelines separadas:

```text
Treino:     transformações aleatórias + tensor + normalização
Val/Teste:  transformações determinísticas + tensor + normalização
```

Visualizar várias versões da mesma imagem com augmentations moderadas, como:

- flips coerentes com imagens aéreas;
- rotações;
- pequenas alterações de cor, brilho ou contraste;
- recorte/redimensionamento quando tecnicamente justificado.

Explicar:

- augmentation aumenta diversidade, não cria informação nova;
- a classe precisa permanecer semanticamente válida;
- transformações agressivas podem destruir sinais úteis;
- augmentation aleatória não deve ser aplicada em validação ou teste;
- estatísticas, parâmetros ou decisões aprendidas com os dados devem usar somente o treino quando isso puder causar leakage.

Incluir uma comparação simples entre amostra original e versões aumentadas, sem criar um benchmark paralelo de dezenas de políticas.

## 7. Análise exploratória e controles de qualidade

Incluir:

- contagem e proporção por classe;
- exemplos representativos de todas as classes;
- distribuição de largura, altura, formato e canais;
- verificação de arquivos corrompidos ou ilegíveis;
- inspeção de possíveis duplicatas, quando viável;
- discussão visual de classes potencialmente parecidas;
- exemplos difíceis ou ambíguos encontrados durante a exploração.

Encerrar a Parte I com uma seção Markdown chamada **O que aprendemos antes de modelar**, resumindo como os pixels foram convertidos até chegarem ao formato consumido pelos modelos.

---

# Organização do Código e da Narrativa

## Padrão das células

Cada seção deverá seguir, quando aplicável, este ciclo:

1. **Objetivo (Markdown):** o que será investigado e por quê.
2. **Teoria mínima (Markdown):** conceito, fórmula ou diagrama necessário.
3. **Código (Python):** implementação curta e legível.
4. **Evidência (saída/gráfico/tabela):** o que ocorreu.
5. **Interpretação (Markdown):** o que o resultado significa e como afeta a próxima decisão.

## Padrão do código

- Centralizar constantes e caminhos em uma célula de configuração.
- Reutilizar, quando adequado, os módulos de `src/` em vez de duplicar lógica no notebook.
- Manter no notebook pequenas demonstrações pedagógicas; mover rotinas extensas e reutilizáveis para funções/módulos.
- Usar nomes descritivos em português ou inglês de forma consistente.
- Adicionar type hints e docstrings às funções reutilizáveis.
- Comentar principalmente o **porquê** de uma decisão; não comentar cada linha óbvia.
- Separar funções de leitura, visualização, transformação, treino e avaliação.
- Utilizar `Dataset`, `DataLoader`, `nn.Module`, `device`, `modelo.train()`, `modelo.eval()` e `torch.no_grad()` explicitamente.
- Evitar números mágicos; registrar configurações em dicionários ou dataclasses simples quando útil.
- Exibir progresso de treinamento sem produzir saída excessiva.
- Definir funções comuns de treino e avaliação para reduzir divergências entre modelos.

## Checkpoints didáticos

Antes de avançar, o notebook deverá permitir verificar:

- **Checkpoint 1:** consigo explicar `H × W × C`, pixel e canal?
- **Checkpoint 2:** consigo acompanhar a mudança `PIL → NumPy → Tensor`?
- **Checkpoint 3:** entendo o que kernel, stride, padding e feature map fazem?
- **Checkpoint 4:** sei por que treino e validação usam transformações diferentes?
- **Checkpoint 5:** entendo por que todos os modelos devem receber os mesmos splits?
- **Checkpoint 6:** consigo distinguir aprendizado do zero, transfer learning, embeddings e LoRA?

Esses checkpoints devem aparecer como pequenos resumos ou perguntas respondidas pelo próprio notebook, e não como exercícios sem solução.

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
