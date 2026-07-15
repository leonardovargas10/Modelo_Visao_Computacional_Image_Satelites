"""Gera o notebook didático e reproduzível do benchmark EuroSAT."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmark_visao_computacional_eurosat.ipynb"


def _source(text: str) -> list[str]:
    text = dedent(text).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str, tags: list[str] | None = None) -> dict:
    metadata = {"tags": tags} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": _source(text),
    }


cells: list[dict] = []


def M(text: str) -> None:
    cells.append(md(text))


def C(text: str, tags: list[str] | None = None) -> None:
    cells.append(code(text, tags))


M(r"""
# Benchmark de Visão Computacional — EuroSAT RGB

**CNN do zero + Optuna · ResNet50 · ViT + LightGBM · ViT + LoRA**

Este notebook é simultaneamente uma aula prática e o ponto de entrada executável do projeto. O percurso é:

> arquivo de imagem → pixels RGB → array NumPy → tensor PyTorch → filtros e convoluções → `Dataset`/`DataLoader` → quatro estratégias de modelagem → benchmark comum

Ao final, saberemos não apenas qual modelo obteve o maior **F1 Macro**, mas também quanto custou treiná-lo, quantos parâmetros foram adaptados e em quais classes ele errou.
""")

M(r"""
## Como usar este notebook

1. Execute primeiro toda a **Parte I**; ela valida os dados e materializa um único split estratificado.
2. Na configuração, mantenha `EXECUTAR_TREINOS = False` para estudar o fluxo sem iniciar tarefas longas.
3. Quando o ambiente estiver pronto, altere a flag para `True` e execute os experimentos em ordem.
4. O teste só é consultado uma vez por modelo, depois de todas as decisões feitas com treino/validação.

Os blocos caros estão marcados com a tag `treino-pesado`. Em CPU, prefira `MODO_RAPIDO = True` para validar o pipeline antes da execução completa.
""")

M(r"""
# Parte I — Fundamentos e exploração guiada

## 1. Apresentação, objetivos e mapa

**Pergunta de aprendizagem:** como números organizados em uma imagem se transformam em uma decisão de classe?

Objetivos:

- interpretar `H × W × C`, pixels, canais, `dtype` e faixa de valores;
- acompanhar `PIL → NumPy → Tensor`;
- entender kernels, feature maps, ativações e pooling;
- construir splits sem vazamento;
- distinguir aprendizado do zero, transfer learning, embeddings e LoRA.
""")

M(r"""
### Mapa experimental

| Estratégia | Representação | O que é treinado |
|---|---|---|
| CNN do zero | filtros aprendidos no EuroSAT | toda a rede |
| ResNet50 | features ImageNet | cabeça e, opcionalmente, último bloco |
| ViT + LightGBM | embedding `[CLS]` congelado | somente LightGBM |
| ViT + LoRA | patches + self-attention | adaptadores LoRA e classificador |

**Conclusão da seção:** todos partem dos mesmos arquivos e índices; o que muda é como a representação visual é obtida e adaptada.
""")

M(r"""
## 2. Problema de negócio e EuroSAT RGB

**Pergunta de aprendizagem:** qual decisão o modelo apoia e o que existe nos dados?

O EuroSAT RGB contém imagens aéreas de uso e cobertura do solo em dez classes. Um classificador pode apoiar triagem cartográfica e monitoramento territorial, mas não substitui validação geoespacial. Este escopo usa **somente RGB**; vermelho, verde e azul não devem ser confundidos com todas as bandas multiespectrais do Sentinel-2.

A unidade de análise é uma imagem e o alvo é uma das dez classes. Como errar uma classe rara importa tanto quanto errar uma frequente, o critério principal será **F1 Macro**.

**Conclusão da seção:** o benchmark mede classificação visual no recorte RGB, não sensoriamento multiespectral completo.
""")

M(r"""
## 3. Bibliotecas, ambiente, dispositivo e reprodutibilidade

**Objetivo:** carregar apenas dependências básicas no início e diagnosticar as opcionais antes dos experimentos.

PyTorch conduz os modelos; `transformers` fornece o ViT; `peft` aplica LoRA; LightGBM classifica embeddings; Optuna busca somente hiperparâmetros da CNN.
""")

C(r"""
import importlib.util

PACOTES_BASE = [
    "matplotlib",
    "numpy",
    "pandas",
    "PIL",
    "seaborn",
    "sklearn",
    "torch",
    "torchvision",
]
ausentes_base = [p for p in PACOTES_BASE if importlib.util.find_spec(p) is None]
if ausentes_base:
    raise RuntimeError(
        "O kernel selecionado não possui as dependências básicas: "
        f"{ausentes_base}. Instale-as nesse ambiente ou selecione o kernel "
        "correto antes de continuar."
    )
print("Dependências básicas encontradas no kernel.")
""")

C(r"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import pickle
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = False
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)
""")

C(r"""
PACOTES_OPCIONAIS = ["optuna", "lightgbm", "transformers", "peft", "safetensors"]
diagnostico = pd.DataFrame({
    "pacote": PACOTES_OPCIONAIS,
    "disponivel": [importlib.util.find_spec(p) is not None for p in PACOTES_OPCIONAIS],
})
display(diagnostico)
print(
    f"Python: {sys.version.split()[0]} | "
    f"PyTorch: {torch.__version__} | "
    f"torchvision: {torchvision.__version__}"
)
print(f"Sistema: {platform.platform()}")
""")

M(r"""
**Interpretação:** dependências opcionais ausentes afetam apenas seus respectivos experimentos. Instale-as no ambiente do kernel antes de ativar os treinos; não escondemos instalações dentro do notebook porque isso dificulta a reprodução.

**Conclusão da seção:** versões e disponibilidade ficam visíveis antes de consumir GPU ou baixar pesos.
""")

M(r"""
## 4. Configurações e funções auxiliares

**Objetivo:** centralizar caminhos, semente e controles de custo. Nada abaixo deve depender do conjunto de teste para tomar decisões.
""")

C(r"""
@dataclass(frozen=True)
class Config:
    seed: int = 42
    img_size: int = 224
    val_size: float = 0.15
    test_size: float = 0.15
    num_workers: int = 0 if os.name == "nt" else 2
    epochs: int = 30
    patience: int = 6
    n_trials: int = 20
    vit_model_id: str = "google/vit-base-patch16-224-in21k"


CFG = Config()
SEED = CFG.seed
ROOT = Path.cwd().resolve()
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SPLIT_FILE = DATA_DIR / "splits_eurosat_seed42.csv"

MODO_RAPIDO = True
EXECUTAR_TREINOS = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for pasta in (ARTIFACT_DIR, REPORT_DIR):
    pasta.mkdir(parents=True, exist_ok=True)

print(f"Projeto: {ROOT}")
print(f"Dados: {DATA_DIR}")
print(f"Dispositivo: {DEVICE} | modo rápido: {MODO_RAPIDO} | treinos: {EXECUTAR_TREINOS}")
""")

C(r"""
def definir_semente(seed: int = SEED) -> None:
    '''Reduz variação acidental entre execuções.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def exigir_pacote(nome: str) -> None:
    '''Produz erro acionável somente quando um experimento opcional é solicitado.'''
    if importlib.util.find_spec(nome) is None:
        raise ImportError(f"Instale '{nome}' no ambiente deste kernel para executar esta seção.")


def worker_seed(worker_id: int) -> None:
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)


def criar_gerador(seed: int = SEED) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def desnormalizar(
    tensor: torch.Tensor,
    media: Iterable[float],
    desvio: Iterable[float],
) -> torch.Tensor:
    media_t = torch.tensor(list(media), device=tensor.device)[:, None, None]
    desvio_t = torch.tensor(list(desvio), device=tensor.device)[:, None, None]
    return (tensor * desvio_t + media_t).clamp(0, 1)


def tamanho_modelo_mb(modelo: nn.Module) -> float:
    bytes_total = sum(p.numel() * p.element_size() for p in modelo.parameters())
    bytes_total += sum(b.numel() * b.element_size() for b in modelo.buffers())
    return bytes_total / 1024**2


def contar_parametros(modelo: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in modelo.parameters())
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return total, treinaveis


def resumo_tensor(nome: str, x: torch.Tensor | np.ndarray) -> None:
    print(f"{nome}: shape={tuple(x.shape)}, dtype={x.dtype}, min={x.min():.4f}, max={x.max():.4f}")


def logits_do_modelo(modelo: nn.Module, imagens: torch.Tensor) -> torch.Tensor:
    saida = modelo(imagens)
    return saida.logits if hasattr(saida, "logits") else saida


def subconjunto_rapido(df: pd.DataFrame, n_por_classe: int = 120) -> pd.DataFrame:
    amostras_por_classe = []

    for _, grupo in df.groupby("classe"):
        quantidade = min(len(grupo), n_por_classe)
        amostras_por_classe.append(
            grupo.sample(
                n=quantidade,
                random_state=SEED,
            )
        )

    amostra_balanceada = pd.concat(
        amostras_por_classe,
        ignore_index=True,
    )
    return amostra_balanceada


def registrar_resultado(
    nome: str,
    metricas: dict,
    treino_s: float,
    inferencia_s: float,
    modelo: nn.Module | None = None,
) -> None:
    total, treinaveis, tamanho = np.nan, np.nan, np.nan
    if modelo is not None:
        total, treinaveis = contar_parametros(modelo)
        tamanho = tamanho_modelo_mb(modelo)

    RESULTADOS[nome] = {
        "modelo": nome,
        **metricas,
        "treino_s": treino_s,
        "inferencia_s": inferencia_s,
        "parametros_totais": total,
        "parametros_treinaveis": treinaveis,
        "tamanho_mb": tamanho,
    }


def assert_splits_disjuntos(df: pd.DataFrame) -> None:
    nomes_splits = ("treino", "validacao", "teste")
    grupos = {
        split: set(df.loc[df["split"].eq(split), "caminho_relativo"])
        for split in nomes_splits
    }
    assert not (grupos["treino"] & grupos["validacao"])
    assert not (grupos["treino"] & grupos["teste"])
    assert not (grupos["validacao"] & grupos["teste"])


def fingerprint_splits(df: pd.DataFrame) -> str:
    colunas = ["caminho_relativo", "classe", "label", "split"]
    conteudo = (
        df[colunas]
        .sort_values("caminho_relativo")
        .to_csv(index=False)
        .encode("utf-8")
    )
    return hashlib.sha256(conteudo).hexdigest()[:16]


def limpar_memoria() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


definir_semente()
RESULTADOS: dict[str, dict] = {}
""")

M(r"""
**Por que não importamos `src`?** Os módulos existentes em `src/` são utilitários de modelagem tabular e binária; forçá-los neste fluxo multiclasse de imagens criaria acoplamento incorreto. As rotinas visuais reutilizáveis permanecem autocontidas até serem estabilizadas e então poderão migrar para um módulo específico.

**Conclusão da seção:** caminhos e decisões experimentais estão em um único lugar, e a semente é aplicada também aos workers.
""")

M(r"""
## 5. Leitura e inventário dos arquivos

**Pergunta de aprendizagem:** como comprovamos que as dez classes e os arquivos esperados realmente existem?
""")

C(r"""
EXTENSOES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Diretório não encontrado: {DATA_DIR}")

registros = []
for pasta_classe in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
    for caminho in sorted(pasta_classe.iterdir()):
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES:
            registros.append({
                "caminho": caminho,
                "caminho_relativo": caminho.relative_to(ROOT).as_posix(),
                "classe": pasta_classe.name,
            })

df_arquivos = pd.DataFrame(registros)
if df_arquivos.empty:
    raise RuntimeError(
        "Nenhuma imagem foi encontrada em data/<classe>/. "
        "Confira a extração do EuroSAT RGB."
    )

CLASSES = sorted(df_arquivos["classe"].unique())
classe_para_id = {nome: i for i, nome in enumerate(CLASSES)}
df_arquivos["label"] = df_arquivos["classe"].map(classe_para_id)

inventario = (df_arquivos.groupby("classe").size().rename("imagens").to_frame()
              .assign(proporcao=lambda x: x["imagens"] / x["imagens"].sum()))
display(inventario)
print(f"Total: {len(df_arquivos):,} | classes: {len(CLASSES)} | {CLASSES}")
assert len(CLASSES) == 10, f"Esperávamos 10 classes, encontramos {len(CLASSES)}."
""")

M(r"""
**Interpretação:** o inventário local deve totalizar 27.000 imagens. Diferenças pequenas ou grandes precisam ser explicadas antes do split; uma pasta ausente muda o problema.

**Conclusão da seção:** agora temos uma tabela explícita `arquivo → classe → label`, base de todas as etapas seguintes.
""")

M(r"""
## 6. Da imagem ao array e ao tensor

**Pergunta de aprendizagem:** o que de fato entra em uma rede neural?

Um JPEG é um arquivo comprimido. A PIL o decodifica; o NumPy organiza intensidades em `(H, W, C)`; PyTorch usa `(C, H, W)`. Em lote, a convenção é `(N, C, H, W)`.
""")

C(r"""
amostra = df_arquivos.sample(1, random_state=SEED).iloc[0]
with Image.open(amostra["caminho"]) as img_aberta:
    imagem_pil = img_aberta.convert("RGB").copy()

imagem_np = np.asarray(imagem_pil)
imagem_tensor = transforms.ToTensor()(imagem_pil)

print(f"Arquivo: {amostra['caminho_relativo']} | classe: {amostra['classe']}")
print(f"PIL: size={imagem_pil.size}, mode={imagem_pil.mode}, format decodificado em RGB")
resumo_tensor("NumPy (H, W, C)", imagem_np)
resumo_tensor("PyTorch (C, H, W)", imagem_tensor)
print("Recorte NumPy [0:3, 0:3, :]:\n", imagem_np[:3, :3, :])
""")

C(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].imshow(imagem_pil)
axes[0].set_title(f"Imagem decodificada — {amostra['classe']}")
axes[0].axis("off")

recorte = imagem_np[:12, :12]
axes[1].imshow(recorte)
axes[1].set_title("Ampliação dos primeiros 12 × 12 pixels")
axes[1].set_xlabel("Largura (W)")
axes[1].set_ylabel("Altura (H)")
plt.tight_layout()
""")

M(r"""
### Checkpoint 1 e 2 — resposta

- `H × W × C` significa altura, largura e canais; cada pixel RGB guarda três intensidades.
- `uint8 [0,255] → float32 [0,1]` ocorre no `ToTensor`.
- o arquivo não entra diretamente na rede: entram números reorganizados de `(H,W,C)` para `(C,H,W)`.

**Conclusão da seção:** a imagem visual e o tensor são duas representações do mesmo conteúdo, com convenções e escalas diferentes.
""")

M(r"""
## 7. Canais RGB, escala de cinza e visualização numérica

**Pergunta de aprendizagem:** que informação cada canal carrega e o que se perde ao reduzir para cinza?
""")

C(r"""
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
axes[0, 0].imshow(imagem_np)
axes[0, 0].set_title("RGB combinado")
axes[0, 0].axis("off")

cores = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
nomes = ["R", "G", "B"]
for i, (nome, cor) in enumerate(zip(nomes, cores)):
    canal = imagem_np[:, :, i]
    axes[0, i + 1].imshow(canal, cmap="gray", vmin=0, vmax=255)
    axes[0, i + 1].set_title(f"Canal {nome} em cinza")
    axes[0, i + 1].axis("off")
    isolada = np.zeros_like(imagem_np)
    isolada[:, :, i] = canal
    axes[1, i].imshow(isolada)
    axes[1, i].set_title(f"Somente {nome}")
    axes[1, i].axis("off")

cinza = np.asarray(imagem_pil.convert("L"))
im = axes[1, 3].imshow(cinza, cmap="gray", vmin=0, vmax=255)
axes[1, 3].set_title("Escala de cinza")
axes[1, 3].axis("off")
fig.colorbar(im, ax=axes[1, 3], fraction=0.046)
plt.tight_layout()
""")

C(r"""
plt.figure(figsize=(9, 4))
for i, (nome, cor) in enumerate(zip(nomes, ["red", "green", "blue"])):
    plt.hist(imagem_np[:, :, i].ravel(), bins=32, alpha=0.45, color=cor, label=nome)
plt.title("Histogramas de intensidade por canal")
plt.xlabel("Intensidade [0, 255]")
plt.ylabel("Quantidade de pixels")
plt.legend(title="Canal")
plt.tight_layout()

pontos = [(0, 0), (imagem_np.shape[0] // 2, imagem_np.shape[1] // 2), (-1, -1)]
pixels_selecionados = [
    {"linha": y, "coluna": x, "RGB": imagem_np[y, x].tolist()}
    for y, x in pontos
]
display(pd.DataFrame(pixels_selecionados))
""")

M(r"""
**Interpretação:** diferenças entre histogramas indicam como cada faixa visível participa da cena. A conversão para cinza combina canais e elimina distinções cromáticas potencialmente úteis. Isso ainda não representa bandas multiespectrais.

**Conclusão da seção:** RGB preserva três medições visíveis por pixel; cinza reduz informação para uma intensidade.
""")

M(r"""
## 8. Transformações geométricas, interpolação e normalização

**Pergunta de aprendizagem:** por que padronizar tamanho e escala, e como fazer isso sem confundir visualização com entrada do modelo?
""")

C(r"""
resize_nearest = transforms.Resize(
    (CFG.img_size, CFG.img_size),
    interpolation=InterpolationMode.NEAREST,
)
resize_bilinear = transforms.Resize(
    (CFG.img_size, CFG.img_size),
    interpolation=InterpolationMode.BILINEAR,
)

t_nearest = resize_nearest(imagem_pil)
t_bilinear = resize_bilinear(imagem_pil)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, img, titulo in zip(
    axes, [imagem_pil, t_nearest, t_bilinear],
    [f"Original {imagem_pil.size}", "Resize nearest", "Resize bilinear"],
):
    ax.imshow(img)
    ax.set_title(titulo)
    ax.set_xlabel("largura")
    ax.set_ylabel("altura")
plt.tight_layout()
""")

C(r"""
MEAN_IMAGENET = (0.485, 0.456, 0.406)
STD_IMAGENET = (0.229, 0.224, 0.225)
to_tensor = transforms.ToTensor()
normalizar = transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET)

x_01 = to_tensor(t_bilinear)
x_norm = normalizar(x_01)
resumo_tensor("Antes da normalização", x_01)
resumo_tensor("Depois da normalização", x_norm)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].imshow(x_01.permute(1, 2, 0))
axes[0].set_title("Tensor em [0,1]")
axes[1].imshow(desnormalizar(x_norm, MEAN_IMAGENET, STD_IMAGENET).permute(1, 2, 0))
axes[1].set_title("Desnormalizado apenas para exibir")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
""")

M(r"""
**Interpretação:** resize altera a grade espacial, não o rótulo. Nearest preserva blocos; bilinear suaviza a interpolação. Normalização pode produzir valores negativos e acima de 1 — isso é esperado. Desnormalizamos somente para plotar.

**Conclusão da seção:** modelos exigem dimensões previsíveis e a normalização esperada pelos pesos pré-treinados.
""")

M(r"""
## 9. Filtros clássicos e convolução passo a passo

**Pergunta de aprendizagem:** como um kernel local produz um mapa de características?

Para entrada de tamanho `H×W`, kernel `K`, padding `P` e stride `S`, a saída por eixo é `floor((H + 2P - K)/S) + 1`.
""")

C(r"""
def convolucao_2d_manual(entrada: np.ndarray, kernel: np.ndarray,
                         stride: int = 1, padding: int = 0) -> np.ndarray:
    '''Convolução 2D didática, sem rotação do kernel (correlação cruzada).'''
    entrada_pad = np.pad(entrada, padding, mode="constant")
    kh, kw = kernel.shape
    oh = (entrada_pad.shape[0] - kh) // stride + 1
    ow = (entrada_pad.shape[1] - kw) // stride + 1
    saida = np.zeros((oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            janela = entrada_pad[i * stride:i * stride + kh, j * stride:j * stride + kw]
            saida[i, j] = np.sum(janela * kernel)
    return saida


janela = cinza[:7, :7].astype(np.float32) / 255
sobel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
produto = janela[:3, :3] * sobel_vertical
print("Janela 3×3:\n", np.round(janela[:3, :3], 3))
print("Kernel:\n", sobel_vertical)
print("Produto elemento a elemento:\n", np.round(produto, 3))
print("Soma / primeira ativação:", produto.sum())
display(pd.DataFrame(convolucao_2d_manual(janela, sobel_vertical)))
""")

C(r"""
cinza_01 = cinza.astype(np.float32) / 255
kernels = {
    "Média": np.ones((3, 3), dtype=np.float32) / 9,
    "Sobel horizontal": np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32),
    "Sobel vertical": sobel_vertical,
    "Nitidez": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32),
}
fig, axes = plt.subplots(1, 5, figsize=(17, 4))
axes[0].imshow(cinza_01, cmap="gray")
axes[0].set_title("Entrada")
axes[0].axis("off")
for ax, (nome, kernel) in zip(axes[1:], kernels.items()):
    mapa = convolucao_2d_manual(cinza_01, kernel, padding=1)
    im = ax.imshow(mapa, cmap="gray")
    ax.set_title(nome)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
""")

C(r"""
x_gray = torch.from_numpy(cinza_01)[None, None]
conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
with torch.no_grad():
    conv.weight.copy_(torch.from_numpy(sobel_vertical)[None, None])
    y_conv = conv(x_gray)

print("Entrada NCHW:", tuple(x_gray.shape), "→ saída NCHW:", tuple(y_conv.shape))
print("Saídas manual e PyTorch próximas:", np.allclose(
    convolucao_2d_manual(cinza_01, sobel_vertical, padding=1), y_conv[0, 0].numpy(), atol=1e-5
))
""")

M(r"""
### Checkpoint 3 — resposta

- kernel define a vizinhança; stride define o salto; padding controla bordas e tamanho;
- um filtro gera um feature map; vários filtros geram vários canais de saída;
- filtros clássicos são fixos, enquanto `Conv2d` aprende os pesos por backpropagation.

**Conclusão da seção:** convolução transforma padrões locais em mapas que realçam propriedades úteis.
""")

M(r"""
## 10. Ativação, pooling, feature maps e data augmentation

**Pergunta de aprendizagem:** como a rede reduz a dimensão e ganha diversidade sem alterar o significado da classe?
""")

C(r"""
matriz_exemplo = [
    [-2.0, 1.0, 0.0, 3.0],
    [4.0, -1.0, 2.0, 0.0],
    [1.0, 5.0, -3.0, 2.0],
    [0.0, 2.0, 1.0, 4.0],
]
exemplo = torch.tensor(matriz_exemplo)[None, None]
relu = F.relu(exemplo)
pooled = F.max_pool2d(relu, kernel_size=2)
print("Entrada:", tuple(exemplo.shape), "\n", exemplo[0, 0])
print("Após ReLU:", tuple(relu.shape), "\n", relu[0, 0])
print("Após MaxPool 2×2:", tuple(pooled.shape), "\n", pooled[0, 0])
""")

C(r"""
transform_treino = transforms.Compose([
    transforms.RandomResizedCrop(
        CFG.img_size,
        scale=(0.85, 1.0),
        interpolation=InterpolationMode.BILINEAR,
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20, interpolation=InterpolationMode.BILINEAR),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET),
])
transform_avaliacao = transforms.Compose([
    transforms.Resize((CFG.img_size, CFG.img_size), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET),
])

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax in axes.flat:
    x_aug = transform_treino(imagem_pil)
    ax.imshow(desnormalizar(x_aug, MEAN_IMAGENET, STD_IMAGENET).permute(1, 2, 0))
    ax.set_title("Variação de treino")
    ax.axis("off")
plt.tight_layout()
""")

M(r"""
### Checkpoint 4 — resposta

ReLU remove ativações negativas; MaxPool reduz custo e resolução espacial. Augmentation aumenta diversidade, não cria informação nova. Treino recebe variações moderadas; validação/teste usam transformação determinística para medir sempre o mesmo problema. Transformações agressivas podem apagar sinais úteis.

**Conclusão da seção:** a pipeline de treino ensina invariâncias plausíveis; a de avaliação mantém a régua estável.
""")

M(r"""
## 11. Análise exploratória e qualidade dos dados

**Pergunta de aprendizagem:** há desbalanceamento, arquivos ilegíveis, formatos inesperados ou duplicatas exatas?
""")

C(r"""
fig, ax = plt.subplots(figsize=(10, 4))
inventario["imagens"].sort_values().plot.barh(ax=ax)
ax.set_title("Quantidade de imagens por classe")
ax.set_xlabel("imagens")
ax.set_ylabel("classe")
plt.tight_layout()

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax, classe in zip(axes.flat, CLASSES):
    linha = df_arquivos[df_arquivos["classe"].eq(classe)].sample(1, random_state=SEED).iloc[0]
    with Image.open(linha["caminho"]) as img:
        ax.imshow(img.convert("RGB"))
    ax.set_title(classe)
    ax.axis("off")
fig.suptitle("Um exemplo por classe", y=1.02)
plt.tight_layout()
""")

C(r"""
def inspecionar_arquivo(caminho: Path) -> dict:
    try:
        with Image.open(caminho) as img:
            img.verify()
        with Image.open(caminho) as img:
            return {"largura": img.width, "altura": img.height, "modo": img.mode, "erro": None}
    except Exception as exc:
        return {"largura": np.nan, "altura": np.nan, "modo": None, "erro": repr(exc)}


if MODO_RAPIDO:
    tamanho_amostra = min(1500, len(df_arquivos))
    amostra_qc = df_arquivos.sample(tamanho_amostra, random_state=SEED)
else:
    amostra_qc = df_arquivos

qc = pd.DataFrame([inspecionar_arquivo(p) for p in amostra_qc["caminho"]])
resumo_qualidade = (
    qc.groupby(["largura", "altura", "modo"], dropna=False)
    .size()
    .rename("arquivos")
    .reset_index()
)
display(resumo_qualidade)
print("Arquivos ilegíveis na verificação:", qc["erro"].notna().sum())
""")

C(r"""
def md5_arquivo(caminho: Path, bloco: int = 1 << 20) -> str:
    h = hashlib.md5()  # usado como impressão digital, não para segurança
    with caminho.open("rb") as f:
        while pedaco := f.read(bloco):
            h.update(pedaco)
    return h.hexdigest()


df_arquivos["md5"] = df_arquivos["caminho"].map(md5_arquivo)
duplicatas_exatas = df_arquivos.loc[df_arquivos["md5"].duplicated(keep=False)].sort_values("md5")
conflitos = duplicatas_exatas.groupby("md5")["classe"].nunique().gt(1)
if conflitos.any():
    raise ValueError(
        "Há imagens idênticas com rótulos diferentes. "
        "Revise esses arquivos antes de modelar."
    )
print(f"Cópias duplicadas encontradas: {len(duplicatas_exatas)}")
display(duplicatas_exatas[["caminho_relativo", "classe", "md5"]].head(20))
""")

M(r"""
**Interpretação:** `AnnualCrop`, `PermanentCrop`, `Pasture` e `HerbaceousVegetation` podem compartilhar texturas; `Highway` e `River` podem compartilhar estruturas lineares. Duplicatas exatas atravessando splits seriam vazamento e devem ser agrupadas ou removidas antes de continuar.

**Conclusão da seção:** qualidade não é só “o arquivo abre”; inclui distribuição, resolução, ambiguidade visual e independência entre amostras.
""")

M(r"""
## 12. Separação estratificada, Dataset e DataLoaders

**Pergunta de aprendizagem:** como garantir exatamente os mesmos splits em todos os experimentos?

Criamos o split uma vez, persistimos caminhos relativos e registramos um fingerprint. Augmentation não participa da divisão.
""")

C(r"""
if SPLIT_FILE.exists():
    df_splits = pd.read_csv(SPLIT_FILE)
    esperadas = {"caminho_relativo", "classe", "label", "split"}
    if not esperadas.issubset(df_splits.columns):
        raise ValueError(f"Split existente inválido: faltam {esperadas - set(df_splits.columns)}")
else:
    # Uma única cópia por conteúdo impede que arquivos idênticos atravessem splits.
    df_para_split = df_arquivos.drop_duplicates("md5", keep="first").copy()
    treino_val, teste = train_test_split(
        df_para_split, test_size=CFG.test_size, random_state=SEED, stratify=df_para_split["classe"]
    )
    fracao_val_relativa = CFG.val_size / (1 - CFG.test_size)
    treino, validacao = train_test_split(
        treino_val, test_size=fracao_val_relativa, random_state=SEED, stratify=treino_val["classe"]
    )
    partes = []
    for nome, parte in (("treino", treino), ("validacao", validacao), ("teste", teste)):
        partes.append(parte.assign(split=nome))
    colunas_split = ["caminho_relativo", "classe", "label", "split"]
    df_splits = pd.concat(partes, ignore_index=True)[colunas_split]
    df_splits.to_csv(SPLIT_FILE, index=False)

df_splits["caminho"] = df_splits["caminho_relativo"].map(lambda p: ROOT / p)
arquivos_existem = df_splits["caminho"].map(Path.exists).all()
assert arquivos_existem, "O arquivo de split aponta para imagens ausentes."
hash_por_caminho = df_arquivos.set_index("caminho_relativo")["md5"]
df_splits["md5"] = df_splits["caminho_relativo"].map(hash_por_caminho)
if df_splits["md5"].isna().any() or df_splits["md5"].duplicated().any():
    raise ValueError(
        "O split persistido contém caminhos desconhecidos ou conteúdo duplicado; "
        "revise-o e regenere."
    )
assert_splits_disjuntos(df_splits)
print("Fingerprint do split:", fingerprint_splits(df_splits))
display(pd.crosstab(df_splits["classe"], df_splits["split"], margins=True))
""")

C(r"""
class EuroSATPaths(Dataset):
    '''Dataset orientado pela tabela persistida de caminhos e splits.'''

    def __init__(
        self,
        tabela: pd.DataFrame,
        transformacao: Callable,
    ) -> None:
        self.tabela = tabela.reset_index(drop=True).copy()
        self.transformacao = transformacao

    def __len__(self) -> int:
        return len(self.tabela)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, int]:
        linha = self.tabela.iloc[indice]

        # A imagem é aberta somente quando solicitada, evitando manter
        # milhares de arquivos simultaneamente na memória.
        with Image.open(linha["caminho"]) as img:
            imagem = img.convert("RGB")

        return self.transformacao(imagem), int(linha["label"])


def tabelas_experimento(
    modo_rapido: bool = MODO_RAPIDO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nomes_splits = ("treino", "validacao", "teste")
    partes = [
        df_splits[df_splits["split"].eq(nome)].copy()
        for nome in nomes_splits
    ]

    if modo_rapido:
        partes = [subconjunto_rapido(parte) for parte in partes]

    return tuple(partes)


def criar_loaders(
    batch_size: int,
    modo_rapido: bool = MODO_RAPIDO,
    transform_train: Callable = transform_treino,
    transform_eval: Callable = transform_avaliacao,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    treino_df, val_df, teste_df = tabelas_experimento(modo_rapido)
    datasets = (
        EuroSATPaths(treino_df, transform_train),
        EuroSATPaths(val_df, transform_eval),
        EuroSATPaths(teste_df, transform_eval),
    )
    loaders = tuple(
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(indice == 0),
            num_workers=CFG.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_seed,
            generator=criar_gerador(SEED),
            persistent_workers=CFG.num_workers > 0,
        )
        for indice, dataset in enumerate(datasets)
    )
    return loaders


train_loader, val_loader, test_loader = criar_loaders(batch_size=32)
x_batch, y_batch = next(iter(train_loader))
print("Batch NCHW:", tuple(x_batch.shape), "| labels N:", tuple(y_batch.shape))
""")

M(r"""
### Checkpoint 5 — resposta

Todos os modelos leem `splits_eurosat_seed42.csv`; somente a transformação e o `batch_size` podem mudar. O teste não seleciona arquitetura, hiperparâmetros, época ou modelo. O fingerprint permite detectar alterações acidentais.

## O que aprendemos antes de modelar

O arquivo comprimido foi decodificado em pixels RGB, convertido de `uint8 HWC` para `float32 CHW`, redimensionado, normalizado e agrupado em lotes NCHW. Vimos filtros fixos, feature maps, ReLU, pooling e augmentation. Agora os modelos podem receber a mesma evidência, sem vazamento.
""")

M(r"""
# Parte II — Modelagem e benchmark

## 13. Baseline e protocolo experimental comum

**Pergunta de aprendizagem:** o que torna a comparação justa?

- split e semente idênticos;
- seleção por **F1 Macro de validação**;
- teste intocado até a versão final;
- métricas comuns: accuracy, precision/recall/F1 Macro, AUC OVR Macro e F1 por classe;
- custo comum: treino, inferência, parâmetros treináveis e tamanho em memória.

Uma baseline majoritária contextualiza o problema, mas não integra as quatro metodologias finais.
""")

C(r"""
def calcular_metricas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    resultado = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_proba is not None:
        resultado["auc_ovr_macro"] = roc_auc_score(
            y_true,
            y_proba,
            multi_class="ovr",
            average="macro",
        )
    else:
        resultado["auc_ovr_macro"] = np.nan

    f1_por_classe = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=range(len(CLASSES)),
        zero_division=0,
    )
    for nome, valor in zip(CLASSES, f1_por_classe):
        resultado[f"f1_{nome}"] = valor
    return resultado


@torch.no_grad()
def predizer(
    modelo: nn.Module,
    loader: DataLoader,
    device: torch.device = DEVICE,
):
    modelo.eval()
    verdade = []
    predicoes = []
    probabilidades = []
    inicio = time.perf_counter()

    for imagens, labels in loader:
        imagens = imagens.to(device, non_blocking=True)
        logits = logits_do_modelo(modelo, imagens)
        proba = logits.softmax(dim=1)

        verdade.append(labels.numpy())
        predicoes.append(proba.argmax(dim=1).cpu().numpy())
        probabilidades.append(proba.cpu().numpy())

    tempo = time.perf_counter() - inicio
    return (
        np.concatenate(verdade),
        np.concatenate(predicoes),
        np.concatenate(probabilidades),
        tempo,
    )


def plotar_avaliacao(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    titulo: str,
) -> None:
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(CLASSES)),
        normalize="true",
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )
    plt.title(f"{titulo} — matriz de confusão normalizada")
    plt.xlabel("Classe predita")
    plt.ylabel("Classe verdadeira")
    plt.tight_layout()


def treinar_epoca(
    modelo: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    clip_grad: float = 1.0,
    scheduler_por_batch=None,
) -> float:
    modelo.train()
    perdas = []

    for imagens, labels in loader:
        # 1) Move o mini-batch para o mesmo dispositivo do modelo.
        imagens = imagens.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # 2) Limpa gradientes antigos antes do novo forward pass.
        optimizer.zero_grad(set_to_none=True)

        # 3) Calcula logits, erro e gradientes desta iteração.
        logits = logits_do_modelo(modelo, imagens)
        loss = criterion(logits, labels)
        loss.backward()

        # 4) Limita gradientes extremos antes de atualizar os pesos.
        nn.utils.clip_grad_norm_(modelo.parameters(), clip_grad)
        optimizer.step()

        # Alguns schedulers, como o warmup do ViT, avançam por mini-batch.
        if scheduler_por_batch is not None:
            scheduler_por_batch.step()

        perdas.append(loss.item())

    return float(np.mean(perdas))


@torch.no_grad()
def validar_loss(
    modelo: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> float:
    modelo.eval()
    perdas = []
    for imagens, labels in loader:
        imagens = imagens.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = logits_do_modelo(modelo, imagens)
        perdas.append(criterion(logits, labels).item())
    return float(np.mean(perdas))


def treinar_com_early_stopping(
    modelo: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    patience: int,
    checkpoint: Path,
    scheduler=None,
    scheduler_por_batch=None,
) -> tuple[nn.Module, pd.DataFrame, float]:
    criterion = nn.CrossEntropyLoss()
    melhor_f1 = -np.inf
    sem_melhora = 0
    historico = []
    inicio = time.perf_counter()

    for epoca in range(1, epochs + 1):
        loss_treino = treinar_epoca(
            modelo,
            train_loader,
            criterion,
            optimizer,
            scheduler_por_batch=scheduler_por_batch,
        )
        loss_val = validar_loss(modelo, val_loader, criterion)
        y_val, p_val, _, _ = predizer(modelo, val_loader)
        f1_val = f1_score(y_val, p_val, average="macro", zero_division=0)

        # ReduceLROnPlateau observa a perda de validação uma vez por época.
        if scheduler is not None:
            scheduler.step(loss_val)

        historico.append(
            {
                "epoca": epoca,
                "loss_treino": loss_treino,
                "loss_val": loss_val,
                "f1_val": f1_val,
            }
        )
        print(
            f"época {epoca:02d} | "
            f"loss treino {loss_treino:.4f} | "
            f"loss val {loss_val:.4f} | "
            f"F1 val {f1_val:.4f}"
        )
        if f1_val > melhor_f1 + 1e-4:
            melhor_f1 = f1_val
            sem_melhora = 0
            torch.save(modelo.state_dict(), checkpoint)
        else:
            sem_melhora += 1
            if sem_melhora >= patience:
                print("Early stopping acionado.")
                break

    melhor_estado = torch.load(
        checkpoint,
        map_location=DEVICE,
        weights_only=True,
    )
    modelo.load_state_dict(melhor_estado)
    return modelo, pd.DataFrame(historico), time.perf_counter() - inicio
""")

C(r"""
_, val_df_base, test_df_base = tabelas_experimento(MODO_RAPIDO)
classe_majoritaria = df_splits.loc[df_splits["split"].eq("treino"), "label"].mode().iloc[0]
pred_majoritaria = np.full(len(val_df_base), classe_majoritaria)
print("Baseline majoritária — F1 Macro (validação):",
      f1_score(val_df_base["label"], pred_majoritaria, average="macro", zero_division=0))
""")

M(r"""
**Interpretação:** accuracy pode parecer aceitável mesmo ignorando classes; F1 Macro pune esse comportamento. AUC usa probabilidades e mede separação OVR, mas não substitui análise da matriz de confusão.

**Conclusão da seção:** o protocolo separa seleção (validação) de estimativa final (teste) e mede desempenho e custo.
""")

M(r"""
## 14. CNN desenvolvida do zero

**Pergunta de aprendizagem:** o que uma CNN consegue aprender sem pesos externos?

Pipeline: `Conv → BatchNorm → ReLU → MaxPool → Dropout → Fully Connected → logits`.
""")

C(r"""
class CNNDoZero(nn.Module):
    def __init__(self, num_classes: int = 10, filtros: int = 32,
                 dimensao_fc: int = 256, dropout: float = 0.35):
        super().__init__()
        canais = [3, filtros, filtros * 2, filtros * 4]
        blocos = []
        for entrada, saida in zip(canais[:-1], canais[1:]):
            blocos.extend([
                nn.Conv2d(entrada, saida, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(saida), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Dropout2d(dropout / 2),
            ])
        self.features = nn.Sequential(*blocos)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(canais[-1] * 4 * 4, dimensao_fc),
            nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(dimensao_fc, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


cnn_demo = CNNDoZero(num_classes=len(CLASSES)).to(DEVICE)
print(cnn_demo)
print("Parâmetros (total, treináveis):", contar_parametros(cnn_demo))
with torch.no_grad():
    print("Entrada", tuple(x_batch.shape), "→ logits", tuple(cnn_demo(x_batch.to(DEVICE)).shape))
""")

M(r"""
**Interpretação:** cada bloco aumenta canais (mais tipos de padrão) e reduz altura/largura (menor custo espacial). `AdaptiveAvgPool2d` impede que a camada densa dependa de uma conta manual frágil.

**Conclusão da seção:** esta arquitetura aprende todos os filtros apenas com o EuroSAT e será otimizada, sem comparar ativações, poolings ou otimizadores.
""")

M(r"""
## 15. Otimização da CNN com Optuna

**Pergunta de aprendizagem:** como escolher poucos hiperparâmetros sem tocar no teste?

Optuna ajusta somente `learning_rate`, `batch_size`, `filtros`, `dimensao_fc`, `dropout` e `weight_decay`. ReLU, MaxPool, BatchNorm, Adam, CrossEntropy e ReduceLROnPlateau permanecem fixos.
""")

C(r"""
def objetivo_cnn(trial):
    definir_semente()
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "filtros": trial.suggest_categorical("filtros", [24, 32, 48]),
        "dimensao_fc": trial.suggest_categorical("dimensao_fc", [128, 256, 384]),
        "dropout": trial.suggest_float("dropout", 0.20, 0.50),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }
    train_l, val_l, _ = criar_loaders(params["batch_size"])
    modelo = CNNDoZero(
        num_classes=len(CLASSES),
        filtros=params["filtros"],
        dimensao_fc=params["dimensao_fc"],
        dropout=params["dropout"],
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        modelo.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=2,
        factor=0.5,
    )
    checkpoint = ARTIFACT_DIR / f"optuna_trial_{trial.number}.pt"
    modelo, hist, _ = treinar_com_early_stopping(
        modelo,
        train_l,
        val_l,
        optimizer,
        epochs=8 if MODO_RAPIDO else 20,
        patience=3,
        checkpoint=checkpoint,
        scheduler=scheduler,
    )
    valor = float(hist["f1_val"].max())
    checkpoint.unlink(missing_ok=True)
    limpar_memoria()
    return valor


if EXECUTAR_TREINOS:
    exigir_pacote("optuna")
    import optuna

    estudo = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    estudo.optimize(objetivo_cnn, n_trials=5 if MODO_RAPIDO else CFG.n_trials)
    print("Melhores hiperparâmetros:", estudo.best_params)
else:
    print("Treino desativado. Defina EXECUTAR_TREINOS=True para iniciar o estudo Optuna.")
""", tags=["treino-pesado"])

C(r"""
if EXECUTAR_TREINOS:
    p = estudo.best_params
    train_cnn, val_cnn, test_cnn = criar_loaders(p["batch_size"])
    cnn_final = CNNDoZero(len(CLASSES), p["filtros"], p["dimensao_fc"], p["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(
        cnn_final.parameters(),
        lr=p["learning_rate"],
        weight_decay=p["weight_decay"],
    )
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=2, factor=0.5)
    cnn_final, hist_cnn, tempo_cnn = treinar_com_early_stopping(
        cnn_final, train_cnn, val_cnn, opt, CFG.epochs if not MODO_RAPIDO else 10,
        CFG.patience, ARTIFACT_DIR / "cnn_eurosat_best.pt", sch,
    )
    y_cnn, pred_cnn, proba_cnn, inf_cnn = predizer(cnn_final, test_cnn)
    met_cnn = calcular_metricas(y_cnn, pred_cnn, proba_cnn)
    registrar_resultado("CNN do zero + Optuna", met_cnn, tempo_cnn, inf_cnn, cnn_final)
    display(pd.DataFrame(hist_cnn).set_index("epoca"))
    plotar_avaliacao(y_cnn, pred_cnn, "CNN do zero + Optuna")
""", tags=["treino-pesado"])

M(r"""
**Como interpretar:** procure convergência conjunta das perdas e o maior F1 de validação, não o menor erro de treino. O checkpoint definitivo é salvo antes do teste.

**Conclusão da seção:** Optuna escolhe a configuração da CNN somente na validação; o teste estima seu desempenho final.
""")

M(r"""
## 16. Transfer Learning com ResNet50

**Pergunta de aprendizagem:** quanto ganhamos reutilizando filtros aprendidos na ImageNet?

Congelamos o backbone e substituímos `fc`. A opção `DESCONGELAR_LAYER4` permite fine-tuning parcial — nunca completo — quando houver GPU e evidência na validação.
""")

C(r"""
def construir_resnet50(num_classes: int, descongelar_layer4: bool = False) -> nn.Module:
    modelo = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in modelo.parameters():
        p.requires_grad = False
    if descongelar_layer4:
        for p in modelo.layer4.parameters():
            p.requires_grad = True
    entrada = modelo.fc.in_features
    modelo.fc = nn.Sequential(nn.Dropout(0.25), nn.Linear(entrada, num_classes))
    return modelo


DESCONGELAR_LAYER4 = False
if EXECUTAR_TREINOS:
    resnet50 = construir_resnet50(len(CLASSES), DESCONGELAR_LAYER4).to(DEVICE)
    print("Parâmetros (total, treináveis):", contar_parametros(resnet50))
else:
    print(
        "A ResNet50 será carregada somente com EXECUTAR_TREINOS=True, "
        "evitando download involuntário dos pesos."
    )
""")

C(r"""
if EXECUTAR_TREINOS:
    train_res, val_res, test_res = criar_loaders(32)
    parametros_treinaveis = filter(
        lambda parametro: parametro.requires_grad,
        resnet50.parameters(),
    )
    opt_res = torch.optim.AdamW(
        parametros_treinaveis,
        lr=3e-4,
        weight_decay=1e-4,
    )
    sch_res = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_res,
        mode="min",
        patience=2,
        factor=0.5,
    )
    resnet50, hist_res, tempo_res = treinar_com_early_stopping(
        resnet50, train_res, val_res, opt_res, 15 if not MODO_RAPIDO else 6, 4,
        ARTIFACT_DIR / "resnet50_eurosat_best.pt", sch_res,
    )
    y_res, pred_res, proba_res, inf_res = predizer(resnet50, test_res)
    met_res = calcular_metricas(y_res, pred_res, proba_res)
    registrar_resultado("ResNet50", met_res, tempo_res, inf_res, resnet50)
    plotar_avaliacao(y_res, pred_res, "ResNet50")
else:
    print("Treino da ResNet50 aguardando EXECUTAR_TREINOS=True.")
""", tags=["treino-pesado"])

M(r"""
**Como interpretar:** uma cabeça congelada tem baixo custo e reduz risco de sobreajuste; descongelar `layer4` aumenta capacidade e custo. A decisão deve vir da validação.

**Conclusão da seção:** transfer learning reaproveita uma representação visual geral e aprende apenas a fronteira necessária para as dez classes.
""")

M(r"""
## 17. Extração de embeddings com Vision Transformer

**Pergunta de aprendizagem:** como representar cada imagem por um vetor sem ajustar o ViT?

O mesmo backbone `google/vit-base-patch16-224-in21k` será usado aqui e no LoRA. A imagem vira patches 16×16, cada patch vira token e o token `[CLS]` resume a sequência após self-attention. O backbone permanece completamente congelado.
""")

C(r"""
def carregar_backbone_vit():
    exigir_pacote("transformers")
    from transformers import AutoImageProcessor, ViTModel
    processor = AutoImageProcessor.from_pretrained(CFG.vit_model_id)
    modelo = ViTModel.from_pretrained(CFG.vit_model_id, attn_implementation="eager").to(DEVICE)
    modelo.eval()
    for p in modelo.parameters():
        p.requires_grad = False
    tamanho = processor.size["height"] if isinstance(processor.size, dict) else int(processor.size)
    transform_vit = transforms.Compose([
        transforms.Resize((tamanho, tamanho), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(tuple(processor.image_mean), tuple(processor.image_std)),
    ])
    return modelo, processor, transform_vit


@torch.no_grad()
def extrair_embeddings_vit(modelo, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, float]:
    vetores, labels = [], []
    inicio = time.perf_counter()
    for imagens, y in loader:
        saida = modelo(pixel_values=imagens.to(DEVICE))
        vetores.append(saida.last_hidden_state[:, 0].cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(vetores), np.concatenate(labels), time.perf_counter() - inicio
""")

C(r"""
if EXECUTAR_TREINOS:
    vit_backbone, vit_processor, transform_vit = carregar_backbone_vit()
    train_vit, val_vit, test_vit = criar_loaders(
        batch_size=32,
        transform_train=transform_vit,
        transform_eval=transform_vit,
    )
    for nome, loader in (("train", train_vit), ("val", val_vit), ("test", test_vit)):
        emb, lab, tempo = extrair_embeddings_vit(vit_backbone, loader)
        np.save(DATA_DIR / f"vit_embeddings_{nome}.npy", emb)
        np.save(DATA_DIR / f"vit_labels_{nome}.npy", lab)
        print(nome, emb.shape, lab.shape, f"{tempo:.1f}s")
else:
    print(
        "Extração aguardando EXECUTAR_TREINOS=True; "
        "os seis arquivos .npy serão salvos em data/."
    )
""", tags=["treino-pesado"])

M(r"""
**Como interpretar:** cada linha do `.npy` corresponde, na mesma ordem, a uma imagem do split; cada coluna é uma dimensão latente. Embedding não é uma classe: é uma representação usada por outro classificador.

**Conclusão da seção:** o ViT congelado transforma imagens em vetores comparáveis sem aprender com os rótulos do EuroSAT.
""")

M(r"""
## 18. LightGBM sobre embeddings

**Pergunta de aprendizagem:** um classificador de árvores consegue separar classes usando apenas a representação congelada?

Testamos três configurações pequenas na validação. Não criamos novo split e não usamos o teste na seleção.
""")

C(r"""
CONFIGS_LGBM = [
    {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "n_estimators": 300,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    },
    {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 10,
        "n_estimators": 500,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    },
    {
        "learning_rate": 0.08,
        "num_leaves": 31,
        "max_depth": 8,
        "n_estimators": 250,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
    },
]

if EXECUTAR_TREINOS:
    exigir_pacote("lightgbm")
    from lightgbm import LGBMClassifier
    X_train = np.load(DATA_DIR / "vit_embeddings_train.npy")
    y_train = np.load(DATA_DIR / "vit_labels_train.npy")
    X_val = np.load(DATA_DIR / "vit_embeddings_val.npy")
    y_val = np.load(DATA_DIR / "vit_labels_val.npy")
    X_test = np.load(DATA_DIR / "vit_embeddings_test.npy")
    y_test = np.load(DATA_DIR / "vit_labels_test.npy")
    candidatos = []
    for params in CONFIGS_LGBM:
        inicio_treino = time.perf_counter()
        modelo = LGBMClassifier(objective="multiclass", num_class=len(CLASSES), random_state=SEED,
                                n_jobs=-1, verbosity=-1, **params).fit(X_train, y_train)
        tempo_treino = time.perf_counter() - inicio_treino
        f1_val = f1_score(y_val, modelo.predict(X_val), average="macro")
        candidatos.append((f1_val, params, modelo, tempo_treino))
    melhor_f1_val, melhor_cfg, lgbm_embeddings, tempo_lgbm = max(candidatos, key=lambda x: x[0])
    inicio = time.perf_counter()
    pred_lgbm = lgbm_embeddings.predict(X_test)
    proba_lgbm = lgbm_embeddings.predict_proba(X_test)
    inf_lgbm = time.perf_counter() - inicio
    met_lgbm = calcular_metricas(y_test, pred_lgbm, proba_lgbm)
    caminho_lgbm = ARTIFACT_DIR / "lightgbm_embeddings.pkl"
    with caminho_lgbm.open("wb") as arquivo:
        pickle.dump(lgbm_embeddings, arquivo)
    registrar_resultado("ViT embeddings + LightGBM", met_lgbm, tempo_lgbm, inf_lgbm)
    RESULTADOS["ViT embeddings + LightGBM"]["tamanho_mb"] = caminho_lgbm.stat().st_size / 1024**2
    print("Configuração selecionada na validação:", melhor_cfg, "| F1 val:", melhor_f1_val)
    plotar_avaliacao(y_test, pred_lgbm, "ViT embeddings + LightGBM")
else:
    print("LightGBM aguardando embeddings e EXECUTAR_TREINOS=True.")
""", tags=["treino-pesado"])

M(r"""
**Como interpretar:** se o LightGBM for competitivo, o embedding já tornou as classes separáveis; se ficar atrás do LoRA, adaptação do backbone provavelmente agregou sinal específico do domínio.

**Conclusão da seção:** embeddings separam representação e classificação, favorecendo reuso e baixo custo de treinamento supervisionado.
""")

M(r"""
## 19. Vision Transformer adaptado com LoRA

**Pergunta de aprendizagem:** como adaptar atenção sem atualizar todos os pesos?

Em self-attention, `Q` consulta, `K` indexa relevância e `V` carrega conteúdo. Multi-head attention aprende relações diferentes entre patches; a rede feed-forward transforma cada token. LoRA representa a atualização de certas projeções como matrizes de baixo posto. Usaremos `rank=8`, `alpha=16` e `dropout=0.10`, treinando LoRA e classificador.
""")

C(r"""
def construir_vit_lora():
    exigir_pacote("transformers")
    exigir_pacote("peft")
    from transformers import AutoImageProcessor, ViTForImageClassification
    from peft import LoraConfig, get_peft_model

    processor = AutoImageProcessor.from_pretrained(CFG.vit_model_id)
    base = ViTForImageClassification.from_pretrained(
        CFG.vit_model_id, num_labels=len(CLASSES),
        id2label={i: c for i, c in enumerate(CLASSES)},
        label2id=classe_para_id, ignore_mismatched_sizes=True,
        attn_implementation="eager",
    )
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.10, bias="none",
        target_modules=["query", "value"], modules_to_save=["classifier"],
    )
    modelo = get_peft_model(base, lora_cfg).to(DEVICE)
    tamanho = processor.size["height"] if isinstance(processor.size, dict) else int(processor.size)
    transform_vit = transforms.Compose([
        transforms.Resize((tamanho, tamanho), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            tuple(processor.image_mean),
            tuple(processor.image_std),
        ),
    ])
    transform_vit_eval = transforms.Compose([
        transforms.Resize((tamanho, tamanho), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(
            tuple(processor.image_mean),
            tuple(processor.image_std),
        ),
    ])
    return modelo, transform_vit, transform_vit_eval
""")

C(r"""
if EXECUTAR_TREINOS:
    vit_lora, transform_lora_train, transform_lora_eval = construir_vit_lora()
    train_lora, val_lora, test_lora = criar_loaders(
        16, transform_train=transform_lora_train, transform_eval=transform_lora_eval
    )
    total, treinaveis = contar_parametros(vit_lora)
    print(f"Treináveis: {treinaveis:,}/{total:,} ({100*treinaveis/total:.3f}%)")
    parametros_lora = filter(
        lambda parametro: parametro.requires_grad,
        vit_lora.parameters(),
    )
    opt_lora = torch.optim.AdamW(
        parametros_lora,
        lr=2e-4,
        weight_decay=1e-4,
    )

    # Warmup linear seguido de decaimento. O número de passos é calculado
    # com o DataLoader real para que a curva de learning rate fique correta.
    exigir_pacote("transformers")
    from transformers import get_linear_schedule_with_warmup
    passos = len(train_lora) * (8 if MODO_RAPIDO else 20)
    scheduler_passos = get_linear_schedule_with_warmup(opt_lora, int(0.1 * passos), passos)

    vit_lora, hist_lora, tempo_lora = treinar_com_early_stopping(
        vit_lora, train_lora, val_lora, opt_lora, 8 if MODO_RAPIDO else 20, 4,
        ARTIFACT_DIR / "vit_lora_eurosat_best.pt", scheduler_por_batch=scheduler_passos,
    )
    y_lora, pred_lora, proba_lora, inf_lora = predizer(vit_lora, test_lora)
    met_lora = calcular_metricas(y_lora, pred_lora, proba_lora)
    registrar_resultado("ViT + LoRA", met_lora, tempo_lora, inf_lora, vit_lora)
    vit_lora.save_pretrained(ARTIFACT_DIR / "vit_lora_adapter")
    plotar_avaliacao(y_lora, pred_lora, "ViT + LoRA")
else:
    print("Treino LoRA aguardando EXECUTAR_TREINOS=True.")
""", tags=["treino-pesado"])

M(r"""
**Como interpretar:** a fração de parâmetros treináveis deve ser muito menor que 100%. LoRA testa se pequenas atualizações nas projeções de atenção bastam para adaptar o ViT ao domínio.

### Checkpoint 6 — resposta

- CNN: representação aprendida do zero;
- ResNet50: representação convolucional transferida;
- embeddings: representação congelada + classificador separado;
- LoRA: representação Transformer adaptada com poucos parâmetros.

**Conclusão da seção:** LoRA ocupa o meio-termo entre backbone congelado e fine-tuning completo, que não faz parte deste benchmark.
""")

M(r"""
## 20. Interpretabilidade

**Pergunta de aprendizagem:** quais regiões e padrões sustentam as predições?

Para a CNN, inspecionamos filtros, feature maps e Grad-CAM. Para o ViT, inspecionamos atenção do `[CLS]` aos patches. Atenção é evidência qualitativa, não explicação causal.
""")

C(r"""
def visualizar_filtros_e_maps(
    modelo: CNNDoZero,
    imagem: torch.Tensor,
    n: int = 8,
) -> None:
    conv = next(m for m in modelo.features if isinstance(m, nn.Conv2d))
    filtros = conv.weight.detach().cpu()
    with torch.no_grad():
        maps = conv(imagem[None].to(DEVICE)).cpu()[0]
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(filtros[i].mean(0), cmap="coolwarm")
        axes[0, i].set_title(f"Filtro {i}")
        axes[1, i].imshow(maps[i], cmap="viridis")
        axes[1, i].set_title(f"Mapa {i}")
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    fig.suptitle("Primeira camada da CNN: pesos e ativações")
    plt.tight_layout()


def grad_cam(
    modelo: nn.Module,
    imagem: torch.Tensor,
    camada: nn.Module,
    classe: int | None = None,
) -> np.ndarray:
    ativacoes, gradientes = [], []
    h1 = camada.register_forward_hook(
        lambda _modulo, _entrada, saida: ativacoes.append(saida)
    )
    h2 = camada.register_full_backward_hook(
        lambda _modulo, _grad_entrada, grad_saida: gradientes.append(
            grad_saida[0]
        )
    )
    modelo.eval()
    modelo.zero_grad(set_to_none=True)
    logits = logits_do_modelo(modelo, imagem[None].to(DEVICE))
    alvo = int(logits.argmax(1).item()) if classe is None else classe
    logits[0, alvo].backward()
    pesos = gradientes[0][0].mean(dim=(1, 2), keepdim=True)
    mapa = F.relu((pesos * ativacoes[0][0]).sum(0))
    mapa = F.interpolate(
        mapa[None, None],
        size=imagem.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    mapa = mapa / (mapa.max() + 1e-8)
    h1.remove()
    h2.remove()
    return mapa.detach().cpu().numpy()
""")

C(r"""
@torch.no_grad()
def mapa_atencao_vit(backbone, imagem: torch.Tensor) -> np.ndarray:
    saida = backbone(pixel_values=imagem[None].to(DEVICE), output_attentions=True)
    atencao = saida.attentions[-1][0].mean(0)[0, 1:]
    lado = int(np.sqrt(atencao.numel()))
    mapa = atencao.reshape(1, 1, lado, lado)
    mapa = F.interpolate(mapa, size=imagem.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    mapa = mapa / (mapa.max() + 1e-8)
    return mapa.cpu().numpy()


if EXECUTAR_TREINOS and "cnn_final" in globals():
    imagem_interpretacao, _ = test_cnn.dataset[0]
    visualizar_filtros_e_maps(cnn_final, imagem_interpretacao)
    ultima_conv = [m for m in cnn_final.modules() if isinstance(m, nn.Conv2d)][-1]
    calor = grad_cam(cnn_final, imagem_interpretacao, ultima_conv)
    base = desnormalizar(imagem_interpretacao, MEAN_IMAGENET, STD_IMAGENET).permute(1, 2, 0)
    plt.figure(figsize=(6, 5))
    plt.imshow(base)
    plt.imshow(calor, cmap="jet", alpha=0.45)
    plt.title("CNN — Grad-CAM")
    plt.axis("off")
    plt.colorbar(label="relevância relativa")
    plt.show()

if EXECUTAR_TREINOS and "vit_backbone" in globals():
    imagem_vit, _ = test_vit.dataset[0]
    attn = mapa_atencao_vit(vit_backbone, imagem_vit)
    base = desnormalizar(
        imagem_vit,
        vit_processor.image_mean,
        vit_processor.image_std,
    ).permute(1, 2, 0)
    plt.figure(figsize=(6, 5))
    plt.imshow(base)
    plt.imshow(attn, cmap="magma", alpha=0.45)
    plt.title("ViT — atenção CLS → patches")
    plt.axis("off")
    plt.colorbar(label="atenção relativa")
    plt.show()
""")

M(r"""
**Como interpretar:** filtros iniciais costumam responder a bordas, cores e texturas; Grad-CAM deve cobrir regiões semanticamente plausíveis. Atenção do ViT pode ser mais distribuída. Mapas instáveis ou concentrados em artefatos sugerem investigação de viés.

**Conclusão da seção:** explicações qualitativas ajudam a auditar padrões, mas não validam sozinhas a decisão.
""")

M(r"""
## 21. Comparação consolidada e análise de erros

**Pergunta de aprendizagem:** qual abordagem equilibra desempenho e custo, e onde cada uma falha?
""")

C(r"""
if RESULTADOS:
    df_resultados = pd.DataFrame(RESULTADOS.values()).sort_values("f1_macro", ascending=False)
    colunas = ["modelo", "accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_ovr_macro",
               "treino_s", "inferencia_s", "parametros_treinaveis", "tamanho_mb"]
    display(df_resultados[colunas].style.format(precision=4))
    ax = df_resultados.plot.bar(
        x="modelo",
        y=["accuracy", "f1_macro", "auc_ovr_macro"],
        figsize=(11, 5),
    )
    ax.set_title("Benchmark EuroSAT — métricas no teste")
    ax.set_ylabel("métrica")
    ax.set_ylim(0, 1)
    ax.legend(title="Métrica")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    df_resultados.to_csv(REPORT_DIR / "benchmark_resultados.csv", index=False)
else:
    print("Ainda não há resultados. Execute os quatro experimentos para consolidar o benchmark.")
""")

C(r"""
def tabela_erros(y_true, y_pred, tabela_split: pd.DataFrame) -> pd.DataFrame:
    tabela = tabela_split.reset_index(drop=True).copy()
    if len(tabela) != len(y_true):
        raise ValueError("Predições e tabela de teste têm tamanhos diferentes.")
    tabela["verdade"] = [CLASSES[i] for i in y_true]
    tabela["predicao"] = [CLASSES[i] for i in y_pred]
    return tabela.loc[tabela["verdade"].ne(tabela["predicao"])]


if EXECUTAR_TREINOS and "y_cnn" in globals():
    erros_cnn = tabela_erros(y_cnn, pred_cnn, tabelas_experimento(MODO_RAPIDO)[2])
    display(pd.crosstab(erros_cnn["verdade"], erros_cnn["predicao"]).head(10))
""")

M(r"""
**Como interpretar:** compare primeiro F1 Macro; depois confirme se o ganho é consistente por classe e se justifica custo, memória e latência. Pares recorrentes na matriz de erros orientam coleta de dados e inspeção visual.

**Conclusão da seção:** “melhor” depende da fronteira entre desempenho, custo, interpretabilidade e simplicidade operacional.
""")

M(r"""
## 22. Trade-offs, conclusões e próximos passos

**Pergunta de aprendizagem:** quando escolher cada estratégia?

- **CNN do zero:** útil para aprender o mecanismo e quando não se pode usar pesos externos; tende a exigir mais dados/épocas.
- **ResNet50:** forte referência de custo-benefício e implantação madura.
- **ViT + LightGBM:** bom quando embeddings serão reutilizados, o treino supervisionado precisa ser barato ou modelos tabulares devem receber features visuais.
- **ViT + LoRA:** adequado quando a representação Transformer precisa de adaptação específica com orçamento menor que fine-tuning completo.

Após executar, substitua hipóteses por evidências: informe o melhor F1 Macro, as classes mais difíceis, a latência no hardware registrado e a proporção de parâmetros treináveis.

### Limitações

- EuroSAT RGB não contém todo o espectro do Sentinel-2;
- split aleatório estratificado não mede generalização geográfica/temporal;
- tempo depende do hardware, cache e tamanho do modo rápido;
- mapas de atenção e Grad-CAM são análises qualitativas;
- o benchmark não avalia calibração, OOD ou deriva.

### Próximos passos seguros

1. executar o modo rápido de ponta a ponta;
2. executar o conjunto completo e congelar versões/artefatos;
3. adicionar validação por região quando metadados geográficos estiverem disponíveis;
4. documentar resultados reais no README e em `reports/benchmark_resultados.csv`.

**Conclusão final:** o notebook mantém uma única régua experimental e compara quatro formas distintas de obter representações visuais, sem transformar a aula em uma busca indiscriminada de arquiteturas.
""")

M(r"""
## Checklist de execução

- [ ] Dependências opcionais instaladas no kernel
- [ ] Inventário com 27.000 imagens e 10 classes
- [ ] Verificação de arquivos e duplicatas revisada
- [ ] Fingerprint do split registrado
- [ ] Modo rápido executado sem erros
- [ ] CNN otimizada somente na validação
- [ ] ResNet50 avaliada no mesmo teste
- [ ] Embeddings salvos por split e LightGBM selecionado na validação
- [ ] LoRA treinou apenas adaptadores e classificador
- [ ] Tabela final contém métricas e custos dos quatro modelos
""")


# Cada célula de código recebe um objetivo explícito. Isso mantém o notebook
# legível mesmo quando uma célula é aberta isoladamente no JupyterLab/VS Code.
OBJETIVOS_CODIGO = [
    "Verificar se o kernel possui as dependências básicas do projeto.",
    "Importar bibliotecas e configurar a apresentação de tabelas e gráficos.",
    "Diagnosticar as dependências opcionais usadas nos experimentos avançados.",
    "Centralizar caminhos, semente, dispositivo e controles de execução.",
    "Definir utilitários compartilhados pelas etapas do notebook.",
    "Inventariar as imagens e criar o mapeamento entre classes e rótulos.",
    "Converter uma imagem real de PIL para NumPy e depois para PyTorch.",
    "Visualizar a imagem completa e ampliar uma pequena região de pixels.",
    "Separar os canais RGB e comparar a informação visual de cada canal.",
    "Comparar histogramas RGB e inspecionar valores de pixels específicos.",
    "Comparar os métodos nearest e bilinear durante o redimensionamento.",
    "Normalizar o tensor e desfazer a normalização somente para visualização.",
    "Implementar manualmente a convolução para observar cada operação.",
    "Aplicar filtros clássicos e comparar seus mapas de características.",
    "Reproduzir o filtro manual com uma camada Conv2d do PyTorch.",
    "Demonstrar separadamente os efeitos de ReLU e Max Pooling.",
    "Construir transformações distintas para treino e para avaliação.",
    "Examinar distribuição e exemplos representativos das dez classes.",
    "Verificar resolução, modo de cor e legibilidade dos arquivos.",
    "Detectar conteúdo duplicado antes da criação dos splits.",
    "Criar ou carregar splits estratificados, persistentes e sem vazamento.",
    "Implementar o Dataset e criar DataLoaders reprodutíveis.",
    "Definir métricas, inferência e o ciclo comum de treinamento.",
    "Calcular uma baseline majoritária usando somente o split de treino.",
    "Construir a CNN do zero e conferir as formas de entrada e saída.",
    "Definir o objetivo do Optuna e executar a busca quando autorizada.",
    "Treinar e avaliar a configuração final da CNN selecionada.",
    "Construir a ResNet50 com backbone congelado e nova cabeça.",
    "Treinar e avaliar a ResNet50 usando o protocolo comum.",
    "Carregar o ViT congelado e definir a extração de embeddings.",
    "Extrair e salvar embeddings dos três splits sem alterar sua ordem.",
    "Selecionar um LightGBM na validação e avaliá-lo uma vez no teste.",
    "Construir o ViT com adaptadores LoRA e classificador treinável.",
    "Treinar e avaliar o ViT com LoRA e warmup por batch.",
    "Visualizar filtros, feature maps e calcular Grad-CAM para a CNN.",
    "Extrair atenção do ViT e sobrepor os mapas às imagens originais.",
    "Consolidar desempenho e custo dos quatro modelos do benchmark.",
    "Organizar erros individuais para análise por pares de classes.",
]


DOCSTRINGS = {
    "Config": "Agrupa as configurações reproduzíveis do benchmark.",
    "worker_seed": "Define uma semente diferente e reproduzível para cada worker.",
    "criar_gerador": "Cria o gerador usado pelo DataLoader para embaralhamento reproduzível.",
    "desnormalizar": "Retorna um tensor à escala visual aproximada de zero a um.",
    "tamanho_modelo_mb": "Estima a memória ocupada por parâmetros e buffers do modelo.",
    "contar_parametros": "Conta parâmetros totais e parâmetros efetivamente treináveis.",
    "resumo_tensor": "Mostra forma, tipo e faixa de valores de um array ou tensor.",
    "logits_do_modelo": "Uniformiza a saída de modelos PyTorch e Hugging Face.",
    "subconjunto_rapido": "Seleciona uma amostra balanceada para validar o pipeline.",
    "registrar_resultado": "Adiciona métricas e custos ao registro consolidado.",
    "assert_splits_disjuntos": "Confirma que nenhum caminho aparece em dois splits.",
    "fingerprint_splits": "Calcula uma assinatura curta e reproduzível dos splits.",
    "limpar_memoria": "Libera o cache da GPU entre experimentos independentes.",
    "inspecionar_arquivo": "Valida a imagem e retorna suas propriedades básicas.",
    "md5_arquivo": "Calcula uma impressão digital para detectar cópias exatas.",
    "tabelas_experimento": "Retorna treino, validação e teste na ordem oficial.",
    "EuroSATPaths": "Lê imagens a partir da tabela persistida de caminhos.",
    "criar_loaders": "Cria os três DataLoaders com transformações apropriadas.",
    "calcular_metricas": "Calcula métricas multiclasse globais e por classe.",
    "predizer": "Executa inferência sem gradientes e mede o tempo total.",
    "plotar_avaliacao": "Exibe a matriz de confusão normalizada por classe real.",
    "treinar_epoca": "Executa uma época completa de atualização dos parâmetros.",
    "validar_loss": "Calcula a perda de validação sem atualizar o modelo.",
    "treinar_com_early_stopping": "Treina, seleciona pela validação e restaura o melhor checkpoint.",
    "CNNDoZero": "CNN didática construída manualmente para o EuroSAT RGB.",
    "objetivo_cnn": "Avalia uma configuração da CNN usando apenas a validação.",
    "construir_resnet50": "Adapta uma ResNet50 pré-treinada para as dez classes.",
    "carregar_backbone_vit": "Carrega o mesmo backbone ViT usado em embeddings e LoRA.",
    "extrair_embeddings_vit": "Extrai o token CLS sem calcular gradientes.",
    "construir_vit_lora": "Adiciona LoRA às projeções de atenção query e value.",
    "visualizar_filtros_e_maps": "Compara filtros aprendidos e ativações da primeira convolução.",
    "grad_cam": "Calcula um mapa Grad-CAM para uma classe e camada escolhidas.",
    "mapa_atencao_vit": "Transforma a atenção CLS-patches em um mapa espacial.",
    "tabela_erros": "Relaciona cada previsão incorreta ao respectivo arquivo.",
}


def adicionar_docstrings(codigo: str) -> str:
    """Insere docstrings didáticas em definições de nível superior."""
    arvore = ast.parse(codigo)
    linhas = codigo.splitlines(keepends=True)
    insercoes: list[tuple[int, str]] = []

    for no in arvore.body:
        if not isinstance(no, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if ast.get_docstring(no) is not None or no.name not in DOCSTRINGS:
            continue

        primeira_linha_do_corpo = no.body[0].lineno - 1
        indentacao = " " * no.body[0].col_offset
        docstring = f'{indentacao}"""{DOCSTRINGS[no.name]}"""\n'
        insercoes.append((primeira_linha_do_corpo, docstring))

    for indice, docstring in reversed(insercoes):
        linhas.insert(indice, docstring)

    return "".join(linhas)


celulas_codigo = [celula for celula in cells if celula["cell_type"] == "code"]
if len(celulas_codigo) != len(OBJETIVOS_CODIGO):
    raise RuntimeError("Atualize OBJETIVOS_CODIGO ao adicionar ou remover células de código.")

for celula, objetivo in zip(celulas_codigo, OBJETIVOS_CODIGO):
    codigo = adicionar_docstrings("".join(celula["source"]))
    cabecalho = (
        "# -----------------------------------------------------------------------------\n"
        f"# Objetivo desta célula: {objetivo}\n"
        "# -----------------------------------------------------------------------------\n"
    )
    celula["source"] = _source(cabecalho + codigo)


for indice, celula in enumerate(cells):
    celula["id"] = f"eurosat-{indice:03d}"


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Gerado: {OUTPUT} ({len(cells)} células)")
