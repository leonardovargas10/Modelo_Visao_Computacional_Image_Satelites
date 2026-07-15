"""Pipeline didático e generalista de Deep Learning em PyTorch.

Abrange dados tabulares, imagens, texto, sequências/séries temporais, sinais e
fine-tuning de Transformers. PyTorch, Torchvision e Transformers são opcionais:
o pacote principal continua importável quando algum backend não está instalado.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import json
import random
import warnings

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, TensorDataset
except ImportError:
    torch = nn = DataLoader = Dataset = TensorDataset = None

try:
    from torchvision import datasets, transforms
except (ImportError, RuntimeError):
    datasets = transforms = None

try:
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
except ImportError:
    AutoImageProcessor = AutoModelForImageClassification = None
    AutoModelForSequenceClassification = AutoTokenizer = None
    Trainer = TrainingArguments = None


def _requer_torch():
    if torch is None:
        raise ImportError(
            "PyTorch não está instalado. Instale a distribuição CPU/CUDA "
            "adequada seguindo https://pytorch.org/get-started/locally/."
        )


def _requer_torchvision():
    _requer_torch()
    if transforms is None:
        raise ImportError("Instale uma versão de torchvision compatível com PyTorch.")


def _requer_transformers():
    _requer_torch()
    if AutoTokenizer is None:
        raise ImportError("Instale transformers e accelerate para fine-tuning.")


@dataclass
class ConfigTreino:
    """Hiperparâmetros comuns aos diferentes tipos de rede."""
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str | None = "plateau"
    patience: int = 7
    min_delta: float = 0.0
    gradient_clip: float | None = 1.0
    mixed_precision: bool = True
    num_workers: int = 0
    seed: int = 42
    checkpoint_path: str | None = None


@dataclass
class ResultadoTreino:
    modelo: object
    historico: pd.DataFrame
    melhor_epoca: int
    melhor_loss: float
    device: str


if nn is not None:
    class MLP(nn.Module):
        """MLP para classificação ou regressão tabular."""
        def __init__(self, n_features, n_outputs=1, hidden_dims=(128, 64, 32),
                     dropout=0.2, batch_norm=True, activation="relu"):
            super().__init__()
            ativacoes = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
                         "leaky_relu": nn.LeakyReLU}
            if activation not in ativacoes:
                raise ValueError(f"activation deve ser uma de {list(ativacoes)}")
            camadas, entrada = [], n_features
            for dimensao in hidden_dims:
                camadas.append(nn.Linear(entrada, dimensao))
                if batch_norm:
                    camadas.append(nn.BatchNorm1d(dimensao))
                camadas.extend([ativacoes[activation](), nn.Dropout(dropout)])
                entrada = dimensao
            camadas.append(nn.Linear(entrada, n_outputs))
            self.network = nn.Sequential(*camadas)

        def forward(self, x):
            return self.network(x)


    class CNN2D(nn.Module):
        """CNN compacta para classificação de imagens de tamanhos variados."""
        def __init__(self, in_channels=3, n_classes=2, channels=(32, 64, 128),
                     dropout=0.3, activation="relu"):
            super().__init__()
            ativacao = nn.GELU if activation == "gelu" else nn.ReLU
            blocos, entrada = [], in_channels
            for saida in channels:
                blocos.extend([
                    nn.Conv2d(entrada, saida, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(saida), ativacao(),
                    nn.Conv2d(saida, saida, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(saida), ativacao(), nn.MaxPool2d(2),
                ])
                entrada = saida
            self.features = nn.Sequential(*blocos)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                            nn.Linear(entrada, n_classes))

        def forward(self, x):
            return self.classifier(self.pool(self.features(x)))


    class CNN1D(nn.Module):
        """CNN 1D para sinais, áudio já vetorizado e séries multivariadas."""
        def __init__(self, in_channels=1, n_outputs=1, channels=(32, 64, 128),
                     kernel_size=5, dropout=0.2):
            super().__init__()
            blocos, entrada = [], in_channels
            for saida in channels:
                blocos.extend([
                    nn.Conv1d(entrada, saida, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(saida), nn.ReLU(), nn.MaxPool1d(2),
                ])
                entrada = saida
            self.features = nn.Sequential(*blocos)
            self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                      nn.Dropout(dropout), nn.Linear(entrada, n_outputs))

        def forward(self, x):
            return self.head(self.features(x))


    class LSTM(nn.Module):
        """LSTM para classificação/regressão de sequências."""
        def __init__(self, input_size, hidden_size=128, n_outputs=1, num_layers=2,
                     dropout=0.2, bidirectional=False, pooling="last"):
            super().__init__()
            self.pooling = pooling
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
            dimensao = hidden_size * (2 if bidirectional else 1)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dimensao, n_outputs))

        def forward(self, x):
            saida, _ = self.lstm(x)
            representacao = saida.mean(dim=1) if self.pooling == "mean" else saida[:, -1]
            return self.head(representacao)
else:
    class _TorchAusente:
        def __init__(self, *args, **kwargs):
            _requer_torch()
    MLP = CNN2D = CNN1D = LSTM = _TorchAusente


class DeepLearning:
    """Fachada única para preparação, modelagem, treino e inferência."""

    @staticmethod
    def configurar_semente(seed=42, deterministic=True):
        _requer_torch()
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def obter_device(preferencia="auto"):
        _requer_torch()
        if preferencia != "auto":
            return torch.device(preferencia)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def criar_mlp(n_features, n_outputs=1, **kwargs):
        _requer_torch(); return MLP(n_features, n_outputs, **kwargs)

    @staticmethod
    def criar_cnn_imagem(in_channels=3, n_classes=2, **kwargs):
        _requer_torch(); return CNN2D(in_channels, n_classes, **kwargs)

    @staticmethod
    def criar_cnn_sinal(in_channels=1, n_outputs=1, **kwargs):
        _requer_torch(); return CNN1D(in_channels, n_outputs, **kwargs)

    @staticmethod
    def criar_lstm(input_size, n_outputs=1, **kwargs):
        _requer_torch(); return LSTM(input_size, n_outputs=n_outputs, **kwargs)

    @staticmethod
    def dataloader_tabular(x, y=None, batch_size=64, shuffle=True,
                           num_workers=0, dtype_x=None):
        _requer_torch()
        dtype_x = dtype_x or torch.float32
        x_tensor = torch.as_tensor(np.asarray(x), dtype=dtype_x)
        tensores = [x_tensor]
        if y is not None:
            y_array = np.asarray(y)
            dtype_y = torch.long if np.issubdtype(y_array.dtype, np.integer) else torch.float32
            tensores.append(torch.as_tensor(y_array, dtype=dtype_y))
        dataset = TensorDataset(*tensores)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available())

    @staticmethod
    def criar_janelas_sequenciais(dados, tamanho_janela, horizonte=1,
                                   colunas_features=None, coluna_target=None, stride=1):
        """Transforma série/DataFrame em X=(amostra, tempo, feature) e y."""
        frame = dados.copy() if isinstance(dados, pd.DataFrame) else pd.DataFrame(dados)
        features = colunas_features or list(frame.columns)
        target = coluna_target or features[0]
        xs, ys, indices = [], [], []
        limite = len(frame) - tamanho_janela - horizonte + 1
        for inicio in range(0, max(limite, 0), stride):
            fim = inicio + tamanho_janela
            xs.append(frame.iloc[inicio:fim][features].to_numpy(dtype=np.float32))
            ys.append(frame.iloc[fim + horizonte - 1][target])
            indices.append(frame.index[fim + horizonte - 1])
        return np.asarray(xs), np.asarray(ys), pd.Index(indices)

    @staticmethod
    def transformacoes_imagem(tamanho=(224, 224), treino=True, normalizacao="imagenet"):
        _requer_torchvision()
        medias, desvios = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        operacoes = [transforms.Resize(tamanho)]
        if treino:
            operacoes.extend([transforms.RandomHorizontalFlip(),
                              transforms.RandomRotation(10)])
        operacoes.append(transforms.ToTensor())
        if normalizacao == "imagenet":
            operacoes.append(transforms.Normalize(medias, desvios))
        return transforms.Compose(operacoes)

    @staticmethod
    def dataloaders_imagem(diretorio, batch_size=32, tamanho=(224, 224),
                           num_workers=0):
        """Lê pastas ``train/valid/test/<classe>`` com ImageFolder."""
        _requer_torchvision()
        raiz = Path(diretorio)
        loaders, datasets_criados = {}, {}
        for split in ("train", "valid", "test"):
            pasta = raiz / split
            if not pasta.exists():
                continue
            ds = datasets.ImageFolder(
                pasta, transform=DeepLearning.transformacoes_imagem(
                    tamanho, treino=split == "train"
                )
            )
            datasets_criados[split] = ds
            loaders[split] = DataLoader(
                ds, batch_size=batch_size, shuffle=split == "train",
                num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            )
        return loaders, datasets_criados

    @staticmethod
    def tokenizar_textos(textos, modelo_pre_treinado, max_length=256, **kwargs):
        _requer_transformers()
        tokenizer = AutoTokenizer.from_pretrained(modelo_pre_treinado)
        tokens = tokenizer(list(textos), truncation=True, padding=True,
                           max_length=max_length, return_tensors="pt", **kwargs)
        return tokenizer, tokens

    @staticmethod
    def criar_transformer_texto(modelo_pre_treinado, n_classes, id2label=None,
                                 label2id=None, **kwargs):
        _requer_transformers()
        return AutoModelForSequenceClassification.from_pretrained(
            modelo_pre_treinado, num_labels=n_classes, id2label=id2label,
            label2id=label2id, **kwargs,
        )

    @staticmethod
    def criar_vit(modelo_pre_treinado, n_classes, id2label=None,
                  label2id=None, **kwargs):
        _requer_transformers()
        processor = AutoImageProcessor.from_pretrained(modelo_pre_treinado)
        model = AutoModelForImageClassification.from_pretrained(
            modelo_pre_treinado, num_labels=n_classes, id2label=id2label,
            label2id=label2id, ignore_mismatched_sizes=True, **kwargs,
        )
        return processor, model

    @staticmethod
    def construir_loss(tarefa="classificacao_binaria", class_weights=None,
                       label_smoothing=0.0):
        _requer_torch()
        if tarefa == "regressao":
            return nn.MSELoss()
        if tarefa == "classificacao_binaria":
            peso = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
            return nn.BCEWithLogitsLoss(pos_weight=peso)
        if tarefa == "classificacao_multiclasse":
            pesos = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
            return nn.CrossEntropyLoss(weight=pesos, label_smoothing=label_smoothing)
        raise ValueError("tarefa inválida.")

    @staticmethod
    def construir_otimizador(modelo, nome="adamw", learning_rate=1e-3,
                             weight_decay=1e-4, **kwargs):
        _requer_torch()
        opcoes = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
                  "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}
        nome = nome.lower()
        if nome not in opcoes:
            raise ValueError(f"Otimizador deve ser um de {list(opcoes)}")
        if nome == "sgd":
            kwargs.setdefault("momentum", 0.9)
        return opcoes[nome](modelo.parameters(), lr=learning_rate,
                            weight_decay=weight_decay, **kwargs)

    @staticmethod
    def _ajusta_saida_target(saida, target, tarefa):
        if tarefa == "classificacao_multiclasse":
            return saida, target.long().ravel()
        return saida.ravel(), target.float().ravel()

    @staticmethod
    def _executar_epoca(modelo, loader, loss_fn, device, tarefa,
                        optimizer=None, gradient_clip=None, scaler=None):
        treinamento = optimizer is not None
        modelo.train() if treinamento else modelo.eval()
        perda_total, quantidade = 0.0, 0
        contexto = torch.enable_grad() if treinamento else torch.no_grad()
        with contexto:
            for lote in loader:
                x, y = lote[0].to(device), lote[1].to(device)
                if treinamento:
                    optimizer.zero_grad(set_to_none=True)
                autocast_ativo = scaler is not None and scaler.is_enabled()
                with torch.autocast(device_type=device.type, enabled=autocast_ativo):
                    saida = modelo(x)
                    saida, y_loss = DeepLearning._ajusta_saida_target(saida, y, tarefa)
                    loss = loss_fn(saida, y_loss)
                if treinamento:
                    if autocast_ativo:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        if gradient_clip:
                            nn.utils.clip_grad_norm_(modelo.parameters(), gradient_clip)
                        scaler.step(optimizer); scaler.update()
                    else:
                        loss.backward()
                        if gradient_clip:
                            nn.utils.clip_grad_norm_(modelo.parameters(), gradient_clip)
                        optimizer.step()
                perda_total += loss.item() * len(y)
                quantidade += len(y)
        return perda_total / max(quantidade, 1)

    @staticmethod
    def treinar(modelo, train_loader, valid_loader, tarefa="classificacao_binaria",
                config=None, loss_fn=None, device="auto"):
        """Loop completo: train/eval, scheduler, AMP, clipping e early stopping."""
        _requer_torch()
        config = config or ConfigTreino()
        DeepLearning.configurar_semente(config.seed)
        device = DeepLearning.obter_device(device)
        modelo = modelo.to(device)
        loss_fn = loss_fn or DeepLearning.construir_loss(tarefa)
        loss_fn = loss_fn.to(device) if hasattr(loss_fn, "to") else loss_fn
        optimizer = DeepLearning.construir_otimizador(
            modelo, config.optimizer, config.learning_rate, config.weight_decay
        )
        scheduler = None
        if config.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        elif config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")
        melhor_estado, melhor_loss, melhor_epoca, espera, linhas = None, np.inf, 0, 0, []
        for epoca in range(1, config.epochs + 1):
            loss_train = DeepLearning._executar_epoca(
                modelo, train_loader, loss_fn, device, tarefa, optimizer,
                config.gradient_clip, scaler,
            )
            loss_valid = DeepLearning._executar_epoca(
                modelo, valid_loader, loss_fn, device, tarefa
            )
            if scheduler is not None:
                scheduler.step(loss_valid) if config.scheduler == "plateau" else scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            linhas.append({"epoca": epoca, "loss_treino": loss_train,
                           "loss_validacao": loss_valid, "learning_rate": lr})
            if loss_valid < melhor_loss - config.min_delta:
                melhor_loss, melhor_epoca, espera = loss_valid, epoca, 0
                melhor_estado = copy.deepcopy(modelo.state_dict())
                if config.checkpoint_path:
                    DeepLearning.salvar_checkpoint(
                        config.checkpoint_path, modelo, optimizer, epoca,
                        metricas={"loss_validacao": loss_valid}, config=config,
                    )
            else:
                espera += 1
                if espera >= config.patience:
                    break
        if melhor_estado is not None:
            modelo.load_state_dict(melhor_estado)
        return ResultadoTreino(modelo, pd.DataFrame(linhas), melhor_epoca,
                               float(melhor_loss), str(device))

    @staticmethod
    def predizer(modelo, loader, tarefa="classificacao_binaria", device="auto"):
        _requer_torch()
        device = DeepLearning.obter_device(device); modelo = modelo.to(device); modelo.eval()
        saidas, targets = [], []
        with torch.no_grad():
            for lote in loader:
                x = lote[0].to(device); logits = modelo(x)
                if tarefa == "classificacao_binaria":
                    pred = torch.sigmoid(logits.ravel())
                elif tarefa == "classificacao_multiclasse":
                    pred = torch.softmax(logits, dim=1)
                else:
                    pred = logits.ravel()
                saidas.append(pred.cpu())
                if len(lote) > 1:
                    targets.append(lote[1].cpu())
        predicoes = torch.cat(saidas).numpy()
        y_true = torch.cat(targets).numpy() if targets else None
        return predicoes, y_true

    @staticmethod
    def fine_tuning_transformer(modelo, train_dataset, eval_dataset,
                                output_dir="./transformer_output", epochs=3,
                                batch_size=16, learning_rate=2e-5,
                                compute_metrics=None, data_collator=None, **kwargs):
        """Cria e executa Hugging Face Trainer para texto ou visão."""
        _requer_transformers()
        argumentos = TrainingArguments(
            output_dir=output_dir, num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate, weight_decay=kwargs.pop("weight_decay", 0.01),
            eval_strategy=kwargs.pop("eval_strategy", "epoch"),
            save_strategy=kwargs.pop("save_strategy", "epoch"),
            load_best_model_at_end=kwargs.pop("load_best_model_at_end", True),
            **kwargs,
        )
        trainer = Trainer(
            model=modelo, args=argumentos, train_dataset=train_dataset,
            eval_dataset=eval_dataset, compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
        trainer.train()
        return trainer

    @staticmethod
    def salvar_checkpoint(caminho, modelo, optimizer=None, epoca=None,
                          metricas=None, config=None):
        _requer_torch()
        caminho = Path(caminho); caminho.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model_state_dict": modelo.state_dict(), "epoca": epoca,
                   "metricas": metricas or {},
                   "config": asdict(config) if hasattr(config, "__dataclass_fields__") else config}
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(payload, caminho)
        return caminho

    @staticmethod
    def carregar_checkpoint(caminho, modelo, optimizer=None, device="cpu"):
        _requer_torch()
        checkpoint = torch.load(caminho, map_location=device, weights_only=False)
        modelo.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return modelo, optimizer, checkpoint

    @staticmethod
    def contar_parametros(modelo):
        _requer_torch()
        total = sum(p.numel() for p in modelo.parameters())
        treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
        return {"total": total, "treinaveis": treinaveis,
                "congelados": total - treinaveis}


__all__ = [
    "DeepLearning", "ConfigTreino", "ResultadoTreino",
    "MLP", "CNN2D", "CNN1D", "LSTM",
]
