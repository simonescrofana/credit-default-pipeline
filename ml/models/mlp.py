"""Define the MLP model family as a scikit-learn-compatible wrapper around PyTorch.

Provide an Estimator Protocol (fit/predict_proba) implementation that
encapsulates the PyTorch training loop, so ``ml.training.trainer`` can treat
the MLP exactly like the baseline and XGBoost, with no changes to its
orchestration logic.

"""

import logging

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

RANDOM_STATE = 202607

DEFAULT_HIDDEN_LAYERS = (64, 32, 16, 8)
DEFAULT_DROPOUT = 0.3
DEFAULT_EPOCHS = 25
DEFAULT_BATCH_SIZE = 1024
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-3


class _MLPNetwork(nn.Module):
    """A feed-forward network with a funnel architecture and dropout.

    Outputs raw logits (no final sigmoid), intended to be used with
    ``nn.BCEWithLogitsLoss`` for numerical stability.

    """

    def __init__(
        self, n_features: int, hidden_layers: tuple[int, ...], dropout: float
    ) -> None:
        """Initialize the multi-layer perceptron architecture.

        Args:
            n_features (int): Number of input features.
            hidden_layers (tuple[int, ...]): Sequence of hidden layer dimensions.
            dropout (float): Dropout probability applied after each linear layer.

        """
        super().__init__()

        layers = []
        input_dim = n_features
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass and return 1D unnormalized logits.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch_size, n_features)``.

        Returns:
            torch.Tensor: Unnormalized logit values of shape ``(batch_size,)``.

        """
        return self.network(x).squeeze(-1)


class MLPClassifier:
    """A scikit-learn-compatible wrapper around a PyTorch MLP for binary classification.

    Implements ``fit``/``predict_proba`` so it can be used as a drop-in
    Estimator by ``ml.training.trainer``, without any trainer-side changes
    for the PyTorch-specific training loop.

    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = DEFAULT_HIDDEN_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        pos_weight: float = 1.0,
    ) -> None:
        """Initialize an MLPClassifier estimator.

        Args:
            hidden_layers (tuple[int, ...], optional): Architecture of hidden layers
                where each element defines the number of units in that layer.
                Defaults to ``DEFAULT_HIDDEN_LAYERS``.
            dropout (float, optional): Dropout probability applied after each
                hidden layer. Defaults to ``DEFAULT_DROPOUT``.
            epochs (int, optional): Number of full training passes over the dataset.
                Defaults to ``DEFAULT_EPOCHS``.
            batch_size (int, optional): Mini-batch size for AdamW optimization.
                Defaults to ``DEFAULT_BATCH_SIZE``.
            learning_rate (float, optional): Learning rate for the AdamW optimizer.
                Defaults to ``DEFAULT_LEARNING_RATE``.
            weight_decay (float, optional): L2 penalty factor for parameter updates.
                Defaults to ``DEFAULT_WEIGHT_DECAY``.
            pos_weight (float, optional): Weight scaling applied to the positive
                class loss in ``BCEWithLogitsLoss`` to handle class imbalance.
                Defaults to ``1.0``.

        """
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network: _MLPNetwork | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "MLPClassifier":
        """Train the MLP on the given features and binary target.

        Logs the training loss per epoch via mlflow.log_metric(..., step=epoch),
        and — if validation data is provided — also logs the validation loss per
        epoch, so both curves can be inspected in the MLflow UI or reconstructed
        in plots.ipynb to guide manual early stopping.

        Args:
            X (pd.DataFrame): The training feature matrix.
            y (pd.Series): The training binary target labels.
            X_val (pd.DataFrame | None, optional): Validation feature matrix,
                used only for per-epoch monitoring, never for training the
                weights. Defaults to None.
            y_val (pd.Series | None, optional): Validation target labels.
                Defaults to None.

        Returns:
            MLPClassifier: self, fitted.

        """
        torch.manual_seed(RANDOM_STATE)

        n_features = X.shape[1]
        self.network = _MLPNetwork(n_features, self.hidden_layers, self.dropout).to(
            self.device
        )

        X_tensor = torch.tensor(X.to_numpy(dtype="float32"), dtype=torch.float32)
        y_tensor = torch.tensor(y.to_numpy(dtype="float32"), dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        has_validation = X_val is not None and y_val is not None
        if has_validation:
            X_val_tensor = torch.tensor(
                X_val.to_numpy(dtype="float32"), dtype=torch.float32
            ).to(self.device)
            y_val_tensor = torch.tensor(
                y_val.to_numpy(dtype="float32"), dtype=torch.float32
            ).to(self.device)

        pos_weight_tensor = torch.tensor(self.pos_weight, dtype=torch.float32).to(
            self.device
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        logger.info(
            "Starting MLP training: %d epochs, batch_size=%d, "
            "pos_weight=%.4f, validation=%s",
            self.epochs,
            self.batch_size,
            self.pos_weight,
            has_validation,
        )

        self.network.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.network(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item() * X_batch.size(0)

            epoch_loss /= len(dataset)

            val_loss = None
            if has_validation:
                self.network.eval()
                with torch.no_grad():
                    val_logits = self.network(X_val_tensor)
                    val_loss = criterion(val_logits, y_val_tensor).item()

            if mlflow.active_run() is not None:
                mlflow.log_metric("train_loss", epoch_loss, step=epoch)
                if val_loss is not None:
                    mlflow.log_metric("val_loss", val_loss, step=epoch)

            if epoch % 10 == 0 or epoch == self.epochs:
                if val_loss is not None:
                    logger.info(
                        "Epoch %d/%d: train_loss=%.4f, val_loss=%.4f",
                        epoch,
                        self.epochs,
                        epoch_loss,
                        val_loss,
                    )
                else:
                    logger.info(
                        "Epoch %d/%d: train_loss=%.4f", epoch, self.epochs, epoch_loss
                    )

        logger.info("MLP training completed.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given features.

        Args:
            X (pd.DataFrame): The feature matrix to score.

        Returns:
            np.ndarray: An array of shape (n_samples, 2) with the
                probability of the negative and positive class, matching
                scikit-learn's ``predict_proba`` convention.

        """
        if self.network is None:
            raise RuntimeError(
                "MLPClassifier must be fitted before calling predict_proba."
            )

        self.network.eval()
        X_tensor = torch.tensor(X.to_numpy(dtype="float32"), dtype=torch.float32).to(
            self.device
        )

        with torch.no_grad():
            logits = self.network(X_tensor)
            positive_proba = torch.sigmoid(logits).cpu().numpy()

        negative_proba = 1.0 - positive_proba
        return np.column_stack([negative_proba, positive_proba])


def build_mlp_model(
    hidden_layers: tuple[int, ...] = DEFAULT_HIDDEN_LAYERS,
    dropout: float = DEFAULT_DROPOUT,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    pos_weight: float = 1.0,
) -> MLPClassifier:
    """Build a fresh, unfitted MLPClassifier instance.

    Mirrors ``ml.models.baseline.build_baseline_model``'s role as a factory
    function passed to ``ml.training.trainer``, ensuring a new, independent
    network is created for every fold instead of reusing mutable state.

    Args:
        hidden_layers (tuple[int, ...], optional): The sizes of the hidden
            layers, in order. Defaults to ``DEFAULT_HIDDEN_LAYERS``.
        dropout (float, optional): The dropout probability applied after
            each hidden layer. Defaults to ``DEFAULT_DROPOUT``.
        epochs (int, optional): The number of training epochs. Defaults to
            ``DEFAULT_EPOCHS``.
        batch_size (int, optional): The mini-batch size. Defaults to
            ``DEFAULT_BATCH_SIZE``.
        learning_rate (float, optional): The AdamW learning rate. Defaults
            to ``DEFAULT_LEARNING_RATE``.
        weight_decay (float, optional): The AdamW weight decay (L2
            regularization). Defaults to ``DEFAULT_WEIGHT_DECAY``.
        pos_weight (float, optional): The positive class weight for
            ``BCEWithLogitsLoss``, addressing class imbalance. Defaults to
            ``1.0`` (no reweighting).

    Returns:
        MLPClassifier: A fresh, unfitted model instance.

    """
    return MLPClassifier(
        hidden_layers=hidden_layers,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        pos_weight=pos_weight,
    )
