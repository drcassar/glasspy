warning = """
To use glasspy.predict you need to have `pytorch` and `pytorch-lightning`
installed.  It is not a hard dependency of GlassPy to reduce the risk of
interfering with your local installation. For pytorch, see the installation
instructions at https://pytorch.org/get-started. For pytorch-lightning, see the
instalation instructions at https://www.pytorchlightning.ai.
"""

try:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    from torch.optim import SGD, Adam, AdamW
    import pytorch_lightning as pl

except ModuleNotFoundError:
    raise ModuleNotFoundError(warning)


import os
import pickle
from abc import ABC, abstractmethod
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple, Union, Any

import numpy as np
import numpy.ma as ma


def _gen_architecture(
    hparams: dict, reverse: bool = False, requires_grad: bool = True
) -> nn.Sequential:
    """Generates a dense network architecture for pytorch.

    Args:
      hparams:
        Dictionary with the hyperparemeters of the network. The possible
        parameters are:
          + "n_features": number of input features (required). Must be a
          positive integer.
          + "num_layers": number of hidden layers (defaults to 1). Must be a
          positive integer.
          + "layer_n_size": number of neurons in layer n (replace n for an
            integer starting at 1, defaults to 10). Must be a positive integer.
          + "layer_n_activation": activation function of layer n (replace n for
            an integer starting at 1, defaults to Tanh). Available values are
            ["Tanh", "Sigmoid", "ReLU", "LeakyReLU", "SELU", "GELU", "ELU",
            "PReLU", "SiLU", "Mish", "Softplus", "Linear"].
          + "layer_n_dropout": dropout of layer n (replace n for an integer
            starting at 1, defaults to False meaning no dropout). Any value
            between 0 and 1 (or False) is permitted.
          + "layer_n_batchnorm": `True` will use batch normalization in layer n,
            `False` will not use batch normalization in layer n (replace n for
            an integer starting at 1, defaults to False meaning no batch
            normalization).
      reverse:
        Reverses the network sequence. Use `True` if creating a decoder, `False`
        otherwise. Default value: False.
      requires_grad:
        `True` if the autograd should record operations on the network tensors,
        `False otherwise`. Default value: True.

    Returns:
      Pytorch Sequetial object.

    Raises:
      AssertionError:
        When the `hparams` dictionary does not have a "n_features" key.
      NotImplementedError:
        When the selected activation function is not one of the permited values.
    """

    assert "n_features" in hparams, "`n_features` is a required hparams key."

    layers = []
    input_dim = hparams["n_features"]

    for n in range(1, hparams.get("num_layers", 1) + 1):

        batchnorm = hparams.get(f"layer_{n}_batchnorm", False)
        bias = False if batchnorm else True
        activation_name = hparams.get(f"layer_{n}_activation", "Tanh")
        layer_size = int(hparams.get(f"layer_{n}_size", 10))
        dropout = hparams.get(f"layer_{n}_dropout", False)

        if not reverse:
            l = [nn.Linear(input_dim, layer_size, bias=bias)]

            if batchnorm and (activation_name != "SELU"):
                l.append(nn.BatchNorm1d(layer_size))

        else:
            l = [nn.Linear(layer_size, input_dim, bias=bias)]

            if batchnorm and (activation_name != "SELU"):
                l.append(nn.BatchNorm1d(input_dim))

        if dropout:
            if activation_name == "SELU":
                l.append(nn.AlphaDropout(dropout))
            else:
                l.append(nn.Dropout(dropout))

        if activation_name == "Tanh":
            l.append(nn.Tanh())
            nn.init.xavier_uniform_(l[0].weight)
        elif activation_name == "Sigmoid":
            l.append(nn.Sigmoid())
            nn.init.xavier_uniform_(l[0].weight)
        elif activation_name == "ReLU":
            l.append(nn.ReLU())
            nn.init.kaiming_uniform_(l[0].weight, nonlinearity="relu")
        elif activation_name == "LeakyReLU":
            l.append(nn.LeakyReLU())
            nn.init.kaiming_uniform_(l[0].weight, nonlinearity="leaky_relu")
        elif activation_name == "GELU":
            l.append(nn.GELU())
        elif activation_name == "SELU":
            l.append(nn.SELU())
        elif activation_name == "ELU":
            l.append(nn.ELU())
        elif activation_name == "PReLU":
            l.append(nn.PReLU())
        elif activation_name == "SiLU":
            l.append(nn.SiLU())
        elif activation_name == "Mish":
            l.append(nn.Mish())
        elif activation_name == "Softplus":
            l.append(nn.Softplus())
        elif activation_name == "Linear":
            l.append(nn.Linear())
        else:
            raise NotImplementedError(
                "Please add this activation to the model class."
            )

        layers.append(nn.Sequential(*l))
        input_dim = layer_size

    if reverse:
        layers.reverse()

    hidden_layers = nn.Sequential(*layers)

    if not requires_grad:
        for param in hidden_layers.parameters():
            param.requires_grad = False

    return hidden_layers


class Domain(NamedTuple):
    """Simple class to store chemical domain information."""

    element: Dict[str, float] = None
    compound: Dict[str, float] = None


class Predict(ABC):
    """Base class for GlassPy predictors."""

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def predict(self):
        pass

    @property
    @abstractmethod
    def domain(self):
        pass

    @abstractmethod
    def is_within_domain(self):
        pass

    @abstractmethod
    def get_training_dataset(self):
        pass

    @abstractmethod
    def get_test_dataset(self):
        pass

    @staticmethod
    def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes the mean squared error.

        Args:
          y_true:
            Array with the true values of y. Can be 1D or 2D.
          y_pred:
            Aray with the predicted values of y. Can be 1D or 2D.

        Returns:
          The mean squared error. Will be 1D if the input arrays are 2D.
          Will be a scalar otherwise.
        """

        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            RMSE = sum((y_true - y_pred) ** 2) / len(y_true)
            return RMSE
        else:
            y_true = ma.masked_invalid(y_true)
            RMSE = np.sum((y_true - y_pred) ** 2, axis=0) / y_true.count(axis=0)
            return RMSE.data

    @staticmethod
    def RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes the root mean squared error.

        Args:
          y_true:
            Array with the true values of y. Can be 1D or 2D.
          y_pred:
            Aray with the predicted values of y. Can be 1D or 2D.

        Returns:
          The root mean squared error. Will be 1D if the input arrays are 2D.
          Will be a scalar otherwise.
        """

        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            RMSE = sqrt(sum((y_true - y_pred) ** 2) / len(y_true))
            return RMSE
        else:
            y_true = ma.masked_invalid(y_true)
            RMSE = np.sqrt(
                np.sum((y_true - y_pred) ** 2, axis=0) / y_true.count(axis=0)
            )
            return RMSE.data

    @staticmethod
    def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes the mean absolute error.

        Args:
          y_true:
            Array with the true values of y. Can be 1D or 2D.
          y_pred:
            Aray with the predicted values of y. Can be 1D or 2D.

        Returns:
          The mean absolute error. Will be 1D if the input arrays are 2D.
          Will be a scalar otherwise.
        """

        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            MAE = np.sum(np.abs(y_true - y_pred)) / len(y_true)
            return MAE
        else:
            y_true = ma.masked_invalid(y_true)
            MAE = np.sum(np.abs(y_true - y_pred), axis=0) / y_true.count(axis=0)
            return MAE.data

    @staticmethod
    def MedAE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes the median absolute error.

        Args:
          y_true:
            Array with the true values of y. Can be 1D or 2D.
          y_pred:
            Aray with the predicted values of y. Can be 1D or 2D.

        Returns:
          The median absolute error. Will be 1D if the input arrays are 2D.
          Will be a scalar otherwise.
        """

        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            MedAE = np.median(np.abs(y_true - y_pred))
            return MedAE
        else:
            y_true = ma.masked_invalid(y_true)
            MedAE = ma.median(np.abs(y_true - y_pred), axis=0)
            return MedAE.data

    @staticmethod
    def PercAE(y_true: np.ndarray, y_pred: np.ndarray, q=75) -> float:
        """Computes the percentile absolute error.

        Args:
          y_true:
            Array with the true values of y. Can be 1D or 2D.
          y_pred:
            Aray with the predicted values of y. Can be 1D or 2D.
          q:
            Percentile to compute.

        Returns:
          The percentile absolute error. Will be 1D if the input arrays are 2D.
          Will be a scalar otherwise.
        """

        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            PercAE = np.percentile(np.abs(y_true - y_pred), q)
            return PercAE
        else:
            PercAE = np.nanpercentile(np.abs(y_true - y_pred), q, axis=0)
            return PercAE

    @staticmethod
    def RD(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes the relative deviation.

        Args:
          y_true:
            1D array with the true values of y.
          y_pred:
            1D array with the predicted values of y.

        Returns:
          The relative deviation.
        """

        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            RD = (100 / len(y_true)) * sum(abs(y_true - y_pred) / y_true)
            return RD
        else:
            y_true = ma.masked_invalid(y_true)
            RD = (100 / y_true.count(axis=0)) * np.sum(
                np.abs(y_true - y_pred) / y_true, axis=0
            )
            return RD.data

    @staticmethod
    def RRMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes the relative root mean squared error.

        Args:
          y_true:
            1D array with the true values of y.
          y_pred:
            1D array with the predicted values of y.

        Returns:
          The relative root mean squared error.
        """

        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            y_mean = sum(y_true) / len(y_true)
            RRMSE = sqrt(
                sum((y_true - y_pred) ** 2) / sum((y_true - y_mean) ** 2)
            )
            return RRMSE
        else:
            y_true = ma.masked_invalid(y_true)
            y_mean = np.sum(y_true, axis=0) / y_true.count(axis=0)
            RRMSE = np.sqrt(
                np.sum((y_true - y_pred) ** 2, axis=0)
                / np.sum((y_true - y_mean) ** 2, axis=0)
            )
            return RRMSE.data

    @staticmethod
    def R2(
        y_true: np.ndarray, y_pred: np.ndarray, one_param: bool = True
    ) -> float:
        """Computes the coefficient of determination.

        Args:
          y_true:
            1D array with the true values of y.
          y_pred:
            1D array with the predicted values of y.
          one_param:
            Determines the relationship between y_true and y_pred. If ´True´
            then it is a relationship with one parameter (y_true = y_pred * c_0
            + error). If ´False´ then it is a relationship with two parameters
            (y_true = y_pred * c_0 + c_1 + error). In most of regression
            problems, the first case is desired.

        Returns:
          The coefficient of determination.
        """

        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            nominator = sum((y_true - y_pred) ** 2)
            if one_param:
                denominator = sum(y_true**2)
            else:
                denominator = sum((y_true - np.mean(y_true)) ** 2)
            R2 = 1 - nominator / denominator
            return R2
        else:
            y_true = ma.masked_invalid(y_true)
            nominator = np.sum((y_true - y_pred) ** 2, axis=0)
            if one_param:
                denominator = np.sum(y_true**2, axis=0)
            else:
                denominator = np.sum(
                    (y_true - np.mean(y_true, axis=0)) ** 2, axis=0
                )
            R2 = 1 - nominator / denominator
            return R2.data


class MLP(pl.LightningModule, Predict):
    """Base class for creating Multilayer Perceptrons.

    Args:
      hparams:
        Dictionary with the hyperparemeters of the network. The possible
        parameters are:
        + "n_features": number of input features (required). Must be a positive
          integer.
        + "num_layers": number of hidden layers (defaults to 1). Must be a
          positive integer.
        + "layer_n_size": number of neurons in layer n (replace n for an integer
          starting at 1, defaults to 10). Must be a positive integer.
        + "layer_n_activation": activation function of layer n (replace n for an
          integer starting at 1, defaults to Tanh). Available values are
          ["Tanh", "Sigmoid", "ReLU", "LeakyReLU", "SELU", "GELU", "ELU",
          "PReLU", "SiLU", "Mish", "Softplus", "Linear"].
        + "layer_n_dropout": dropout of layer n (replace n for an integer
          starting at 1, defaults to False meaning no dropout). Any value
          between 0 and 1 (or False) is permitted.
        + "layer_n_batchnorm": `True` will use batch normalization in layer n,
          `False` will not use batch normalization in layer n (replace n for an
          integer starting at 1, defaults to False meaning no batch
          normalization).
        + "loss": loss function to use for the backpropagation algorithm
          (defaults to `mse`). Use `mse` for mean squared error loss (L2) or
          `huber` for a smooth L1 loss.
        + "optimizer": optimizer algorithm to use (defaults `SGD`). Use `SGD`
          for stochastic gradient descend, `Adam` for Adam, or `AdamW` for
          weighted Adam.
        + "lr": optimizer learning rate (defaults to 1e-4 if optimizer is `SGD`
          or 1e-3 if optimizer is `Adam` or `AdamW`).
        + "momentum": momentum to use when optmizer is `SGD` (defaults to 0).
        + "optimizer_Adam_eps": eps to use for Adam or AdamW optimizers
          (defaults to 1e-8).

    Raises:
      NotImplementedError:
        When the selected hyperparameters is not one of the permited values.
    """

    learning_curve_train = []
    learning_curve_val = []

    def __init__(self, hparams: Dict[str, Any]):
        super().__init__()

        self.hidden_layers = _gen_architecture(hparams, reverse=False)

        if hparams.get("loss", "mse") == "mse":
            self.loss_fun = F.mse_loss
        elif hparams["loss"] == "huber":
            self.loss_fun = F.smooth_l1_loss
        else:
            raise NotImplementedError(
                "Please add this loss function to the model class."
            )

    @property
    def domain(self) -> Domain:
        # TODO
        raise NotImplementedError("GlassPy error: not implemented.")

    def is_within_domain(self):
        # TODO
        raise NotImplementedError("GlassPy error: not implemented.")

    def get_training_dataset(self):
        # TODO
        raise NotImplementedError("GlassPy error: not implemented.")

    def get_test_dataset(self):
        # TODO
        raise NotImplementedError("GlassPy error: not implemented.")

    def distance_from_training(self):
        # TODO
        raise NotImplementedError("GlassPy error: not implemented.")

    def configure_optimizers(self):
        if "optimizer" not in self.hparams:
            optimizer = SGD(self.parameters(), lr=1e-4)

        elif self.hparams["optimizer"] == "SGD":
            optimizer = SGD(
                self.parameters(),
                lr=self.hparams.get("lr", 1e-4),
                momentum=self.hparams.get("momentum", 0),
            )

        elif self.hparams["optimizer"] == "Adam":
            optimizer = Adam(
                self.parameters(),
                lr=self.hparams.get("lr", 1e-3),
                eps=self.hparams.get("optimizer_Adam_eps", 1e-08),
            )

        elif self.hparams["optimizer"] == "AdamW":
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.get("lr", 1e-3),
                eps=self.hparams.get("optimizer_Adam_eps", 1e-08),
            )

        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fun(self(x), y)
        return {
            "loss": loss,
        }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.learning_curve_train.append(float(avg_loss))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fun(self(x), y)
        return {
            "val_loss_step": loss,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss_step"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        self.learning_curve_val.append(float(avg_loss))

    def save_training(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (
            self.state_dict(),
            self.learning_curve_train,
            self.learning_curve_val,
            self.hparams,
        )
        pickle.dump(data, open(path, "wb"))

    def load_training(self, path):
        state_dict, learning_train, learning_val, hparams = pickle.load(
            open(path, "rb")
        )
        self.load_state_dict(state_dict)
        self.learning_curve_train = learning_train
        self.learning_curve_val = learning_val

        return hparams


class MTL(MLP):
    """Base class for creating Multi-task Learning NN.

    Args:
      hparams:
        Dictionary with the hyperparemeters of the network. The possible
        parameters are:
        + "n_features": number of input features (required). Must be a positive
          integer.
        + "num_layers": number of hidden layers (defaults to 1). Must be a
          positive integer.
        + "layer_n_size": number of neurons in layer n (replace n for an integer
          starting at 1, defaults to 10). Must be a positive integer.
        + "layer_n_activation": activation function of layer n (replace n for an
          integer starting at 1, defaults to Tanh). Available values are
          ["Tanh", "Sigmoid", "ReLU", "LeakyReLU", "SELU", "GELU", "ELU",
          "PReLU", "SiLU", "Mish", "Softplus", "Linear"].
        + "layer_n_dropout": dropout of layer n (replace n for an integer
          starting at 1, defaults to False meaning no dropout). Any value
          between 0 and 1 (or False) is permitted.
        + "layer_n_batchnorm": `True` will use batch normalization in layer n,
          `False` will not use batch normalization in layer n (replace n for an
          integer starting at 1, defaults to False meaning no batch
          normalization).
        + "loss": loss function to use for the backpropagation algorithm
          (defaults to `mse`). Use `mse` for mean squared error loss (L2) or
          `huber` for a smooth L1 loss.
        + "optimizer": optimizer algorithm to use (defaults `SGD`). Use `SGD`
          for stochastic gradient descend, `Adam` for Adam, or `AdamW` for
          weighted Adam.
        + "lr": optimizer learning rate (defaults to 1e-4 if optimizer is `SGD`
          or 1e-3 if optimizer is `Adam` or `AdamW`).
        + "momentum": momentum to use when optmizer is `SGD` (defaults to 0).
        + "optimizer_Adam_eps": eps to use for Adam or AdamW optimizers
          (defaults to 1e-8).

    Raises:
      NotImplementedError:
        When the selected hyperparameters is not one of the permited values.
    """

    def __init__(self, hparams: Dict[str, Any]):
        super().__init__(hparams)

        self.n_outputs = hparams["n_targets"]
        self.loss_weights = nn.Parameter(
            torch.ones(self.n_outputs, requires_grad=True)
        )

    def _compute_loss(self, yhat, y):
        """Computes the loss of multi-task learning with missing values.

        Reference:
          Liebel, L., and Körner, M. (2018). Auxiliary Tasks in Multi-task
          Learning (arXiv).
        """

        not_nan = y.isnan().logical_not()
        good_cols = not_nan.any(dim=0)
        y = y[:, good_cols]
        yhat = yhat[:, good_cols]
        weights = self.loss_weights[good_cols]
        not_nan = not_nan[:, good_cols]

        loss = torch.sum(
            torch.stack(
                [
                    self.loss_fun(yhat[not_nan[:, i], i], y[not_nan[:, i], i])
                    * 0.5
                    / (weights[i] ** 2)
                    + torch.log(1 + weights[i] ** 2)
                    for i in range(len(weights))
                ]
            )
        )

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self._compute_loss(self(x), y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self._compute_loss(self(x), y)
        return {"val_loss_step": loss}


class AE(pl.LightningModule, Predict):
    """Base class for creating Autoencoders.

    Args:
      hparams:
        Dictionary with the hyperparemeters of the network. The possible
        parameters are:
        + "n_features": number of input features (required). Must be a positive
          integer. Will be the same as the number of output features.
        + "num_layers": number of encoder hidden layers (defaults to 1). Must be
          a positive integer. *NOTE*: The decoder will have the same number of
          hidden layers.
        + "layer_n_size": number of neurons in layer n of the encoder (replace n
          for an integer starting at 1, defaults to 10). Must be a positive
          integer. *NOTE*: The decoder architecture will be the same as the
          encoder, but mirrored.
        + "layer_n_activation": activation function of layer n of the encoder
          (replace n for an integer starting at 1, defaults to Tanh). Available
          values are ["Tanh", "Sigmoid", "ReLU", "LeakyReLU", "SELU", "GELU",
          "ELU", "PReLU", "SiLU", "Mish", "Softplus", "Linear"].
        + "layer_n_dropout": dropout of layer n of the encoder (replace n for an
          integer starting at 1, defaults to False meaning no dropout). Any
          value between 0 and 1 (or False) is permitted.
        + "layer_n_batchnorm": `True` will use batch normalization in layer n of
          the encoder, `False` will not use batch normalization in layer n
          (replace n for an integer starting at 1, defaults to False meaning no
          batch normalization).
        + "loss": loss function to use for the backpropagation algorithm
          (defaults to `mse`). Use `mse` for mean squared error loss (L2) or
          `huber` for a smooth L1 loss.
        + "optimizer": optimizer algorithm to use (defaults `SGD`). Use `SGD`
          for stochastic gradient descend, `Adam` for Adam, or `AdamW` for
          weighted Adam.
        + "lr": optimizer learning rate (defaults to 1e-4 if optimizer is `SGD`
          or 1e-3 if optimizer is `Adam` or `AdamW`).
        + "momentum": momentum to use when optmizer is `SGD` (defaults to 0).
        + "optimizer_Adam_eps": eps to use for Adam or AdamW optimizers
          (defaults to 1e-8).

    Raises:
      NotImplementedError:
        When the selected hyperparameters is not one of the permited values.
    """

    learning_curve_train = []
    learning_curve_val = []

    def __init__(self, hparams: Dict[str, Any]):
        super().__init__()

        self.encoder = _gen_architecture(hparams, reverse=False)
        self.decoder = _gen_architecture(hparams, reverse=True)

        if hparams.get("loss", "mse") == "mse":
            self.loss_fun = F.mse_loss
        elif hparams["loss"] == "huber":
            self.loss_fun = F.smooth_l1_loss
        else:
            raise NotImplementedError(
                "Please add this loss function to the model class."
            )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def predict(self, x):
        pass

    @property
    def domain(self) -> Domain:
        # TODO
        raise NotImplementedError("GlassPy error: not implemented.")

    def is_within_domain():
        # TODO
        raise NotImplementedError("GlassPy error: not implemented.")

    def get_training_dataset(self):
        # TODO
        raise NotImplementedError("GlassPy error: not implemented.")

    def get_validation_dataset(self):
        # TODO
        raise NotImplementedError("GlassPy error: not implemented.")

    def get_test_dataset(self):
        # TODO
        raise NotImplementedError("GlassPy error: not implemented.")

    def distance_from_training(self):
        # TODO
        raise NotImplementedError("GlassPy error: not implemented.")

    def configure_optimizers(self):
        if "optimizer" not in self.hparams:
            optimizer = SGD(self.parameters(), lr=1e-4)

        elif self.hparams["optimizer"] == "SGD":
            optimizer = SGD(
                self.parameters(),
                lr=self.hparams.get("lr", 1e-4),
                momentum=self.hparams.get("momentum", 0),
            )

        elif self.hparams["optimizer"] == "Adam":
            optimizer = Adam(
                self.parameters(),
                lr=self.hparams.get("lr", 1e-3),
                eps=self.hparams.get("optimizer_Adam_eps", 1e-08),
            )

        elif self.hparams["optimizer"] == "AdamW":
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.get("lr", 1e-3),
                eps=self.hparams.get("optimizer_Adam_eps", 1e-08),
            )

        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.loss_fun(batch, self(batch))
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.learning_curve_train.append(float(avg_loss))

    def validation_step(self, batch, batch_idx):
        loss = self.loss_fun(batch, self(batch))
        return {"val_loss_step": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss_step"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        self.learning_curve_val.append(float(avg_loss))

    def save_training(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (
            self.state_dict(),
            self.learning_curve_train,
            self.learning_curve_val,
            self.hparams,
        )
        pickle.dump(data, open(path, "wb"))

    def load_training(self, path):
        state_dict, learning_train, learning_val, hparams = pickle.load(
            open(path, "rb")
        )
        self.load_state_dict(state_dict)
        self.learning_curve_train = learning_train
        self.learning_curve_val = learning_val
        self.hparams = hparams
