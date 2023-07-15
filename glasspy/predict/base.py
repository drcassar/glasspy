"""Module with base classes for building predictive models."""

import os
import pickle
from abc import ABC, abstractmethod
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple, Union, Any

import numpy as np
import numpy.ma as ma
import joblib
import torch
import torch.nn as nn
import pandas as pd
from torch.nn import functional as F
from torch.optim import SGD, Adam, AdamW
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from scipy.optimize import least_squares
from scipy.stats import theilslopes
from torch.nn import functional as F


from glasspy.viscosity.equilibrium_log import myega_alt
from glasspy.chemistry import physchem_featurizer, CompositionLike
from glasspy.data import SciGlass

_BASEMODELPATH = Path(os.path.dirname(__file__)) / "models"

_VISCOSITY_COLUMNS_FOR_REGRESSION = [
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",
    "T6",
    "T7",
    "T8",
    "T9",
    "T10",
    "T11",
    "T12",
    "Viscosity773K",
    "Viscosity873K",
    "Viscosity973K",
    "Viscosity1073K",
    "Viscosity1173K",
    "Viscosity1273K",
    "Viscosity1373K",
    "Viscosity1473K",
    "Viscosity1573K",
    "Viscosity1673K",
    "Viscosity1773K",
    "Viscosity1873K",
    "Viscosity2073K",
    "Viscosity2273K",
    "Tg",
    "TLittletons",
]

GLASSNET_TARGETS = [
    "T0",
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",
    "T6",
    "T7",
    "T8",
    "T9",
    "T10",
    "T11",
    "T12",
    "Viscosity773K",
    "Viscosity873K",
    "Viscosity973K",
    "Viscosity1073K",
    "Viscosity1173K",
    "Viscosity1273K",
    "Viscosity1373K",
    "Viscosity1473K",
    "Viscosity1573K",
    "Viscosity1673K",
    "Viscosity1773K",
    "Viscosity1873K",
    "Viscosity2073K",
    "Viscosity2273K",
    "Viscosity2473K",
    "Tg",
    "Tmelt",
    "Tliquidus",
    "TLittletons",
    "TAnnealing",
    "Tstrain",
    "Tsoft",
    "TdilatometricSoftening",
    "AbbeNum",
    "RefractiveIndex",
    "RefractiveIndexLow",
    "RefractiveIndexHigh",
    "MeanDispersion",
    "Permittivity",
    "TangentOfLossAngle",
    "TresistivityIs1MOhm.m",
    "Resistivity273K",
    "Resistivity373K",
    "Resistivity423K",
    "Resistivity573K",
    "Resistivity1073K",
    "Resistivity1273K",
    "Resistivity1473K",
    "Resistivity1673K",
    "YoungModulus",
    "ShearModulus",
    "Microhardness",
    "PoissonRatio",
    "Density293K",
    "Density1073K",
    "Density1273K",
    "Density1473K",
    "Density1673K",
    "ThermalConductivity",
    "ThermalShockRes",
    "CTEbelowTg",
    "CTE328K",
    "CTE373K",
    "CTE433K",
    "CTE483K",
    "CTE623K",
    "Cp293K",
    "Cp473K",
    "Cp673K",
    "Cp1073K",
    "Cp1273K",
    "Cp1473K",
    "Cp1673K",
    "TMaxGrowthVelocity",
    "MaxGrowthVelocity",
    "CrystallizationPeak",
    "CrystallizationOnset",
    "SurfaceTensionAboveTg",
    "SurfaceTension1173K",
    "SurfaceTension1473K",
    "SurfaceTension1573K",
    "SurfaceTension1673K",
]


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


def _myega_residuals(x, T, y):
    return myega_alt(T, *x) - y


def _load_data_glassnet():
    """Returns the data used to train GlassNet.

    Returns:
        DataFrame with the all the data used to train GlassNet."""

    remove_dupe_decimals = 3

    # fmt: off
    removed_compounds = [
        "", "Al2O3+Fe2O3", "MoO3+WO3", "CaO+MgO", "FeO+Fe2O3", "Li2O+Na2O+K2O",
        "Na2O+K2O", "F2O-1", "FemOn", "HF+H2O", "R2O", "R2O3", "R2O3", "RO",
        "RmOn",
    ]

    removed_elements = [
        "Ac", "Am", "Ar", "At", "Bk", "Cf", "Cm", "Es", "Fm", "Fr", "He", "Kr",
        "Ne", "Np", "Pa", "Pm", "Po", "Pu", "Ra", "Rn", "Th", "U", "Xe",
    ]
    # fmt: on

    treatment = {
        "AbbeNum": {
            "max": 115,
        },
        "Cp293K": {
            "max": 2000,
        },
        "Cp473K": {
            "max": 2000,
        },
        "Cp673K": {
            "max": 3000,
        },
        "Cp1073K": {
            "min": 500,
            "max": 2500,
        },
        "Cp1273K": {
            "min": 500,
            "max": 3000,
        },
        "Cp1473K": {
            "min": 500,
            "max": 3000,
        },
        "Cp1673K": {
            "min": 500,
            "max": 2250,
        },
        "CTE328K": {
            "min": 10**-6.5,
            "log": True,
        },
        "CTE373K": {
            "min": 10**-6.5,
            "log": True,
        },
        "CTE433K": {
            "min": 10**-8,
            "log": True,
        },
        "CTE483K": {
            "min": 10**-7,
            "log": True,
        },
        "CTEbelowTg": {
            "min": 0,
            "log": True,
        },
        "Density293K": {
            "min": 1,
            "max": 10,
        },
        "CTE623K": {
            "log": True,
        },
        "MaxGrowthVelocity": {
            "min": 1e-10,
            "log": True,
        },
        "MeanDispersion": {
            "log": True,
        },
        "Microhardness": {
            "max": 15,
        },
        "Permittivity": {
            "max": 50,
        },
        "PoissonRatio": {
            "min": 0,
            "max": 1,
        },
        "RefractiveIndex": {
            "max": 4,
        },
        "RefractiveIndexHigh": {
            "min": 1.7,
            "max": 3.5,
        },
        "Resistivity273K": {
            "log": True,
            "max": 1e40,
        },
        "Resistivity373K": {
            "log": True,
            "max": 1e28,
        },
        "Resistivity423K": {
            "log": True,
        },
        "Resistivity573K": {
            "log": True,
        },
        "Resistivity1073K": {
            "max": 10**4,
            "log": True,
        },
        "Resistivity1273K": {
            "max": 10**5,
            "log": True,
        },
        "Resistivity1473K": {
            "log": True,
        },
        "Resistivity1673K": {
            "log": True,
        },
        "SurfaceTension1473K": {
            "max": 0.5,
        },
        "SurfaceTension1573K": {
            "max": 0.7,
        },
        "SurfaceTension1673K": {
            "max": 0.7,
        },
        "SurfaceTensionAboveTg": {
            "max": 0.8,
        },
        "Viscosity773K": {
            "log": True,
        },
        "Viscosity873K": {
            "log": True,
        },
        "Viscosity973K": {
            "log": True,
        },
        "Viscosity1073K": {
            "log": True,
        },
        "Viscosity1173K": {
            "log": True,
        },
        "Viscosity1273K": {
            "log": True,
        },
        "Viscosity1373K": {
            "log": True,
        },
        "Viscosity1473K": {
            "log": True,
        },
        "Viscosity1573K": {
            "log": True,
        },
        "Viscosity1673K": {
            "log": True,
        },
        "Viscosity1773K": {
            "max": 10**10,
            "log": True,
        },
        "Viscosity1873K": {
            "max": 10**10,
            "log": True,
        },
        "Viscosity2073K": {
            "max": 10**8,
            "log": True,
        },
        "Viscosity2273K": {
            "max": 10**8,
            "log": True,
        },
        "Viscosity2473K": {
            "log": True,
        },
        "T3": {
            "max": 2350,
        },
        "T4": {
            "max": 2000,
        },
        "TangentOfLossAngle": {
            "min": 1e-4,
            "max": 0.16,
            "log": True,
        },
        "ThermalConductivity": {
            "max": 6,
        },
        "TresistivityIs1MOhm.m": {
            "max": 2000,
        },
        "YoungModulus": {
            "max": 175,
        },
        "Tsoft": {
            "max": 1600,
        },
    }

    min_cols = [
        col for col in treatment if treatment[col].get("min", None) is not None
    ]

    max_cols = [
        col for col in treatment if treatment[col].get("max", None) is not None
    ]

    log_cols = [
        ("property", col)
        for col in treatment
        if treatment[col].get("log", False)
    ]

    propconf = {"keep": GLASSNET_TARGETS}

    compconf = {
        "acceptable_sum_deviation": 1,
        "final_sum": 1,
        "return_weight": False,
        "dropline": removed_compounds,
        "drop_compound_with_element": removed_elements,
    }

    sg = SciGlass(False, propconf, compconf)

    for col in min_cols:
        logic = sg.data["property"][col] < treatment[col]["min"]
        sg.data.loc[logic, ("property", col)] = np.nan

    for col in max_cols:
        logic = sg.data["property"][col] > treatment[col]["max"]
        sg.data.loc[logic, ("property", col)] = np.nan

    sg.data[log_cols] = sg.data[log_cols].apply(np.log10, axis=1)

    sg.elements_from_compounds(final_sum=1, compounds_in_weight=False)

    sg.remove_duplicate_composition(
        scope="elements",
        decimals=remove_dupe_decimals,
        aggregator="median",
    )

    return sg.data


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
            MSE = sum((y_true - y_pred) ** 2) / len(y_true)
            return MSE
        else:
            y_true = ma.masked_invalid(y_true)
            MSE = np.sum((y_true - y_pred) ** 2, axis=0) / y_true.count(axis=0)
            return MSE.data

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
            MAE = np.sum(np.abs(y_true - y_pred), axis=0) / y_true.count(
                axis=0
            )
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

        # for lightning
        self.training_step_outputs = []
        self.validation_step_outputs = []

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
        self.training_step_outputs.append(loss)
        return {
            "loss": loss,
        }

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", avg_loss)
        self.learning_curve_train.append(float(avg_loss))
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fun(self(x), y)
        self.validation_step_outputs.append(loss)
        return {
            "val_loss_step": loss,
        }

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log("val_loss", avg_loss)
        self.learning_curve_val.append(float(avg_loss))
        self.validation_step_outputs.clear()  # free memory

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


class _BaseViscNet(MLP):
    """Base class for creating ViscNet-like models.

    References:
      [1] D.R. Cassar, ViscNet: Neural network for predicting the fragility
        index and the temperature-dependency of viscosity, Acta Materialia. 206
        (2021) 116602. https://doi.org/10.1016/j.actamat.2020.116602.
        https://arxiv.org/abs/2007.03719
    """

    def __init__(self, parameters_range, hparams={}, x_mean=0, x_std=1):
        super().__init__(hparams)

        self.hparams = hparams
        self.x_mean = x_mean
        self.x_std = x_mean
        self.parameters_range = parameters_range

        input_dim = int(
            self.hparams.get(
                f'layer_{self.hparams.get("num_layers",1)}_size', 10
            )
        )

        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, len(self.parameters_range)),
            nn.Sigmoid(),
        )

    @abstractmethod
    def log_viscosity_fun(self) -> torch.tensor:
        pass

    @abstractmethod
    def featurizer(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
    ) -> np.ndarray:
        pass

    def viscosity_parameters_from_tensor(
        self,
        feature_tensor,
        return_tensor=False,
    ):
        """Predicts the viscosity parameters from a feature tensor.

        Consider using other methods for predicting the viscosity parameters.
        """

        xf = self.hidden_layers((feature_tensor - self.x_mean) / self.x_std)
        xf = self.output_layer(xf)

        parameters = {}

        for i, (p_name, p_range) in enumerate(self.parameters_range.items()):
            # Scaling the viscosity parameters to be within the parameter range
            parameters[p_name] = torch.add(
                torch.ones(xf.shape[0]).mul(p_range[0]),
                xf[:, i],
                alpha=p_range[1] - p_range[0],
            )

        if not return_tensor:
            parameters = {k: v.detach().numpy() for k, v in parameters.items()}

        return parameters

    def viscosity_parameters(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
        return_tensor: bool = False,
    ):
        """Compute the viscosity parameters.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          return_tensor:
            If `True`, the dictionary values will be torch tensors. If `False`,
            the dictionary values will be numpy arrays.

        Returns:
          Dictionary with the viscosity parameters as keys and the viscosity
          parameters as values.
        """

        features = self.featurizer(composition, input_cols)
        features = torch.from_numpy(features).float()
        parameters = self.viscosity_parameters_from_tensor(
            features, return_tensor
        )
        return parameters

    def viscosity_parameters_dist(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
        num_samples: int = 100,
    ):
        """Compute the distribution of the viscosity parameters.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          num_samples:
            Number of samples to draw for the Monte Carlo dropout computation.
            The higher the better, but also more computational expensive.

        Returns:
          Dictionary with the viscosity parameters as keys and an array with the
          the parameters distribution as values.
        """

        features = self.featurizer(composition, input_cols)
        features = torch.from_numpy(features).float()

        is_training = self.training
        if not is_training:
            self.train()

        dist = defaultdict(list)
        with torch.no_grad():
            for _ in range(num_samples):
                pdict = self.viscosity_parameters_from_tensor(features, False)
                for k, v in pdict.items():
                    dist[k].append(v)

        if not is_training:
            self.eval()

        dist = {k: np.array(v).T for k, v in dist.items()}

        return dist

    def viscosity_parameters_unc(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
        confidence: float = 0.95,
        num_samples: int = 100,
    ) -> Dict[str, np.ndarray]:
        """Compute the confidence bands of the viscosity parameters.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          confidence:
            Confidence level. Accepts values between 0 and 1 (inclusive).
          num_samples:
            Number of samples to draw for the Monte Carlo dropout computation.
            The higher the better, but also more computational expensive.

        Returns:
          Dictionary with the viscosity parameters as keys and an array with the
          the confidence band as values.
        """

        q = [(100 - 100 * confidence) / 2, 100 - (100 - 100 * confidence) / 2]
        dist = self.viscosity_parameters_dist(
            composition, input_cols, num_samples
        )
        bands = {k: np.percentile(v, q, axis=1).T for k, v in dist.items()}
        return bands

    def forward(self, x):
        """Method used for training the neural network.

        Consider using the other methods for prediction.

        Args:
          x:
            Feature tensor with the last column being the temperature in Kelvin.

        Returns
          Tensor with the viscosity prediction.
        """

        T = x[:, -1].detach().clone()
        parameters = self.viscosity_parameters_from_tensor(x[:, :-1], True)
        log_viscosity = self.log_viscosity_fun(T, **parameters)
        return log_viscosity

    def predict(
        self,
        T: Union[float, List[float], np.ndarray],
        composition: CompositionLike,
        input_cols: List[str] = [],
        table_mode: bool = False,
    ):
        """Prediction of the base-10 logarithm of viscosity.

        Args:
          T:
            Temperature to compute the viscosity. If this is a numpy array, it
            must have only one dimension.
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          table_mode:
            This argument is only relevant when the number of compositions is
            the same as the number of items in the temperature list. If True,
            then the code assumes that each composition is associated with its
            own temperature, which would be the case of tabular data. If False,
            then no such assumption is made and all temperatures will be
            considered for all compositions.

        Returns:
          Predicted values of the base-10 logarithm of viscosity.
        """

        parameters = self.viscosity_parameters(composition, input_cols, True)
        num_compositions = len(list(parameters.values())[0])

        with torch.no_grad():
            if isinstance(T, Iterable):
                if len(T) == num_compositions and table_mode:
                    T = torch.tensor(T).float()
                    params = [p for p in parameters.values()]
                    log_viscosity = self.log_viscosity_fun(T, *params).numpy()

                else:
                    T = torch.tensor(T).float()
                    params = [
                        p.expand(len(T), len(p)).T for p in parameters.values()
                    ]
                    log_viscosity = self.log_viscosity_fun(T, *params).numpy()

            else:
                T = torch.tensor(T).float()
                params = [p for p in parameters.values()]
                log_viscosity = self.log_viscosity_fun(T, *params).numpy()

        return log_viscosity

    def predict_log10_viscosity(
        self,
        T: Union[float, List[float], np.ndarray],
        composition: CompositionLike,
        input_cols: List[str] = [],
        table_mode: bool = False,
    ):
        """Prediction of the base-10 logarithm of viscosity.

        Args:
          T:
            Temperature to compute the viscosity. If this is a numpy array, it
            must have only one dimension.
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          table_mode:
            This argument is only relevant when the number of compositions is
            the same as the number of items in the temperature list. If True,
            then the code assumes that each composition is associated with its
            own temperature, which would be the case of tabular data. If False,
            then no such assumption is made and all temperatures will be
            considered for all compositions.

        Returns:
          Predicted values of the base-10 logarithm of viscosity.
        """

        return self.predict(T, composition, input_cols, table_mode)

    def predict_viscosity(
        self,
        T: Union[float, List[float], np.ndarray],
        composition: CompositionLike,
        input_cols: List[str] = [],
        table_mode: bool = False,
    ):
        """Prediction of the viscosity.

        Args:
          T:
            Temperature to compute the viscosity. If this is a numpy array, it
            must have only one dimension.
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          table_mode:
            This argument is only relevant when the number of compositions is
            the same as the number of items in the temperature list. If True,
            then the code assumes that each composition is associated with its
            own temperature, which would be the case of tabular data. If False,
            then no such assumption is made and all temperatures will be
            considered for all compositions.

        Returns:
          Predicted values of the viscosity.
        """

        return 10 ** self.predict(T, composition, input_cols, table_mode)

    def predict_fragility(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
    ):
        """Prediction of the liquid fragility.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.

        Returns:
          Predicted values of the liquid fragility.
        """

        parameters = self.viscosity_parameters(composition, input_cols)
        fragility = parameters["m"]
        return fragility

    def predict_Tg(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
    ):
        """Prediction of the glass transition temperature.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.

        Returns:
          Predicted values of the glass transition temperature.
        """

        parameters = self.viscosity_parameters(composition, input_cols)
        Tg = parameters["Tg"]
        return Tg

    def predict_log10_eta_infinity(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
    ):
        """Prediction of the base-10 logarithm of the asymptotic viscosity.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.

        Returns:
          Predicted values of the base-10 logarithm of the asymptotic viscosity.
        """

        parameters = self.viscosity_parameters(composition, input_cols)
        log_eta_inf = parameters["log_eta_inf"]
        return log_eta_inf

    def predict_eta_infinity(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
    ):
        """Prediction of the asymptotic viscosity.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.

        Returns:
          Predicted values of the asymptotic viscosity.
        """

        return 10 ** self.predict_log10_eta_infinity(composition, input_cols)

    def predict_Tg_unc(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
        confidence: float = 0.95,
        num_samples: int = 100,
    ):
        """Confidence bands of the glass transition temperature.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          confidence:
            Confidence level. Accepts values between 0 and 1 (inclusive).
          num_samples:
            Number of samples to draw for the Monte Carlo dropout computation.
            The higher the better, but also more computational expensive.

        Returns:
          Confidence bands of the glass transition temperature.
        """

        parameters_unc = self.viscosity_parameters_unc(
            composition, input_cols, confidence, num_samples
        )
        Tg = parameters_unc["Tg"]
        return Tg

    def predict_fragility_unc(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
        confidence: float = 0.95,
        num_samples: int = 100,
    ):
        """Confidence bands of the liquid fragility.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          confidence:
            Confidence level. Accepts values between 0 and 1 (inclusive).
          num_samples:
            Number of samples to draw for the Monte Carlo dropout computation.
            The higher the better, but also more computational expensive.

        Returns:
          Confidence bands of the liquid fragility.
        """

        parameters_unc = self.viscosity_parameters_unc(
            composition, input_cols, confidence, num_samples
        )
        m = parameters["m"]
        return m

    def predict_log10_eta_infinity_unc(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
        confidence: float = 0.95,
        num_samples: int = 100,
    ) -> np.ndarray:
        """Confidence bands of the base-10 logarithm of the asymptotic viscosity.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          confidence:
            Confidence level. Accepts values between 0 and 1 (inclusive).
          num_samples:
            Number of samples to draw for the Monte Carlo dropout computation.
            The higher the better, but also more computational expensive.

        Returns:
          Confidence bands of the base-10 logarithm of the asymptotic viscosity.
        """

        parameters_unc = self.viscosity_parameters_unc(
            composition, input_cols, confidence, num_samples
        )
        log_eta_inf = parameters_unc["log_eta_inf"]
        return log_eta_inf

    def predict_viscosity_unc(
        self,
        T: Union[float, List[float], np.ndarray],
        composition: CompositionLike,
        input_cols: List[str] = [],
        confidence: float = 0.95,
        num_samples: int = 100,
        table_mode: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.array]]:
        """Compute the confidence bands of the viscosity prediction.

        Args:
          T:
            Temperature to compute the viscosity. If this is a numpy array, it
            must have only one dimension.
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          confidence:
            Confidence level. Accepts values between 0 and 1 (inclusive).
          num_samples:
            Number of samples to draw for the Monte Carlo dropout computation.
            The higher the better, but also more computational expensive.
          table_mode:
            This argument is only relevant when the number of compositions is
            the same as the number of items in the temperature list. If True,
            then the code assumes that each composition is associated with its
            own temperature, which would be the case of tabular data. If False,
            then no such assumption is made and all temperatures will be
            considered for all compositions.

        Returns:
          bands:
            Confidence bands of the base-10 logarithm of viscosity. If more than
            one composition was given, then this is a 3-dimension numpy array.
            In this case, the first dimension is the composition, the second is
            the confidence band (lower and higher value, respectively), and the
            third is the temperature. Differently, if only one composition was
            given, then this is a 2-dimension array, with the first dimension
            being the confidence band and the second being the temperature. If
            the number of compositions is the same as the lenght of variable T
            and table_mode is True, then this variable is also a 2-dimension
            array.
          param_bands:
            Dictionary with the uncertainty of the viscosity parameters for each
            composition.
          pdistribution:
            Dictionary with the distribution of the viscosity parameters.
        """

        pdistribution = self.viscosity_parameters_dist(
            composition, input_cols, num_samples
        )
        parameters = []
        for param in self.parameters_range.keys():
            parameters.append(pdistribution[param])
        parameters = np.array(parameters)

        num_compositions = parameters.shape[1]

        if isinstance(T, Iterable):
            if len(T) == num_compositions and table_mode:
                all_curves = np.zeros((1, num_compositions, num_samples))
                T = torch.tensor(T).float()
                for i in range(num_samples):
                    params = [
                        torch.from_numpy(p).float()
                        for p in parameters[:, :, i]
                    ]
                    viscosity = self.log_viscosity_fun(T, *params).numpy()
                    all_curves[:, :, i] = viscosity

            else:
                all_curves = np.zeros((len(T), num_compositions, num_samples))
                T = torch.tensor(T).float()
                for i in range(num_samples):
                    params = [
                        torch.from_numpy(
                            np.broadcast_to(p, (len(T), len(p))).T
                        ).float()
                        for p in parameters[:, :, i]
                    ]
                    viscosity = self.log_viscosity_fun(T, *params).numpy()
                    all_curves[:, :, i] = viscosity.T

        else:
            all_curves = np.zeros((1, num_compositions, num_samples))
            T = torch.tensor(T).float()
            for i in range(num_samples):
                params = [
                    torch.from_numpy(p).float() for p in parameters[:, :, i]
                ]
                viscosity = self.log_viscosity_fun(T, *params).numpy()
                all_curves[:, :, i] = viscosity

        q = [(100 - 100 * confidence) / 2, 100 - (100 - 100 * confidence) / 2]

        bands = np.percentile(all_curves, q, axis=2)
        bands = np.transpose(bands, (2, 0, 1))

        if num_compositions == 1:
            bands = bands[0, :, :]

        param_bands = {
            k: np.percentile(v, q, axis=1).T for k, v in pdistribution.items()
        }

        return bands, param_bands, pdistribution

    @staticmethod
    def citation(bibtex: bool = False) -> str:
        if bibtex:
            c = "@article{Cassar_2021, title={ViscNet: Neural network for "
            "predicting the fragility index and the temperature-dependency of "
            "viscosity}, volume={206}, ISSN={1359-6454}, "
            "DOI={10.1016/j.actamat.2020.116602}, journal={Acta Materialia}, "
            "author={Cassar, Daniel R.}, year={2021}, month={Mar}, "
            "pages={116602}}"
        else:
            c = "D.R. Cassar, ViscNet: Neural network for predicting the "
            "fragility index and the temperature-dependency of viscosity, "
            "Acta Materialia. 206 (2021) 116602. "
            "https://doi.org/10.1016/j.actamat.2020.116602."
        return c


class _BaseGlassNet(MTL):
    """Base class for GlassNet-like networks.

    Args:
      hparams:
        Dictionary of the hyperparameters of the neural network.
    """

    # fmt: off
    element_features = [
        "Ag", "Al", "As", "B", "Ba", "Be", "Bi", "Br", "C", "Ca", "Cd", "Ce",
        "Cl", "Co", "Cr", "Cs", "Cu", "Dy", "Er", "Eu", "Fe", "Ga", "Gd", "Ge",
        "H", "Hf", "Hg", "Ho", "I", "In", "K", "La", "Li", "Lu", "Mg", "Mn",
        "Mo", "N", "Na", "Nb", "Nd", "Ni", "P", "Pb", "Pr", "Rb", "S", "Sb",
        "Sc", "Se", "Sm", "Sn", "Sr", "Ta", "Tb", "Te", "Ti", "Tl", "V", "W",
        "Y", "Yb", "Zn", "Zr",
    ]
    # fmt: on

    weighted_features = [
        ("c6_gb", "min"),
        ("ElectronAffinity", "min"),
        ("NUnfilled", "std1"),
        ("NdValence", "min"),
        ("NfValence", "min"),
        ("NpUnfilled", "min"),
        ("NsUnfilled", "min"),
        ("nvalence", "max"),
        ("atomic_volume", "min"),
        ("GSbandgap", "min"),
        ("GSenergy_pa", "max"),
        ("FusionEnthalpy", "min"),
    ]

    absolute_features = [
        ("num_oxistates", "max"),
        ("NUnfilled", "min"),
        ("NdUnfilled", "max"),
        ("NdValence", "max"),
        ("NfUnfilled", "sum"),
        ("NfValence", "sum"),
        ("NpUnfilled", "sum"),
        ("NpUnfilled", "max"),
        ("NpValence", "max"),
        ("NsUnfilled", "sum"),
        ("NsValence", "min"),
        ("nvalence", "max"),
        ("vdw_radius_uff", "max"),
        ("atomic_radius_rahm", "max"),
        ("GSbandgap", "min"),
        ("GSestFCClatcnt", "std1"),
        ("GSmagmom", "sum"),
        ("en_Sanderson", "max"),
        ("en_Tardini_Organov", "min"),
        ("zeff", "std1"),
        ("boiling_point", "std1"),
        ("FusionEnthalpy", "max"),
    ]

    scaler_x_file = _BASEMODELPATH / "GlassNet_scaler_x.joblib"
    scaler_y_file = _BASEMODELPATH / "GlassNet_scaler_y.joblib"

    def __init__(self, hparams: dict):
        super().__init__(self.hparams)
        self.scaler_x = joblib.load(self.scaler_x_file)
        self.scaler_y = joblib.load(self.scaler_y_file)

    def get_all_data(self):
        """Loads the data used to train GlassNet.

        This method takes some time to run, that is why it is not run
        automatically when a new instance of the class is created. After running
        once, the instance will have a `data` attribute that is a DataFrame with
        all the data.

        Returns:
          DataFrame with the all the data used to train and validate
          GlassNet.
        """
        if not hasattr(self, "data"):
            self.data = _load_data_glassnet()
        return self.data

    def get_training_dataset(self):
        """Gets the training dataset used in the GlassNet paper.

        Returns:
          DataFrame containing the training dataset used in the GlassNet
          paper.
        """
        self.get_all_data()
        indices = self.data.index
        train_idx, _ = train_test_split(
            indices,
            test_size=0.1,
            random_state=61455,
            shuffle=True,
        )
        return self.data.loc[train_idx]

    def get_test_dataset(self):
        """Gets the holdout dataset used in the GlassNet paper.

        Returns:
          DataFrame containing the holdout dataset used in the GlassNet
          paper.
        """
        self.get_all_data()
        indices = self.data.index
        _, test_idx = train_test_split(
            indices,
            test_size=0.1,
            random_state=61455,
            shuffle=True,
        )
        return self.data.loc[test_idx]

    def featurizer(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
        auto_scale: bool = True,
        return_cols: bool = False,
    ) -> np.ndarray:
        """Compute the features used for input.

        Args:
          composition:
            Any composition-like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          auto_scale:
            If `True`, then the features are automatically scaled and ready to
            use with GlassNet. If `False`, then the features will not be scaled
            and should not be used to make predictions with Glassnet. Default
            value is `True`.
          return_cols:
            If `True`, then a list with the name of the feature columns is also
            returned.

        Returns:
          Array with the computed features. Optionally, the name of the coulmns
          can also be returned if `return_cols` is True.
        """
        feat_array, cols = physchem_featurizer(
            x=composition,
            input_cols=input_cols,
            elemental_features=self.element_features,
            weighted_features=self.weighted_features,
            absolute_features=self.absolute_features,
            rescale_to_sum=1,
            order="eaw",
        )

        if auto_scale:
            feat_array = self.scaler_x.transform(feat_array)

        if return_cols:
            return feat_array, cols
        else:
            return feat_array

    def forward(self, x):
        """Method used for training the neural network.

        Consider using the other methods for prediction.

        Args:
          x:
            Feature tensor.

        Returns
          Tensor with the predictions.
        """
        return self.output_layer(self.hidden_layers(x))

    def predict(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
        return_dataframe: bool = True,
    ):
        """Makes prediction of properties.

        Args:
          composition:
            Any composition-like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          return_dataframe:
            If `True`, then returns a pandas DataFrame, else returns an array.
            Default value is `True`.

        Returns:
          Predicted values of properties. Will be a DataFrame if
          `return_dataframe` is True, otherwise will be an array.
        """
        is_training = self.training
        if is_training:
            self.eval()

        with torch.no_grad():
            features = self.featurizer(composition, input_cols)
            features = torch.from_numpy(features).float()
            y_pred = self.scaler_y.inverse_transform(self(features).detach())

        if is_training:
            self.train()

        if return_dataframe:
            return pd.DataFrame(y_pred, columns=self.targets)
        else:
            return y_pred

    @staticmethod
    def citation(bibtex: bool = False) -> str:
        raise NotImplementedError("Not implemented yet.")


class _BaseGlassNetViscosity(ABC):
    @abstractmethod
    def predict(self):
        pass

    def _viscosity_table_single(
        self,
        prediction,
        log_visc_limit: float = 12,
        columns: list = _VISCOSITY_COLUMNS_FOR_REGRESSION,
    ):
        """Viscosity table prediction.

        Args:
          log_visc_limit:
            Maximum value of log10(viscosity) (viscosity in Pa.s) that is
            acceptable. Predictions of values greater than 12 for the log10 of
            viscosity are prone to higher errors. Default value is 12.
          columns:
            Which targets of GlassNet that will be considered to build the
            table.

        Returns:
          Viscosity table.
        """

        data = []

        for col in columns:
            value = prediction[self.target_trans[col]]

            if col.startswith("Viscosity"):
                T = int(col[9:-1])
                log_visc = value

            elif col.startswith("T"):
                T = value

                if col[1].isdigit():
                    log_visc = int(col[1:])

                else:
                    if col == "Tg":
                        log_visc = 12
                    elif col == "TLittletons":
                        log_visc = 6.6
                    elif col == "TAnnealing":
                        log_visc = 12.4
                    elif col == "Tstrain":
                        log_visc = 13.5
                    elif col == "TdilatometricSoftening":
                        log_visc = 10
                    elif col == "Tsoft":
                        log_visc = 6.6
                    elif col == "Tmelt":
                        log_visc = 1
                    else:
                        log_visc = np.nan

            if log_visc > log_visc_limit:
                log_visc = np.nan

            if np.isfinite(log_visc):
                data.append((T, log_visc))

        data = np.array(sorted(data))

        return data

    def _viscosity_param_single(
        self,
        prediction,
        log_visc_limit: float = 12,
        columns: list = _VISCOSITY_COLUMNS_FOR_REGRESSION,
        n_points_low: int = 10,
    ):
        """Generator of viscosity tables.

        Args:
          composition:
            Any composition-like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          log_visc_limit:
            Maximum value of log10(viscosity) (viscosity in Pa.s) that is
            acceptable. Predictions of values greater than 12 for the log10 of
            viscosity are prone to higher errors. Default value is 12.
          columns:
            Which targets of GlassNet that will be considered to build the
            table.
          return_dataframe:
            If `True`, returns a pandas DataFrame. Else returns a numpy array.
          n_points_low:
            Number of viscosity data points to consider to guess the fragility
            index. Must be a positive integer. Default value is 10.

        Return:
          Viscosity parameters.
        """

        data = self._viscosity_table_single(
            prediction, log_visc_limit, columns
        )

        try:
            slope, _, _, _ = theilslopes(
                data[:n_points_low, 1],
                1 / data[:n_points_low, 0],
            )

            closest_to_Tg = np.argmin(np.abs(data[:, 1] - 12))

            guess_Tg = data[closest_to_Tg, 0]
            guess_m = slope / guess_Tg
            guess_ninf = data[-1, 1]

            regression = least_squares(
                _myega_residuals,
                [guess_ninf, guess_Tg, guess_m],
                loss="cauchy",
                f_scale=0.5,  # inlier residuals are lower than this value
                args=(data[:, 0], data[:, 1]),
                bounds=([-np.inf, 0, 10], [11, np.inf, np.inf]),
            )

            log_eta_inf, Tg_MYEGA, fragility = regression.x

            return log_eta_inf, Tg_MYEGA, fragility

        except ValueError:
            return np.nan, np.nan, np.nan

    def viscosity_parameters(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
        log_visc_limit: float = 12,
        columns: list = _VISCOSITY_COLUMNS_FOR_REGRESSION,
        n_points_low: int = 10,
        return_dataframe: bool = True,
    ):
        """Predict the MYEGA viscosity parameters.

        This method performs a robust non-linear regression of the MYEGA
        equation on viscosity data predicted by GlassNet. If the regression
        cannot be performed, the parameters returned will be NaN.

        Args:
          composition:
            Any composition-like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          log_visc_limit:
            Maximum value of log10(viscosity) (viscosity in Pa.s) that is
            acceptable. Predictions of values greater than 12 for the log10 of
            viscosity are prone to higher errors. Default value is 12.
          columns:
            Which targets of GlassNet that will be considered to build the
            table.
          n_points_low:
            Number of viscosity data points to consider to guess the fragility
            index. Must be a positive integer. Default value is 10.
          return_dataframe:
            If `True`, then returns a pandas DataFrame, else returns an array.
            Default value is `True`.

        Returns:
          parameters:
            Numpy array or DataFrame of the viscosity parameters. The first
            column is the asymptotic viscosity (in base-10 logarithm of Pa.s);
            the second column is the temperature where the viscosity is 10^12
            Pa.s; and the third column is the fragility index.
        """

        predictions = self.predict(composition, input_cols, False)
        parameters = [
            self._viscosity_param_single(
                i, log_visc_limit, columns, n_points_low
            )
            for i in predictions
        ]

        if return_dataframe:
            return pd.DataFrame(parameters, columns=self._visc_parameters)
        else:
            return np.array(parameters)

    def predict_log10_viscosity(
        self,
        T: Union[float, List[float], np.ndarray],
        composition: CompositionLike,
        input_cols: List[str] = [],
        log_visc_limit: float = 12,
        columns: list = _VISCOSITY_COLUMNS_FOR_REGRESSION,
        n_points_low: int = 10,
        return_parameters: bool = False,
    ):
        """Predict the base-10 logarithm of viscosity (viscosity in Pa.s).

        This method performs a robust non-linear regression of the MYEGA
        equation on viscosity data predicted by GlassNet. If the regression
        cannot be performed, the predicted viscosity will be NaN.

        Args:
          T:
            Temperature (in Kelvin) to compute the viscosity. If this is a numpy
            array, it must have only one dimension.
          composition:
            Any composition-like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          log_visc_limit:
            Maximum value of log10(viscosity) (viscosity in Pa.s) that is
            acceptable for the regression. Predictions of values greater than 12
            for the log10 of viscosity are prone to higher errors. Default value
            is 12.
          columns:
            Which targets of GlassNet that will be considered to build the
            table.
          n_points_low:
            Number of viscosity data points to consider to guess the fragility
            index. Must be a positive integer. Default value is 10.
          return_parameters:
            If `True`, also returns the viscosity parameters array.

        Returns:
          log_10_viscosity:
            Numpy array of the predicted base-10 logarithm of viscosity
          parameters:
            Numpy array of the viscosity parameters. The first column is the
            asymptotic viscosity (in base-10 logarithm of Pa.s); the second
            column is the temperature where the viscosity is 10^12 Pa.s; and the
            third column is the fragility index. Only returned if
            `return_parameters` is `True`.
        """

        parameters = self.viscosity_parameters(
            composition,
            input_cols,
            log_visc_limit,
            columns,
            n_points_low,
            return_dataframe=False,
        )

        log10_viscosity = myega_alt(
            T, parameters[:, 0], parameters[:, 1], parameters[:, 2]
        )

        if return_parameters:
            return log10_viscosity, parameters
        else:
            return log10_viscosity

    def predict_viscosity(
        self,
        T: Union[float, List[float], np.ndarray],
        composition: CompositionLike,
        input_cols: List[str] = [],
        log_visc_limit: float = 12,
        columns: list = _VISCOSITY_COLUMNS_FOR_REGRESSION,
        n_points_low: int = 10,
        return_parameters: bool = False,
    ):
        """Predict the viscosity in Pa.s.

        This method performs a robust non-linear regression of the MYEGA
        equation on viscosity data predicted by GlassNet. If the regression
        cannot be performed, the predicted viscosity will be NaN.

        Args:
          T:
            Temperature (in Kelvin) to compute the viscosity. If this is a numpy
            array, it must have only one dimension.
          composition:
            Any composition-like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.
          log_visc_limit:
            Maximum value of log10(viscosity) (viscosity in Pa.s) that is
            acceptable for the regression. Predictions of values greater than 12
            for the log10 of viscosity are prone to higher errors. Default value
            is 12.
          columns:
            Which targets of GlassNet that will be considered to perform the
            regression.
          n_points_low:
            Number of viscosity data points to consider to guess the fragility
            index for the regression. Must be a positive integer. Default value
            is 10.
          return_parameters:
            If `True`, also returns the viscosity parameters array.

        Returns:
          viscosity:
            Numpy array of the predicted viscosity in Pa.s.
          parameters:
            Numpy array of the viscosity parameters. The first column is the
            asymptotic viscosity (in base-10 logarithm of Pa.s); the second
            column is the temperature where the viscosity is 10^12 Pa.s; and the
            third column is the fragility index. Only returned if
            `return_parameters` is `True`.
        """

        log10_viscosity, parameters = self.predict_log10_viscosity(
            T,
            composition,
            input_cols,
            log_visc_limit,
            columns,
            n_points_low,
            return_parameters=True,
        )

        if return_parameters:
            return 10**log10_viscosity, parameters
        else:
            return 10**log10_viscosity
