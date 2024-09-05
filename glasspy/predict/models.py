"""Predictive models offered by GlassPy."""

from typing import List

import numpy as np
import pandas as pd
import torch
from glasspy.chemistry import CompositionLike, physchem_featurizer
from torch.nn import functional as F

from .base import (
    _BASEMODELPATH,
    GLASSNET_TARGETS,
    GLASSNET_HP,
    _BaseGlassNet,
    _BaseGlassNetViscosity,
    _BaseViscNet,
)


class ViscNet(_BaseViscNet):
    """ViscNet predictor of viscosity and viscosity parameters.

    ViscNet is a physics-informed neural network that has the MYEGA [1]
    viscosity equation embedded in it. See Ref. [2] for the original
    publication.

    References:
      [1] J.C. Mauro, Y. Yue, A.J. Ellison, P.K. Gupta, D.C. Allan, Viscosity of
          glass-forming liquids., Proceedings of the National Academy of
          Sciences of the United States of America. 106 (2009) 19780–19784.
          https://doi.org/10.1073/pnas.0911705106.
      [2] D.R. Cassar, ViscNet: Neural network for predicting the fragility
          index and the temperature-dependency of viscosity, Acta Materialia.
          206 (2021) 116602. https://doi.org/10.1016/j.actamat.2020.116602.
          https://arxiv.org/abs/2007.03719
    """

    parameters_range = {
        "log_eta_inf": [-18, 5],
        "Tg": [400, 1400],
        "m": [10, 130],
    }

    _hparams = {
        "batch_size": 64,
        "layer_1_activation": "ReLU",
        "layer_1_batchnorm": False,
        "layer_1_dropout": 0.07942161101271952,
        "layer_1_size": 192,
        "layer_2_activation": "Tanh",
        "layer_2_batchnorm": False,
        "layer_2_dropout": 0.05371454289414608,
        "layer_2_size": 48,
        "loss": "mse",
        "lr": 0.0011695226458761677,
        "max_epochs": 500,
        "n_features": 35,
        "num_layers": 2,
        "optimizer": "AdamW",
        "patience": 9,
    }

    absolute_features = [
        ("ElectronAffinity", "std1"),
        ("FusionEnthalpy", "std1"),
        ("GSenergy_pa", "std1"),
        ("GSmagmom", "std1"),
        ("NdUnfilled", "std1"),
        ("NfValence", "std1"),
        ("NpUnfilled", "std1"),
        ("atomic_radius_rahm", "std1"),
        ("c6_gb", "std1"),
        ("lattice_constant", "std1"),
        ("mendeleev_number", "std1"),
        ("num_oxistates", "std1"),
        ("nvalence", "std1"),
        ("vdw_radius_alvarez", "std1"),
        ("vdw_radius_uff", "std1"),
        ("zeff", "std1"),
    ]

    weighted_features = [
        ("FusionEnthalpy", "min"),
        ("GSbandgap", "max"),
        ("GSmagmom", "mean"),
        ("GSvolume_pa", "max"),
        ("MiracleRadius", "std1"),
        ("NValence", "max"),
        ("NValence", "min"),
        ("NdUnfilled", "max"),
        ("NdValence", "max"),
        ("NsUnfilled", "max"),
        ("SpaceGroupNumber", "max"),
        ("SpaceGroupNumber", "min"),
        ("atomic_radius", "max"),
        ("atomic_volume", "max"),
        ("c6_gb", "max"),
        ("c6_gb", "min"),
        ("max_ionenergy", "min"),
        ("num_oxistates", "max"),
        ("nvalence", "min"),
    ]

    # fmt: off
    x_mean = torch.tensor([
        5.7542068e01, 2.2090124e01, 2.0236173e00, 3.6860932e-02,
        3.2620981e-01, 1.4419103e00, 2.0164611e00, 3.4407539e01,
        1.2352635e03, 1.4792695e00, 4.2045139e01, 8.4130961e-01,
        2.3045063e00, 4.7984661e01, 5.6983612e01, 1.1145602e00,
        9.2185795e-02, 2.1363457e-01, 2.2581025e-04, 5.8149724e00,
        1.2964197e01, 3.7008467e00, 1.3742505e-01, 1.8369867e-02,
        3.2303229e-01, 7.1324579e-02, 5.0019447e01, 4.3720217e00,
        3.6445999e01, 8.4036522e00, 2.0281389e02, 7.5613613e00,
        1.2259276e02, 6.7182720e-01, 1.0508456e-01
    ]).float()

    x_std = torch.tensor([
        7.6420674e00, 4.7180834e00, 4.5827568e-01, 1.6872969e-01,
        9.7033477e-01, 2.7695282e00, 3.3152765e-01, 6.4520807e00,
        6.3392188e02, 4.0605769e-01, 1.1776709e01, 2.8129578e-01,
        7.9213607e-01, 7.5883222e00, 1.1334700e01, 2.8822860e-01,
        4.4787191e-02, 1.1219133e-01, 1.2392291e-03, 1.1634388e00,
        2.9513965e00, 4.7246245e-01, 3.1958127e-01, 8.8973150e-02,
        6.7548370e-01, 6.2869355e-02, 1.0003569e01, 2.7434089e00,
        1.9245349e00, 3.4734800e-01, 1.2475176e02, 3.2667663e00,
        1.5287421e02, 7.3510885e-02, 1.6187511e-01
    ]).float()
    # fmt: on

    def __init__(self):
        super().__init__(
            self.parameters_range, self._hparams, self.x_mean, self.x_std
        )

        state_dict_path = _BASEMODELPATH / "ViscNet.pth"
        self.load_training(state_dict_path)
        self.eval()

    def log_viscosity_fun(self, T, log_eta_inf, Tg, m):
        """Computes the base-10 logarithm of viscosity using the MYEGA equation."""

        log_viscosity = (
            log_eta_inf
            + (12 - log_eta_inf)
            * (Tg / T)
            * ((m / (12 - log_eta_inf) - 1) * (Tg / T - 1)).exp()
        )
        return log_viscosity

    def featurizer(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
    ) -> np.ndarray:
        """Compute the chemical features used for viscosity prediction.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.

        Returns:
          Array with the computed chemical features
        """

        feat_array, _ = physchem_featurizer(
            x=composition,
            input_cols=input_cols,
            weighted_features=self.weighted_features,
            absolute_features=self.absolute_features,
            rescale_to_sum=1,
            order="aw",
        )
        return feat_array


class ViscNetHuber(ViscNet):
    """ViscNet-Huber predictor of viscosity and viscosity parameters.

    ViscNet-Huber is a physics-informed neural network that has the MYEGA [1]
    viscosity equation embedded in it. The difference between this model and
    ViscNet is the loss function: this model has a robust smooth-L1 loss
    function, while ViscNet has a MSE (L2) loss function. See Ref. [2] for the
    original publication.

    References:
      [1] J.C. Mauro, Y. Yue, A.J. Ellison, P.K. Gupta, D.C. Allan, Viscosity of
          glass-forming liquids., Proceedings of the National Academy of
          Sciences of the United States of America. 106 (2009) 19780–19784.
          https://doi.org/10.1073/pnas.0911705106.
      [2] D.R. Cassar, ViscNet: Neural network for predicting the fragility
          index and the temperature-dependency of viscosity, Acta Materialia.
          206 (2021) 116602. https://doi.org/10.1016/j.actamat.2020.116602.
          https://arxiv.org/abs/2007.03719

    """

    def __init__(self):
        super().__init__()

        self.loss_fun = F.smooth_l1_loss
        state_dict_path = _BASEMODELPATH / "ViscNetHuber.pth"
        self.load_training(state_dict_path)


class ViscNetVFT(ViscNet):
    """ViscNet-VFT predictor of viscosity and viscosity parameters.

    ViscNet-VFT is a physics-informed neural network that has the VFT [1-3]
    viscosity equation embedded in it. See Ref. [4] for the original
    publication.

    References:
      [1] H. Vogel, Das Temperatureabhängigketsgesetz der Viskosität von
          Flüssigkeiten, Physikalische Zeitschrift. 22 (1921) 645–646.
      [2] G.S. Fulcher, Analysis of recent measurements of the viscosity of
          glasses, Journal of the American Ceramic Society. 8 (1925) 339–355.
          https://doi.org/10.1111/j.1151-2916.1925.tb16731.x.
      [3] G. Tammann, W. Hesse, Die Abhängigkeit der Viscosität von der
          Temperatur bie unterkühlten Flüssigkeiten, Z. Anorg. Allg. Chem. 156
          (1926) 245–257. https://doi.org/10.1002/zaac.19261560121.
      [4] D.R. Cassar, ViscNet: Neural network for predicting the fragility
          index and the temperature-dependency of viscosity, Acta Materialia.
          206 (2021) 116602. https://doi.org/10.1016/j.actamat.2020.116602.
          https://arxiv.org/abs/2007.03719
    """

    def __init__(self):
        super().__init__()

        state_dict_path = _BASEMODELPATH / "ViscNetVFT.pth"
        self.load_training(state_dict_path)

    def log_viscosity_fun(self, T, log_eta_inf, Tg, m):
        """Computes the base-10 logarithm of viscosity using the VFT equation.

        Reference:
          [1] H. Vogel, Das Temperatureabhängigketsgesetz der Viskosität von
            Flüssigkeiten, Physikalische Zeitschrift. 22 (1921) 645–646.
          [2] G.S. Fulcher, Analysis of recent measurements of the viscosity of
            glasses, Journal of the American Ceramic Society. 8 (1925) 339–355.
            https://doi.org/10.1111/j.1151-2916.1925.tb16731.x.
          [3] G. Tammann, W. Hesse, Die Abhängigkeit der Viscosität von der
            Temperatur bie unterkühlten Flüssigkeiten, Z. Anorg. Allg. Chem. 156
            (1926) 245–257. https://doi.org/10.1002/zaac.19261560121.
        """

        log_viscosity = log_eta_inf + (12 - log_eta_inf) ** 2 / (
            m * (T / Tg - 1) + (12 - log_eta_inf)
        )
        return log_viscosity


class GlassNetMTMLP(_BaseGlassNet, _BaseGlassNetViscosity):
    """Multitask neural network for predicting glass properties.

    This is the MT-MLP model.

    """

    def __init__(self):
        self.targets = GLASSNET_TARGETS.copy()
        self.target_trans = {p: i for i, p in enumerate(self.targets)}
        self._visc_parameters = [
            "log10_eta_infinity (Pa.s)",
            "Tg_MYEGA (K)",
            "fragility",
        ]
        super().__init__(GLASSNET_HP)
        state_dict_path = _BASEMODELPATH / "GlassNet.pth"
        learning_curve_path = _BASEMODELPATH / "GlassNet_lc.p"
        self.load_training(state_dict_path, learning_curve_path)


class GlassNetMTMH(_BaseGlassNet, _BaseGlassNetViscosity):
    """Multitask neural network for predicting glass properties.

    This is the MT-MH model.

    """

    def __init__(self):
        self.targets = GLASSNET_TARGETS.copy()
        self.target_trans = {p: i for i, p in enumerate(self.targets)}
        self._visc_parameters = [
            "log10_eta_infinity (Pa.s)",
            "Tg_MYEGA (K)",
            "fragility",
        ]
        super().__init__(GLASSNET_HP, 10)
        state_dict_path = _BASEMODELPATH / "GlassNetMH.pth"
        learning_curve_path = _BASEMODELPATH / "GlassNetMH_lc.p"
        self.load_training(state_dict_path, learning_curve_path)


class GlassNetSTNN(_BaseGlassNet):
    """Single-task neural network for predicting glass properties.

    This is the ST-NN model.

    """

    def __init__(self, model_name):
        hparams = GLASSNET_HP.copy()
        hparams["n_targets"] = 1
        super().__init__(hparams, 10)
        state_dict_path = _BASEMODELPATH / f"st-nn/{model_name}.pth"
        learning_curve_path = _BASEMODELPATH / f"st-nn/{model_name}_lc.p"
        self.load_training(state_dict_path, learning_curve_path)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fun(self(x).ravel(), y.ravel())
        self.training_step_outputs.append(loss)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fun(self(x).ravel(), y.ravel())
        self.validation_step_outputs.append(loss)
        self.log("loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fun(self(x).ravel(), y.ravel())
        self.log("test_loss", loss)
        return loss


class GlassNet(GlassNetMTMH):
    """Hybrid neural network for predicting glass properties.

    This hybrid model has a multitask neural network to compute most of the
    properties and especialized neural networks to predict selected properties.

    Args:
      st_models:
        List of the properties to use especialized models instead of using the
        multitask network. If `default`, then the model uses those properties
        that performed better than the multitask model.
    """

    def __init__(self, st_models="default"):
        super().__init__()

        if st_models == "default":
            self._st_models = [
                "AbbeNum",
                "CrystallizationOnset",
                "CrystallizationPeak",
                "Density293K",
                "MeanDispersion",
                "Microhardness",
                "PoissonRatio",
                "RefractiveIndex",
                "Resistivity1273K",
                "Resistivity1473K",
                "Resistivity1673K",
                "SurfaceTension1473K",
                "SurfaceTension1673K",
                "SurfaceTensionAboveTg",
                "TdilatometricSoftening",
                "Tg",
                "Tliquidus",
                "TresistivityIs1MOhm.m",
                "Tsoft",
                "Viscosity1273K",
                "Viscosity1473K",
                "Viscosity1673K",
                "YoungModulus",
                "T0",
                "Cp293K",
            ]
        else:
            self._st_models = st_models

        self.st_dict = {}
        for target in self._st_models:
            _model = GlassNetSTNN(target)
            _model.eval()
            self.st_dict[target] = _model

        self.eval()

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
        with torch.no_grad():
            features = self.featurizer(composition, input_cols)
            features = torch.from_numpy(features).float()

            y_pred = self(features).detach()

            for target in self.st_dict:
                pos = self.target_trans[target]
                y_pred[:, pos] = (
                    self.st_dict[target](features).ravel().detach()
                )

            y_pred = self.scaler_y.inverse_transform(y_pred)

        if return_dataframe:
            return pd.DataFrame(y_pred, columns=self.targets)
        else:
            return y_pred
