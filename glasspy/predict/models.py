"""Predictive models offered by GlassPy."""

import gzip
import warnings
from typing import List

import joblib
import numpy as np
import onnxruntime as rt
import pandas as pd
import torch
from glasspy.chemistry import CompositionLike, physchem_featurizer
from glasspy.utils import get_from_zenodo

# Required for joblib deserialization of RandomForestClassifier models
from sklearn.ensemble import RandomForestClassifier  # noqa: F401
from torch.nn import functional as F

from .base import (
    _BASEMODELPATH,
    GLASSNET_HP,
    GLASSNET_TARGETS,
    Predict,
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
        super().__init__(self.parameters_range, self._hparams, self.x_mean, self.x_std)

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
                y_pred[:, pos] = self.st_dict[target](features).ravel().detach()

            y_pred = self.scaler_y.inverse_transform(y_pred)

        if return_dataframe:
            return pd.DataFrame(y_pred, columns=self.targets)
        else:
            return y_pred


class VITRIFY(Predict):
    """Predictor of the probability of forming a glass using the VITRIFY model.

    VITRIFY stands for *Vitrification Inference Tool for Rapid Identification
    of glass-Forming abilitY*.

    The original model was trained using the scikit-learn framework. During
    conversion to ONNX, all float64 values were cast to float32. As a result,
    compositions very close to decision boundaries may yield different
    predictions due to floating-point rounding.

    .. warning::
        When using the `sklearn` framework, scikit-learn version 1.7.1 must be
        installed. Other versions may produce inconsistent results.

    Args:
        model (str):
          Name of the glass-formation predictive model to use.  Available
          options are: `FEATENG`, `FEATENG+GS`, `CHEM`, and `GS`. Defaults to
          `"CHEM"`.
        framework (str):
          Inference framework to use. Accepted values are `"onnx"` and
          `"sklearn"`. Defaults to `"onnx"` if not specified. If `"sklearn"` is
          selected, ensure scikit-learn version 1.7.1 is installed.
    """

    def __init__(self, model="CHEM", framework="onnx"):
        super().__init__()

        self.model_name = model
        self.framework = framework

        # fmt: off
        pc_feats = [
            'A|vdw_radius_uff|std', 'W|atomic_radius_rahm|mean',
            'W|NpValence|mean', 'A|pettifor_number|sum', 'W|FusionEnthalpy|sum',
            'W|num_oxistates|mean', 'A|FusionEnthalpy|mean', 'W|NdValence|mean',
            'W|NsValence|sum', 'A|GSbandgap|std', 'W|GSvolume_pa|mean',
            'W|GSbandgap|sum', 'A|num_oxistates|std', 'W|NdUnfilled|max',
            'W|NpUnfilled|sum', 'A|melting_point|std', 'A|NUnfilled|sum',
            'A|num_oxistates|max', 'W|covalent_radius_cordero|std',
            'A|en_Martynov_Batsanov|std'
        ]

        abs_feats = [
            ("vdw_radius_uff", "std"),
            ("pettifor_number", "sum"),
            ("FusionEnthalpy", "mean"),
            ("GSbandgap", "std"),
            ("num_oxistates", "std"),
            ("melting_point", "std"),
            ("NUnfilled", "sum"),
            ("num_oxistates", "max"),
            ("en_Martynov_Batsanov", "std"),
        ]

        wei_feats = [
            ("atomic_radius_rahm", "mean"),
            ("NpValence", "mean"),
            ("FusionEnthalpy", "sum"),
            ("num_oxistates", "mean"),
            ("NdValence", "mean"),
            ("NsValence", "sum"),
            ("GSvolume_pa", "mean"),
            ("GSbandgap", "sum"),
            ("NdUnfilled", "max"),
            ("NpUnfilled", "sum"),
            ("covalent_radius_cordero", "std"),
        ]

        gs_feats = ['Kw_Tx', 'Kw_Tc', 'Kh_Tc', 'H_Tx', 'gamma_Tc', 'jezica']

        chem_feats = [
            'Ag', 'Al', 'As', 'B', 'Ba', 'Be', 'Bi', 'Br', 'Ca', 'Cd', 'Ce',
            'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Er', 'Eu', 'F', 'Fe', 'Ga',
            'Gd', 'Ge', 'H', 'Hf', 'Hg', 'Ho', 'I', 'In', 'K', 'La', 'Li', 'Lu',
            'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nb', 'Nd', 'Ni', 'O', 'P', 'Pb', 'Pr',
            'Rb', 'S', 'Sb', 'Sc', 'Se', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb',
            'Te', 'Ti', 'Tl', 'Tm', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr'
        ]
        # fmt: on

        if framework == "onnx":
            suffix = ".onnx.gz"
        elif framework == "sklearn":
            suffix = ".pkl"
        else:
            raise ValueError("Invalid framework name.")

        self.comp_pc = False
        self.comp_gs = False
        self.comp_chem = False

        match model:
            case "FEATENG":
                self.features = pc_feats
                self.abs_feats = abs_feats
                self.wei_feats = wei_feats
                self.comp_pc = True
                model_name = "FEATENG" + suffix
            case "GS":
                self.features = gs_feats
                self.glassnet = GlassNet()
                self.comp_gs = True
                model_name = "GS" + suffix
            case "CHEM":
                self.features = chem_feats
                self.comp_chem = True
                model_name = "CHEM" + suffix
            case "FEATENG+GS":
                self.features = pc_feats + gs_feats
                self.glassnet = GlassNet()
                self.abs_feats = abs_feats
                self.wei_feats = wei_feats
                self.comp_pc = True
                self.comp_gs = True
                model_name = "FEATENG_GS" + suffix
            case _:
                raise ValueError("GlassPy: Invalid GFA model name.")

        get_from_zenodo(
            "18964978",
            model_name,
            _BASEMODELPATH / model_name,
            verbose=True,
        )

        if framework == "onnx":
            with gzip.open(_BASEMODELPATH / model_name, "rb") as f:
                self.model = rt.InferenceSession(f.read())

        elif framework == "sklearn":
            self.model = joblib.load(_BASEMODELPATH / model_name)

    def _featurizer(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
    ):
        dfs = []

        if self.comp_chem:
            x, cols = physchem_featurizer(
                composition,
                input_cols,
                elemental_features=self.features,
                rescale_to_sum=1,
            )
            chem = pd.DataFrame(x, columns=cols)
            if any(chem.sum(axis=1).round(3) < 1):
                warnings.warn(
                    "Chemical composition outside of training domain.", UserWarning
                )
            dfs.append(chem)

        if self.comp_pc:
            x, cols = physchem_featurizer(
                composition,
                input_cols,
                weighted_features=self.wei_feats,
                absolute_features=self.abs_feats,
            )
            pc = pd.DataFrame(x, columns=cols)
            dfs.append(pc)

        if self.comp_gs:
            preds = self.glassnet.predict(composition, input_cols)
            Tg = preds["Tg"]
            Tc = preds["CrystallizationPeak"]
            Tx = preds["CrystallizationOnset"]
            Tm = preds["Tliquidus"]
            eta_Tm = self.glassnet.predict_viscosity(Tm, composition, input_cols)
            gs = pd.DataFrame(
                {
                    "Kw_Tx": (Tx - Tg) / Tm,
                    "Kw_Tc": (Tc - Tg) / Tm,
                    "Kh_Tc": (Tc - Tg) / (Tm - Tc),
                    "H_Tx": (Tx - Tg) / Tg,
                    "gamma_Tc": Tc / (Tg + Tm),
                    "jezica": eta_Tm / Tm**2,
                }
            )
            dfs.append(gs)

        X = pd.concat(dfs, axis=1).reindex(self.features, axis=1)

        return X

    def predict(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
    ):
        """Classifies the compositions with respect of glass formation.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.

        Returns:
          Array with zeros and ones. Zero represent a prediction that the
          composition will not form a glass while one represents the opposite.
        """

        X = self._featurizer(composition, input_cols)

        if self.framework == "onnx":
            dict = {"float_input": X.to_numpy(dtype=np.float32)}
            return self.model.run(None, dict)[0]
        else:
            return self.model.predict(X.to_numpy())

    def predict_proba(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
    ):
        """Returns the predicted probabilities of the model.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.

        Returns:
          2D Numpy array with values between 0 and 1. First column represents
          the probability of not forming a glass and second column represents
          the probability of forming a glass.
        """
        X = self._featurizer(composition, input_cols)

        if self.framework == "onnx":
            dict_ = {"float_input": X.to_numpy(dtype=np.float32)}
            proba = self.model.run(None, dict_)[1]
            proba = np.array([[d[0], d[1]] for d in proba])
            return proba
        else:
            return self.model.predict_proba(X.to_numpy())

    def predict_proba_glass(
        self,
        composition: CompositionLike,
        input_cols: List[str] = [],
    ):
        """Returns the predicted probability of forming a glass.

        Args:
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of `composition`. Necessary only when `composition` is a list
            or array, ignored otherwise.

        Returns:
          Numpy array with values between 0 and 1 representing the predicted
          probability of glass formation.
        """
        return self.predict_proba(composition, input_cols)[:, 1]

    @property
    def domain(self):
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

    @staticmethod
    def citation(bibtex: bool = False) -> str:
        if bibtex:
            c = "@unpublished{Carvalho2026glass, author={Carvalho, Diogo P. L. and Loponi, Ana C. B. and Cassar, Daniel R.}, title={Will it form a glass? {T}ackling glass formation using binary classification}, note={Currently in peer review}, year  ={2026},}"
        else:
            c = "Diogo P. L. Carvalho, Ana C. B. Loponi, Daniel R. Cassar. Will it form a glass? Tackling glass formation using binary classification. Currently in peer review. 2026."
        return c
