from .base import MLP
from glasspy.chemistry import physchem_featurizer, CompositionLike

from typing import Dict, List, Tuple, NamedTuple, Union, Any

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


_basemodelpath = Path(os.path.dirname(__file__)) / "models"


class BaseViscNet(MLP):
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


class ViscNet(BaseViscNet):
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

    hparams = {
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

    x_mean = torch.tensor(
        [
            5.7542068e01,
            2.2090124e01,
            2.0236173e00,
            3.6860932e-02,
            3.2620981e-01,
            1.4419103e00,
            2.0164611e00,
            3.4407539e01,
            1.2352635e03,
            1.4792695e00,
            4.2045139e01,
            8.4130961e-01,
            2.3045063e00,
            4.7984661e01,
            5.6983612e01,
            1.1145602e00,
            9.2185795e-02,
            2.1363457e-01,
            2.2581025e-04,
            5.8149724e00,
            1.2964197e01,
            3.7008467e00,
            1.3742505e-01,
            1.8369867e-02,
            3.2303229e-01,
            7.1324579e-02,
            5.0019447e01,
            4.3720217e00,
            3.6445999e01,
            8.4036522e00,
            2.0281389e02,
            7.5613613e00,
            1.2259276e02,
            6.7182720e-01,
            1.0508456e-01,
        ]
    ).float()

    x_std = torch.tensor(
        [
            7.6420674e00,
            4.7180834e00,
            4.5827568e-01,
            1.6872969e-01,
            9.7033477e-01,
            2.7695282e00,
            3.3152765e-01,
            6.4520807e00,
            6.3392188e02,
            4.0605769e-01,
            1.1776709e01,
            2.8129578e-01,
            7.9213607e-01,
            7.5883222e00,
            1.1334700e01,
            2.8822860e-01,
            4.4787191e-02,
            1.1219133e-01,
            1.2392291e-03,
            1.1634388e00,
            2.9513965e00,
            4.7246245e-01,
            3.1958127e-01,
            8.8973150e-02,
            6.7548370e-01,
            6.2869355e-02,
            1.0003569e01,
            2.7434089e00,
            1.9245349e00,
            3.4734800e-01,
            1.2475176e02,
            3.2667663e00,
            1.5287421e02,
            7.3510885e-02,
            1.6187511e-01,
        ]
    ).float()

    state_dict_path = _basemodelpath / "ViscNet_SD.p"

    def __init__(self):
        super().__init__(
            self.parameters_range, self.hparams, self.x_mean, self.x_std
        )

        state_dict = pickle.load(open(self.state_dict_path, "rb"))
        self.load_state_dict(state_dict)

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

        self.hparams = self.hparams.copy()
        self.hparams["loss"] = "huber"
        self.loss_fun = F.smooth_l1_loss

        state_dict_path = _basemodelpath / "ViscNetHuber_SD.p"
        state_dict = pickle.load(open(state_dict_path, "rb"))
        self.load_state_dict(state_dict)


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

        state_dict_path = _basemodelpath / "ViscNetVFT_SD.p"
        state_dict = pickle.load(open(state_dict_path, "rb"))
        self.load_state_dict(state_dict)

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
