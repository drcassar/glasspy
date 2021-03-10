from abc import ABC, abstractmethod
from collections import namedtuple, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple, Union
import os
import pickle

from torch.nn import functional as F
from torch.optim import SGD, Adam, AdamW
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
try:
    import tensorflow as tf
except ModuleNotFoundError:
    print('WARNING: Some models require that tensorflow is installed.\n'
          '  It is not a hard dependency of GlassPy to reduce the '
          'risk of interfering\n  with your local copies')

from glasspy.chemistry import convert, featurizer, CompositionLike
from glasspy.typing import CompositionLike


_basemodelpath = Path(os.path.dirname(__file__)) / 'models'
_basesupportpath = Path(os.path.dirname(__file__)) / 'support'

class Domain(NamedTuple):
    element: Dict[str, float] = None
    compound: Dict[str, float] = None


class Predict(ABC):
    '''Base class for GlassPy predictors.'''
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
    def get_validation_dataset(self):
        pass

    @abstractmethod
    def get_test_dataset(self):
        pass

    def compute_metrics(self):
        # TODO
        pass


class PredictElementalInputTF(Predict):
    '''Base Predictor for models that use chemical composition as features.'''
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def info(self):
        pass

    @abstractmethod
    def convert_input(self):
        pass

    @property
    def domain(self) -> Domain:
        '''Get information about the training domain of the model.

        Returns:
          Instance of Domain.

        '''
        domain = Domain(
            element = self.info.get('domain_el', None),
            compound = self.info.get('domain_comp', None),
            )
        return domain

    def is_within_domain(
            self,
            x: CompositionLike,
            input_cols: List[str] = [],
            convert_wt_to_mol: bool = False,
            check_only_hard_domain: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        '''Checks if composition is inside the training domain.

        Args:
          x:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of x. Necessary only when x is a list or array, ignored
            otherwise.
          convert_wt_to_mol:
            Set to True if x is is weight%, False otherwise.
          check_only_hard_domain:
            If True, the function checks only if the chemical elements in x are
            present in the domain of training, but don't check if they are
            within the training domain range. If False, then the function also
            checks if the quantity of each chemical element is within the
            training chemical domain range.

        Returns:
          is_within_domain:
            A boolean numpy array. Substances within the domain are True and
            outside the domain are False.
          x:
            A 2D array containing the representation of x in terms of chemical
            elements.
          elements:
            List of strings representing the chemical entities related to each
            column of x.

        '''
        if convert_wt_to_mol:
            x, elements = convert.to_element_array(x, input_cols, 'all')
            x = convert.wt_to_mol(x, elements, 1)
        else:
            x, elements = convert.to_element_array(x, input_cols, 'all',
                                                   rescale_to_sum=1)

        elements_not_in_hard_domain = [
            elements.index(el) for el in elements
            if el not in self.info['features']
        ]

        is_within_domain = np.logical_not(
            x[:, elements_not_in_hard_domain].sum(axis=1).astype(bool)
        )

        if not check_only_hard_domain:
            elements_in_hard_domain = [
                elements.index(el) for el in self.info['features']
            ]

            x = x[:,elements_in_hard_domain]

            for n, el in enumerate(self.info['features']):
                col = x[:,n]
                greater_than_min = np.greater(col, self.domain.element[el][0])
                less_than_max = np.less(col, self.domain.element[el][1])
                el_is_within_domain = np.logical_and(
                    greater_than_min, less_than_max
                )
                is_within_soft_domain = np.logical_and(
                    is_within_domain, el_is_within_domain
                )

            elements = self.info['features']

        return is_within_domain, x, elements

    def predict(
            self,
            x: CompositionLike,
            input_cols: List[str] = [],
            convert_wt_to_mol: bool = False,
            check_only_hard_domain: bool = False,
    ) -> Union[np.ndarray, float]:
        '''Predicts the relevant property of substance(s) using the model. 

        Args:
          x:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of x. Necessary only when x is a list or array, ignored
            otherwise.
          convert_wt_to_mol:
            Set to True if x is is weight%, False otherwise.
          check_only_hard_domain:
            If True, the function checks only if the chemical elements in x are
            present in the domain of training, but don't check if they are
            within the training domain range. If False, then the function also
            checks if the quantity of each chemical element is within the
            training chemical domain range.

        Returns:
          Predicted values of the relenant property. If the input data was only
          one substance, then returns a float. Otherwise returns a 1D numpy
          array. Predictions of substances outside of the domain are represented
          by nan (not a number).

        '''
        is_within_domain, x, elements = \
            self.is_within_domain(x, input_cols, convert_wt_to_mol,
                                  check_only_hard_domain)

        elements_in_hard_domain = [
            elements.index(el) for el in self.info['features']
        ]

        x = x[:,elements_in_hard_domain]

        x_mean = self.info['x_mean']
        x_std = self.info['x_std']
        y_mean = self.info['y_mean']
        y_std = self.info['y_std']
        model = self.info['model']

        x_scaled = self.convert_input((x - x_mean) / x_std)
        y_scaled = model.predict(x_scaled)
        y = (y_scaled * y_std + y_mean).flatten()
        y[~is_within_domain] = np.nan

        if len(y) == 1:
            y = y[0]

        return y

    def get_training_dataset(self):
        # TODO
        raise NotImplementedError('GlassPy error: not implemented.')

    def get_validation_dataset(self):
        # TODO
        raise NotImplementedError('GlassPy error: not implemented.')

    def get_test_dataset(self):
        # TODO
        raise NotImplementedError('GlassPy error: not implemented.')


class MLP(pl.LightningModule, Predict):
    '''Base Predictor for models that use chemical composition as features.'''
    def __init__(self):
        super().__init__()

        layers = []
        hparams = self.hparams
        input_dim = hparams['n_features']

        for n in range(1, hparams['num_layers'] + 1):

            l = [
                nn.Linear(
                    input_dim, int(hparams[f'layer_{n}_size']),
                    bias=False if hparams[f'layer_{n}_batchnorm'] else True)
            ]

            if hparams[f'layer_{n}_batchnorm']:
                l.append(nn.BatchNorm1d(int(hparams[f'layer_{n}_size'])))

            if hparams[f'layer_{n}_dropout']:
                l.append(nn.Dropout(hparams[f'layer_{n}_dropout']))

            if hparams[f'layer_{n}_activation'] == 'Tanh':
                l.append(nn.Tanh())
            elif hparams[f'layer_{n}_activation'] == 'ReLU':
                l.append(nn.ReLU())
            else:
                raise NotImplementedError(
                    'Please add this activation to the model class.'
                )

            layers.append(nn.Sequential(*l))
            input_dim = int(hparams[f'layer_{n}_size'])

        self.hidden_layers = nn.Sequential(*layers)

        if hparams['loss'] == 'mse':
            self.loss_fun = F.mse_loss
        elif hparams['loss'] == 'huber':
            self.loss_fun = F.smooth_l1_loss
        else:
            raise NotImplementedError(
                'Please add this loss function to the model class.'
            )

    @property
    @abstractmethod
    def hparams(self):
        pass

    @property
    def domain(self) -> Domain:
        # TODO
        raise NotImplementedError('GlassPy error: not implemented.')

    def is_within_domain():
        # TODO
        raise NotImplementedError('GlassPy error: not implemented.')

    def get_training_dataset(self):
        # TODO
        raise NotImplementedError('GlassPy error: not implemented.')

    def get_validation_dataset(self):
        # TODO
        raise NotImplementedError('GlassPy error: not implemented.')

    def get_test_dataset(self):
        # TODO
        raise NotImplementedError('GlassPy error: not implemented.')

    def distance_from_training(self):
        # TODO
        raise NotImplementedError('GlassPy error: not implemented.')

    def configure_optimizers(self):
        if 'optimizer' not in self.hparams:
            optimizer = SGD(self.parameters())

        elif self.hparams['optimizer'] == 'SGD':
            optimizer = SGD( 
                self.parameters(),
                lr=self.hparams['lr'],
                momentum=self.hparams['momentum'],
            )

        elif self.hparams['optimizer'] == 'Adam':
            optimizer = Adam( 
                self.parameters(),
                lr=self.hparams['lr'],
            )

        elif self.hparams['optimizer'] == 'AdamW':
            optimizer = AdamW( 
                self.parameters(),
                lr=self.hparams['lr'],
            )

        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fun(self(x), y)
        return {'loss': loss,}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fun(self(x), y)
        return {'val_loss_step': loss,}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss_step"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)


class RefractiveIndexOxide2010(PredictElementalInputTF):
    model_id = 'nn_nd_oxide_20-10'

    def __init__(self):
        super().__init__()

        model_path = _basemodelpath / (self.model_id + '.h5')
        info_path = _basesupportpath / (self.model_id + '.p')

        features, domain, x_mean, x_std, y_mean, y_std = \
            pickle.load(open(info_path, "rb"))

        self.info_dict = {
            'domain_el': domain,
            'features': features,
            'input': 'elements',
            'output': 'refractive index',
            'unit': '',
            'x_mean': x_mean,
            'x_std': x_std,
            'y_mean': y_mean,
            'y_std': y_std,
            'model': tf.keras.models.load_model(model_path),
        }

    @property
    def info(self):
        return self.info_dict

    def convert_input(self, x):
        return tf.convert_to_tensor(x, dtype=tf.float32)


class AbbeNumberOxide2010(PredictElementalInputTF):
    model_id = 'nn_abbe_oxide_20-10'

    def __init__(self):
        super().__init__()

        model_path = _basemodelpath / (self.model_id + '.h5')
        info_path = _basesupportpath / (self.model_id + '.p')

        features, domain, x_mean, x_std, y_mean, y_std = \
            pickle.load(open(info_path, "rb"))

        self.info_dict = {
            'domain_el': domain,
            'features': features,
            'input': 'elements',
            'output': 'Abbe number',
            'unit': '',
            'x_mean': x_mean,
            'x_std': x_std,
            'y_mean': y_mean,
            'y_std': y_std,
            'model': tf.keras.models.load_model(model_path),
        }

    @property
    def info(self):
        return self.info_dict

    def convert_input(self, x):
        return tf.convert_to_tensor(x, dtype=tf.float32)


class GlassTransitionOxide2010(PredictElementalInputTF):
    model_id = 'nn_Tg_oxide_20-10'

    def __init__(self):
        super().__init__()

        model_path = _basemodelpath / (self.model_id + '.h5')
        info_path = _basesupportpath / (self.model_id + '.p')

        features, domain, x_mean, x_std, y_mean, y_std = \
            pickle.load(open(info_path, "rb"))

        self.info_dict = {
            'domain_el': domain,
            'features': features,
            'input': 'elements',
            'output': 'glass transtion',
            'unit': 'K',
            'x_mean': x_mean,
            'x_std': x_std,
            'y_mean': y_mean,
            'y_std': y_std,
            'model': tf.keras.models.load_model(model_path),
        }

    @property
    def info(self):
        return self.info_dict

    def convert_input(self, x):
        return tf.convert_to_tensor(x, dtype=tf.float32)


class ViscNet(MLP):

    parameters_range = {
        'log_eta_inf': [-18, 5],
        'Tg': [400, 1400],
        'm': [10, 130],
    }

    hparams = {
        'batch_size': 64,
        'layer_1_activation': 'ReLU',
        'layer_1_batchnorm': False,
        'layer_1_dropout': 0.07942161101271952,
        'layer_1_size': 192,
        'layer_2_activation': 'Tanh',
        'layer_2_batchnorm': False,
        'layer_2_dropout': 0.05371454289414608,
        'layer_2_size': 48,
        'loss': 'mse',
        'lr': 0.0011695226458761677,
        'max_epochs': 500,
        'n_features': 35,
        'num_layers': 2,
        'optimizer': 'AdamW',
        'patience': 9,
    }

    absolute_features = [
        ('ElectronAffinity', 'std'),
        ('FusionEnthalpy', 'std'),
        ('GSenergy_pa', 'std'),
        ('GSmagmom', 'std'),
        ('NdUnfilled', 'std'),
        ('NfValence', 'std'),
        ('NpUnfilled', 'std'),
        ('atomic_radius_rahm', 'std'),
        ('c6_gb', 'std'),
        ('lattice_constant', 'std'),
        ('mendeleev_number', 'std'),
        ('num_oxistates', 'std'),
        ('nvalence', 'std'),
        ('vdw_radius_alvarez', 'std'),
        ('vdw_radius_uff', 'std'),
        ('zeff', 'std'),
    ]

    weighted_features = [
        ('FusionEnthalpy', 'min'),
        ('GSbandgap', 'max'),
        ('GSmagmom', 'mean'),
        ('GSvolume_pa', 'max'),
        ('MiracleRadius', 'std'),
        ('NValence', 'max'),
        ('NValence', 'min'),
        ('NdUnfilled', 'max'),
        ('NdValence', 'max'),
        ('NsUnfilled', 'max'),
        ('SpaceGroupNumber', 'max'),
        ('SpaceGroupNumber', 'min'),
        ('atomic_radius', 'max'),
        ('atomic_volume', 'max'),
        ('c6_gb', 'max'),
        ('c6_gb', 'min'),
        ('max_ionenergy', 'min'),
        ('num_oxistates', 'max'),
        ('nvalence', 'min'),
    ]

    x_mean = [5.7542e+01, 2.2090e+01, 2.0236e+00, 3.6861e-02, 3.2621e-01,
              1.4419e+00, 2.0165e+00, 3.4408e+01, 1.2353e+03, 1.4793e+00,
              4.2045e+01, 8.4131e-01, 2.3045e+00, 4.7985e+01, 5.6984e+01,
              1.1146e+00, 9.2186e-02, 2.1363e-01, 2.2581e-04, 5.8150e+00,
              1.2964e+01, 3.7008e+00, 1.3743e-01, 1.8370e-02, 3.2303e-01,
              7.1325e-02, 5.0019e+01, 4.3720e+00, 3.6446e+01, 8.4037e+00,
              2.0281e+02, 7.5614e+00, 1.2259e+02, 6.7183e-01, 1.0508e-01]

    x_std = [7.6421e+00, 4.7181e+00, 4.5828e-01, 1.6873e-01, 9.7033e-01,
             2.7695e+00, 3.3153e-01, 6.4521e+00, 6.3392e+02, 4.0606e-01,
             1.1777e+01, 2.8130e-01, 7.9214e-01, 7.5883e+00, 1.1335e+01,
             2.8823e-01, 4.4787e-02, 1.1219e-01, 1.2392e-03, 1.1634e+00,
             2.9514e+00, 4.7246e-01, 3.1958e-01, 8.8973e-02, 6.7548e-01,
             6.2869e-02, 1.0004e+01, 2.7434e+00, 1.9245e+00, 3.4735e-01,
             1.2475e+02, 3.2668e+00, 1.5287e+02, 7.3511e-02, 1.6188e-01]

    state_dict_path = _basemodelpath / 'ViscNet_SD.p' 

    def __init__(self):
        super().__init__()

        input_dim = int(self.hparams[f'layer_{self.hparams["num_layers"]}_size'])

        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, len(self.parameters_range)),
            nn.Sigmoid(),
        )

        state_dict = pickle.load(open(self.state_dict_path, 'rb'))
        self.load_state_dict(state_dict)

        self.x_mean = torch.tensor(self.x_mean).float()
        self.x_std = torch.tensor(self.x_std).float()

    def log_viscosity_fun(self, T, log_eta_inf, Tg, m):
        log_viscosity = log_eta_inf + (12 - log_eta_inf)*(Tg / T) * \
            ((m / (12 - log_eta_inf) - 1) * (Tg / T - 1)).exp()
        return log_viscosity

    def viscosity_parameters_from_tensor(
            self,
            feature_tensor,
            return_tensor=False,
    ):

        xf = self.hidden_layers((feature_tensor - self.x_mean) / self.x_std)
        xf = self.output_layer(xf)

        parameters = {}

        for i, (p_name, p_range) in enumerate(self.parameters_range.items()):
            # Scaling the viscosity parameters to be within the parameter range
            parameters[p_name] = torch.add(
                torch.ones(xf.shape[0]).mul(p_range[0]),
                xf[:,i],
                alpha=p_range[1] - p_range[0],
            )

        if not return_tensor:
            parameters = {k: v.detach().numpy() for k,v in parameters.items()}

        return parameters

    def viscosity_parameters(
            self,
            composition: CompositionLike,
            input_cols: List[str] = [],
    ):
        features = self.featurizer(composition, input_cols)
        features = torch.from_numpy(features).float()
        parameters = self.viscosity_parameters_from_tensor(features, False)
        return parameters

    def viscosity_parameters_dist(
            self,
            composition: CompositionLike,
            input_cols: List[str] = [],
            num_samples: int = 100,
    ):
        features = self.featurizer(composition, input_cols)
        features = torch.from_numpy(features).float()

        is_training = self.training
        if not is_training:
            self.train()

        dist = defaultdict(list)
        with torch.no_grad():
            for _ in range(num_samples):
                pdict = self.viscosity_parameters_from_tensor(features, False)
                for k,v in pdict.items():
                    dist[k].append(v)

        if not is_training:
            self.eval()

        dist = {k: np.array(v).T for k,v in dist.items()}

        return dist

    def viscosity_parameters_unc(
            self,
            composition: CompositionLike,
            input_cols: List[str] = [],
            confidence: float = 0.95, 
            num_samples: int = 100,
    ):
        q = [(100 - 100 * confidence) / 2, 100 - (100 - 100 * confidence) / 2]
        dist = self.viscosity_parameters_dist(composition, input_cols,
                                              num_samples)
        bands = {k: np.percentile(v, q, axis=1).T for k,v in dist.items()}
        return bands

    def forward(self, x):
        T = x[:, -1].detach().clone()
        parameters = self.viscosity_parameters_from_tensor(x[:, :-1], True)
        log_viscosity = self.log_viscosity_fun(T, **parameters)
        return log_viscosity

    def featurizer(
            self,
            composition: CompositionLike,
            input_cols: List[str] = [],
    ):
        (feat_array,
         feat_names) = featurizer.extract_chem_feats(composition, input_cols,
                                                     self.weighted_features,
                                                     self.absolute_features, 1)
        feat_idx = (
            list(range(
                len(self.weighted_features),
                len(self.weighted_features) + len(self.absolute_features)
            )) + list(range(len(self.weighted_features)))
        )
        feat_array = feat_array[:,feat_idx]
        return feat_array

    def predict(
            self,
            T,
            composition: CompositionLike,
            input_cols: List[str] = [],
    ):
        features = self.featurizer(composition, input_cols)
        x = np.concatenate((features, T), axis=1)
        x = torch.from_numpy(x).float()
        log_viscosity = self(x).detach().numpy()
        return log_viscosity

    def predict_log10_viscosity(
            self,
            T,
            composition: CompositionLike,
            input_cols: List[str] = [],
    ):
        return self.predict(T, composition, input_cols)

    def predict_viscosity(
            self,
            T,
            composition: CompositionLike,
            input_cols: List[str] = [],
    ):
        return 10**self.predict(T, composition, input_cols)

    def predict_fragility(
            self,
            composition: CompositionLike,
            input_cols: List[str] = [],
    ):
        parameters = self.viscosity_parameters(composition, input_cols)
        fragility = parameters['m']
        return fragility

    def predict_Tg(
            self,
            composition: CompositionLike,
            input_cols: List[str] = [],
    ):
        parameters = self.viscosity_parameters(composition, input_cols)
        Tg = parameters['Tg']
        return Tg

    def predict_log10_eta_infinity(
            self,
            composition: CompositionLike,
            input_cols: List[str] = [],
    ):
        parameters = self.viscosity_parameters(composition, input_cols)
        log_eta_inf = parameters['log_eta_inf']
        return log_eta_inf

    def predict_eta_infinity(
            self,
            composition: CompositionLike,
            input_cols: List[str] = [],
    ):
        return 10**self.predict_log10_eta_infinity(composition, input_cols)

    def predict_Tg_unc(
            self,
            composition: CompositionLike,
            input_cols: List[str] = [],
            confidence: float = 0.95, 
            num_samples: int = 100,
    ):
        parameters_unc = self.viscosity_parameters_unc(composition, input_cols,
                                                       confidence, num_samples)
        Tg = parameters_unc['Tg']
        return Tg

    def predict_fragility_unc(
            self,
            composition: CompositionLike,
            input_cols: List[str] = [],
            confidence: float = 0.95, 
            num_samples: int = 100,
    ):
        parameters_unc = self.viscosity_parameters_unc(composition, input_cols,
                                                       confidence, num_samples)
        m = parameters['m']
        return m

    def predict_log10_eta_infinity_unc(
            self,
            composition: CompositionLike,
            input_cols: List[str] = [],
            confidence: float = 0.95, 
            num_samples: int = 100,
    ):
        parameters_unc = self.viscosity_parameters_unc(composition, input_cols,
                                                       confidence, num_samples)
        log_eta_inf = parameters_unc['log_eta_inf']
        return log_eta_inf

    def predict_viscosity_unc(
            self,
            T: Union[float, List[float], np.ndarray],
            composition: CompositionLike,
            input_cols: List[str] = [],
            confidence: float = 0.95, 
            num_samples: int = 100,
            table_mode: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.array]]:
        '''Compute the confidence bands of the viscosity prediction.

        Args:
          T:
            Temperature to compute the viscosity. If this is a numpy array, it
            must have only one dimension.
          composition:
            Any composition like object.
          input_cols:
            List of strings representing the chemical entities related to each
            column of 'composition'. Necessary only when 'composition' is a list
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

        Returns
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
          dist:
            Dictionary with the distribution of the viscosity parameters.

        '''
        dist = self.viscosity_parameters_dist(composition, input_cols,
                                              num_samples)
        parameters = []
        for param in self.parameters_range.keys():
            parameters.append(dist[param])
        parameters = np.array(parameters)

        num_compositions = parameters.shape[1]

        if isinstance(T, Iterable):
            if len(T) == num_compositions and table_mode:
                all_curves = np.zeros((1, num_compositions, num_samples))
                T = torch.tensor(T).float()
                for i in range(num_samples):
                    params = [
                        torch.from_numpy(p).float()
                        for p in parameters[:,:,i]
                    ]
                    viscosity = self.log_viscosity_fun(T, *params).numpy()
                    all_curves[:,:,i] = viscosity

            else:
                all_curves = np.zeros((len(T), num_compositions, num_samples))
                T = torch.tensor(T).float()
                for i in range(num_samples):
                    params = [
                        torch.from_numpy(
                            np.broadcast_to(p, (len(T), len(p))).T
                        ).float()
                        for p in parameters[:,:,i]
                    ]
                    viscosity = self.log_viscosity_fun(T, *params).numpy()
                    all_curves[:,:,i] = viscosity.T

        else:
            all_curves = np.zeros((1, num_compositions, num_samples))
            T = torch.tensor(T).float()
            for i in range(num_samples):
                params = [torch.from_numpy(p).float() for p in parameters[:,:,i]]
                viscosity = self.log_viscosity_fun(T, *params).numpy()
                all_curves[:,:,i] = viscosity

        q = [(100 - 100 * confidence) / 2, 100 - (100 - 100 * confidence) / 2]

        bands = np.percentile(all_curves, q, axis=2)
        bands = np.transpose(bands, (2, 0, 1))

        if num_compositions == 1:
            bands = bands[0,:,:]

        param_bands = {k: np.percentile(v, q, axis=1).T for k,v in dist.items()}

        return bands, param_bands, dist


class ViscNetHuber(ViscNet):
    def __init__(self):
        super().__init__()

        self.hparams['loss'] = 'huber'
        self.loss_fun = F.smooth_l1_loss

        state_dict_path = _basemodelpath / 'ViscNetHuber_SD.p' 
        state_dict = pickle.load(open(state_dict_path, 'rb'))
        self.load_state_dict(state_dict)


class ViscNetVFT(ViscNet):
    def __init__(self):
        super().__init__()

        state_dict_path = _basemodelpath / 'ViscNetVFT_SD.p' 
        state_dict = pickle.load(open(state_dict_path, 'rb'))
        self.load_state_dict(state_dict)

    def log_viscosity_fun(self, T, log_eta_inf, Tg, m):
        log_viscosity = log_eta_inf + (12 - log_eta_inf)**2 / \
            (m * (T / Tg - 1) + (12 - log_eta_inf))
        return log_viscosity
