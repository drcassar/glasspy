from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple
import os
import pickle
import typing

import numpy as np
try:
    import tensorflow as tf
except ModuleNotFoundError:
    print('WARNING: Some models require that tensorflow is installed.\n'
          '  It is not a hard dependency of GlassPy to reduce the '
          'risk of interfering\n  with your local copies')

from glasspy.chemistry import convert
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


class PredictElementalInputNN(Predict):
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
            x, elements = convert.any_to_element_array(x, input_cols)
            x = convert.wt_to_mol(x, elements, 1)
        else:
            x, elements = convert.any_to_element_array(x, input_cols,
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


class RefractiveIndexOxide2010(PredictElementalInputNN):
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


class AbbeNumberOxide2010(PredictElementalInputNN):
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


class GlassTransitionOxide2010(PredictElementalInputNN):
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

