'''Classes for regression of nucleation density data.'''

import pandas as pd
import numpy as np
from numpy import log10
from scipy.stats import theilslopes
from scipy.constants import pi
from lmfit import Model, Parameters

from .density import wakeshima, kashchiev, shneidman


class _BaseDensityRegression:
    '''
    Base class for density regression.

    Parameters
    ----------

    table : pandas DataFrame, optional, must be a named argument
        DataFrame with a 'time' and a 'density' column. If an 'ID' column is
        also present, then each unique ID is considered a separate dataset. A
        RuntimeError is raised if the class is initiated without a table *or*
        without both time and density arguments.

    time : array_like, optional, must be a named argument
        Elapsed time. A RuntimeError is raised if the class is initiated
        without a table *or* without both time and density arguments. If
        'table' is given then this argument is ignored.

    density : array_like, optional, must be a named argument
        Nuclei density. A RuntimeError is raised if the class is initiated
        without a table *or* without both time and density arguments. If
        'table' is given then this argument is ignored.

    ID : array_like, optional, must be a named argument
        The ID for each of the 'time' and 'density' data. Each unique ID is
        treated as a separate dataset. If 'table' is given then this argument
        is ignored.
    '''
    def __init__(self, **kwargs):

        if 'table' in kwargs:

            table = kwargs['table']
            columns = table.columns.values

            if 'time' not in columns:
                name = self.__name__
                msg = f'The {name} class was initiated with a table with no "time" column'
                raise RuntimeError(msg)

            if 'density' not in columns:
                name = self.__name__
                msg = f'The {name} class was initiated with a table with no "density" column'
                raise RuntimeError(msg)

            if 'ID' not in columns:
                table['ID'] = np.zeros(len(table)).astype(bool)

        elif 'time' in kwargs and 'density' in kwargs:

            table = pd.DataFrame({
                'time': kwargs['time'],
                'density': kwargs['density'],
            })

            if 'ID' in kwargs:
                table['ID'] = kwargs['ID']
            else:
                table['ID'] = np.zeros(len(table)).astype(bool)

            if 'temperature' in kwargs:
                table['temperature'] = kwargs['temperature']

        else:
            name = self.__name__
            msg = f'The {name} class was initiated with insuficient arguments'
            raise RuntimeError(msg)

        self.table = table

    def dataGenerator(self):
        '''
        Generator that yields each individual dataset

        Yields
        ------
        ID
            The identification of the yielded dataset

        time : array_like
            Elapsed time of the yielded dataset

        density : array_like
            Nuclei density of the yielded dataset

        table_slice : pandas DataFrame
            An instance of the table DataFrame of the yielded dataset
        '''
        for ID in self.table['ID'].unique():

            logic = self.table['ID'] == ID
            table_slice = self.table[logic]
            time = table_slice['time'].values
            density = table_slice['density'].values

            yield ID, time, density, table_slice

    def guess(self, time, density):
        '''
        Guess the steady-state nucleation rate and induction time

        The guess is based on a robust linear regression of the density over
        time data. Guesses are useful for non-linear regressions.

        Parameters
        ----------
        time : array_like
            Elapsed time.

        density : array_like
            Nuclei density.

        Returns
        -------
        guess_rate : float
            Guess for the steady-state nucleatio rate for the yielded dataset

        guess_induction_time : float
            Guess for the induction time for the yielded dataset
        '''
        slope, intercept, _, _ = theilslopes(x=time, y=density)
        guess_rate, guess_induction_time = slope, -intercept / slope

        if guess_induction_time < 1e-3:
            guess_induction_time = 1e-3

        return guess_rate, guess_induction_time

    def dataGeneratorWithGuess(self):
        '''
        Generate individual datasets and guesses for nucl. rate and induct. time

        Yields
        ------
        ID
            The identification of the yielded dataset

        time : array_like
            Elapsed time of the yielded dataset

        density : array_like
            Nuclei density of the yielded dataset

        guess_rate : float
            Guess for the steady-state nucleatio rate for the yielded dataset

        guess_induction_time : float
            Guess for the induction time for the yielded dataset
        '''
        for ID in self.table['ID'].unique():

            logic = self.table['ID'] == ID
            table_slice = self.table[logic]
            time = table_slice['time'].values
            density = table_slice['density'].values

            guess_rate, guess_induction_time = self.guess(time, density)

            yield ID, time, density, guess_rate, guess_induction_time

    def _fit(self,
             model,
             time,
             density,
             density_weights=None,
             params=None,
             fitmethod='leastsq'):
        '''
        Regression of nucleation density data.

        Parameters
        ----------
        model : instance of lmfit's Model class
            Model to fit the data. The independent variable must be named
            'time'.

        time : array_like
            Elapsed time.

        density : array_like
            Nuclei density.

        density_weights : array_like or None, optional
            The weights of density to use during the regression. Default value
            is None.

        params : instance of lmfit's Parameters class or None, optional
            Optional Parameters instance to pass to the fit function. If None
            then the model will generate the Parameters class during fitting.
            Default value is None.

        fitmethod : str, optional
            Method to use for the regression. See lmfit's documentation for
            more information. Default value is 'leastsq'.

        Returns
        -------
        fitresult : instance of lmfit's ModelResult class
            Result of the regression. See lmfit for documentation on the
            ModelResult class.
        '''
        fitresult = model.fit(density,
                              time=time,
                              method=fitmethod,
                              params=params,
                              weights=density_weights,
                              nan_policy='propagate')

        return fitresult

    def _fitBrute(self,
                  model,
                  time,
                  density,
                  time_weights=None,
                  params=None,
                  fitmethod='leastsq'):
        '''
        Brute-force regression of nucleation density data.

        Some combinations of dataset and models are quite difficult to fit.
        This method is a brute-force approach to find an answer that converges.

        Parameters
        ----------
        model : instance of lmfit's Model class
            Model to fit the data. The independent variable must be named
            'time'.

        time : array_like
            Elapsed time.

        density : array_like
            Nuclei density.

        density_weights : array_like or None, optional
            The weights of density to use during the regression. Default value
            is None.

        params : instance of lmfit's Parameters class or None, optional
            Optional Parameters instance to pass to the fit function. If None
            then the model will generate the Parameters class during fitting.
            Default value is None.

        fitmethod : str, optional
            Method to use for the final regression. See lmfit's documentation
            for more information. Default value is 'leastsq'.

        Returns
        -------
        fitresult : instance of lmfit's ModelResult class
            Result of the regression. See lmfit for documentation on the
            ModelResult class.
        '''
        fitresult = self._fit(model,
                              time,
                              density,
                              time_weights,
                              params,
                              fitmethod='brute')

        fitresult = self._fit(model,
                              time,
                              density,
                              time_weights,
                              fitresult.params,
                              fitmethod='differential_evolution')

        fitresult = self._fit(model,
                              time,
                              density,
                              time_weights,
                              fitresult.params,
                              fitmethod=fitmethod)

        return fitresult

    def __confidenceInterval(self, fitresult, ci_names, ci_sigmas=(0.95, )):
        # TODO
        pass


class Wakeshima(_BaseDensityRegression):
    '''
    Class for density regression considering the Wakeshima equation.

    Parameters
    ----------

    table : pandas DataFrame, optional, must be a named argument
        DataFrame with a 'time' and a 'density' column. If an 'ID' column is
        also present, then each unique ID is considered a separate dataset. A
        RuntimeError is raised if the class is initiated without a table *or*
        without both time and density arguments.

    time : array_like, optional, must be a named argument
        Elapsed time. A RuntimeError is raised if the class is initiated
        without a table *or* without both time and density arguments. If
        'table' is given then this argument is ignored.

    density : array_like, optional, must be a named argument
        Nuclei density. A RuntimeError is raised if the class is initiated
        without a table *or* without both time and density arguments. If
        'table' is given then this argument is ignored.

    ID : array_like, optional, must be a named argument
        The ID for each of the 'time' and 'density' data. Each unique ID is
        treated as a separate dataset. If 'table' is given then this argument
        is ignored.

    References
    ----------
    [1] Wakeshima, H. (1954). Time Lag in the Self‐Nucleation. The Journal of
        Chemical Physics 22, 1614–1615.
    '''
    def getModel(self, guess_rate, guess_induction_time):
        '''
        Creates a model for regression of the Wakeshima equation.

        Parameters
        ----------
        guess_rate : float
            Guess for the steady-state nucleation rate.

        guess_induction_time : float
            Guess for the nucleation induction time.

        Returns
        -------
        model : instance of lmfit's Model class
            Wakeshima model to fit experimental data.
        '''
        model = Model(wakeshima)

        model.set_param_hint('log_rate',
                             vary=True,
                             max=25,
                             value=log10(guess_rate))
        model.set_param_hint('steady_state_rate',
                             vary=False,
                             expr=r'10**log_rate')
        model.set_param_hint('log_induction_time',
                             vary=True,
                             value=log10(guess_induction_time))
        model.set_param_hint('time_lag',
                             vary=False,
                             expr=r'10**log_induction_time')
        model.set_param_hint('induction_time',
                             vary=False,
                             expr=r'10**log_induction_time')

        return model

    def fit(self,
            time,
            density,
            density_weights=None,
            params=None,
            fitmethod='leastsq'):
        '''
        Regression of nucleation density data.

        Parameters
        ----------
        time : array_like
            Elapsed time.

        density : array_like
            Nuclei density.

        density_weights : array_like or None, optional
            The weights of density to use during the regression. Default value
            is None.

        params : instance of lmfit's Parameters class or None, optional
            Optional Parameters instance to pass to the fit function. If None
            then the model will generate the Parameters class during fitting.
            Default value is None.

        fitmethod : str, optional
            Method to use for the regression. See lmfit's documentation for
            more information. Default value is 'leastsq'.

        Returns
        -------
        fitresult : instance of lmfit's ModelResult class
            Result of the regression. See lmfit for documentation on the
            ModelResult class.

        model : instance of lmfit's Model class
            Wakeshima model to fit experimental data.
        '''
        guess_rate, guess_induction_time = self.guess(time, density)
        model = self.getModel(guess_rate, guess_induction_time)
        fitresult = self._fit(model, time, density, density_weights, params,
                              fitmethod)

        return fitresult, model


class Kashchiev(_BaseDensityRegression):
    '''
    Class for density regression considering the Kashchiev equation.

    Parameters
    ----------

    table : pandas DataFrame, optional, must be a named argument
        DataFrame with a 'time' and a 'density' column. If an 'ID' column is
        also present, then each unique ID is considered a separate dataset. A
        RuntimeError is raised if the class is initiated without a table *or*
        without both time and density arguments.

    time : array_like, optional, must be a named argument
        Elapsed time. A RuntimeError is raised if the class is initiated
        without a table *or* without both time and density arguments. If
        'table' is given then this argument is ignored.

    density : array_like, optional, must be a named argument
        Nuclei density. A RuntimeError is raised if the class is initiated
        without a table *or* without both time and density arguments. If
        'table' is given then this argument is ignored.

    ID : array_like, optional, must be a named argument
        The ID for each of the 'time' and 'density' data. Each unique ID is
        treated as a separate dataset. If 'table' is given then this argument
        is ignored.

    summation_ub : int, optional
        Upper boundary of the infinite summation. It is advisable to choose an
        even integer. Default value is 1000.

    use_time_shift : bool, optional
        If False then the original Kashchiev equation is used (equivalent to
        setting the time_shift to zero). If True then the expression with a
        time-shift will be considered for the regression. The time-shift is a
        parameter related to the dissolution of supercritical nuclei in a
        double-stage nucleation treatment. Default value is False.

    References
    ----------
    [1] Kashchiev, D. (1969). Solution of the non-steady state problem in
        nucleation kinetics. Surface Science 14, 209–220.

    [2] Kashchiev, D. (2000). Nucleation basic theory with applications.
    '''
    def __init__(self, **kwargs):
        _BaseDensityRegression.__init__(self, **kwargs)
        self.summation_ub = kwargs.get('summation_ub', 1000)
        self.use_time_shift = kwargs.get('use_time_shift', False)

    def __str__(self):
        if self.use_time_shift:
            time_shift_info = ' considering the time-shift'
        else:
            time_shift_info = ''
        return f'Kaschiev{time_shift_info} with {self.summation_ub} summation terms'

    def getModel(self, guess_rate, guess_induction_time, time=None):
        '''
        Creates a model for regression of the Kashchiev equation.

        Parameters
        ----------
        guess_rate : float
            Guess for the steady-state nucleation rate.

        guess_induction_time : float
            Guess for the nucleation induction time.

        time : array_like, optional
            Elapsed time. Only necessary if self.use_time_shift is True. Used
            to give a maximum value for the time_shift.

        Returns
        -------
        model : instance of lmfit's Model class
            Kashchiev model to fit experimental data.
        '''
        model = Model(kashchiev)

        model.set_param_hint('log_rate',
                             vary=True,
                             min=5,
                             max=25,
                             value=log10(guess_rate),
                             brute_step=1)
        model.set_param_hint('steady_state_rate',
                             vary=False,
                             expr=r'10**log_rate')
        model.set_param_hint('log_time_lag',
                             vary=True,
                             min=-1,
                             max=9,
                             value=log10(guess_induction_time * 6 / pi**2),
                             brute_step=0.5)
        model.set_param_hint('time_lag', vary=False, expr=r'10**log_time_lag')
        model.set_param_hint('summation_ub',
                             vary=False,
                             value=self.summation_ub)

        if self.use_time_shift:
            time_shift_guess = min(min(time), guess_induction_time) / 2

            model.set_param_hint('log_time_shift',
                                 vary=True,
                                 min=-1,
                                 max=log10(max(time)),
                                 value=log10(time_shift_guess),
                                 brute_step=0.5)
            model.set_param_hint('time_shift',
                                 vary=False,
                                 expr=r'10**log_time_shift')
        else:
            model.set_param_hint('time_shift', vary=False, value=0)

        model.set_param_hint('induction_time',
                             vary=False,
                             expr='time_lag*pi**2/6 + time_shift')
        return model

    def fit(self,
            time,
            density,
            density_weights=None,
            params=None,
            fitmethod='leastsq',
            time_shift_threshold=0,
            time_lag_threshold=0):
        '''
        Regression of nucleation density data.

        Parameters
        ----------
        time : array_like
            Elapsed time.

        density : array_like
            Nuclei density.

        density_weights : array_like or None, optional
            The weights of density to use during the regression. Default value
            is None.

        params : instance of lmfit's Parameters class or None, optional
            Optional Parameters instance to pass to the fit function. If None
            then the model will generate the Parameters class during fitting.
            Default value is None.

        fitmethod : str, optional
            Method to use for the regression. See lmfit's documentation for
            more information. Default value is 'leastsq'.

        time_shift_threshold : float, optional
            If the time_shift obtained from the regression is lower than the
            threshold value, then the time_shift is fixed at zero and a new
            regression is done. Default value is zero.

        time_shift_threshold : float, optional
            If the time_lag obtained from the regression is lower than the
            threshold value, then the time_lag is fixed at zero and a new
            regression is done. Default value is zero.

        Notes
        -----
        When considering the Kashchiev equation with time-shift, the fitting
        procedure checks if the standard deviation of the time-shift is greater
        than the actual value of the time-shift. If so, then the time-shift is
        forced to zero.

        Returns
        -------
        fitresult : instance of lmfit's ModelResult class
            Result of the regression. See lmfit for documentation on the
            ModelResult class.

        model : instance of lmfit's Model class
            Kashchiev model to fit experimental data.
        '''
        guess_rate, guess_induction_time = self.guess(time, density)
        model = self.getModel(guess_rate, guess_induction_time, time)
        fitresult = self._fit(model, time, density, density_weights, params,
                              fitmethod)

        if self.use_time_shift:
            fitparams = fitresult.params
            time_shift = fitparams['time_shift'].value
            time_shift_std = fitparams['time_shift'].stderr
            if time_shift < time_shift_threshold or time_shift_std > time_shift:
                fitparams.add('time_shift', vary=False, value=0)
                fitparams.add('log_time_shift', vary=False, value=-np.inf)
                fitresult = self._fit(model, time, density, density_weights,
                                      fitparams, fitmethod)

        fitparams = fitresult.params
        time_lag = fitparams['time_lag'].value
        time_lag_std = fitparams['time_lag'].stderr
        if time_lag < time_lag_threshold:
            fitparams.add('time_lag', vary=False, value=0)
            fitparams.add('log_time_lag', vary=False, value=-np.inf)
            fitresult = self._fit(model, time, density, density_weights,
                                  fitparams, fitmethod)

        return fitresult, model


class __Shneidman(_BaseDensityRegression):
    '''
    Class for density regression considering the Shneidman equation.

    Parameters
    ----------

    table : pandas DataFrame, optional, must be a named argument
        DataFrame with a 'time' and a 'density' column. If an 'ID' column is
        also present, then each unique ID is considered a separate dataset. A
        RuntimeError is raised if the class is initiated without a table *or*
        without both time and density arguments.

    time : array_like, optional, must be a named argument
        Elapsed time. A RuntimeError is raised if the class is initiated
        without a table *or* without both time and density arguments. If
        'table' is given then this argument is ignored.

    density : array_like, optional, must be a named argument
        Nuclei density. A RuntimeError is raised if the class is initiated
        without a table *or* without both time and density arguments. If
        'table' is given then this argument is ignored.

    ID : array_like, optional, must be a named argument
        The ID for each of the 'time' and 'density' data. Each unique ID is
        treated as a separate dataset. If 'table' is given then this argument
        is ignored.

    detectable_radius : float, optional
        The size of the detectable radius. See ref. [1] for more details. The
        default value is 1e-6.

    time_ratio_fun : function or None, optional
        The function to compute the ratio between the incubation time and the
        time-lag. The function must accept two key arguments: steady_state_rate
        and time_lag. If, instead, time_ratio_fun is None then the adjustable
        parameters will be considered free and independent. Default value is
        None.

    References
    ----------
    [1] Shneidman, V.A. (1988). Establishment of a steady-state nucleation
        regime. Theory and comparison with experimental data for glasses. Sov.
        Phys. Tech. Phys. 33, 1338–1342.
    '''
    def __init__(self, **kwargs):
        _BaseDensityRegression.__init__(self, **kwargs)
        self.detectable_radius = kwargs.get('detectable_radius', 1e-6)
        self.time_ratio_fun = kwargs.get('time_ratio_fun', None)

    def __str__(self):
        if self.time_ratio_fun:
            return 'Shneidman with 2 free adjustable parameteres'
        else:
            return 'Shneidman with 3 free adjustable parameters'

    def getModel(self, guess_rate, guess_induction_time):
        '''
        Creates a model for regression of the Shneidman equation.

        Parameters
        ----------
        guess_rate : float
            Guess for the steady-state nucleation rate.

        guess_induction_time : float
            Guess for the nucleation induction time.

        time : array_like, optional
            Elapsed time. Only necessary if self.use_time_shift is True. Used
            to give a maximum value for the time_shift.

        Returns
        -------
        model : instance of lmfit's Model class
            Kashchiev model to fit experimental data.
        '''
        model = Model(shneidman)

        params = Parameters(
            usersyms={
                'euler_gamma': np.euler_gamma,
                'time_ratio_fun': self.time_ratio_fun,
                'detectable_radius': self.detectable_radius,
            })

        params.add('log_rate',
                   vary=True,
                   min=5,
                   max=25,
                   value=log10(guess_rate),
                   brute_step=1)
        params.add('steady_state_rate', vary=False, expr=r'10**log_rate')
        params.add('log_time_lag',
                   vary=True,
                   min=-1,
                   max=9,
                   value=log10(guess_induction_time / (2 * np.euler_gamma)),
                   brute_step=0.5)
        params.add('time_lag', vary=False, expr=r'10**log_time_lag')

        if self.time_ratio_fun:
            expr = r'time_lag*time_ratio_fun(steady_state_rate, time_lag, detectable_radius)'
            params.add('incubation_time', vary=False, expr=expr)

        else:
            params.add('log_incubation_time',
                       vary=True,
                       min=-1,
                       max=9,
                       value=log10(guess_induction_time / 2),
                       brute_step=0.5)
            params.add('incubation_time',
                       vary=False,
                       expr=r'10**log_incubation_time')

        params.add('induction_time',
                   vary=False,
                   expr=r'incubation_time + euler_gamma*time_lag')

        return model, params

    def fit(self,
            time,
            density,
            density_weights=None,
            params=None,
            fitmethod='leastsq',
            time_shift_threshold=0):
        '''
        Regression of nucleation density data.

        Parameters
        ----------
        time : array_like
            Elapsed time.

        density : array_like
            Nuclei density.

        density_weights : array_like or None, optional
            The weights of density to use during the regression. Default value
            is None.

        params : instance of lmfit's Parameters class or None, optional
            Optional Parameters instance to pass to the fit function. If None
            then the model will generate the Parameters class during fitting.
            Default value is None.

        fitmethod : str, optional
            Method to use for the regression. See lmfit's documentation for
            more information. Default value is 'leastsq'.

        time_shift_threshold : float, optional
            If the time_shift obtained from the regression is lower than the
            threshold value, then the time_shift is fixed at zero and a new
            regression is done. Default value is zero.

        Returns
        -------
        fitresult : instance of lmfit's ModelResult class
            Result of the regression. See lmfit for documentation on the
            ModelResult class.

        model : instance of lmfit's Model class
            Shneidman model to fit experimental data.
        '''
        guess_rate, guess_induction_time = self.guess(time, density)
        model, params_ = self.getModel(guess_rate, guess_induction_time)

        if params is None:
            params = params_

        fitresult = self._fit(model, time, density, density_weights, params,
                              fitmethod)

        fitparams = fitresult.params
        incubation_time = fitparams['incubation_time'].value
        if incubation_time < 1:
            fitparams.add('incubation_time', vary=False, value=0)
            fitparams.add('log_incubation_time', vary=False, value=-np.inf)
            fitresult = self._fit(model, time, density, density_weights,
                                  fitparams, fitmethod)

        fitparams = fitresult.params
        time_lag = fitparams['time_lag'].value
        if time_lag < 1:
            fitparams.add('time_lag', vary=False, value=0)
            fitparams.add('log_time_lag', vary=False, value=-np.inf)
            fitresult = self._fit(model, time, density, density_weights,
                                  fitparams, fitmethod)

        return fitresult, model
