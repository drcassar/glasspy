'''Classes for regression of viscosity data.'''

from abc import ABC, abstractmethod
from numpy import log
from scipy.constants import R
from scipy.misc import derivative as diff
from scipy.optimize import brentq
from scipy.stats import linregress
from pandas import DataFrame
from lmfit import Model

import glasspy.viscosity.equilibrium_log as eq


class _BaseViscosityRegression(ABC):
    '''
    Base class for viscosity regression.

    Parameters
    ----------
    verbose : boolean, optional
        If 'True' then the fitresults are printed when the regression is
        performed. Default value is True.

    autofit : boolean, optional
        'True' if the regression should be performed during the Class initiation.
        'False' otherwise. Default value is True.

    table : pandas DataFrame, optional, must be a named argument
        DataFrame with a 'temperature' column and a 'log_viscosity' column. A
        RuntimeError is raised if the class is initiated without a table *or*
        without both temperature and log_viscosity arguments.

    temperature : array_like, optional, must be a named argument
        Temperature in Kelvin. A RuntimeError is raised if the class is initiated
        without a table *or* without both time and density arguments. If
        'table' is given then this argument is ignored.

    log_viscosity : array_like, optional, must be a named argument
        Base-10 logarithm of viscosity. It is highly recommended to use
        viscosity in units of Pascal second. A RuntimeError is raised if the
        class is initiated without a table *or* without both time and density
        arguments. If 'table' is given then this argument is ignored.

    '''
    def __init__(self, verbose=True, **kwargs):
        super().__init__()

        self.verbose = verbose

        if 'table' in kwargs:

            table = kwargs['table']
            columns = table.columns.values

            if 'temperature' not in columns:
                name = self.__name__()
                msg = f'The {name} class was initiated with a table with no temperature data'
                raise RuntimeError(msg)

            if 'log_viscosity' not in columns:
                name = self.__name__()
                msg = f'The {name} class was initiated with a table with no log_viscosity data'
                raise RuntimeError(msg)

        elif 'temperature' in kwargs and 'log_viscosity' in kwargs:

            table = DataFrame({
                'temperature': kwargs['temperature'],
                'log_viscosity': kwargs['log_viscosity'],
            })

        else:
            name = self.__name__()
            msg = f'The {name} class was initiated with insuficient arguments'
            raise RuntimeError(msg)

        self.table = table

        if kwargs.get('autofit', True):
            self.fit()

    def guess(self):
        '''
        Guess T12, fragility, and the logarithm of infinite viscosity.

        Returns
        -------
        guess_T12 : float
            Guess for the temperature were the log_10 of viscosity is 12

        guess_fragility : float
            Guess for the fragility index

        guess_log_eta_inf : float
            Guess for the viscosity limit at infinite temperature.

        '''
        temperature = self.table['temperature']
        log_viscosity = self.table['log_viscosity']
        slope, intercept, _, _, _ = linregress(x=1 / temperature,
                                               y=log_viscosity)

        guess_T12 = slope / (12 - intercept)
        guess_T12 = guess_T12 if guess_T12 > 0 else min(temperature)
        guess_log_eta_inf = -3
        guess_fragility = 50

        model = Model(eq.MYEGA_alt)
        fitresult = model.fit(log_viscosity,
                              T=temperature,
                              log_eta_inf=guess_log_eta_inf,
                              T12=guess_T12,
                              m=guess_fragility)

        guess_T12 = fitresult.params['T12'].value
        guess_fragility = fitresult.params['m'].value
        guess_log_eta_inf = fitresult.params['log_eta_inf'].value

        return guess_T12, guess_fragility, guess_log_eta_inf

    @abstractmethod
    def getModel(self):
        pass

    @abstractmethod
    def fit(self, model, weights=None, params=None, fitmethod='leastsq',
            extra_computation=True):
        '''
        Regression of nucleation density data.

        Parameters
        ----------
        model : instance of lmfit's Model class
            Model to fit the data. The independent variable must be named 'T'.

        weights : array_like or None, optional
            The weights of log_viscosity to use during the regression. If None
            then no weights are applied. Default value is None.

        params : instance of lmfit's Parameters class or None, optional
            Optional Parameters instance to pass to the fit function. If None
            then the model will generate the Parameters class during fitting.
            Default value is None.

        fitmethod : str, optional
            Method to use for the regression. See lmfit's documentation for
            more information. Default value is 'leastsq'.

        extra_computation : bool, optional
            If True then T12 and fragility are also computed. Default value is
            True.

        Returns
        -------
        fitresult : instance of lmfit's ModelResult class
            Result of the regression. See lmfit for documentation on the
            ModelResult class.

        '''
        temperature = self.table['temperature']
        log_viscosity = self.table['log_viscosity']
        fitresult = model.fit(log_viscosity,
                              T=temperature,
                              method=fitmethod,
                              params=params,
                              weights=weights,
                              nan_policy='propagate')

        self.model = model
        self.fitresult = fitresult

        if self.verbose:
            print(fitresult.fit_report())

        if extra_computation:
            self.getT12()
            if 'm' in fitresult.params:
                self.m = fitresult.params['m'].value
            else:
                self.m = self.fragility()

        return fitresult

    def evallog(self, T):
        '''
        Computes the base-10 log of viscosity at a given temperature.

        Parameters
        ----------
        T : float or array_like
            Temperature. Unit: Kelvin.

        Returns
        -------
        log10_viscosity : float or array_like
            Returns the base-10 logarithm of viscosity.

        '''
        params = self.fitresult.params
        model = self.model
        log10_viscosity = model.eval(params, T=T)

        return log10_viscosity

    def eval(self, T):
        '''
        Computes the viscosity at a given temperature.

        Parameters
        ----------
        T : float or array_like
            Temperature. Unit: Kelvin.

        Returns
        -------
        viscosity : float or array_like
            Returns the viscosity.

        '''
        params = self.fitresult.params
        model = self.model
        viscosity = 10**model.eval(params, T=T)

        return viscosity

    def plot(self):
        '''
        Plot the datapoints with the regression and residuals.

        Returns
        -------
        fig : instance of matplotlib's Figure object.

        axe : instance of matplotlib's GridSpec object.

        '''
        fig, axe = self.fitresult.plot(
            xlabel='Temperature',
            ylabel='Log of Viscosity',
            numpoints=100
        )

        return fig, axe

    def getT12(self):
        '''
        Compute T12 using the fitted viscosity model.

        Returns
        -------
        T12 : float
            T12 is the temperature where viscosity is equal to 10^12 Pa.s.
        
        '''
        params = self.fitresult.params
        model = self.model

        if 'T12' in params:
            T12 = params['T12'].value
            self.T12 = T12

        else:
            if 'T0' in params:
                min_T, max_T = params['T0'].value + 5, 5000
            else:
                min_T, max_T = 100, 5000
                
            def fun(T):
                return model.eval(params, T=T) - 12

            T12 = brentq(fun, min_T, max_T)
            self.T12 = T12

        return T12

    def activationEnergy(self, T):
        '''
        Computes the effective activation energy in units of R.

        Parameters
        ----------
        T : float or numpy array
            Temperature. Unit: Kelvin.

        Returns
        -------
        activation_energy : float or numpy array
            Effective activation energy for viscous flow in units of R.

        '''
        params = self.fitresult.params
        def fun(inverse_temperature):
            return self.model.eval(params, T=1/inverse_temperature)

        activation_energy = diff(fun, 1/T, dx=1e-6, n=1) * log(10) * R

        return activation_energy

    def fragility(self):
        '''
        Computes the fragility index of the liquid, as proposed by Angell.

        Returns
        -------
        fragility : float or array_like
            Angell's fragility index of the liquid.

        References
        ----------
        [1] Angell, C.A. (1985). Strong and fragile liquids. In Relaxation in
            Complex Systems, K.L. Ngai, and G.B. Wright, eds. (Springfield:
            Naval Research Laboratory), pp. 3–12.

        '''
        T12 = self.T12
        fragility = self.activationEnergy(T12) / (R * T12 * log(10))

        return fragility

    def mod_fragility(self, T, melting_point):
        '''
        Computes the modified fragility index of the liquid.

        Parameters
        ----------
        T : float or numpy array
            Temperature. Unit: Kelvin.

        melting_point : float
            Temperature of the melting point. Unit: Kelvin.

        Returns
        -------
        mod_fragility : float or numpy array
            Modified fragility index. See eq. (21) in ref. [1].

        References
        ----------
        [1] Schmelzer, J.W.P., Abyzov, A.S., Fokin, V.M., Schick, C., and
            Zanotto, E.D. (2015). Crystallization in glass-forming liquids:
            Effects of fragility and glass transition temperature. Journal of
            Non-Crystalline Solids 428, 68–74.

        '''
        mod_fragility = self.activationEnergy(T) / (R * melting_point)

        return mod_fragility


class MYEGA(_BaseViscosityRegression):
    '''
    Class for performing the MYEGA regression.

    Parameters
    ----------

    autofit : boolean, optional
        'True' if the regression should be performed during the Class initiation.
        'False' otherwise. Default value is True.

    table : pandas DataFrame, optional, must be a named argument
        DataFrame with a 'temperature' column and a 'log_viscosity' column. A
        RuntimeError is raised if the class is initiated without a table *or*
        without both temperature and log_viscosity arguments.

    temperature : array_like, optional, must be a named argument
        Temperature in Kelvin. A RuntimeError is raised if the class is initiated
        without a table *or* without both time and density arguments. If
        'table' is given then this argument is ignored.

    log_viscosity : array_like, optional, must be a named argument
        Base-10 logarithm of viscosity. It is highly recommended to use
        viscosity in units of Pascal second. A RuntimeError is raised if the
        class is initiated without a table *or* without both time and density
        arguments. If 'table' is given then this argument is ignored.

    '''
    def __init__(self, autofit=True, **kwargs):
        _BaseViscosityRegression.__init__(self, **kwargs)

    def __str__(self):
        return 'MYEGA'

    def getModel(self, guess_T12, guess_fragility, guess_log_eta_inf):
        '''
        Creates a model for regression.

        Parameters
        ----------
        guess_T12 : float
            Guess for the temperature were the viscosity is 10^12 Pa.s.

        guess_fragility : float
            Guess for the fragility index.

        guess_log_eta_inf : array_like, optional
            Guess for the base-10 logarithm of the infinite viscosity.

        Notes
        -----
        The parameters 'K' and 'C' from Eq. (6) from Ref. [1] are also added in
        the model paremeters.

        Returns
        -------
        model : instance of lmfit's Model class.

        '''
        model = Model(eq.MYEGA_alt, name=self.__str__())

        model.set_param_hint('T12', vary=True, min=0, value=guess_T12)
        model.set_param_hint('m', vary=True, min=0, value=guess_fragility)
        model.set_param_hint('log_eta_inf',
                             vary=True,
                             max=11.99,
                             value=guess_log_eta_inf)
        model.set_param_hint(
            'K',
            vary=False,
            expr=r'(12-log_eta_inf)*T12*exp(1-m/(12-log_eta_inf))')
        model.set_param_hint('C',
                             vary=False,
                             expr=r'T12*(m/(12-log_eta_inf)-1)')

        return model

    def fit(self, weights=None, params=None, fitmethod='leastsq'):
        '''
        Regression of the viscosity data.

        Parameters
        ----------
        weights : array_like or None, optional
            The weights of log_viscosity to use during the regression. If None
            then no weights are applied. Default value is None.

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

        model : instance of lmfit's Model class.

        '''
        guess_T12, guess_fragility, guess_log_eta_inf = self.guess()
        model = self.getModel(guess_T12, guess_fragility, guess_log_eta_inf)
        fitresult = super().fit(model, weights, params, fitmethod)

        return fitresult, model


class VFT(_BaseViscosityRegression):
    '''
    Class for performing the VFT regression.

    Parameters
    ----------

    autofit : boolean, optional
        'True' if the regression should be performed during the Class initiation.
        'False' otherwise. Default value is True.

    table : pandas DataFrame, optional, must be a named argument
        DataFrame with a 'temperature' column and a 'log_viscosity' column. A
        RuntimeError is raised if the class is initiated without a table *or*
        without both temperature and log_viscosity arguments.

    temperature : array_like, optional, must be a named argument
        Temperature in Kelvin. A RuntimeError is raised if the class is initiated
        without a table *or* without both time and density arguments. If
        'table' is given then this argument is ignored.

    log_viscosity : array_like, optional, must be a named argument
        Base-10 logarithm of viscosity. It is highly recommended to use
        viscosity in units of Pascal second. A RuntimeError is raised if the
        class is initiated without a table *or* without both time and density
        arguments. If 'table' is given then this argument is ignored.

    '''
    def __init__(self, autofit=True, **kwargs):
        _BaseViscosityRegression.__init__(self, **kwargs)

    def __str__(self):
        return 'VFT'

    def getModel(self, guess_T12, guess_fragility, guess_log_eta_inf):
        '''
        Creates a model for regression.

        Parameters
        ----------
        guess_T12 : float
            Guess for the temperature were the viscosity is 10^12 Pa.s.

        guess_fragility : float
            Guess for the fragility index.

        guess_log_eta_inf : array_like, optional
            Guess for the base-10 logarithm of the infinite viscosity.

        Notes
        -----
        The parameters 'T0' and 'A' are also added in the model paremeters.

        Returns
        -------
        model : instance of lmfit's Model class.

        '''
        model = Model(eq.VFT_alt, name=self.__str__())

        model.set_param_hint('T12', vary=True, min=0, value=guess_T12)
        model.set_param_hint('m', vary=True, min=0, value=guess_fragility)
        model.set_param_hint(
            'log_eta_inf',
            vary=True,
            max=11.99,
            value=guess_log_eta_inf
        )
        model.set_param_hint(
            'T0',
            vary=False,
            expr=r'T12 * (1 - (12 - log_eta_inf) / m)'
        )
        model.set_param_hint(
            'A',
            vary=False,
            expr=r'T12 * (12 - log_eta_inf)**2 / m'
        )

        return model

    def fit(self, weights=None, params=None, fitmethod='leastsq'):
        '''
        Regression of the viscosity data.

        Parameters
        ----------
        weights : array_like or None, optional
            The weights of log_viscosity to use during the regression. If None
            then no weights are applied. Default value is None.

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

        model : instance of lmfit's Model class.

        '''
        guess_T12, guess_fragility, guess_log_eta_inf = self.guess()
        model = self.getModel(guess_T12, guess_fragility, guess_log_eta_inf)
        fitresult = super().fit(model, weights, params, fitmethod)

        return fitresult, model


class CLU(_BaseViscosityRegression):
    '''
    Class for performing the CLU regression.

    Parameters
    ----------

    autofit : boolean, optional
        'True' if the regression should be performed during the Class initiation.
        'False' otherwise. Default value is True.

    table : pandas DataFrame, optional, must be a named argument
        DataFrame with a 'temperature' column and a 'log_viscosity' column. A
        RuntimeError is raised if the class is initiated without a table *or*
        without both temperature and log_viscosity arguments.

    temperature : array_like, optional, must be a named argument
        Temperature in Kelvin. A RuntimeError is raised if the class is initiated
        without a table *or* without both time and density arguments. If
        'table' is given then this argument is ignored.

    log_viscosity : array_like, optional, must be a named argument
        Base-10 logarithm of viscosity. It is highly recommended to use
        viscosity in units of Pascal second. A RuntimeError is raised if the
        class is initiated without a table *or* without both time and density
        arguments. If 'table' is given then this argument is ignored.

    '''
    def __init__(self, autofit=True, **kwargs):
        _BaseViscosityRegression.__init__(self, **kwargs)

    def __str__(self):
        return 'CLU'

    def getModel(self, guess_T12, guess_fragility, guess_log_eta_inf):
        '''
        Creates a model for regression.

        Parameters
        ----------
        guess_T12 : float
            Guess for the temperature were the viscosity is 10^12 Pa.s.

        guess_fragility : float
            Guess for the fragility index.

        guess_log_eta_inf : array_like, optional
            Guess for the base-10 logarithm of the infinite viscosity.

        Notes
        -----
        The parameters 'T0' and 'A' are also added in the model paremeters.

        Returns
        -------
        model : instance of lmfit's Model class.

        '''
        model = Model(eq.CLU, name=self.__str__())

        model.set_param_hint('T12', vary=True, min=0, value=guess_T12)
        model.set_param_hint('m', vary=True, min=0, value=guess_fragility)
        model.set_param_hint(
            'log_pre_exp',
            vary=True,
            value=guess_log_eta_inf
        )
        model.set_param_hint(
            'T0',
            vary=False,
            expr=r'T12-(12-log_pre_exp-log10(T12)/2)*(T12/(m+1/(2*log(10))))'
        )
        model.set_param_hint(
            'A',
            vary=False,
            expr=r'(12-log_pre_exp-log10(T12)/2)**2 * T12/(m+1/(2*log(10)))'
        )

        return model

    def fit(self, weights=None, params=None, fitmethod='leastsq'):
        '''
        Regression of the viscosity data.

        Parameters
        ----------
        weights : array_like or None, optional
            The weights of log_viscosity to use during the regression. If None
            then no weights are applied. Default value is None.

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

        model : instance of lmfit's Model class.

        '''
        guess_T12, guess_fragility, guess_log_eta_inf = self.guess()
        model = self.getModel(guess_T12, guess_fragility, guess_log_eta_inf)
        fitresult = super().fit(model, weights, params, fitmethod)

        return fitresult, model


class BS(_BaseViscosityRegression):
    '''
    Class for performing the BS regression.

    Parameters
    ----------

    autofit : boolean, optional
        'True' if the regression should be performed during the Class initiation.
        'False' otherwise. Default value is True.

    table : pandas DataFrame, optional, must be a named argument
        DataFrame with a 'temperature' column and a 'log_viscosity' column. A
        RuntimeError is raised if the class is initiated without a table *or*
        without both temperature and log_viscosity arguments.

    temperature : array_like, optional, must be a named argument
        Temperature in Kelvin. A RuntimeError is raised if the class is initiated
        without a table *or* without both time and density arguments. If
        'table' is given then this argument is ignored.

    log_viscosity : array_like, optional, must be a named argument
        Base-10 logarithm of viscosity. It is highly recommended to use
        viscosity in units of Pascal second. A RuntimeError is raised if the
        class is initiated without a table *or* without both time and density
        arguments. If 'table' is given then this argument is ignored.

    '''
    def __init__(self, autofit=True, **kwargs):
        _BaseViscosityRegression.__init__(self, **kwargs)

    def __str__(self):
        return 'BS'

    def getModel(self, guess_T12, guess_fragility, guess_log_eta_inf):
        '''
        Creates a model for regression.

        Parameters
        ----------
        guess_T12 : float
            Guess for the temperature were the viscosity is 10^12 Pa.s.

        guess_fragility : float
            Guess for the fragility index.

        guess_log_eta_inf : array_like, optional
            Guess for the base-10 logarithm of the infinite viscosity.

        Returns
        -------
        model : instance of lmfit's Model class.

        '''
        model = Model(eq.BS, name=self.__str__())

        m = guess_fragility
        n = guess_log_eta_inf

        guess_T0 = guess_T12*(3*(n-12) + 2*m)/(2*m)
        guess_A = (12-n)*(3*guess_T12*(12-n)/(2*m))**(3/2)

        model.set_param_hint('gamma', vary=False, value=1)
        model.set_param_hint(
            'log_eta_inf',
            vary=True,
            max=11.99,
            value=guess_log_eta_inf
        )
        model.set_param_hint(
            'T0',
            vary=True,
            value=guess_T0,
        )
        model.set_param_hint(
            'A',
            vary=True,
            value=guess_A,
        )

        return model

    def fit(self, weights=None, params=None, fitmethod='leastsq'):
        '''
        Regression of the viscosity data.

        Parameters
        ----------
        weights : array_like or None, optional
            The weights of log_viscosity to use during the regression. If None
            then no weights are applied. Default value is None.

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

        model : instance of lmfit's Model class.

        '''
        guess_T12, guess_fragility, guess_log_eta_inf = self.guess()
        model = self.getModel(guess_T12, guess_fragility, guess_log_eta_inf)
        fitresult = super().fit(model, weights, params, fitmethod)

        return fitresult, model


class Dienes(_BaseViscosityRegression):
    '''
    Class for performing the Dienes regression.

    Parameters
    ----------

    autofit : boolean, optional
        'True' if the regression should be performed during the Class initiation.
        'False' otherwise. Default value is True.

    table : pandas DataFrame, optional, must be a named argument
        DataFrame with a 'temperature' column and a 'log_viscosity' column. A
        RuntimeError is raised if the class is initiated without a table *or*
        without both temperature and log_viscosity arguments.

    temperature : array_like, optional, must be a named argument
        Temperature in Kelvin. A RuntimeError is raised if the class is initiated
        without a table *or* without both time and density arguments. If
        'table' is given then this argument is ignored.

    log_viscosity : array_like, optional, must be a named argument
        Base-10 logarithm of viscosity. It is highly recommended to use
        viscosity in units of Pascal second. A RuntimeError is raised if the
        class is initiated without a table *or* without both time and density
        arguments. If 'table' is given then this argument is ignored.

    '''
    def __init__(self, autofit=True, **kwargs):
        _BaseViscosityRegression.__init__(self, **kwargs)

    def __str__(self):
        return 'Dienes'

    def getModel(self, guess_T12, guess_fragility, guess_log_eta_inf):
        '''
        Creates a model for regression.

        Parameters
        ----------
        guess_T12 : float
            Guess for the temperature were the viscosity is 10^12 Pa.s.

        guess_fragility : float
            Guess for the fragility index.

        guess_log_eta_inf : array_like, optional
            Guess for the base-10 logarithm of the infinite viscosity.

        Notes
        -----
        The parameters 'T0' and 'A' are also added in the model paremeters.

        Returns
        -------
        model : instance of lmfit's Model class.

        '''
        model = Model(eq.Dienes, name=self.__str__())

        T12 = guess_T12
        m = guess_fragility
        n = guess_log_eta_inf

        guess_T0 = T12 * (1 - (12 - n) / m)
        guess_A = -log(10)*(12-n-m)/(guess_T0/(T12-guess_T0)**2)
        guess_B = T12*log(10)/guess_T0*(T12*(12-n-m) + guess_T0*m)

        model.set_param_hint(
            'log_eta_inf',
            vary=True,
            max=11.99,
            value=guess_log_eta_inf
        )
        model.set_param_hint(
            'T0',
            vary=True,
            min=0,
            value=guess_T0,
        )
        model.set_param_hint(
            'A',
            vary=True,
            value=guess_A,
        )
        model.set_param_hint(
            'B',
            vary=True,
            value=guess_B,
        )

        return model


    def fit(self, weights=None, params=None, fitmethod='leastsq'):
        '''
        Regression of the viscosity data.

        Parameters
        ----------
        weights : array_like or None, optional
            The weights of log_viscosity to use during the regression. If None
            then no weights are applied. Default value is None.

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

        model : instance of lmfit's Model class.

        '''
        guess_T12, guess_fragility, guess_log_eta_inf = self.guess()
        model = self.getModel(guess_T12, guess_fragility, guess_log_eta_inf)
        fitresult = super().fit(model, weights, params, fitmethod)

        return fitresult, model


class DML(_BaseViscosityRegression):
    '''
    Class for performing the DML regression.

    Parameters
    ----------

    autofit : boolean, optional
        'True' if the regression should be performed during the Class initiation.
        'False' otherwise. Default value is True.

    table : pandas DataFrame, optional, must be a named argument
        DataFrame with a 'temperature' column and a 'log_viscosity' column. A
        RuntimeError is raised if the class is initiated without a table *or*
        without both temperature and log_viscosity arguments.

    temperature : array_like, optional, must be a named argument
        Temperature in Kelvin. A RuntimeError is raised if the class is initiated
        without a table *or* without both time and density arguments. If
        'table' is given then this argument is ignored.

    log_viscosity : array_like, optional, must be a named argument
        Base-10 logarithm of viscosity. It is highly recommended to use
        viscosity in units of Pascal second. A RuntimeError is raised if the
        class is initiated without a table *or* without both time and density
        arguments. If 'table' is given then this argument is ignored.

    '''
    def __init__(self, autofit=True, **kwargs):
        _BaseViscosityRegression.__init__(self, **kwargs)

    def __str__(self):
        return 'DML'

    def getModel(self, guess_T12, guess_fragility, guess_log_eta_inf):
        '''
        Creates a model for regression.

        Parameters
        ----------
        guess_T12 : float
            Guess for the temperature were the viscosity is 10^12 Pa.s.

        guess_fragility : float
            Guess for the fragility index.

        guess_log_eta_inf : array_like, optional
            Guess for the base-10 logarithm of the infinite viscosity.

        Notes
        -----
        The parameters 'T0' and 'A' are also added in the model paremeters.

        Returns
        -------
        model : instance of lmfit's Model class.

        '''
        model = Model(eq.DML, name=self.__str__())

        T12 = guess_T12
        m = guess_fragility
        n = guess_log_eta_inf

        guess_T0 = T12 * (1 - (12 - n) / m)
        guess_A = -log(10)*(12-n-m)/(guess_T0/(T12-guess_T0)**2)
        guess_B = T12*log(10)/guess_T0*(T12*(12-n-m) + guess_T0*m)

        model.set_param_hint(
            'log_eta_inf',
            vary=True,
            max=11.99,
            value=guess_log_eta_inf
        )
        model.set_param_hint(
            'T0',
            vary=True,
            min=0,
            value=guess_T0,
        )
        model.set_param_hint(
            'A',
            vary=True,
            value=guess_A,
        )
        model.set_param_hint(
            'B',
            vary=True,
            value=guess_B,
        )

        return model

    def fit(self, weights=None, params=None, fitmethod='leastsq'):
        '''
        Regression of the viscosity data.

        Parameters
        ----------
        weights : array_like or None, optional
            The weights of log_viscosity to use during the regression. If None
            then no weights are applied. Default value is None.

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

        model : instance of lmfit's Model class.

        '''
        guess_T12, guess_fragility, guess_log_eta_inf = self.guess()
        model = self.getModel(guess_T12, guess_fragility, guess_log_eta_inf)
        fitresult = super().fit(model, weights, params, fitmethod)

        return fitresult, model
