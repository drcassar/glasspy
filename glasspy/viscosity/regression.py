'''Classes for regression of viscosity data.'''

from scipy.stats import linregress
from lmfit import Model
from .equilibrium import logMYEGA


class _BaseViscosityRegression:
    '''
    Base class for viscosity regression.

    Parameters
    ----------

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
    def __init__(self, **kwargs):

        if 'table' in kwargs:

            table = kwargs['table']
            columns = table.columns.values

            if 'temperature' not in columns:
                name = self.__name__
                msg = f'The {name} class was initiated with a table with no temperature data'
                raise RuntimeError(msg)

            if 'log_viscosity' not in columns:
                name = self.__name__
                msg = f'The {name} class was initiated with a table with no log_viscosity data'
                raise RuntimeError(msg)

        elif 'temperature' in kwargs and 'log_viscosity' in kwargs:

            table = pd.DataFrame({
                'temperature': kwargs['temperature'],
                'log_viscosity': kwargs['log_viscosity'],
            })

        else:
            name = self.__name__
            msg = f'The {name} class was initiated with insuficient arguments'
            raise RuntimeError(msg)

        self.table = table

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

        '''
        temperature = self.table['temperature']
        log_viscosity = self.table['log_viscosity']
        slope, intercept, _, _, _ = linregress(x=1 / temperature,
                                               y=log_viscosity)

        guess_T12 = slope / (12 - intercept)
        guess_T12 = guess_T12 if guess_T12 > 0 else min(temperature)
        guess_log_eta_inf = -3
        guess_fragility = 50

        model = Model(logMYEGA)
        fitresult = model.fit(log_viscosity,
                              T=temperature,
                              log_eta_inf=guess_log_eta_inf,
                              T12=guess_T12,
                              m=guess_fragility)

        guess_T12 = fitresult.params['T12'].value
        guess_fragility = fitresult.params['m'].value
        guess_log_eta_inf = fitresult.params['log_eta_inf'].value

        return guess_T12, guess_fragility, guess_log_eta_inf

    def _fit(self, model, weights=None, params=None, fitmethod='leastsq'):
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

        return fitresult


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
        if autofit:
            self.fitresult, self.model = self.fit()

    def __str__(self):
        return 'MYEGA'

    def getModel(self, guess_T12, guess_fragility, guess_log_eta_inf):
        '''
        Creates a model for regression of the MYEGA equation.

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
        model : instance of lmfit's Model class
            MYEGA model to fit experimental data.

        References
        ----------
        [1] Mauro, J.C., Yue, Y., Ellison, A.J., Gupta, P.K., and Allan, D.C.
            (2009). Viscosity of glass-forming liquids. Proceedings of the National
            Academy of Sciences of the United States of America 106, 19780–19784.

        '''
        model = Model(logMYEGA, name="MYEGA")

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

        model : instance of lmfit's Model class
            MYEGA model to fit experimental data.

        '''
        guess_T12, guess_fragility, guess_log_eta_inf = self.guess()
        model = self.getModel(guess_T12, guess_fragility, guess_log_eta_inf)
        fitresult = self._fit(model, weights, params, fitmethod)

        return fitresult, model
