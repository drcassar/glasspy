import numpy as np
import warnings
from numpy import exp, log10
from copy import deepcopy as copy
from operator import gt, lt
from scipy.constants import N_A, pi, k
from scipy.integrate import odeint, solve_ivp
from scipy.misc import derivative
from scipy.optimize import brentq


def clusterPopGenOnlyMonomers(lenght, number_of_monomers=N_A, **kwargs):
    '''Generate a cluster population consisting of only monomers.

    Parameters
    ----------
    lenght : scalar
        Maximum number of particles possible in a cluster.

    number_of_monomers : scalar, optional
        Total number of monomers available. Default value is Avagadro's number.

    Returns
    -------
    N : array
        Array of size 'lenght' with all values being zero, except the first,
        for which is 'numebr_of_monomers'.

    '''
    N = np.zeros(lenght)
    N[0] = number_of_monomers
    return N


class KineticModelIsotropicSphere:
    '''Kinetic Model for isotropic sphere nucleation.

    Parameters
    ----------
    max_cluster_size : integer, optional
        Maximum number of particles possible in a cluster. Default value is
        500.

    min_cluster_size : integer, optional
        Minimum number of particles possible in a cluster. Default value is
        1.

    init_cluster_pop_function : callable, optional
        Function used to generate the initial cluster population distribution.
        The first argument of this function must be the maximum cluster size.
        All other arguments of the function are passed via
        'init_cluster_dictionary'. Default value is "clusterPopGenOnlyMonomers".

    init_cluster_dictionary : dictionary, optional
        Dictionaly with the arguments to pass to 'init_cluster_pop_function'.
        Default value is an empty dictionary.

    accuracy : 'cluster density' or 'cluster distribution' or None, optional
        It is not possible to be accurate in both the cluster density
        calculation and the cluster distribution calculation. If 'accuracy' is
        'cluster density' then the density calculation will be accurate. If
        'accuracy' is 'cluster distribution', then the cluster distribution
        will be accurate. If 'accuracy' is None then both Nv and cluster size
        distribution will be innacurate for cluster sizes close to the
        max_cluster_size. Default value is 'cluster density'

    References
    ----------
    [1] Kelton, K., and Greer, A.L. (2010). Nucleation in condensed matter:
        applications in materials and biology (Amsterdam: Elsevier).

    '''
    def __init__(self,
                 max_cluster_size=500,
                 min_cluster_size=1,
                 init_cluster_dictionary={},
                 init_cluster_pop_function=clusterPopGenOnlyMonomers,
                 accuracy='cluster density'):

        self.max_cluster_size = max_cluster_size
        self.min_cluster_size = min_cluster_size
        self.accuracy = accuracy
        initial_cluster_distribution = init_cluster_pop_function(
            self.max_cluster_size, **init_cluster_dictionary)
        self.cluster_distribution = [initial_cluster_distribution]
        self.time = [0]
        self.temperature = [np.nan]
        self.cluster_size = np.arange(self.min_cluster_size,
                                      self.max_cluster_size + 1)
        self.critical_cluster_size = [np.nan]
        self.supercritical_cluster_density = [np.nan]

    def W(self, cluster_size, driving_force, surface_energy, monomer_volume):
        '''Compute the work of formation of a cluster of size n.

        Parameters
        ----------
        cluster_size : scalar or array-like
            Number of monomers in a cluster.

        driving_force : float
            Thermodynamic driving force for nucleation. This value should be
            positive for temperatures below the melting temperature. Units of
            Joules per monomer.

        surface_energy : float
            Surface energy between a cluster and the ambient phase. Units of
            Joule per squared meter.

        monomer_volume : float
            Volume of a monomer. Units of cubic meter per monomer.

        Returns
        -------
        work_of_formation : scalar or array-like
            Work of formation of a cluster of size n. If n is an array, then
            this is the array of work of formation of all the clustes sizes in
            the array.

        Notes
        -----
        Equation for a formation of a spherical and isotropic cluster. See
        Equation (13) in  page 26 of Ref. [1].

        References
        ----------
        [1] Kelton, K., and Greer, A.L. (2010). Nucleation in condensed matter:
            applications in materials and biology (Amsterdam: Elsevier).

        '''
        surface = (36 * pi * (cluster_size * monomer_volume)**2)**(1 / 3)
        work_of_formation = surface_energy * surface \
            - cluster_size * driving_force
        return work_of_formation

    def _generate_matrix_K(self, temperature, diffusion_coeff, driving_force,
                           surface_energy, monomer_volume, jump_distance):
        '''Generate the K matrix.

        Parameters
        ----------
        temperature : scalar
            Absolute temperature.

        diffusion_coeff : scalar
            Diffusion coefficient.

        driving_force : float
            Thermodynamic driving force for nucleation. This value should be
            positive for temperatures below the melting temperature. Units of
            Joules per monomer.

        surface_energy : float
            Surface energy between a cluster and the ambient phase. Units of
            Joule per squared meter.

        monomer_volume : float
            Volume of a monomer. Units of cubic meter per monomer.

        jump_distance : float
            Average distance for an effective attachment or deattachment event.
            Units of meter.

        Returns
        -------
        K : array-like
            Matrix K used for the kinetic model.

        References
        ----------
        [1] Kelton, K., and Greer, A.L. (2010). Nucleation in condensed matter:
            applications in materials and biology (Amsterdam: Elsevier).

        '''
        n = np.arange(self.min_cluster_size - 1, self.max_cluster_size + 2)
        W = self.W(n, driving_force, surface_energy, monomer_volume)
        delW = W[1:] - W[:-1]
        O = 4 * n**(2 / 3)  # see pages 32 and 33 of Ref. [1]
        gamma = 6 * diffusion_coeff / jump_distance**2  # Eq. 30, page 33 of Ref. [1]

        kplus = O[1:-1] * gamma * exp(-delW[1:] / (2 * k * temperature))
        kminus = O[:-2] * gamma * exp(delW[:-1] / (2 * k * temperature))

        K = np.zeros((self.max_cluster_size - self.min_cluster_size + 1,
                      self.max_cluster_size - self.min_cluster_size + 1))
        i, j = np.indices(K.shape)

        K[i == j] = -(kplus + kminus)
        K[i == j - 1] = kminus[1:]
        K[i == j + 1] = kplus[:-1]

        if self.accuracy.lower() in ['cluster density', 'density']:
            K[-1, -1] += kplus[-1]
        elif self.accuracy.lower() in ['cluster distribution', 'distribution']:
            K[-1, -1] += kminus[-1]

        return K

    def computeCriticalClusterSize(self, driving_force, surface_energy,
                                   monomer_volume):
        '''Computes the cluster size (number of monomers) of a critical nucleus.

        A warning is raised if the critical cluster size is above the maximum
        cluster size.

        Parameters
        ----------
        driving_force : float
            Thermodynamic driving force for nucleation. This value should be
            positive for temperatures below the melting temperature. Units of
            Joules per monomer.

        surface_energy : float
            Surface energy between a cluster and the ambient phase. Units of
            Joule per squared meter.

        monomer_volume : float
            Volume of a monomer. Units of cubic meter per monomer.

        Returns
        -------
        critical_cluster_size : integer or NaN
            Number of monomers in a critical cluster size. Returns NaN if the
            critical cluster size is above the maximum cluster size.

        '''
        def Wfun(n):
            return self.W(n, driving_force, surface_energy, monomer_volume)

        def dW_dn(n):
            return derivative(Wfun, n, dx=1e-6)

        try:
            critical_cluster_size = brentq(dW_dn, self.min_cluster_size,
                                           self.max_cluster_size)
            critical_cluster_size = int(np.ceil(critical_cluster_size))

        except ValueError:
            msg = 'Critical cluster size is above the maximum cluster size'
            critical_cluster_size = np.nan
            warnings.warn(msg)

        return critical_cluster_size

    def _computeSupercriticalClusterDensity(self, cluster_distribution,
                                            critical_cluster_size,
                                            monomer_volume):
        '''Compute the supercritical cluster density.

        cluster_distribution : array-like
            This argument can be an array or array of arrays of the cluster
            distribution.

        critical_cluster_size : integer
            The size of a critical cluster.

        monomer_volume : float
            Volume of a monomer. Units of cubic meter per monomer.

        Returns
        -------
        supercritical_cluster_density : scalar or array-like
            Number of supercrcitical clusters per unity of volume.

        '''
        number_of_atoms = cluster_distribution.sum(axis=1)
        total_volume = monomer_volume * number_of_atoms
        critical_cluster_index = self.cluster_size.tolist().index(
            critical_cluster_size)
        supercritical_cluster_distribution = \
            cluster_distribution[:, critical_cluster_index:]
        supercritical_cluster_density = np.sum(
            np.round(supercritical_cluster_distribution),
            axis=1) / total_volume

        return supercritical_cluster_density

    def isothermalTreatment(self,
                            temperature,
                            time,
                            diffusion_coeff,
                            surface_energy,
                            driving_force,
                            monomer_volume,
                            jump_distance,
                            time_resolution='default'):
        ''' Perform an isothermal treatment.

        Parameters
        ----------
        temperature : scalar
            Absolute temperature.

        time : float
            Total time of the isothermal treatment in seconds.

        diffusion_coeff : float
            Diffusion coefficient.

        driving_force : float
            Thermodynamic driving force for nucleation. This value should be
            positive for temperatures below the melting temperature. Units of
            Joules per monomer.

        surface_energy : float
            Surface energy between a cluster and the ambient phase. Units of
            Joule per squared meter.

        monomer_volume : float
            Volume of a monomer. Units of cubic meter per monomer.

        jump_distance : float
            Average distance for an effective attachment or deattachment event.
            Units of meter.

        time_resolution : 'default' or integer
            If default then the time resolution is set automatically. If
            integer then it is the number of times to consider for the
            isothermal treatment is.


        References
        ----------
        [1] Kelton, K., and Greer, A.L. (2010). Nucleation in condensed matter:
            applications in materials and biology (Amsterdam: Elsevier).

        '''
        K = self._generate_matrix_K(temperature, diffusion_coeff,
                                    driving_force, surface_energy,
                                    monomer_volume, jump_distance)
        current_cluster_distribution = copy(self.cluster_distribution[-1])

        def dN_dt(N, t, K):
            return K @ N  # dot product of K and N

        if time_resolution is 'default':
            time_resolution_ = max(10, int(np.ceil(log10(time) * 5)))
        else:
            time_resolution_ = time_resolution

        times = np.logspace(0, log10(time), time_resolution_)
        cluster_distribution_array = odeint(dN_dt,
                                            current_cluster_distribution,
                                            times,
                                            args=(K, ))

        critical_cluster_size = self.computeCriticalClusterSize(
            driving_force, surface_energy, monomer_volume)

        if np.isfinite(critical_cluster_size):
            supercritical_cluster_density = \
                self._computeSupercriticalClusterDensity(
                    cluster_distribution_array,
                    critical_cluster_size,
                    monomer_volume)
            self.critical_cluster_size.extend([critical_cluster_size] *
                                              time_resolution_)
            self.supercritical_cluster_density.extend(
                supercritical_cluster_density)
        else:
            self.critical_cluster_size.extend([np.nan] * time_resolution_)
            self.supercritical_cluster_density.extend([np.nan] *
                                                      time_resolution_)

        self.time.extend(times + self.time[-1])
        self.temperature.extend([temperature] * time_resolution_)
        self.cluster_distribution = np.concatenate(
            (self.cluster_distribution, cluster_distribution_array), axis=0)

    def _linearRamp(self,
                    initial_temperature,
                    final_temperature,
                    temperature_change_rate,
                    diffusion_coeff_fun,
                    surface_energy_fun,
                    driving_force_fun,
                    jump_distance_fun,
                    monomer_volume_fun,
                    temperature_resolution=1):

        T = initial_temperature

        T2 = final_temperature
        rate = abs(temperature_change_rate)
        resolution = temperature_resolution
        default_time = resolution / rate

        sign = np.sign(final_temperature - initial_temperature)

        if sign > 0:
            compare_fun = lt
        else:
            compare_fun = gt

        # the ramp stays half the default time at the initial temperature
        time = default_time / 2

        while T != T2:

            self.isothermalTreatment(T, time, diffusion_coeff_fun(T),
                                     surface_energy_fun(T),
                                     driving_force_fun(T),
                                     monomer_volume_fun(T),
                                     jump_distance_fun(T))

            if compare_fun(T + sign * resolution, T2):
                time = default_time
                next_T = T + sign * resolution
            else:
                time = abs(T - T2) / rate
                next_T = T2

            T = next_T

        # the ramp stays half the default time at the final temperature
        self.isothermalTreatment(T, default_time / 2, diffusion_coeff_fun(T),
                                 surface_energy_fun(T), driving_force_fun(T),
                                 monomer_volume_fun(T), jump_distance_fun(T))
