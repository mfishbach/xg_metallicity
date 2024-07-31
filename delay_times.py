import numpy as np

from scipy.integrate import cumtrapz
import jax.scipy.stats as ss
from jax.scipy.special import erf
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

from gw_pop_numpyro import cosmo_utils, jacobians

np.random.seed(213)

ZMAX = 15

ZSUN = 0.0142 #default in COMPAS

cosmo_dict = cosmo_utils.interp_cosmology(zmax = ZMAX + 15)

def redshift_at_age_Gyr(age_Gyr):
    age_Myr = age_Gyr * 1e3
    lookback_time = cosmo_dict["age_of_universe"] - age_Myr

    z = cosmo_dict["z_at_lookback_time"](lookback_time)

    return z

def probz_from_merger_rate_versus_age(age_Gyr, merger_rate, return_normalization = False):

    z = redshift_at_age_Gyr(age_Gyr)

    #sel = z < ZMAX

    z = z[::-1] #change from decreasing to increasing
    merger_rate = merger_rate[::-1]

    #z = z[sel]
    #merger_rate = merger_rate[sel]
    
    #dVc/dz * 1/(1+z)
    pz = merger_rate * cosmo_dict["differential_comoving_volume"](z) * (1 + z)**-1 * 1e-9 #units Gpc**3
    norm_fact = np.trapz(pz, z) #this times the observing time T is the number of observed events 

    pz /= norm_fact

    if return_normalization:
        return z, pz, merger_rate, norm_fact

    return z, pz, merger_rate

def rate_from_probz(z, pz, zhigh = None):

    if zhigh is not None:
        volume_term = 1e-9 * 0.5 * (cosmo_dict["differential_comoving_volume"](z)/(1+z) + cosmo_dict["differential_comoving_volume"](zhigh)/(1+zhigh))

    else:
        volume_term = cosmo_dict["differential_comoving_volume"](z)/(1+z) * 1e-9

    rate = pz / volume_term #units: Gpc**-3

    return rate

def merger_from_formation_rate_v_age(age_m_grid, tau_grid, ptau_grid, formation_rate_at_age_func):
    '''
    age_m_grid: 1-d array of times at which to evaluate merger rate
    tau_grid: 1-d array of delay times used in integration
    ptau_grid: 1-d array of delay time pdf evaluated at tau_grid, same shape as tau_grid
    formation_rate_at_age_func: function that gives formation rate at a given time (age)
    returns: merger rate evaluated at age_m_grid
    '''

    age_m_2d = age_m_grid[:, np.newaxis] #time at merger, elevate to a 2-d array

    age_f_2d = age_m_2d - tau_grid #time at formation, has shape (len(age_m_grid), len(tau_grid))

    formation_rate_2d = formation_rate_at_age_func(age_f_2d) #formation rate at formation time

    merger_rate = np.trapz(formation_rate_2d * ptau_grid, tau_grid, axis = -1) #integrate over delay time distribution

    return merger_rate


def draw_redshifts_from_formation_delay(nevents, age_m_grid_Gyr, tau_grid, ptau_grid, formation_rate_at_age_func):

    merger_rate = merger_from_formation_rate_v_age(age_m_grid_Gyr, tau_grid, ptau_grid, formation_rate_at_age_func)

    z_grid, pz, merger_rate_grid, norm_fact = probz_from_merger_rate_versus_age(age_m_grid_Gyr, merger_rate, return_normalization = True)

    rs = np.random.rand(nevents)

    zs = np.interp(rs,cumtrapz(pz, z_grid, initial = 0), z_grid)

    return zs, z_grid, pz, merger_rate_grid, norm_fact


def formation_from_merger_rate_v_age(merger_rate_grid, tau_grid, ptau_grid):

    '''
    merger_rate_grid: shape (len(tau_grid), n). Merger rate evaluated at each age in tau_grid. Can be n posterior samples per tau.
    tau_grid: 1D array, Ages corresponding to merger times. ***Also grid of delay times***
    ptau_grid: 1D array, same dimensions at tau_grid. Delay time distribution evaluated on tau_grid (Probability not pdf, i.e. pdf*delta_tau)

    return: formation_rate_grid, same dimension as merger_rate_grid

    tau_grid runs from minimum delay time to age of Universe today
    merger_rate_grid: merger rate from the minimum delay time to age of Universe today
    formation_rate_grid: runs from 0 to age of Universe - minimum delay time
    '''
    

    #ptau_grid /= np.sum(ptau_grid) #make sure ptau_grid sums to 1

    formation_rate_grid = np.zeros_like(merger_rate_grid)
    
    #Rf[0] = merger_rate_grid[1] / ptau_grid[0]
    #Rf[1] = (merger_rate_grid[2] - Rf[0] * ptau_grid[1]) / ptau_grid[0]
    #Rf[2] = (merger_rate_grid[3] - Rf[1] * ptau_grid[1] - Rf[0] * ptau_grid[2]) / ptau_grid[0]

    for i in range(len(tau_grid) - 1):

        subtract = 0

        for j in range(i):

            subtract += formation_rate_grid[j] * ptau_grid[i + 1 - j]

        formation_rate_grid[i] = (merger_rate_grid[i + 1] - subtract) / ptau_grid[0]

    return formation_rate_grid #last element in formation_rate_grid is always 0

def delay_time_from_merger_rate_v_age(merger_rate_grid, formation_rate_grid, tau_grid):

    '''
    merger_rate_grid: shape (len(tau_grid), n). Merger rate evaluated at each age in tau_grid. Can be n posterior samples per tau.
    tau_grid: 1D array, Ages corresponding to merger times.
    formation_rate_grid: Formation rates corresponding to merger times - minimum delay time. Same dimensions as either tau_grid or merger rate_grid

    return: ptau_grid, Delay time distribution evaluated on tau_grid (Probability not pdf, i.e. pdf*delta_tau)

    tau_grid runs from minimum delay time + minimum formation time to age of Universe today
    ptau_grid: Delay time probabilities from the minimum delay time to age of Universe today. Minimum delay time is given by subtracting first nonzero merger rate time from first nonzero formation rate time.
    formation_rate_grid: runs from minimum formation time  to age of Universe - minimum delay time
    '''
    ptau_grid = np.zeros_like(merger_rate_grid)

    for i in range(len(tau_grid) - 1):

        subtract = 0

        for j in range(i):

            subtract += formation_rate_grid[j] * ptau_grid[i + 1 - j]

        ptau_grid[i] = (merger_rate_grid[i] - subtract) / formation_rate_grid[0] 
    
    return ptau_grid

def formation_rate_at_age_func_Kats(tau, a = 1.83, b = -0.48):

    return np.where((tau > 0) & (tau < cosmo_dict['age_of_universe']/1000), tau**a * np.exp(tau * b), 0)

def norm_cdf(x):
    #return 0.5 * (1 + erf(x/jnp.sqrt(2)))
    return ss.norm.cdf(x)

def p_logZ_versus_redshift(logZ, z, alpha = -1.778, mu0 = 0.025, muz = -0.049, omega0 = 1.122, omegaz = 0.049): 
    '''
    logZ: log10(Z/Zsun) 
    logZ follows a skew normal distribution that evolves with redshift.
    normalized over range of logZ
    mu controls center, omega controls the width
    from van Son et al. 2023
    both logZ and zred are arrays, will integrate over logZ
    '''
    
    lnZ = jnp.log(10**logZ * ZSUN)
    
    omega = omega0* 10**(omegaz*z)
    mu = mu0 * 10**(muz * z)


    beta = alpha * (1 + alpha**2)**-0.5

    xi = jnp.log(mu / (2 * norm_cdf(beta * omega) )) - omega**2/2 

    #elevate the redshift arrays to nd+1 arrays to allow for logZ array
    omega = jnp.expand_dims(omega, axis = -1)
    xi = jnp.expand_dims(xi, axis = -1)


    x = (lnZ - xi)/omega
    pdf_lnZ = 2/omega * ss.norm.pdf(x) * norm_cdf(alpha * x)

    norm_fact = jnp.trapz(pdf_lnZ, lnZ, axis = -1)

    return pdf_lnZ / jnp.expand_dims(norm_fact, axis = -1)


def efficiency_versus_logmet(logZ, y = 3.5e-5, wm1 = -0.15):
    '''parameterization follows Fishbach & van Son (2023), 
    with y representing the efficiency at low metallicities,
    and wm1 (for w - 1) representing the metallicity location of the sharp drop in efficiency
    '''

    x = wm1 - logZ
    posx = (jnp.abs(x) + x) / 2 #same as x if x is positive, zero if x is negative
    ef = y * jnp.log(posx + 1)

    return ef
    #return jnp.where( logZ < wm1 , y * jnp.log(wm1 + 1 - logZ), 0) 

def efficiency_versus_redshift(z, log10Z_grid = np.linspace(-5, 1, 512), y = 3.5e-5, wm1 = -0.15, alpha = -1.778, mu0 = 0.025, muz = -0.049, omega0 = 1.122, omegaz = 0.049): 
    
    efficiency_at_log10Z_grid = efficiency_versus_logmet(log10Z_grid, y, wm1)

    pdf_lnZ = p_logZ_versus_redshift(log10Z_grid, z, alpha, mu0, muz, omega0, omegaz)

    lnZ_grid = jnp.log(10**log10Z_grid * ZSUN)

    #integrate over metallicity grid
    out = jnp.trapz(pdf_lnZ * efficiency_at_log10Z_grid[None, :], lnZ_grid, axis = -1)

    return out


def efficiency_versus_redshift_step(z, log10Zlow = -5, log10Zhigh = -1, log10Z_grid = np.linspace(-5, 1, 512), alpha = -1.778, mu0 = 0.025, muz = -0.049, omega0 = 1.122, omegaz = 0.049):

    indexlow = np.arange(len(log10Z_grid))[log10Z_grid >= log10Zlow][0]
    indexhigh = np.arange(len(log10Z_grid))[log10Z_grid >= log10Zhigh][0]

    pdf_lnZ = p_logZ_versus_redshift(log10Z_grid, z, alpha, mu0, muz, omega0, omegaz)

    lnZ_grid = np.log(10**log10Z_grid * ZSUN)

    out = np.trapz(pdf_lnZ[: , indexlow:indexhigh], lnZ_grid[indexlow:indexhigh], axis = -1)

    return out

def formation_rate_at_redshift_MD(z, Rz0 = 1e7, a = 1.5, b = 4.4, zp = 3.4, zmax = ZMAX):

    return np.where((z <= zmax) & (z >= 0), Rz0 * (1.0+(1.0+zp)**(-a-b))*(1+z)**a/(1.0+((1.0+z)/(1.0+zp))**(a+b)), 0) 

def formation_rate_at_age_func(tau, Rz0 = 1e7, a = 1.5, b = 4.4, zp = 3.4, zmax = ZMAX, efficiency_func = efficiency_versus_redshift):

    z = redshift_at_age_Gyr(tau)

    #units are same as Rz0, Gpc^-3 yr^-1
    formation_rate = np.where((z <= zmax) & (z >= 0), efficiency_func(z) * Rz0 * (1.0+(1.0+zp)**(-a-b))*(1+z)**a/(1.0+((1.0+z)/(1.0+zp))**(a+b)), 0)
    return formation_rate

def test_delay_time_from_rate():

    tau_grid = np.linspace(0.01, cosmo_dict['age_of_universe']/1000, 10000) #in Gyr

    merger_rate_grid = merger_from_formation_rate_v_age(tau_grid, tau_grid, tau_grid**-1/np.trapz(tau_grid**-1, tau_grid), formation_rate_at_age_func_Kats)

    formation_rate_grid = formation_rate_at_age_func_Kats(tau_grid)

    Ptau_grid = delay_time_from_merger_rate_v_age(merger_rate_grid, formation_rate_grid, tau_grid)

    plt.plot(tau_grid, Ptau_grid / (tau_grid[1] - tau_grid[0]), label = 'inferred')
    plt.plot(tau_grid, tau_grid**-1/np.trapz(tau_grid**-1, tau_grid), ls = '--', label = 'Truth')

    plt.legend(loc = 'best')

    plt.xlabel(r'$\tau$ (Gyr)')
    plt.ylabel(r'delay time distribution $p(\tau)$')
    plt.yscale('log')

    plt.show()

def test():

    tau_grid = np.linspace(0.01, cosmo_dict['age_of_universe']/1000, 10000) #in Gyr

    ptau_grid = tau_grid**-1/np.trapz(tau_grid**-1, tau_grid) * (tau_grid[1] - tau_grid[0]) #Ptau (probability) rather than probability density as expected by merger_rate_from_formation_rate_v_age function

    #print(ptau_grid)

    merger_rate_grid = merger_from_formation_rate_v_age(tau_grid, tau_grid, tau_grid**-1/np.trapz(tau_grid**-1, tau_grid), formation_rate_at_age_func)

    formation_rate_grid = formation_from_merger_rate_v_age(merger_rate_grid, tau_grid, ptau_grid)

    #print(merger_rate_grid)
    #print(formation_rate_grid)

    plt.plot(tau_grid, merger_rate_grid)

    plt.plot(tau_grid - tau_grid[0], formation_rate_grid, ls = '--')
    plt.plot(tau_grid - tau_grid[0], formation_rate_at_age_func(tau_grid - tau_grid[0]), alpha = 0.5)

    plt.yscale('log')
    plt.ylim(1e-2,10)

    plt.show()

    return formation_rate_grid

def test_draw_redshifts():

    tau_grid = np.linspace(0.01, cosmo_dict['age_of_universe']/1000, 10000) #in Gyr

    ptau_grid = tau_grid**-1/np.trapz(tau_grid**-1, tau_grid)

    age_m_grid_Gyr = tau_grid

    nevents = 100000

    zs, z_grid, pz = draw_redshifts_from_formation_delay(nevents, age_m_grid_Gyr, tau_grid, ptau_grid, formation_rate_at_age_func)

    #plt.hist(np.log(1 + zs), bins = 20, density = True)

    #plt.hist(1 + zs, bins = np.logspace(0, 1, 20), density = True)

    #plt.plot(np.log(1 + z_grid), pz * (1 + z_grid))

    counts, bins = np.histogram(np.log(1 + zs), bins = 30, density = True)

    counts = counts[::-1]
    bins = bins[::-1]

    age_bin_Gyr = (cosmo_dict['age_of_universe'] - cosmo_dict["lookback_time"](np.exp(bins) - 1))/1000

    bin_centers = 0.5 * (bins[1:] + bins[0:-1])

    z_bin_centers = np.exp(bin_centers) - 1

    age_bin_centers_Myr = cosmo_dict['age_of_universe'] - cosmo_dict["lookback_time"](z_bin_centers)

    age_bin_centers_Gyr = age_bin_centers_Myr/1000

    probz_from_counts = counts/ (1 + z_bin_centers)

    merger_rate_from_histogram = rate_from_probz(z_bin_centers, probz_from_counts)

    Ptau_bin_centers = age_bin_centers_Gyr**-1/np.trapz(age_bin_centers_Gyr**-1, age_bin_centers_Gyr) * (age_bin_Gyr[1:] - age_bin_Gyr[0:-1])

    formation_rate_from_histogram = formation_from_merger_rate_v_age(merger_rate_from_histogram, age_bin_centers_Gyr, Ptau_bin_centers)

    plt.plot(age_bin_centers_Gyr, formation_rate_from_histogram/max(formation_rate_from_histogram))

    plt.plot(tau_grid, formation_rate_at_age_func(tau_grid)/ max(formation_rate_at_age_func(tau_grid)))

    #rate = rate_from_probz(z_grid, pz)

    #plt.plot(z_grid, rate)

    #plt.plot(z_bin_centers, rate_from_histogram)

    #print(z_bin_centers)

    plt.show()

#test_draw_redshifts()
test_delay_time_from_rate()
