
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os
import math
import datetime
import numpy as np
# Plotting imports
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
# tudatpy imports
from tudatpy.kernel import constants
from tudatpy.kernel.astro import time_conversion
from tudatpy.kernel.astro import element_conversion
# Problem-specific imports
from tests import validation_util as Util

### MATPLOTLIB STANDARD SETTINGS ###
# Grid
plt.rcParams['axes.grid'] = True
plt.rc('grid', linestyle='-.', alpha=0.5)
# Font and Size
plt.rc('font', family='serif', size=18)
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['figure.figsize'] = (16, 9)
# Line- and marker-size, tick position
plt.rcParams.update({'lines.linewidth': 3})
plt.rcParams.update({'lines.markersize': 5})
plt.rcParams['axes.axisbelow'] = True
# Tight layout and always save as png
plt.rcParams['figure.autolayout'] = True
plt.rcParams['savefig.format'] = 'png'

# Get path of current directory
current_dir = os.path.dirname(__file__)
image_save_path = os.path.join(current_dir, 'validation_plots')
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

###########################################################################
# DYNAMICAL MODEL VALIDATION ##############################################
###########################################################################

def plot_dynamical_model_validation():
    ### LOAD DATA ###
    # Load state histories
    propagation_validation_load_path = os.path.join(current_dir, 'validation_data/dynamical_validation',
                                                    'propagated_state_history.dat')
    ephemeris_validation_load_path = os.path.join(current_dir, 'validation_data/dynamical_validation',
                                                  'ephemeris_state_history.dat')

    propagation_state_history = np.loadtxt(propagation_validation_load_path)
    ephemeris_state_history = np.loadtxt(ephemeris_validation_load_path)

    state_history_difference = propagation_state_history[:, 1:] - ephemeris_state_history[:, 1:]
    position_difference = {'Io': state_history_difference[:, 0:3],
                           'Europa': state_history_difference[:, 6:9],
                           'Ganymede': state_history_difference[:, 12:15],
                           'Callisto': state_history_difference[:, 18:21]}

    # Load Kepler elements
    propagation_kepler_load_path = os.path.join(current_dir, 'validation_data/dynamical_validation',
                                                'propagated_keplerian_states.dat')
    ephemeris_kepler_load_path = os.path.join(current_dir, 'validation_data/dynamical_validation',
                                              'ephemeris_keplerian_states.dat')

    propagation_kepler_elements = np.loadtxt(propagation_kepler_load_path)
    ephemeris_kepler_elements = np.loadtxt(ephemeris_kepler_load_path)

    propagated_kepler_elements_dict = {'Io': propagation_kepler_elements[:, 1:7],
                                       'Europa': propagation_kepler_elements[:, 7:13],
                                       'Ganymede': propagation_kepler_elements[:, 13:19],
                                       'Callisto': propagation_kepler_elements[:, 19:25]}
    ephemeris_kepler_elements_dict = {'Io': ephemeris_kepler_elements[:, 1:7],
                                      'Europa': ephemeris_kepler_elements[:, 7:13],
                                      'Ganymede': ephemeris_kepler_elements[:, 13:19],
                                      'Callisto': ephemeris_kepler_elements[:, 19:25]}

    propagated_mean_longitude_dict = Util.calculate_mean_longitude(propagated_kepler_elements_dict)
    ephemeris_mean_longitude_dict = Util.calculate_mean_longitude(ephemeris_kepler_elements_dict)

    mean_longitude_difference = dict()
    for moon in propagated_kepler_elements_dict.keys():
        diff_temp = propagated_mean_longitude_dict[moon] - ephemeris_mean_longitude_dict[moon]
        diff_temp[diff_temp > math.pi] = diff_temp[diff_temp > math.pi] - 2 * math.pi
        diff_temp[diff_temp < -1 * math.pi] = diff_temp[diff_temp < -1 * math.pi] + 2 * math.pi
        mean_longitude_difference[moon] = diff_temp

    mean_motion_difference_perc = dict()
    # Gravitational parameter taken from Folkner et al. (2017)
    jupiter_gravitational_parameter = 1.26686533E17
    for moon in propagated_kepler_elements_dict.keys():
        mean_motion_prop = np.sqrt(jupiter_gravitational_parameter / (propagated_kepler_elements_dict[moon][:, 0]) ** 3)
        mean_motion_ephem = np.sqrt(jupiter_gravitational_parameter / (ephemeris_kepler_elements_dict[moon][:, 0]) ** 3)
        mean_motion_difference_perc[moon] = (mean_motion_prop - mean_motion_ephem) / mean_motion_ephem

    raan_difference = dict()
    for moon in propagated_kepler_elements_dict.keys():
        raan_difference[moon] = (propagated_kepler_elements_dict[moon][:, 4] -
                                 ephemeris_kepler_elements_dict[moon][:, 4])

    mean_anomaly_difference = dict()
    for moon in propagated_kepler_elements_dict.keys():
        propagated_mean_anomaly = list()
        kepler_elements_per_moon = propagated_kepler_elements_dict[moon]
        # For every epoch get the mean anomaly of the moon
        for i in range(len(propagated_kepler_elements_dict[moon])):
            propagated_mean_anomaly.append(element_conversion.true_to_mean_anomaly(
                eccentricity=kepler_elements_per_moon[i, 1],
                true_anomaly=kepler_elements_per_moon[i, 5]))
        propagated_mean_anomaly = np.array(propagated_mean_anomaly)
        propagated_mean_anomaly[propagated_mean_anomaly < 0] = (
                propagated_mean_anomaly[propagated_mean_anomaly < 0] + 2 * math.pi)

        ephemeris_mean_anomaly = list()
        kepler_elements_per_moon = ephemeris_kepler_elements_dict[moon]
        # For every epoch get the mean anomaly of the moon
        for i in range(len(ephemeris_kepler_elements_dict[moon])):
            ephemeris_mean_anomaly.append(element_conversion.true_to_mean_anomaly(
                eccentricity=kepler_elements_per_moon[i, 1],
                true_anomaly=kepler_elements_per_moon[i, 5]))
        ephemeris_mean_anomaly = np.array(ephemeris_mean_anomaly)
        ephemeris_mean_anomaly[ephemeris_mean_anomaly < 0] = (
                ephemeris_mean_anomaly[ephemeris_mean_anomaly < 0] + 2 * math.pi)

        diff_temp = propagated_mean_anomaly - ephemeris_mean_anomaly
        diff_temp[diff_temp > math.pi] = diff_temp[diff_temp > math.pi] - 2 * math.pi
        diff_temp[diff_temp < -1 * math.pi] = diff_temp[diff_temp < -1 * math.pi] + 2 * math.pi
        mean_anomaly_difference[moon] = diff_temp

    ### PLOTTING ###
    time2plt = list()
    epochs_julian_seconds = propagation_state_history[:, 0]
    for epoch in epochs_julian_seconds:
        epoch_days = constants.JULIAN_DAY_ON_J2000 + epoch / constants.JULIAN_DAY
        time2plt.append(time_conversion.julian_day_to_calendar_date(epoch_days))

    lower_limit = time_conversion.julian_day_to_calendar_date(
        constants.JULIAN_DAY_ON_J2000 + 1 +
        (31.0 * constants.JULIAN_YEAR + 172.0 * constants.JULIAN_DAY) / constants.JULIAN_DAY)
    upper_limit = time_conversion.julian_day_to_calendar_date(
        constants.JULIAN_DAY_ON_J2000 - 1 +
        (34.0 * constants.JULIAN_YEAR + 365.0 * constants.JULIAN_DAY) / constants.JULIAN_DAY)

    fig = plt.figure(figsize=(16, 14))

    gs = gridspec.GridSpec(3, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    ax1.plot(time2plt, np.linalg.norm(position_difference['Io'], axis=1) * 1E-3,
             label=r'Io ($i=1$)', c='#A50034')
    ax1.plot(time2plt, np.linalg.norm(position_difference['Europa'], axis=1) * 1E-3,
             label=r'Europa ($i=2$)', c='#0076C2')
    ax1.plot(time2plt, np.linalg.norm(position_difference['Ganymede'], axis=1) * 1E-3,
             label=r'Ganymede ($i=3$)', c='#EC6842')
    ax1.plot(time2plt, np.linalg.norm(position_difference['Callisto'], axis=1) * 1E-3,
             label=r'Callisto ($i=4$)', c='#009B77')
    ax1.set_title(r'Difference in Position (C-M)')
    ax1.set_xlim(lower_limit, upper_limit)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    ax1.set_ylabel(r'Difference [km]')
    ax1.legend()

    ax2.plot(time2plt, mean_longitude_difference['Io'] * (180 / math.pi), c='#A50034')
    ax2.plot(time2plt, mean_longitude_difference['Europa'] * (180 / math.pi), c='#0076C2')
    ax2.plot(time2plt, mean_longitude_difference['Ganymede'] * (180 / math.pi), c='#EC6842')
    ax2.plot(time2plt, mean_longitude_difference['Callisto'] * (180 / math.pi), c='#009B77')
    ax2.set_title(r'$\Delta M_i$ (C-M)')
    ax2.set_xlim(lower_limit, upper_limit)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    ax2.set_ylabel(r'Difference [deg]')
    ax2.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

    ax3.plot(time2plt, mean_motion_difference_perc['Io'], c='#A50034')
    ax3.plot(time2plt, mean_motion_difference_perc['Europa'], c='#0076C2')
    ax3.plot(time2plt, mean_motion_difference_perc['Ganymede'], c='#EC6842')
    ax3.plot(time2plt, mean_motion_difference_perc['Callisto'], c='#009B77')
    ax3.set_title(r'$\Delta n_i$ (C-M)')
    ax3.set_xlim(lower_limit, upper_limit)
    ax3.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    ax3.xaxis.set_minor_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    ax3.set_ylabel(r'Difference [%]')
    ax3.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

    ax4.plot(time2plt, raan_difference['Io'] * (180 / math.pi), c='#A50034')
    ax4.plot(time2plt, raan_difference['Europa'] * (180 / math.pi), c='#0076C2')
    ax4.plot(time2plt, raan_difference['Ganymede'] * (180 / math.pi), c='#EC6842')
    ax4.plot(time2plt, raan_difference['Callisto'] * (180 / math.pi), c='#009B77')
    ax4.set_title(r'$\Delta \Omega_i$ (C-M)')
    ax4.set_xlim(lower_limit, upper_limit)
    ax4.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    ax4.xaxis.set_minor_locator(mdates.MonthLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    ax4.set_ylabel(r'Difference [deg]')
    ax4.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

    ax5.plot(time2plt, mean_anomaly_difference['Ganymede'] * (180 / math.pi), c='#EC6842')
    ax5.plot(time2plt, mean_anomaly_difference['Io'] * (180 / math.pi), c='#A50034')
    ax5.plot(time2plt, mean_anomaly_difference['Callisto'] * (180 / math.pi), c='#009B77')
    ax5.plot(time2plt, mean_anomaly_difference['Europa'] * (180 / math.pi), c='#0076C2')
    ax5.set_title(r'$\Delta \lambda_i$ (C-M)')
    ax5.set_xlim(lower_limit, upper_limit)
    ax5.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    ax5.xaxis.set_minor_locator(mdates.MonthLocator())
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    ax5.set_ylabel(r'Difference [deg]')
    ax5.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

    fig.align_ylabels([ax1, ax2, ax4])
    fig.align_ylabels([ax3, ax5])

    plt.savefig(os.path.join(image_save_path, 'dynamical_model_validation.png'))

def plot_laplace_resonance_validation():
    ### LOAD DATA ###
    # Load Kepler elements
    propagation_kepler_load_path = os.path.join(current_dir, 'validation_data/dynamical_validation',
                                                'propagated_keplerian_states.dat')
    ephemeris_kepler_load_path = os.path.join(current_dir, 'validation_data/dynamical_validation',
                                              'ephemeris_keplerian_states.dat')

    propagation_kepler_elements = np.loadtxt(propagation_kepler_load_path)
    ephemeris_kepler_elements = np.loadtxt(ephemeris_kepler_load_path)

    propagated_kepler_elements_dict = {'Io': propagation_kepler_elements[:, 1:7],
                                       'Europa': propagation_kepler_elements[:, 7:13],
                                       'Ganymede': propagation_kepler_elements[:, 13:19],
                                       'Callisto': propagation_kepler_elements[:, 19:25]}
    ephemeris_kepler_elements_dict = {'Io': ephemeris_kepler_elements[:, 1:7],
                                      'Europa': ephemeris_kepler_elements[:, 7:13],
                                      'Ganymede': ephemeris_kepler_elements[:, 13:19],
                                      'Callisto': ephemeris_kepler_elements[:, 19:25]}

    # Calculate propagated coefficients of the Laplace resonance
    mean_longitude_dict_prop = Util.calculate_mean_longitude(propagated_kepler_elements_dict)
    theta_11_prop = Util.resonance_theta(1, 1, propagated_kepler_elements_dict)
    theta_11_prop[theta_11_prop > 1.8 * math.pi] = theta_11_prop[theta_11_prop > 1.8 * math.pi] - 2 * math.pi
    theta_12_prop = Util.resonance_theta(1, 2, propagated_kepler_elements_dict)
    theta_22_prop = Util.resonance_theta(2, 2, propagated_kepler_elements_dict)
    theta_22_prop[theta_22_prop > 1.8 * math.pi] = theta_22_prop[theta_22_prop > 1.8 * math.pi] - 2 * math.pi
    theta_23_prop = Util.resonance_theta(2, 3, propagated_kepler_elements_dict)
    for i in range(len(theta_23_prop) - 1):
        if np.abs(theta_23_prop[i + 1] - theta_23_prop[i]) > math.pi and theta_23_prop[i] > theta_23_prop[i + 1]:
            theta_23_prop[i] = theta_23_prop[i] - 2 * math.pi
        elif np.abs(theta_23_prop[i + 1] - theta_23_prop[i]) > math.pi and theta_23_prop[i] < theta_23_prop[i + 1]:
            theta_23_prop[i + 1] = theta_23_prop[i + 1] - 2 * math.pi
    theta_23_prop[theta_23_prop < 0] = theta_23_prop[theta_23_prop < 0] + 2 * math.pi
    theta_23_prop[:-1][np.abs(np.diff(theta_23_prop)) > math.pi] = np.nan

    # Calculate propagated Laplace stability
    laplace_stability_prop = (mean_longitude_dict_prop['Io']
                              - 3 * mean_longitude_dict_prop['Europa']
                              + 2 * mean_longitude_dict_prop['Ganymede'])
    laplace_stability_prop = np.mod(laplace_stability_prop, 2 * math.pi)

    # Calculate ephemeris coefficients of the Laplace resonance
    mean_longitude_dict_ephem = Util.calculate_mean_longitude(ephemeris_kepler_elements_dict)
    theta_11_ephem = Util.resonance_theta(1, 1, ephemeris_kepler_elements_dict)
    theta_11_ephem[theta_11_ephem > 1.8 * math.pi] = theta_11_ephem[theta_11_ephem > 1.8 * math.pi] - 2 * math.pi
    theta_12_ephem = Util.resonance_theta(1, 2, ephemeris_kepler_elements_dict)
    theta_22_ephem = Util.resonance_theta(2, 2, ephemeris_kepler_elements_dict)
    theta_22_ephem[theta_22_ephem > 1.8 * math.pi] = theta_22_ephem[theta_22_ephem > 1.8 * math.pi] - 2 * math.pi
    theta_23_ephem = Util.resonance_theta(2, 3, ephemeris_kepler_elements_dict)
    for i in range(len(theta_23_ephem) - 1):
        if np.abs(theta_23_ephem[i + 1] - theta_23_ephem[i]) > math.pi and theta_23_ephem[i] > theta_23_ephem[i + 1]:
            theta_23_ephem[i] = theta_23_ephem[i] - 2 * math.pi
        elif np.abs(theta_23_ephem[i + 1] - theta_23_ephem[i]) > math.pi and theta_23_ephem[i] < theta_23_ephem[i + 1]:
            theta_23_ephem[i + 1] = theta_23_ephem[i + 1] - 2 * math.pi

    # Calculate ephemeris Laplace stability
    laplace_stability_ephem = (mean_longitude_dict_ephem['Io']
                               - 3 * mean_longitude_dict_ephem['Europa']
                               + 2 * mean_longitude_dict_ephem['Ganymede'])
    laplace_stability_ephem = np.mod(laplace_stability_ephem, 2 * math.pi)

    # Calculate differences in theta
    diff_11 = theta_11_prop - theta_11_ephem
    diff_12 = theta_12_prop - theta_12_ephem
    diff_22 = theta_22_prop - theta_22_ephem
    diff_23 = theta_23_prop - theta_23_ephem
    diff_23[diff_23 > 1.8 * math.pi] = diff_23[diff_23 > 1.8 * math.pi] - 2 * math.pi
    diff_23[diff_23 < -1.8 * math.pi] = diff_23[diff_23 < -1.8 * math.pi] + 2 * math.pi

    diff_11_perc = diff_11 / math.pi
    diff_12_perc = diff_12 / math.pi
    diff_22_perc = diff_22 / math.pi
    diff_23_perc = diff_23 / math.pi

    ### PLOTTING ###
    time2plt = list()
    epochs_julian_seconds = propagation_kepler_elements[:, 0]
    for epoch in epochs_julian_seconds:
        epoch_days = constants.JULIAN_DAY_ON_J2000 + epoch / constants.JULIAN_DAY
        time2plt.append(time_conversion.julian_day_to_calendar_date(epoch_days))

    lower_limit = time_conversion.julian_day_to_calendar_date(
        constants.JULIAN_DAY_ON_J2000 + 1 +
        (31.0 * constants.JULIAN_YEAR + 172.0 * constants.JULIAN_DAY) / constants.JULIAN_DAY)
    upper_limit = time_conversion.julian_day_to_calendar_date(
        constants.JULIAN_DAY_ON_J2000 - 1 +
        (34.0 * constants.JULIAN_YEAR + 365.0 * constants.JULIAN_DAY) / constants.JULIAN_DAY)

    fig = plt.figure(figsize=(16, 9))

    gs = gridspec.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax1.plot(time2plt, laplace_stability_prop * 180 / math.pi, label='Propagated', c='#A50034')
    ax1.plot(time2plt, laplace_stability_ephem * 180 / math.pi, label='NOE-5-2021', c='#EC6842',
             linestyle=(0, (5, 10)))
    ax1.set_title(r'$\Phi_L=\lambda_I-3 \lambda_E+2 \lambda_G$')
    ax1.set_xlim(lower_limit, upper_limit)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    ax1.set_ylabel(r'Laplace $\Phi_L$ [deg]')
    ax1.legend()

    ax2.plot(time2plt, theta_11_prop * 180 / math.pi, label=r'$\theta_{11}$', c='#A50034')
    ax2.plot(time2plt, theta_12_prop * 180 / math.pi, label=r'$\theta_{12}$', c='#0076C2')
    ax2.plot(time2plt, theta_22_prop * 180 / math.pi, label=r'$\theta_{22}$', c='#EC6842')
    ax2.plot(time2plt, theta_23_prop * 180 / math.pi, label=r'$\theta_{23}$', c='#009B77')
    ax2.set_title(r'$\theta_{i j}=\lambda_i-2 \lambda_j+\bar{\omega}_i$')
    ax2.set_xlim(lower_limit, upper_limit)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    ax2.set_ylabel(r'$\theta_{i j}$ [deg]')
    ax2.legend()

    ax3.plot(time2plt, diff_23_perc, label=r'$\theta_{23}$', c='#009B77')
    ax3.plot(time2plt, diff_11_perc, label=r'$\theta_{11}$', c='#A50034', linewidth=0.2)
    ax3.plot(time2plt, diff_12_perc, label=r'$\theta_{12}$', c='#0076C2')
    ax3.plot(time2plt, diff_22_perc, label=r'$\theta_{22}$', c='#EC6842', linestyle=(0, (1, 3)))
    ax3.set_title(r'$\Delta\theta_{i j}$ (C-M)')
    ax3.set_xlim(lower_limit, upper_limit)
    ax3.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    ax3.xaxis.set_minor_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    ax3.set_ylabel(r'Difference in $\theta_{i j}$ [%]')

    fig.align_ylabels([ax1, ax2])

    plt.savefig(os.path.join(image_save_path, 'laplace_resonance_validation.png'))

###########################################################################
# JUICE FLYBY VALIDATION ##################################################
###########################################################################

def plot_juice_closest_approaches():
    ### LOAD DATA ###
    satellites_state_history_load_path = (
        os.path.join(current_dir, 'validation_data/juice_validation_data/satellites_closest_approaches_state_history.dat'))
    juice_state_history_load_path = (
        os.path.join(current_dir, 'validation_data/juice_validation_data/juice_closest_approaches_state_history.dat'))
    closest_approach_input_path = (
        os.path.join(current_dir, 'validation_data/juice_validation_data/relative_distances_juice_moons.dat'))

    satellites_state_history = np.loadtxt(satellites_state_history_load_path)
    juice_state_history = np.loadtxt(juice_state_history_load_path)

    satellites_state_history_dict = dict()
    for i in range(len(satellites_state_history)):
        satellites_state_history_dict[satellites_state_history[i, 0]] = satellites_state_history[i, 1:]

    flybys_dict = Util.determine_closest_approaches(closest_approach_input_path)

    fig, ax1 = plt.subplots(1, 1, figsize=(16, 9))
    ax1.axis('equal')

    color_map_moons = ['#A50034', '#0076C2', '#EC6842', '#009B77']
    for moon_idx, moon in enumerate(flybys_dict.keys()):
        ax1.plot(satellites_state_history[:, (moon_idx * 6 + 1)] * 1E-3,
                 satellites_state_history[:, (moon_idx * 6 + 2)] * 1E-3,
                 linewidth=3, label=moon, c=color_map_moons[moon_idx], zorder=0)
        moon_flybys = flybys_dict[moon]
        for central_instants in moon_flybys:
            ax1.scatter(satellites_state_history_dict[central_instants][(moon_idx * 6 + 0)] * 1E-3,
                        satellites_state_history_dict[central_instants][(moon_idx * 6 + 1)] * 1E-3,
                        marker='o', s=50, c='#6F1D77', zorder=1)

    ax1.plot(juice_state_history[:, 1] * 1E-3, juice_state_history[:, 2] * 1E-3,
             c='#000000', alpha=0.4, linewidth=1, label='JUICE', zorder=-1)

    ax1.set_xlabel(r'$\Delta x_{0 i}$ [km]')
    ax1.set_ylabel(r'$\Delta y_{0 i}$ [km]')
    ax1.set_xlim(-2.5E6, 2.5E6)
    ax1.set_ylim(-2.5E6, 2.5E6)
    ax1.legend()

    plt.savefig(os.path.join(image_save_path, 'juice_flyby_verification.png'))

###########################################################################
# SPACE-BASED ASTROMETRY VERIFICATION #####################################
###########################################################################

def plot_space_based_astrometry_verification():
    ### LOAD DATA ###
    # Dependent variables history
    dependent_variables_load_path = os.path.join(current_dir, 'validation_data/space_based_astrometry_data',
                                                 'constraints_dependent_variables.dat')
    dependent_variable_history = np.loadtxt(dependent_variables_load_path)
    # Space-based astrometry predicted epochs
    epochs_with_errors_load_path = os.path.join(current_dir, 'validation_data/space_based_astrometry_data',
                                                'epochs_with_errors_list.dat')
    epochs_with_errors = np.loadtxt(epochs_with_errors_load_path)
    possible_epochs = epochs_with_errors[:, 0]

    ### MANIPULATE DATA ###
    ganymede_orbit_insertion_epoch = 34 * constants.JULIAN_YEAR + 335 * constants.JULIAN_DAY
    # Extract epochs and dependent variables from loaded array
    epochs = dependent_variable_history[:, 0]
    cut_off_idx = next(idx for idx, value in enumerate(epochs) if value >= ganymede_orbit_insertion_epoch)
    epochs = epochs[:cut_off_idx]
    dependent_variables = dependent_variable_history[:cut_off_idx, 1:]
    # Split dependent variables into individual lists
    distance_juice_io = dependent_variables[:cut_off_idx, 0:3]
    distance_juice_sun = dependent_variables[:cut_off_idx, 3:6]
    distance_juice_jupiter = dependent_variables[:cut_off_idx, 6:9]
    distance_jupiter_io = dependent_variables[:cut_off_idx, 9:12]
    distance_io_sun = dependent_variables[:cut_off_idx, 12:15]
    jupiter_radius = 71492E3

    sun_spacecraft_moon_angle = dict()
    jupiter_limb_spacecraft_moon_angle_large = dict()
    jupiter_limb_spacecraft_moon_angle_small = dict()
    sun_moon_spacecraft_angle = dict()

    cmap = list()

    for idx in range(len(dependent_variables)):
        ### SUN-SPACECRAFT-MOON ANGLE ###
        sun_spacecraft_moon_angle[epochs[idx]] = \
            Util.angle_between(distance_juice_sun[idx], distance_juice_io[idx]) * 180 / math.pi

        ### JUPITER-LIMB-SPACECRAFT-MOON ANGLE ###
        apparent_size_jupiter = 2 * np.arcsin(jupiter_radius / np.linalg.norm(distance_juice_jupiter[idx]))
        if apparent_size_jupiter > 4 * math.pi / 180:
            distance_juice_jupiter_limb = np.add(distance_juice_jupiter[idx],
                                                 jupiter_radius * Util.unit_vector(distance_jupiter_io[idx]))
            jupiter_limb_spacecraft_moon_angle_large[epochs[idx]] = \
                Util.angle_between(distance_juice_jupiter_limb, distance_juice_io[idx]) * 180 / math.pi
            jupiter_limb_spacecraft_moon_angle_small[epochs[idx]] = np.nan
        else:
            distance_juice_jupiter_limb = np.add(distance_juice_jupiter[idx],
                                                 jupiter_radius * Util.unit_vector(distance_jupiter_io[idx]))
            jupiter_limb_spacecraft_moon_angle_small[epochs[idx]] = \
                Util.angle_between(distance_juice_jupiter_limb, distance_juice_io[idx]) * 180 / math.pi
            jupiter_limb_spacecraft_moon_angle_large[epochs[idx]] = np.nan

        ### SUN-MOON-SPACECRAFT ANGLE ###
        sun_moon_spacecraft_angle[epochs[idx]] = \
            Util.angle_between(distance_io_sun[idx], -1 * distance_juice_io[idx]) * 180 / math.pi

        ### COLORMAP ###
        if epochs[idx] in possible_epochs:
            cmap.append('#0076C2')
        else:
            cmap.append('#A50034')

    ### PLOTTING ###
    time2plt = list()
    for epoch in epochs:
        epoch_days = constants.JULIAN_DAY_ON_J2000 + epoch / constants.JULIAN_DAY
        time2plt.append(time_conversion.julian_day_to_calendar_date(epoch_days))

    fig, (ax1, ax3, ax4, ax2) = plt.subplots(4, 1, figsize=(15, 20.5), sharex=True)
    x_lim_min = time2plt[0] - datetime.timedelta(days=30)
    x_lim_max = time2plt[-1] + datetime.timedelta(days=30)
    x = np.arange(x_lim_min, x_lim_max, (time2plt[1]-time2plt[0]))

    ax1.scatter(time2plt, sun_spacecraft_moon_angle.values(), s=1, c=cmap)
    ax1.fill_between(x=x, y1=30, y2=-5, color='grey', alpha=0.35, zorder=-1)
    ax1.hlines(y=30, xmin=x_lim_min, xmax=x_lim_max, color='grey', linewidth=1.5, linestyle='-')

    ax1.set_title(r'Sun-Spacecraft-Moon Angle')
    ax1.set_ylabel(r'Angle [deg]')
    ax1.set_ylim(-5, 185)

    ax3.scatter(time2plt, sun_moon_spacecraft_angle.values(), s=1, c=cmap)
    ax3.fill_between(x=x, y1=130, y2=185, color='grey', alpha=0.35, zorder=-1)
    ax3.hlines(y=130, xmin=x_lim_min, xmax=x_lim_max, color='grey', linewidth=1.5, linestyle='-')

    ax3.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    ax3.xaxis.set_minor_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    ax3.set_title(r'Sun-Moon-Spacecraft Angle')
    ax3.set_ylabel(r'Angle [deg]')
    ax3.set_xlim(x_lim_min, x_lim_max)
    ax3.set_ylim(-5, 185)

    ax2.scatter(time2plt, jupiter_limb_spacecraft_moon_angle_large.values(), s=1, c=cmap)
    ax2.fill_between(x=x, y1=5, y2=-1, color='grey', alpha=0.35, zorder=-1)
    ax2.hlines(y=5, xmin=x_lim_min, xmax=x_lim_max, color='grey', linewidth=1.5, linestyle='-')

    ax2.set_title(r'Jupiter-Limb-Spacecraft-Moon Angle - Apparent Size > 4 Degrees')
    ax2.set_ylabel(r'Angle [deg]')
    ax2.set_ylim(-1, 32)

    ax4.scatter(time2plt, jupiter_limb_spacecraft_moon_angle_small.values(), s=1, c=cmap)
    ax4.fill_between(x=x, y1=10, y2=-1, color='grey', alpha=0.35, zorder=-1)
    ax4.hlines(y=10, xmin=x_lim_min, xmax=x_lim_max, color='grey', linewidth=1.5, linestyle='-')

    ax4.set_title(r'Jupiter-Limb-Spacecraft-Moon Angle - Apparent Size < 4 Degrees')
    ax4.set_ylabel(r'Angle [deg]')
    ax4.set_ylim(-1, 32)

    plt.savefig(os.path.join(image_save_path, 'space_based_astrometry_verification.png'))

def plot_space_based_astrometry_arcs():
    ### LOAD DATA ###
    # Dependent variables history
    dependent_variables_load_path = os.path.join(current_dir, 'validation_data/space_based_astrometry_data',
                                                 'constraints_dependent_variables.dat')
    dependent_variable_history = np.loadtxt(dependent_variables_load_path)
    # Space-based astrometry predicted epochs
    epochs_with_errors_load_path = os.path.join(current_dir, 'validation_data/space_based_astrometry_data',
                                                'epochs_with_errors_list.dat')
    epochs_with_errors = np.loadtxt(epochs_with_errors_load_path)

    ### MANIPULATE DATA ###
    ganymede_orbit_insertion_epoch = 34 * constants.JULIAN_YEAR + 335 * constants.JULIAN_DAY
    # Extract epochs and dependent variables from loaded array
    epochs2investigate = dependent_variable_history[:, 0]
    cut_off_idx = next(idx for idx, value in enumerate(epochs2investigate) if value >= ganymede_orbit_insertion_epoch)
    epochs2investigate = epochs2investigate[:cut_off_idx]

    uncertainty_history_astrometry = list()
    distance_history_astrometry = list()
    idx = 0
    for epoch_idx, epoch in enumerate(epochs2investigate):
        if epoch in epochs_with_errors[:, 0]:
            uncertainty_history_astrometry.append([epochs_with_errors[idx, 1],
                                                   epochs_with_errors[idx, 2]])
            distance_history_astrometry.append(epochs_with_errors[idx, 3])
            idx += 1
        else:
            uncertainty_history_astrometry.append([np.nan, np.nan])
            distance_history_astrometry.append(np.nan)

    uncertainty_history_astrometry = np.array(uncertainty_history_astrometry)
    distance_history_astrometry = np.array(distance_history_astrometry)

    ### PLOTTING ###
    time2plt = list()
    for epoch in epochs2investigate:
        epoch_days = constants.JULIAN_DAY_ON_J2000 + epoch / constants.JULIAN_DAY
        time2plt.append(time_conversion.julian_day_to_calendar_date(epoch_days))

    fig, axs = plt.subplots(4, 1, figsize=(16, 20.5), sharex=True, constrained_layout=True)

    x_lim_min = time2plt[0] - datetime.timedelta(days=30)
    x_lim_max = time_conversion.julian_day_to_calendar_date(constants.JULIAN_DAY_ON_J2000 - 1
                                                            + (35 * constants.JULIAN_YEAR) / constants.JULIAN_DAY)

    y_lim_arcsec_min = np.nanmin(uncertainty_history_astrometry[:, 0:2] * 3600 * 180 / math.pi) * 0.95
    delta_y_lim_arcsec_min = np.nanmin(uncertainty_history_astrometry[:, 0:2] * 3600 * 180 / math.pi) * 0.05
    y_lim_arcsec_max = np.nanmax(uncertainty_history_astrometry[:, 0:2] * 3600 * 180 / math.pi) + delta_y_lim_arcsec_min

    y_lim_metric_min = \
        np.nanmin(2 * np.multiply(np.tan(0.5 * uncertainty_history_astrometry[:, 1]),
                                  distance_history_astrometry * 1E-3)) * 0.90
    delta_y_lim_metric_min = \
        np.nanmin(2 * np.multiply(np.tan(0.5 * uncertainty_history_astrometry[:, 1]),
                                  distance_history_astrometry * 1E-3)) * 0.10
    y_lim_metric_max = \
        np.nanmax(2 * np.multiply(np.tan(0.5 * uncertainty_history_astrometry[:, 0]),
                                  distance_history_astrometry * 1E-3)) + delta_y_lim_metric_min

    plt1 = axs[0].scatter(time2plt, uncertainty_history_astrometry[:, 1] * 3600 * 180 / math.pi, s=1,
                          c=distance_history_astrometry * 1E-3, cmap='viridis')

    axs[0].set_ylim(y_lim_arcsec_min, y_lim_arcsec_max)
    axs[0].set_ylabel(r'Uncertainty DEC $\sigma_{\delta}$ [arcsec]')

    axs[1].scatter(
        time2plt, 2 * np.tan(0.5 * uncertainty_history_astrometry[:, 1]) * distance_history_astrometry * 1E-3,
        s=1, c=distance_history_astrometry * 1E-3, cmap='viridis')

    axs[1].set_ylim(y_lim_metric_min, y_lim_metric_max)
    axs[1].set_ylabel(r'Uncertainty DEC $\sigma_{\alpha}$ [km]')

    axs[2].scatter(time2plt, uncertainty_history_astrometry[:, 0] * 3600 * 180 / math.pi, s=1,
                   c=distance_history_astrometry * 1E-3, cmap='viridis')

    axs[2].set_ylim(y_lim_arcsec_min, y_lim_arcsec_max)
    axs[2].set_ylabel(r'Uncertainty RA $\sigma_{\alpha}$ [arcsec]')

    axs[3].scatter(
        time2plt, 2 * np.tan(0.5 * uncertainty_history_astrometry[:, 0]) * distance_history_astrometry * 1E-3,
        s=1, c=distance_history_astrometry * 1E-3, cmap='viridis')

    axs[3].xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    axs[3].xaxis.set_minor_locator(mdates.MonthLocator())
    axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    axs[3].set_xlim(x_lim_min, x_lim_max)
    axs[3].set_ylim(y_lim_metric_min, y_lim_metric_max)
    axs[3].set_ylabel(r'Uncertainty RA $\sigma_{\alpha}$ [km]')

    fig.align_ylabels(axs)

    cbar = fig.colorbar(plt1, ax=axs, shrink=0.95, location='bottom', pad=0.015, aspect=60)
    cbar.set_label('Distance JUICE-Io [km]', labelpad=10)
    cbar.formatter.set_powerlimits((-3, 3))

    plt.savefig(os.path.join(image_save_path, 'space_based_astrometry_uncertainty_alpha_and_delta.png'))
