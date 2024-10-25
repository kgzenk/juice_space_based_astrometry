
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os
import math
import datetime
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
# tudatpy imports
from tudatpy.kernel import constants
from tudatpy.util import redirect_std
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro import time_conversion
from tudatpy.kernel.interface import spice_interface
# Problem-specific imports
import src.core.space_based_astrometry.space_based_astrometry_utilities as Util
from src.core.tudat_setup_util.propagation.prop_full_setup import create_global_environment
# Estimation utilities
from src.misc.estimation_dynamical_model import estimate_satellites_initial_state

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
image_save_path = os.path.join(current_dir, 'estimation_plots')

###########################################################################
# DEFINE GLOBAL VARIABLES #################################################
###########################################################################

### DEFINE SIMULATION SETTINGS ###
# JUICE Jupiter Orbit Insertion July 2031
simulation_start_epoch = 31.0 * constants.JULIAN_YEAR + 182.0 * constants.JULIAN_DAY
# Ganymede Orbit Insertion December 2034 (1st)
simulation_end_epoch = 34.0 * constants.JULIAN_YEAR + 355.0 * constants.JULIAN_DAY

### CREATE ENVIRONMENT ###
spice_interface.load_kernel('../tudat_setup_util/spice/JUICE/juice_crema_5_1_150lb_23_1.tm.txt')

bodies, body_settings, propagator_settings = \
    create_global_environment(simulation_start_epoch, simulation_end_epoch)

### ESTIMATE BETTER INITIAL STATE OF IO ###
initial_state_estimated = (
    estimate_satellites_initial_state(bodies, simulation_start_epoch, simulation_end_epoch, propagator_settings))

### UPDATE TABULATED EPHEMERIS OF IO ###
propagator_settings.initial_states = initial_state_estimated
propagator_settings.processing_settings.set_integrated_result = True
# Run propagation
with redirect_std():
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
# Retrieve nominal state history
nominal_state_history = dynamics_simulator.state_history

###########################################################################
# MAIN ####################################################################
###########################################################################

def main():
    ### MANIPULATE DATE AND DEFINE COHERENT ASTROMETRY-ARCS ###
    epochs_with_errors_dict, epochs_with_errors_metric_dict, juice_wrt_to_io_normalized, single_arcs_astrometry, \
        single_arcs_astrometry_idx_list = Util.get_astrometry_arcs(simulation_end_epoch=simulation_end_epoch)

    total_number_of_observations = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]

    observation_epochs_per_number_of_obs = dict()

    for total_number_of_observations_iter in total_number_of_observations:
        # Initialize containers
        observation_epochs_random_brute_force = list()
        observation_epochs_geometry_brute_force = list()
        observation_epochs_uncertainty_brute_force = list()
        observation_epochs_uncertainty_hybrid = list()

        ### RANDOM BRUTE-FORCE ###
        epochs_comprehensive = np.ravel(list(epochs_with_errors_dict.keys()))

        ### GEOMETRY DRIVEN ###
        observation_angles_dict = dict()
        observation_angles_comprehensive = list()

        for epoch in epochs_comprehensive:
            observation_angle_with_optical_plane = \
                np.arccos(np.clip(np.dot(juice_wrt_to_io_normalized[epoch], np.array([0.0, 1.0, 0.0])), -1.0, 1.0))
            observation_angles_dict[epoch] = observation_angle_with_optical_plane
            observation_angles_comprehensive.append(observation_angle_with_optical_plane)

        observation_angles_comprehensive = np.array(observation_angles_comprehensive)

        ### UNCERTAINTY DRIVEN ###
        uncertainty_right_ascension_metric = \
            np.vstack(np.array(list(epochs_with_errors_metric_dict.values()))[:, 0])

        uncertainty_right_ascension_comprehensive = np.vstack(
            np.array(list(epochs_with_errors_metric_dict.values()))[:, 0])
        epochs_uncertainty_concatenated_metric = \
            np.concatenate((np.vstack(epochs_comprehensive), uncertainty_right_ascension_comprehensive), axis=1)
        epochs_sorted = \
            epochs_uncertainty_concatenated_metric[epochs_uncertainty_concatenated_metric[:, 1].argsort()][:, 0]

        for global_seed in range(42, 52):
            ### RANDOM BRUTE-FORCE ###
            observation_epochs_random_brute_force_temp = dict()
            # Initialise first random Generator for uniform selection of the arcs of observation
            rng_uniform_random_brute_force = np.random.default_rng(seed=global_seed)
            epochs_of_observation = \
                np.sort(rng_uniform_random_brute_force.choice(epochs_comprehensive,
                                                              size=total_number_of_observations_iter, replace=False))
            for epoch in epochs_of_observation:
                observation_epochs_random_brute_force_temp[epoch] = epochs_with_errors_dict[epoch]

            observation_epochs_random_brute_force.append(observation_epochs_random_brute_force_temp)

            ### GEOMETRY DRIVEN BRUTE FORCE ###
            observation_epochs_geometry_brute_force_temp = dict()

            geometric_angle_spread = \
                np.divide(np.max(observation_angles_comprehensive) - np.min(observation_angles_comprehensive),
                          (total_number_of_observations_iter - 1))

            rng_uniform_arcs = np.random.default_rng(seed=global_seed)
            initial_epoch_of_observation = \
                np.sort(rng_uniform_arcs.choice(epochs_comprehensive, size=1, replace=False))

            current_angle = observation_angles_dict[initial_epoch_of_observation[0]]
            desired_angles = [current_angle]

            for i in range(total_number_of_observations_iter - 1):
                current_angle = current_angle + geometric_angle_spread
                if current_angle > np.max(observation_angles_comprehensive):
                    current_angle = np.min(observation_angles_comprehensive) + \
                                    (current_angle - np.max(observation_angles_comprehensive))
                desired_angles.append(current_angle)
            desired_angles = np.array(desired_angles)

            angles_array_temp = observation_angles_comprehensive.copy()
            epochs_array_temp = epochs_comprehensive.copy()

            for angle in desired_angles:
                observation_epoch_idx = (np.abs(angles_array_temp - angle)).argmin()
                angles_array_temp = np.delete(angles_array_temp, observation_epoch_idx)
                epoch = epochs_array_temp[observation_epoch_idx]
                epochs_array_temp = np.delete(epochs_array_temp, observation_epoch_idx)
                observation_epochs_geometry_brute_force_temp[epoch] = epochs_with_errors_dict[epoch]

            observation_epochs_geometry_brute_force.append(observation_epochs_geometry_brute_force_temp)

            ### UNCERTAINTY DRIVEN BRUTE FORCE ###
            observation_epochs_uncertainty_brute_force_temp = dict()

            # Initialise first random Generator for uniform selection of the arcs of observation
            rng_uniform_arcs = np.random.default_rng(seed=global_seed)
            epochs_of_observation = \
                np.sort(rng_uniform_arcs.choice(epochs_sorted[:int(1.2 * total_number_of_observations_iter)],
                                                size=total_number_of_observations_iter, replace=False))

            for epoch in epochs_of_observation:
                observation_epochs_uncertainty_brute_force_temp[epoch] = epochs_with_errors_dict[epoch]

            observation_epochs_uncertainty_brute_force.append(observation_epochs_uncertainty_brute_force_temp)

            ### UNCERTAINTY DRIVEN HYBRID ###
            observation_epochs_hybrid_temp = dict()

            geometric_angle_spread = \
                np.divide(np.max(observation_angles_comprehensive) - np.min(observation_angles_comprehensive),
                          total_number_of_observations_iter - 1)

            rng_uniform_arcs = np.random.default_rng(seed=global_seed)
            initial_epoch_of_observation = \
                np.sort(rng_uniform_arcs.choice(epochs_sorted, size=1, replace=False))
            current_angle = observation_angles_dict[initial_epoch_of_observation[0]]
            desired_angles = [current_angle]
            for i in range(total_number_of_observations_iter - 1):
                current_angle = current_angle + geometric_angle_spread
                if current_angle > np.max(observation_angles_comprehensive):
                    current_angle = np.min(observation_angles_comprehensive) + \
                                    (current_angle - np.max(observation_angles_comprehensive))
                desired_angles.append(current_angle)
            desired_angles = np.array(desired_angles)

            angle_error_bound = 0.1
            constant_factor = total_number_of_observations_iter
            epochs_sorted_hybrid = epochs_sorted.copy()
            epochs_sorted_leftover = epochs_sorted.copy()

            multiply_factor_range = len(epochs_sorted_hybrid) // constant_factor

            while len(desired_angles) != 0:
                for multiply_factor in range(1, multiply_factor_range):
                    if len(desired_angles) != 0:
                        lower_boundary = (multiply_factor - 1) * constant_factor
                        upper_boundary = multiply_factor * constant_factor
                        epochs_sorted_temp = epochs_sorted_hybrid[lower_boundary:upper_boundary]
                        observation_angles_temp = list()
                        for epoch in epochs_sorted_temp:
                            observation_angles_temp.append(observation_angles_dict[epoch])
                        observation_angles_temp = np.array(observation_angles_temp)

                        desired_angles_leftover = list()
                        for idx, angle in enumerate(desired_angles):
                            if (1 - angle_error_bound) * angle <= \
                                    observation_angles_temp[np.abs(observation_angles_temp - angle).argmin()] \
                                    <= (1 + angle_error_bound) * angle:
                                observation_epoch_idx = (np.abs(observation_angles_temp - angle)).argmin()
                                epoch = epochs_sorted_temp[observation_epoch_idx]
                                observation_angles_temp = np.delete(observation_angles_temp, observation_epoch_idx)
                                epochs_sorted_temp = np.delete(epochs_sorted_temp, observation_epoch_idx)
                                epochs_sorted_leftover = np.delete(epochs_sorted_leftover,
                                                                   np.abs(epochs_sorted_leftover - epoch).argmin())
                                observation_epochs_hybrid_temp[epoch] = epochs_with_errors_dict[epoch]
                            else:
                                desired_angles_leftover.append(angle)
                        desired_angles = np.array(desired_angles_leftover).copy()
                    else:
                        break
                # Loosen error-bound for angular spread
                if 2 * angle_error_bound <= 1:
                    angle_error_bound = 2 * angle_error_bound
                else:
                    angle_error_bound = 1
                # Update the list of still available epochs
                epochs_sorted_hybrid = epochs_sorted_leftover.copy()

            observation_epochs_uncertainty_hybrid.append(observation_epochs_hybrid_temp)

        observation_epochs_per_number_of_obs[total_number_of_observations_iter] = {
            0: np.array(observation_epochs_random_brute_force),
            1: np.array(observation_epochs_geometry_brute_force),
            2: np.array(observation_epochs_uncertainty_brute_force),
            3: np.array(observation_epochs_uncertainty_hybrid)
        }

    plot_angles_test(observation_epochs_per_number_of_obs)


def plot_angles_test(observation_epochs_per_number_of_obs):
    ### MANIPULATE DATE AND DEFINE COHERENT ASTROMETRY-ARCS ###
    epochs_with_errors_dict, epochs_with_errors_metric_dict, juice_wrt_to_io_normalized, single_arcs_astrometry, \
        single_arcs_astrometry_idx_list = Util.get_astrometry_arcs(simulation_end_epoch=simulation_end_epoch)

    juice_angle_with_optical_plane = Util.get_angles_with_unit_vectors(juice_wrt_to_io_normalized)

    epochs_comprehensive = np.vstack(list(epochs_with_errors_metric_dict.keys()))
    uncertainty_right_ascension_comprehensive = np.vstack(np.array(list(epochs_with_errors_metric_dict.values()))[:, 0])
    uncertainty_declination_comprehensive = np.vstack(np.array(list(epochs_with_errors_metric_dict.values()))[:, 1])
    epochs_uncertainty_concatenated_metric = \
        np.concatenate((epochs_comprehensive, uncertainty_right_ascension_comprehensive,
                        uncertainty_declination_comprehensive), axis=1)

    ### PROCESSING & WRITING OUTPUTS ###
    time2plt = list()
    angles2plot = list()
    epochs_with_errors_metric_cbar = list()
    for epoch in epochs_with_errors_dict.keys():
        angles2plot.append(juice_angle_with_optical_plane[epoch])
        epochs_with_errors_metric_cbar.append(epochs_with_errors_metric_dict[epoch][0])
        epoch_days = constants.JULIAN_DAY_ON_J2000 + epoch / constants.JULIAN_DAY
        time2plt.append(time_conversion.julian_day_to_calendar_date(epoch_days))
    time2plt = np.array(time2plt)
    angles2plot = np.array(angles2plot)

    lower_limit = time2plt[0] - datetime.timedelta(days=30)
    upper_limit = time_conversion.julian_day_to_calendar_date(constants.JULIAN_DAY_ON_J2000 - 1
                                                              + (35 * constants.JULIAN_YEAR) / constants.JULIAN_DAY)

    ### OBSERVATION ANGLE THREE-DIMENSIONAL ###
    fig, axs = plt.subplots(3, 1, figsize=(7.6, 19.95), sharex=True, sharey=True, constrained_layout=True)

    plt1 = axs[0].scatter(time2plt, angles2plot[:, 0] * 180 / math.pi, s=2,
                          c=np.multiply(epochs_with_errors_metric_cbar, 1E-3), cmap='viridis',
                          vmin=np.min(epochs_uncertainty_concatenated_metric[:, 1]) * 1E-3,
                          vmax=np.max(epochs_uncertainty_concatenated_metric[:, 1]) * 1E-3)
    axs[0].set_ylim(-10, 190)
    axs[0].set_xlim(lower_limit, upper_limit)
    axs[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    axs[0].xaxis.set_minor_locator(mdates.MonthLocator())
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    axs[0].set_title(r'Radial')
    axs[0].set_ylabel(r'Observation Angle [deg]')

    axs[1].scatter(time2plt, angles2plot[:, 1] * 180 / math.pi, s=2,
                   c=np.multiply(epochs_with_errors_metric_cbar, 1E-3), cmap='viridis',
                   vmin=np.min(epochs_uncertainty_concatenated_metric[:, 1]) * 1E-3,
                   vmax=np.max(epochs_uncertainty_concatenated_metric[:, 1]) * 1E-3)
    axs[1].set_ylim(-10, 190)
    axs[1].set_xlim(lower_limit, upper_limit)
    axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    axs[1].xaxis.set_minor_locator(mdates.MonthLocator())
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    axs[1].set_title(r'Along-Track')
    axs[1].set_ylabel(r'Observation Angle [deg]')

    axs[2].scatter(time2plt, angles2plot[:, 2] * 180 / math.pi, s=2,
                   c=np.multiply(epochs_with_errors_metric_cbar, 1E-3), cmap='viridis',
                   vmin=np.min(epochs_uncertainty_concatenated_metric[:, 1]) * 1E-3,
                   vmax=np.max(epochs_uncertainty_concatenated_metric[:, 1]) * 1E-3)
    axs[2].set_ylim(-10, 190)
    axs[2].set_xlim(lower_limit, upper_limit)
    axs[2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    axs[2].xaxis.set_minor_locator(mdates.MonthLocator())
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    axs[2].set_title(r'Normal')
    axs[2].set_ylabel(r'Observation Angle [deg]')

    cbar = fig.colorbar(plt1, ax=axs, shrink=0.90, location='bottom', pad=0.01, aspect=30)
    cbar.set_label(r'Uncertainty Right Ascension $\sigma_{\alpha}$ [km]', labelpad=10)

    plt.savefig(os.path.join(image_save_path, 'observation_angles_three_dimensional.png'))

    ### IMAGINARY SQUARE PLOT ###
    fig, axs = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
    observation_epochs_per_number_of_obs_temp = observation_epochs_per_number_of_obs[2560]#

    axs[0, 0].set_title('Purely Randomised')
    axs[0, 1].set_title('Geometry Driven')
    axs[1, 0].set_title('Uncertainty Driven')
    axs[1, 1].set_title('Hybrid Approach')

    axs[0, 0].set_ylabel(r'Along-Track Observation Angle [deg]')
    axs[1, 0].set_ylabel(r'Along-Track Observation Angle [deg]')
    axs[1, 0].set_xlabel(r'Radial Observation-Angle [deg]')
    axs[1, 1].set_xlabel(r'Radial Observation-Angle [deg]')

    axs[0, 0].xaxis.set_tick_params(labelbottom=False)
    axs[0, 1].xaxis.set_tick_params(labelbottom=False)
    axs[0, 1].yaxis.set_tick_params(labelleft=False)
    axs[1, 1].yaxis.set_tick_params(labelleft=False)

    for idx, ax in enumerate(np.ravel(axs)):

        ### PROCESSING INPUTS ###
        current_observation_epochs = observation_epochs_per_number_of_obs_temp[idx][0]
        angles2plot = list()
        angles2plot_red = list()
        epochs_with_errors_metric_cbar = list()
        for epoch in epochs_with_errors_dict.keys():
            if epoch in current_observation_epochs:
                angles2plot_red.append(juice_angle_with_optical_plane[epoch])
                epochs_with_errors_metric_cbar.append(epochs_with_errors_metric_dict[epoch][0])
            else:
                angles2plot.append(juice_angle_with_optical_plane[epoch])

        angles2plot = np.array(angles2plot)
        angles2plot_red = np.array(angles2plot_red)

        ax.axis('equal')
        ax.set_ylim(-5, 185)
        ax.xaxis.set_ticks(np.arange(-30, 215, 30))
        ax.yaxis.set_ticks(np.arange(0, 185, 30))
        ax.scatter(angles2plot[:, 0] * 180 / math.pi, angles2plot[:, 1] * 180 / math.pi, c='#919393', alpha=0.6)
        if idx == 0:
            plt1 = ax.scatter(angles2plot_red[:, 0] * 180 / math.pi, angles2plot_red[:, 1] * 180 / math.pi,
                              c=np.multiply(epochs_with_errors_metric_cbar, 1E-3), cmap='viridis',
                              vmin=np.min(epochs_uncertainty_concatenated_metric[:, 1]) * 1E-3,
                              vmax=np.max(epochs_uncertainty_concatenated_metric[:, 1]) * 1E-3)
        else:
            ax.scatter(angles2plot_red[:, 0] * 180 / math.pi, angles2plot_red[:, 1] * 180 / math.pi,
                       c=np.multiply(epochs_with_errors_metric_cbar, 1E-3), cmap='viridis',
                       vmin=np.min(epochs_uncertainty_concatenated_metric[:, 1]) * 1E-3,
                       vmax=np.max(epochs_uncertainty_concatenated_metric[:, 1]) * 1E-3)

    cbar = fig.colorbar(plt1, ax=axs, shrink=0.95, location='bottom', pad=0.015, aspect=60)
    cbar.set_label(r'Uncertainty Right Ascension $\sigma_{\alpha}$ [km]', labelpad=10)

    plt.savefig(os.path.join(image_save_path, 'angles_with_orbital_plane.png'))


if __name__ == "__main__":
    main()
