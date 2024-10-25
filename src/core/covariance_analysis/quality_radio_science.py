
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os
import numpy as np
import multiprocessing as mp
from functools import partial
# tudatpy imports
from tudatpy.kernel import constants
from tudatpy.util import redirect_std
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import estimation, estimation_setup
# Problem-specific imports
from src.core.tudat_setup_util.propagation.prop_full_setup import \
    create_global_environment
import src.core.space_based_astrometry.space_based_astrometry_utilities as Util
# Estimation utilities
from src.misc.estimation_dynamical_model import \
    estimate_satellites_initial_state, covariance_initial_state_all_moons

# Get path of current directory
current_dir = os.path.dirname(__file__)
raw_data_output_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'data_cov_analysis')

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

output_path_initial_state = os.path.join(raw_data_output_path, 'uncertainty_variations')

# Check if directory already exists, otherwise create
if not os.path.exists(output_path_initial_state):
    os.makedirs(output_path_initial_state)

np.savetxt(os.path.join(output_path_initial_state, 'initial_state.dat'), initial_state_estimated, fmt='%.18e')

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
    epochs_with_errors_dict, epochs_with_errors_metric_dict, juice_wrt_to_io_normalized, \
        single_arcs_astrometry, single_arcs_astrometry_idx_list = Util.get_astrometry_arcs()

    total_number_of_observations = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]

    observation_epochs_per_number_of_obs = dict()

    for total_number_of_observations_iter in total_number_of_observations:
        # Initialize containers
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
        uncertainty_right_ascension_comprehensive = np.vstack(
            np.array(list(epochs_with_errors_metric_dict.values()))[:, 0])
        epochs_uncertainty_concatenated_metric = \
            np.concatenate((np.vstack(epochs_comprehensive), uncertainty_right_ascension_comprehensive), axis=1)
        epochs_sorted = \
            epochs_uncertainty_concatenated_metric[epochs_uncertainty_concatenated_metric[:, 1].argsort()][:, 0]

        for global_seed in range(42, 142):
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

        observation_epochs_per_number_of_obs[total_number_of_observations_iter] = \
            np.array(observation_epochs_uncertainty_hybrid)

    epoch_true_to_formal_ratio_iterable = list()

    for total_number_of_observations_iter in total_number_of_observations:
        true_to_formal_error_ratio = [1, 4, 7, 10, 13, 16, 19, 22, 25]
        space_based_astrometry_uncertainty_ratio = [1.0]
        observation_epochs_comprehensive = observation_epochs_per_number_of_obs[total_number_of_observations_iter]
        epoch_true_to_formal_ratio_iterable.append(np.stack(
            np.meshgrid(observation_epochs_comprehensive,
                        true_to_formal_error_ratio,
                        space_based_astrometry_uncertainty_ratio), -1).reshape(-1, 3))

    epoch_true_to_formal_ratio_iterable = np.array(epoch_true_to_formal_ratio_iterable)

    ### A-PRIORI COVARIANCE IO ###
    apriori_covariance = covariance_initial_state_all_moons(bodies, simulation_start_epoch, simulation_end_epoch,
                                                            propagator_settings, nominal_state_history)

    np.savetxt(os.path.join(output_path_initial_state, 'apriori_covariance.dat'), apriori_covariance, fmt='%.18e')

    # Loop over different total numbers of observation
    for idx, total_number_of_observations_iter in enumerate(total_number_of_observations):
        # Get current dict with observation epochs per method
        epoch_true_to_formal_ratio_iterable_temp = epoch_true_to_formal_ratio_iterable[idx]
        epoch_true_to_formal_ratio_iterable_temp = \
            epoch_true_to_formal_ratio_iterable_temp[np.lexsort((epoch_true_to_formal_ratio_iterable_temp[:, 2],
                                                                 epoch_true_to_formal_ratio_iterable_temp[:, 1]))]

        ### PARALLELIZE ESTIMATION OF PARAMETERS ###
        n_cores = mp.cpu_count() // 2
        with mp.get_context("spawn").Pool(n_cores) as pool:
            true_to_formal_error_ratio, uncertainty_factor_space_based_astrometry, formal_errors, correlations = \
                zip(*pool.starmap(partial(covariance_com_cof_effect_true_to_formal,
                                          apriori_covariance=apriori_covariance),
                                  iterable=epoch_true_to_formal_ratio_iterable_temp))

        output_path = (
            os.path.join(raw_data_output_path, 'uncertainty_variations', str(total_number_of_observations_iter)))

        # Check if directory already exists, otherwise create
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        ### SAVE DATA ###
        np.savetxt(os.path.join(output_path, 'formal_errors.dat'), formal_errors, fmt='%.18e')
        np.savetxt(os.path.join(output_path, 'correlations.dat'), correlations, fmt='%.18e')
        np.savetxt(os.path.join(output_path, 'scaling_factors.dat'),
                   np.concatenate([np.vstack(true_to_formal_error_ratio),
                                   np.vstack(uncertainty_factor_space_based_astrometry)], axis=1), fmt='%.18e')

###########################################################################
# ESTIMATION FUNCTIONALITY ################################################
###########################################################################

def covariance_com_cof_effect_true_to_formal(observation_epochs_manual,
                                             true_to_formal_error_ratio,
                                             uncertainty_factor_space_based_astrometry,
                                             apriori_covariance,
                                             com_cof_offset_local=None):

    ### ADD COF TO THE ENVIRONMENT ###
    if com_cof_offset_local is None:
        com_cof_offset_local = [0.0, 0.0, 0.0]
    environment_setup.add_ground_station(bodies.get_body('Io'), 'COF', com_cof_offset_local)

    ### CREATE LINK END FOR IO ###
    link_ends_cof = dict()
    link_ends_cof[estimation_setup.observation.transmitter] = (estimation_setup.observation.
                                                               body_reference_point_link_end_id('Io', 'COF'))
    link_ends_cof[estimation_setup.observation.receiver] = (estimation_setup.observation.
                                                            body_origin_link_end_id('JUICE'))
    link_definition_cof = estimation_setup.observation.LinkDefinition(link_ends_cof)

    ### OBSERVATION MODEL SETTINGS ###
    angular_position_observation_settings_cof = [estimation_setup.observation.angular_position(link_definition_cof)]

    ### OBSERVATIONS SIMULATION SETTINGS ###
    # Define epochs at which the ephemerides shall be checked
    observation_times = np.sort(np.array((list(observation_epochs_manual.keys()))), axis=0)
    observation_simulation_settings = list()
    noise_level = list()
    for current_epoch in observation_times:
        # Retrieve associated uncertainties for both right ascension and declination at current epoch
        uncertainty_right_ascension = (uncertainty_factor_space_based_astrometry *
                                       observation_epochs_manual[current_epoch][0])
        uncertainty_declination = (uncertainty_factor_space_based_astrometry *
                                   observation_epochs_manual[current_epoch][1])
        noise_level.append(uncertainty_right_ascension)
        noise_level.append(uncertainty_declination)
        # Create the observation simulation settings for Io at the current epoch
        observation_simulation_settings_per_epoch = estimation_setup.observation.tabulated_simulation_settings(
            estimation_setup.observation.angular_position_type,
            link_definition_cof,
            [current_epoch])
        observation_simulation_settings.append(observation_simulation_settings_per_epoch)

    ### SIMULATE SPACE-BASED ASTROMETRIC OBSERVATIONS OF IO ###
    # Create observation simulators
    angular_position_observation_simulators = estimation_setup.create_observation_simulators(
        angular_position_observation_settings_cof, bodies)
    # Get angular position measurements as ObservationCollection
    angular_position_measurements_cof = estimation.simulate_observations(
        observation_simulation_settings,
        angular_position_observation_simulators,
        bodies)

    ### PARAMETERS TO ESTIMATE ###
    # Perturb initial state and update set of estimable parameters
    parameters_to_estimate_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
    parameters_to_estimate_settings.append(estimation_setup.parameter.ground_station_position('Io', 'COF'))
    parameters_to_estimate = estimation_setup.create_parameter_set(parameters_to_estimate_settings, bodies)

    ### PERFORM THE ESTIMATION ###
    # Create the estimator
    estimator = numerical_simulation.Estimator(bodies, parameters_to_estimate,
                                               angular_position_observation_settings_cof, propagator_settings)

    """
    apriori_covariance_scaled = np.zeros(np.shape(apriori_covariance))
    for row_idx, row in enumerate(apriori_covariance):
        for column_idx, value in enumerate(row):
            if row_idx < 6 and column_idx < 6:
                apriori_covariance_scaled[row_idx, column_idx] = (true_to_formal_error_ratio ** 2) * value
            elif row_idx < 6 or column_idx < 6:
                apriori_covariance_scaled[row_idx, column_idx] = true_to_formal_error_ratio * value
            else:
                apriori_covariance_scaled[row_idx, column_idx] = value
                
    """

    ### ALTER THE A PRIORI COVARIANCE MATRIX ###
    inverse_apriori_covariance = np.linalg.inv((true_to_formal_error_ratio ** 2) * apriori_covariance)

    # Add additional rows of zero
    columns_to_be_added = np.zeros((3, len(inverse_apriori_covariance)))
    inverse_apriori_covariance = \
        np.hstack((inverse_apriori_covariance, np.atleast_2d(columns_to_be_added).T))

    for n in range(-3, 0):
        additional_row_zeros = np.zeros((1, len(inverse_apriori_covariance[0])))
        additional_row_zeros[0, n] = 0
        additional_row_with_correlation = additional_row_zeros
        inverse_apriori_covariance = \
            np.r_[inverse_apriori_covariance, np.atleast_2d(additional_row_with_correlation)]

    # Create input object for the estimation
    covariance_input = estimation.CovarianceAnalysisInput(
        angular_position_measurements_cof,
        inverse_apriori_covariance=inverse_apriori_covariance)

    # Set methodological options
    covariance_input.define_covariance_settings(save_design_matrix=False, print_output_to_terminal=True)

    weight_vector = np.power(noise_level, -2)
    covariance_input.set_total_single_observable_and_link_end_vector_weight(
        observable_type=estimation_setup.observation.angular_position_type,
        link_ends=link_ends_cof,
        weight_vector=weight_vector
    )

    # Perform the covariance analysis
    covariance_output = estimator.compute_covariance(covariance_input)
    formal_errors = covariance_output.formal_errors
    correlations = covariance_output.correlations

    return true_to_formal_error_ratio, uncertainty_factor_space_based_astrometry,\
        np.ravel(np.array(formal_errors)), np.ravel(correlations)


if __name__ == "__main__":
    mp.freeze_support()
    main()
