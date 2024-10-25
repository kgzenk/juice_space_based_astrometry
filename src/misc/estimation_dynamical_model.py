
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os
import numpy as np
# tudatpy imports
from tudatpy.util import redirect_std
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro import frame_conversion
from tudatpy.kernel.numerical_simulation import estimation, estimation_setup

###########################################################################
# ESTIMATION DYNAMICAL MODEL - ALL FOUR MOONS #############################
###########################################################################

def estimate_satellites_initial_state(bodies,
                                      simulation_start_epoch,
                                      simulation_end_epoch,
                                      propagator_settings,
                                      force_estimate=False):
    """
    This function estimates the initial states of the propagated moons such that the resulting propagation will
    closely resemble the performance of the ephemerides.

    Parameters
    ----------
    bodies : tudatpy.kernel.numerical_simulation.environment_setup.SystemOfBodies
        Object that contains a set of body objects and associated frame information.
    simulation_start_epoch : float
        Start time of the simulation, in seconds since J2000
    simulation_end_epoch : float
        End time of the simulation, in seconds since J2000
    propagator_settings : (...).numerical_simulation.propagation_setup.propagator.TranslationalStatePropagatorSettings
        Settings of the propagator.
    force_estimate : bool (default=False)
        Force the program to re-estimate the initial states, independent of whether there are already
        other results saved to file. File will be overwritten with new values.
    """
    ### SETTING IN- AND OUTPUT-PATHS ###
    current_dir = os.path.dirname(__file__)
    dynamical_validation_output_path = os.path.join(os.path.split(current_dir)[0], 'data_cov_analysis')
    optimised_initial_states_path = (
        os.path.join(dynamical_validation_output_path, 'initial_states', 'estimated_initial_states_moons.dat'))

    ### CHECK EXISTENCE OF EARLIER OPTIMISED STATES ###
    if os.path.isfile(optimised_initial_states_path):
        optimised_initial_states_list = np.loadtxt(optimised_initial_states_path)
        if optimised_initial_states_list.ndim == 1:
            optimised_initial_states_list = [optimised_initial_states_list]
        if len(optimised_initial_states_list) == 1:
            optimised_initial_states_list = np.array(optimised_initial_states_list)
        for idx in range(len(optimised_initial_states_list)):
            if simulation_start_epoch == optimised_initial_states_list[idx, 0] and \
                    simulation_end_epoch == optimised_initial_states_list[idx, 1]:
                if force_estimate:
                    list2save_idx = idx
                    break
                else:
                    initial_states_updated = optimised_initial_states_list[idx, 2:]
                    return initial_states_updated

    ### CREATE LINK ENDS FOR MOONS ###
    link_ends_io = dict()
    link_ends_io[estimation_setup.observation.observed_body] = estimation_setup.observation. \
        body_origin_link_end_id('Io')
    link_definition_io = estimation_setup.observation.LinkDefinition(link_ends_io)

    link_ends_europa = dict()
    link_ends_europa[estimation_setup.observation.observed_body] = estimation_setup.observation. \
        body_origin_link_end_id('Europa')
    link_definition_europa = estimation_setup.observation.LinkDefinition(link_ends_europa)

    link_ends_ganymede = dict()
    link_ends_ganymede[estimation_setup.observation.observed_body] = estimation_setup.observation. \
        body_origin_link_end_id('Ganymede')
    link_definition_ganymede = estimation_setup.observation.LinkDefinition(link_ends_ganymede)

    link_ends_callisto = dict()
    link_ends_callisto[estimation_setup.observation.observed_body] = estimation_setup.observation. \
        body_origin_link_end_id('Callisto')
    link_definition_callisto = estimation_setup.observation.LinkDefinition(link_ends_callisto)

    ### OBSERVATION MODEL SETTINGS ###
    position_observation_settings = [estimation_setup.observation.cartesian_position(link_definition_io),
                                     estimation_setup.observation.cartesian_position(link_definition_europa),
                                     estimation_setup.observation.cartesian_position(link_definition_ganymede),
                                     estimation_setup.observation.cartesian_position(link_definition_callisto)]

    ### OBSERVATIONS SIMULATION SETTINGS ###
    # Define epochs at which the ephemerides shall be checked
    observation_times = np.arange(simulation_start_epoch, simulation_end_epoch, 3.0 * 3600)
    # Create the observation simulation settings per moon
    observation_simulation_settings_io = estimation_setup.observation.tabulated_simulation_settings(
        estimation_setup.observation.position_observable_type,
        link_definition_io,
        observation_times,
        reference_link_end_type=estimation_setup.observation.observed_body)
    observation_simulation_settings_europa = estimation_setup.observation.tabulated_simulation_settings(
        estimation_setup.observation.position_observable_type,
        link_definition_europa,
        observation_times,
        reference_link_end_type=estimation_setup.observation.observed_body)
    observation_simulation_settings_ganymede = estimation_setup.observation.tabulated_simulation_settings(
        estimation_setup.observation.position_observable_type,
        link_definition_ganymede,
        observation_times,
        reference_link_end_type=estimation_setup.observation.observed_body)
    observation_simulation_settings_callisto = estimation_setup.observation.tabulated_simulation_settings(
        estimation_setup.observation.position_observable_type,
        link_definition_callisto,
        observation_times,
        reference_link_end_type=estimation_setup.observation.observed_body)
    # Create conclusive list of observation simulation settings
    observation_simulation_settings = [observation_simulation_settings_io,
                                       observation_simulation_settings_europa,
                                       observation_simulation_settings_ganymede,
                                       observation_simulation_settings_callisto]

    ### "SIMULATE" EPHEMERIS STATES OF SATELLITES ###
    # Create observation simulators
    ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
        position_observation_settings, bodies)
    # Get ephemeris states as ObservationCollection
    print('Checking ephemerides...')
    ephemeris_satellite_states = estimation.simulate_observations(
        observation_simulation_settings,
        ephemeris_observation_simulators,
        bodies)

    ### PARAMETERS TO ESTIMATE ###
    parameters_to_estimate_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
    parameters_to_estimate = estimation_setup.create_parameter_set(parameters_to_estimate_settings, bodies)
    original_parameter_vector = parameters_to_estimate.parameter_vector

    ### PERFORM THE ESTIMATION ###
    # Create the estimator
    print('Running propagation...')
    with redirect_std():
        estimator = numerical_simulation.Estimator(
            bodies, parameters_to_estimate, position_observation_settings, propagator_settings)
    # Create input object for the estimation
    estimation_input = estimation.EstimationInput(ephemeris_satellite_states)
    # Perform the estimation
    print('Performing the estimation...')
    print(f'Original initial states: {original_parameter_vector}')
    estimation_output = estimator.perform_estimation(estimation_input)
    initial_states_updated = parameters_to_estimate.parameter_vector
    print('Done with the estimation...')
    print(f'Updated initial states: {initial_states_updated}')

    # Save optimised initial state together with simulation beginning and duration to file
    list2save = np.concatenate(([simulation_start_epoch], [simulation_end_epoch], initial_states_updated), axis=0)
    if 'optimised_initial_states_list' in locals():
        if force_estimate and 'list2save_idx' in locals():
            optimised_initial_states_list[list2save_idx] = list2save
            optimised_initial_states_list = sorted(optimised_initial_states_list, key=lambda x: (x[0], x[1]))
        else:
            optimised_initial_states_list = np.append(optimised_initial_states_list, [list2save], axis=0)
            optimised_initial_states_list = sorted(optimised_initial_states_list, key=lambda x: (x[0], x[1]))
        np.savetxt(optimised_initial_states_path, optimised_initial_states_list)
    else:
        np.savetxt(optimised_initial_states_path, list2save, newline=' ', delimiter=',')

    return np.array(initial_states_updated)


def covariance_initial_state_all_moons(bodies,
                                       simulation_start_epoch,
                                       simulation_end_epoch,
                                       propagator_settings,
                                       nominal_state_history,
                                       auto_plotting=False):
    """
    This function estimates the covariance matrix of the initial states of the propagated moons such that the resulting
    the evolution of the formal a priori radio science uncertainties in the position of the Galilean moons over the
    duration of the Jovian orbital phase of JUICE will closely resemble that of Fayolle et al. (2023).

    Parameters
    ----------
    bodies : tudatpy.kernel.numerical_simulation.environment_setup.SystemOfBodies
        Object that contains a set of body objects and associated frame information.
    simulation_start_epoch : float
        Start time of the simulation, in seconds since J2000
    simulation_end_epoch : float
        End time of the simulation, in seconds since J2000
    propagator_settings : (...).numerical_simulation.propagation_setup.propagator.TranslationalStatePropagatorSettings
        Settings of the propagator.
    nominal_state_history : bool (default=False)

    auto_plotting : bool (default=False)
        Setting whether to return the state_transition_interface and state_history
        next to the covariance_analysis for propagation and plotting purposes.
    """
    ### CREATE LINK ENDS FOR MOONS ###
    link_ends_io = dict()
    link_ends_io[estimation_setup.observation.observed_body] = estimation_setup.observation. \
        body_origin_link_end_id('Io')
    link_definition_io = estimation_setup.observation.LinkDefinition(link_ends_io)

    link_ends_europa = dict()
    link_ends_europa[estimation_setup.observation.observed_body] = estimation_setup.observation. \
        body_origin_link_end_id('Europa')
    link_definition_europa = estimation_setup.observation.LinkDefinition(link_ends_europa)

    link_ends_ganymede = dict()
    link_ends_ganymede[estimation_setup.observation.observed_body] = estimation_setup.observation. \
        body_origin_link_end_id('Ganymede')
    link_definition_ganymede = estimation_setup.observation.LinkDefinition(link_ends_ganymede)

    link_ends_callisto = dict()
    link_ends_callisto[estimation_setup.observation.observed_body] = estimation_setup.observation. \
        body_origin_link_end_id('Callisto')
    link_definition_callisto = estimation_setup.observation.LinkDefinition(link_ends_callisto)

    ### OBSERVATION MODEL SETTINGS ###
    position_observation_settings = [estimation_setup.observation.cartesian_position(link_definition_io),
                                     estimation_setup.observation.cartesian_position(link_definition_europa),
                                     estimation_setup.observation.cartesian_position(link_definition_ganymede),
                                     estimation_setup.observation.cartesian_position(link_definition_callisto)]

    ### OBSERVATIONS SIMULATION SETTINGS ###
    # Define epochs at which the ephemerides shall be checked
    observation_times = np.arange(simulation_start_epoch, simulation_end_epoch, 3.0 * 3600)
    number_of_observations = len(observation_times)
    # Create the observation simulation settings per moon
    observation_simulation_settings_io = estimation_setup.observation.tabulated_simulation_settings(
        estimation_setup.observation.position_observable_type,
        link_definition_io,
        observation_times,
        reference_link_end_type=estimation_setup.observation.observed_body)
    observation_simulation_settings_europa = estimation_setup.observation.tabulated_simulation_settings(
        estimation_setup.observation.position_observable_type,
        link_definition_europa,
        observation_times,
        reference_link_end_type=estimation_setup.observation.observed_body)
    observation_simulation_settings_ganymede = estimation_setup.observation.tabulated_simulation_settings(
        estimation_setup.observation.position_observable_type,
        link_definition_ganymede,
        observation_times,
        reference_link_end_type=estimation_setup.observation.observed_body)
    observation_simulation_settings_callisto = estimation_setup.observation.tabulated_simulation_settings(
        estimation_setup.observation.position_observable_type,
        link_definition_callisto,
        observation_times,
        reference_link_end_type=estimation_setup.observation.observed_body)
    # Create conclusive list of observation simulation settings
    observation_simulation_settings = [observation_simulation_settings_io,
                                       observation_simulation_settings_europa,
                                       observation_simulation_settings_ganymede,
                                       observation_simulation_settings_callisto]

    ### "SIMULATE" EPHEMERIS STATES OF SATELLITES ###
    # Create observation simulators
    ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
        position_observation_settings, bodies)
    # Get ephemeris states as ObservationCollection
    ephemeris_satellite_states = estimation.simulate_observations(
        observation_simulation_settings,
        ephemeris_observation_simulators,
        bodies)

    ### PARAMETERS TO ESTIMATE ###
    parameters_to_estimate_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
    parameters_to_estimate = estimation_setup.create_parameter_set(parameters_to_estimate_settings, bodies)

    ### PREPARE THE ESTIMATION ###
    averaged_uncertainties_dict = {
        'Io': [[2, 137, 387], [85, 1000, 55], link_ends_io],
        'Europa': [[3, 27, 130], [98, 27, 50], link_ends_europa],
        'Ganymede': [[2, 8, 46], [96, 35, 50], link_ends_ganymede],
        'Callisto': [[4, 17, 139], [70, 43, 50], link_ends_callisto]
    }

    # create weight matrix with correlations but (n, n) shape with zeros filled up
    weight_matrix = np.zeros((12 * len(observation_times), 12 * len(observation_times)))

    inverse_apriori_covariance = np.zeros((24, 24))

    # Create input object for the estimation
    estimation_input = estimation.EstimationInput(ephemeris_satellite_states,
                                                  inverse_apriori_covariance=inverse_apriori_covariance)

    moons_alphabetical = list(averaged_uncertainties_dict.keys())
    moons_alphabetical.sort()

    moon_idx_nominal_history_dict = {'Io': 0, 'Europa': 1, 'Ganymede': 2, 'Callisto': 3}

    for moon_idx, moon in enumerate(moons_alphabetical):
        rsw_to_inertial = list()
        moon_idx_nom = moon_idx_nominal_history_dict[moon]
        for epoch_seconds in observation_times:
            rsw_to_inertial.append(frame_conversion.rsw_to_inertial_rotation_matrix(
                nominal_state_history[epoch_seconds][moon_idx_nom*6:(moon_idx_nom+1)*6]))
        rsw_to_inertial = np.array(rsw_to_inertial)

        weights_rsw = np.multiply(averaged_uncertainties_dict[moon][1], np.power(averaged_uncertainties_dict[moon][0], 2))

        weight_matrix_rsw = np.zeros((3, 3))
        np.fill_diagonal(weight_matrix_rsw, weights_rsw)

        for idx, rsw_to_inertial_matrix in enumerate(rsw_to_inertial):

            noise_level_temp = np.multiply(np.sqrt(number_of_observations),
                                           np.matmul(np.matmul(rsw_to_inertial_matrix, weight_matrix_rsw),
                                                     np.transpose(rsw_to_inertial_matrix)))

            noise_level = np.linalg.inv(noise_level_temp)

            for row_idx_temp, row in enumerate(noise_level):
                for column_idx_temp, noise_value in enumerate(row):
                    weight_matrix[3 * (moon_idx * number_of_observations + idx) + row_idx_temp,
                                  3 * (moon_idx * number_of_observations + idx) + column_idx_temp] \
                        = noise_value

            if idx == 0:
                weight_vector = np.diagonal(noise_level)
            else:
                weight_vector = np.append(weight_vector, np.diagonal(noise_level))

        estimation_input.set_total_single_observable_and_link_end_vector_weight(
            observable_type=estimation_setup.observation.position_observable_type,
            link_ends=averaged_uncertainties_dict[moon][2],
            weight_vector=weight_vector
        )

    estimation_input.define_estimation_settings(print_output_to_terminal=False, save_state_history_per_iteration=True)

    ### PERFORM THE ESTIMATION ###
    # Create the estimator
    print('Calculating covariance...')
    with redirect_std():
        estimator = numerical_simulation.Estimator(
            bodies, parameters_to_estimate, position_observation_settings, propagator_settings)

    # Perform the estimation
    estimation_output = estimator.perform_estimation(estimation_input)
    covariance_matrix_test = estimation_output.covariance
    # Get normalized design matrix
    normalized_design_matrix = estimation_output.normalized_design_matrix
    # Normalize inverse a-priori matrix
    normalization_terms = estimation_output.normalization_terms
    normalized_inverse_apriori_covariance = np.zeros(np.shape(inverse_apriori_covariance))
    for j in range(len(parameters_to_estimate.parameter_vector)):
        for k in range(len(parameters_to_estimate.parameter_vector)):
            normalized_inverse_apriori_covariance[j, k] = \
                inverse_apriori_covariance[j, k] / (normalization_terms[j] * normalization_terms[k])
    # Compute normalized covariance matrix
    normalized_covariance_matrix = \
        np.linalg.inv(np.add(normalized_inverse_apriori_covariance,
                             np.matmul(np.matmul(np.transpose(normalized_design_matrix), weight_matrix),
                                       normalized_design_matrix)))
    # Denormalize covariance matrix
    covariance_matrix = np.zeros(np.shape(normalized_covariance_matrix))
    for j in range(len(parameters_to_estimate.parameter_vector)):
        for k in range(len(parameters_to_estimate.parameter_vector)):
            covariance_matrix[j, k] = \
                normalized_covariance_matrix[j, k] / (normalization_terms[j] * normalization_terms[k])

    if auto_plotting:
        state_transition_interface = estimator.state_transition_interface
        simulator_object = estimation_output.simulation_results_per_iteration[-1]
        state_history = simulator_object.dynamics_results.state_history

        return np.array(covariance_matrix), state_transition_interface, state_history

    return np.array(covariance_matrix)


###########################################################################
# ESTIMATION DYNAMICAL MODEL - IO ONLY ####################################
###########################################################################

def estimate_single_satellite_initial_state(bodies,
                                            simulation_start_epoch,
                                            simulation_end_epoch,
                                            propagator_settings,
                                            force_estimate=False):
    """
    This function estimates the initial states of the propagated moons such that the resulting propagation will
    closely resemble the performance of the ephemerides.

    Parameters
    ----------
    bodies : tudatpy.kernel.numerical_simulation.environment_setup.SystemOfBodies
        Object that contains a set of body objects and associated frame information.
    simulation_start_epoch : float
        Start time of the simulation, in seconds since J2000
    simulation_end_epoch :
        End time of the simulation, in seconds since J2000
    propagator_settings : tudatpy.kernel.numerical_simulation.propagation_setup.propagator.
                          TranslationalStatePropagatorSettings
        Settings of the propagator.
    force_estimate : bool (default=False)
        Whether to force the program to re-estimate the initial states, independent of whether there are already
        other results saved to file. File will be overwritten with new values.
    """
    ### SETTING IN- AND OUTPUT-PATHS ###
    current_dir = os.path.dirname(__file__)
    dynamical_validation_output_path = os.path.join(os.path.split(current_dir)[0], 'data_cov_analysis')
    optimised_initial_state_path = (
        os.path.join(dynamical_validation_output_path, 'initial_states', 'estimated_initial_states_io.dat'))

    ### CHECK EXISTENCE OF EARLIER OPTIMISED STATES ###
    if os.path.isfile(optimised_initial_state_path):
        optimised_initial_state_list = np.loadtxt(optimised_initial_state_path)
        if optimised_initial_state_list.ndim == 1:
            optimised_initial_state_list = [optimised_initial_state_list]
        if len(optimised_initial_state_list) == 1:
            optimised_initial_state_list = np.array(optimised_initial_state_list)
        for idx in range(len(optimised_initial_state_list)):
            if simulation_start_epoch == optimised_initial_state_list[idx, 0] and \
                    simulation_end_epoch == optimised_initial_state_list[idx, 1]:
                if force_estimate:
                    list2save_idx = idx
                    break
                else:
                    initial_states_updated = optimised_initial_state_list[idx, 2:]
                    return np.array(initial_states_updated)

    ### CREATE LINK ENDS FOR IO ###
    link_ends_io = dict()
    link_ends_io[estimation_setup.observation.observed_body] = estimation_setup.observation. \
        body_origin_link_end_id('Io')
    link_definition_io = estimation_setup.observation.LinkDefinition(link_ends_io)

    ### OBSERVATION MODEL SETTINGS ###
    position_observation_settings = [estimation_setup.observation.cartesian_position(link_definition_io)]

    ### OBSERVATIONS SIMULATION SETTINGS ###
    # Define epochs at which the ephemerides shall be checked
    observation_times = np.arange(simulation_start_epoch, simulation_end_epoch, 3.0 * 3600)
    # Create the observation simulation settings per moon
    observation_simulation_settings_io = estimation_setup.observation.tabulated_simulation_settings(
        estimation_setup.observation.position_observable_type,
        link_definition_io,
        observation_times,
        reference_link_end_type=estimation_setup.observation.observed_body)
    # Create conclusive list of observation simulation settings
    observation_simulation_settings = [observation_simulation_settings_io]

    ### "SIMULATE" EPHEMERIS STATES OF SATELLITES ###
    # Create observation simulators
    ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
        position_observation_settings, bodies)
    # Get ephemeris states as ObservationCollection
    print('Checking ephemerides...')
    ephemeris_satellite_states = estimation.simulate_observations(
        observation_simulation_settings,
        ephemeris_observation_simulators,
        bodies)

    ### PARAMETERS TO ESTIMATE ###
    parameters_to_estimate_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
    parameters_to_estimate = estimation_setup.create_parameter_set(parameters_to_estimate_settings, bodies)
    original_parameter_vector = parameters_to_estimate.parameter_vector

    ### PERFORM THE ESTIMATION ###
    # Create the estimator
    print('Running propagation...')
    with redirect_std():
        estimator = numerical_simulation.Estimator(bodies, parameters_to_estimate,
                                                   position_observation_settings, propagator_settings)
    # Create input object for the estimation
    estimation_input = estimation.EstimationInput(ephemeris_satellite_states)
    # Perform the estimation
    print('Performing the estimation...')
    print(f'Original initial states: {original_parameter_vector}')
    estimation_output = estimator.perform_estimation(estimation_input)
    initial_state_updated = parameters_to_estimate.parameter_vector
    print('Done with the estimation...')
    print(f'Updated initial states: {initial_state_updated}')

    # Save optimised initial state together with simulation beginning and duration to file
    list2save = np.concatenate(([simulation_start_epoch], [simulation_end_epoch], initial_state_updated), axis=0)
    if 'optimised_initial_state_list' in locals():
        if force_estimate and 'list2save_idx' in locals():
            optimised_initial_state_list[list2save_idx] = list2save
            optimised_initial_state_list = sorted(optimised_initial_state_list, key=lambda x: (x[0], x[1]))
        else:
            optimised_initial_state_list = np.append(optimised_initial_state_list, [list2save], axis=0)
            optimised_initial_state_list = sorted(optimised_initial_state_list, key=lambda x: (x[0], x[1]))
        np.savetxt(optimised_initial_state_path, optimised_initial_state_list)
    else:
        np.savetxt(optimised_initial_state_path, list2save, newline=' ', delimiter=',')

    return np.array(initial_state_updated)


def covariance_single_satellite_initial_state(bodies,
                                              simulation_start_epoch,
                                              simulation_end_epoch,
                                              propagator_settings,
                                              nominal_state_history,
                                              auto_plotting=False,
                                              radiometric_accuracy_best_case=False,
                                              radiometric_accuracy_realistic_case=False):
    """
    This function estimates the initial states of the propagated moons such that the resulting propagation will
    closely resemble the performance of the ephemerides.

    Parameters
    ----------
    """
    ### CREATE LINK ENDS FOR IO ###
    link_ends_io = dict()
    link_ends_io[estimation_setup.observation.observed_body] = estimation_setup.observation. \
        body_origin_link_end_id('Io')
    link_definition_io = estimation_setup.observation.LinkDefinition(link_ends_io)

    ### OBSERVATION MODEL SETTINGS ###
    position_observation_settings = [estimation_setup.observation.cartesian_position(link_definition_io)]

    ### OBSERVATIONS SIMULATION SETTINGS ###
    # Define epochs at which the ephemerides shall be checked
    observation_times = np.arange(simulation_start_epoch, simulation_end_epoch, 3.0 * 3600)
    number_of_observations = len(observation_times)
    # Create the observation simulation settings per moon
    observation_simulation_settings_io = estimation_setup.observation.tabulated_simulation_settings(
        estimation_setup.observation.position_observable_type,
        link_definition_io,
        observation_times,
        reference_link_end_type=estimation_setup.observation.observed_body)
    # Create conclusive list of observation simulation settings
    observation_simulation_settings = [observation_simulation_settings_io]

    ### "SIMULATE" EPHEMERIS STATES OF SATELLITES ###
    # Create observation simulators
    ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
        position_observation_settings, bodies)
    # Get ephemeris states as ObservationCollection
    ephemeris_satellite_states = estimation.simulate_observations(
        observation_simulation_settings,
        ephemeris_observation_simulators,
        bodies)

    ### PARAMETERS TO ESTIMATE ###
    parameters_to_estimate_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
    parameters_to_estimate = estimation_setup.create_parameter_set(parameters_to_estimate_settings, bodies)
    original_parameter_vector = parameters_to_estimate.parameter_vector

    ### PERFORM THE ESTIMATION ###
    # Create the estimator
    print('Calculating covariance...')
    estimator = numerical_simulation.Estimator(bodies, parameters_to_estimate,
                                               position_observation_settings, propagator_settings)

    rsw_to_inertial = list()
    for epoch_seconds in observation_times:
        rsw_to_inertial.append(frame_conversion.rsw_to_inertial_rotation_matrix(
            nominal_state_history[epoch_seconds]))
    rsw_to_inertial = np.array(rsw_to_inertial)

    apriori_covariance_rsw = np.zeros((3, 3))
    apriori_covariance_rsw[0, 0] = 0.98 ** 2
    apriori_covariance_rsw[1, 1] = 0.14 ** 2
    apriori_covariance_rsw[2, 2] = 0.72 ** 2

    apriori_covariance_rsw = np.array(apriori_covariance_rsw)
    apriori_covariance_temp = np.matmul(np.matmul(rsw_to_inertial[0], apriori_covariance_rsw),
                                        np.transpose(rsw_to_inertial[0]))

    inverse_apriori_covariance_temp = np.linalg.inv(apriori_covariance_temp)

    inverse_apriori_covariance = np.zeros((6, 6))
    for row_idx, row in enumerate(inverse_apriori_covariance_temp):
        for column_idx, column in enumerate(row):
            inverse_apriori_covariance[row_idx + 3, column_idx + 3] = column

    # create weight matrix with correlations but n, n shape with zeros filled up
    weight_matrix = np.zeros((3 * len(observation_times), 3 * len(observation_times)))

    if radiometric_accuracy_best_case:
        weights_rsw = np.multiply(50.0, np.power(np.array([0.4, 30, 91]), 2))

    elif radiometric_accuracy_realistic_case:
        weights_rsw = np.multiply(50.0, np.power(np.array([1.3, 160, 390]), 2))

    else:
        weights_rsw = np.multiply(50.0, np.power(np.array([15E3, 15E3, 15E3]), 2))

    maries_stupid_weight_matrix = np.zeros((3, 3))
    np.fill_diagonal(maries_stupid_weight_matrix, weights_rsw)

    for idx, rsw_to_inertial_matrix in enumerate(rsw_to_inertial):

        noise_level_temp = np.multiply(np.sqrt(number_of_observations),
                                       np.matmul(np.matmul(rsw_to_inertial_matrix, maries_stupid_weight_matrix),
                                                 np.transpose(rsw_to_inertial_matrix)))

        noise_level = np.linalg.inv(noise_level_temp)

        for row_idx_temp, row in enumerate(noise_level):
            for column_idx_temp, noise_value in enumerate(row):
                weight_matrix[3 * idx + row_idx_temp, 3 * idx + column_idx_temp] = noise_value

    weight_vector = np.diagonal(weight_matrix)

    # Create input object for the estimation
    estimation_input = estimation.EstimationInput(ephemeris_satellite_states,
                                                  inverse_apriori_covariance=inverse_apriori_covariance)

    estimation_input.define_estimation_settings(save_design_matrix=True,
                                                print_output_to_terminal=False,
                                                save_state_history_per_iteration=True)

    estimation_input.set_total_single_observable_and_link_end_vector_weight(
        observable_type=estimation_setup.observation.position_observable_type,
        link_ends=link_ends_io,
        weight_vector=weight_vector
    )

    # Perform the estimation
    estimation_output = estimator.perform_estimation(estimation_input)
    covariance_matrix_test = estimation_output.covariance
    # Get normalized design matrix
    normalized_design_matrix = estimation_output.normalized_design_matrix
    # Normalize inverse a-priori matrix
    normalization_terms = estimation_output.normalization_terms
    normalized_inverse_apriori_covariance = np.zeros(np.shape(inverse_apriori_covariance))
    for j in range(len(parameters_to_estimate.parameter_vector)):
        for k in range(len(parameters_to_estimate.parameter_vector)):
            normalized_inverse_apriori_covariance[j, k] = \
                inverse_apriori_covariance[j, k] / (normalization_terms[j] * normalization_terms[k])
    # Compute normalized covariance matrix
    normalized_covariance_matrix = \
        np.linalg.inv(np.add(normalized_inverse_apriori_covariance,
                             np.matmul(np.matmul(np.transpose(normalized_design_matrix), weight_matrix),
                                       normalized_design_matrix)))
    # Denormalize covariance matrix
    covariance_matrix = np.zeros(np.shape(normalized_covariance_matrix))
    for j in range(len(parameters_to_estimate.parameter_vector)):
        for k in range(len(parameters_to_estimate.parameter_vector)):
            covariance_matrix[j, k] = \
                normalized_covariance_matrix[j, k] / (normalization_terms[j] * normalization_terms[k])

    if auto_plotting:
        state_transition_interface = estimator.state_transition_interface
        simulator_object = estimation_output.simulation_results_per_iteration[-1]
        state_history = simulator_object.dynamics_results.state_history

        return np.array(covariance_matrix), state_transition_interface, state_history

    return np.array(covariance_matrix)
