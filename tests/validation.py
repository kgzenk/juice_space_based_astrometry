
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os
import numpy as np
# tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.util import redirect_std
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import create_dynamics_simulator
# Problem-specific imports
# tudat set-up
from src.core.tudat_setup_util.env.env_full_setup import \
    create_simulation_bodies
from src.core.tudat_setup_util.propagation.prop_full_setup import \
    get_acceleration_models_moons
from src.core.tudat_setup_util.propagation.prop_general_utilities import \
    get_integrator_settings, get_termination_settings
# Validation
from src.misc.estimation_dynamical_model import \
    estimate_satellites_initial_state
# Space-based astrometry
from src.core.space_based_astrometry.space_based_astrometry_utilities import \
    space_based_astrometry_epochs_errors
# Plotting
from tests import validation_plotting as Util

# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

### Plotting Settings ###
auto_plot = True

### Dynamical Model Validation ###
run_dynamical_model_validation = False

### Determine Closest Approaches ###
get_juice_closest_approaches = False

### Space-Based Astrometry Validation ###
run_space_based_astrometry = False

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# JUICE Jupiter Orbit Insertion July 2031 / Ganymede Orbit Insertion December 2034 (1st)
simulation_start_epoch = 31.0 * constants.JULIAN_YEAR + 182.0 * constants.JULIAN_DAY
simulation_end_epoch = 34.0 * constants.JULIAN_YEAR + 335.0 * constants.JULIAN_DAY

simulation_duration = simulation_end_epoch - simulation_start_epoch

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Load most current SPICE cassini_kernels for JUICE (CReMA 5.1) obtained from
# https://www.cosmos.esa.int/web/spice/spice-for-juice
spice_interface.load_kernel('../tudat_setup_util/spice/JUICE/juice_crema_5_1_150lb_23_1.tm.txt')

###########################################################################
# CREATE PROPAGATION SETTINGS #############################################
###########################################################################

# Define bodies that are propagated, and their central bodies of propagation
bodies_to_propagate = ['Io', 'Europa', 'Ganymede', 'Callisto']
central_bodies = ['Jupiter', 'Jupiter', 'Jupiter', 'Jupiter']
body_settings, bodies = create_simulation_bodies(simulation_start_epoch, simulation_end_epoch, bodies_to_propagate)
# Get global accelerations dictionary
acceleration_settings_moons = get_acceleration_models_moons(bodies_to_propagate)
acceleration_settings = acceleration_settings_moons
# Create acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies)

# Define initial state
initial_states = list()
for body in bodies_to_propagate:
    initial_states.append(spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name=body,
        observer_body_name='Jupiter',
        reference_frame_name='ECLIPJ2000',
        aberration_corrections='none',
        ephemeris_time=simulation_start_epoch))
initial_states = np.concatenate(initial_states)

###########################################################################
# DYNAMICAL MODEL VALIDATION ##############################################
###########################################################################

if run_dynamical_model_validation:
    # Define overall output-path for all tests-related validation_data
    dynamical_validation_output_path = os.path.join(current_dir, 'validation_data',
                                                    'dynamical_validation')
    if not os.path.exists(dynamical_validation_output_path):
        os.makedirs(dynamical_validation_output_path)

    # Define termination condition(s)
    termination_settings_validation = get_termination_settings(simulation_start_epoch, simulation_duration)

    # Define integrator settings for propagation
    integrator_settings_validation = get_integrator_settings()

    # Create propagator settings
    propagator_settings_estimation = propagation_setup.propagator. \
        translational(central_bodies=central_bodies,
                      acceleration_models=acceleration_models,
                      bodies_to_integrate=bodies_to_propagate,
                      initial_states=initial_states,
                      initial_time=simulation_start_epoch,
                      integrator_settings=integrator_settings_validation,
                      termination_settings=termination_settings_validation)

    ### ESTIMATE BETTER INITIAL STATES ###
    initial_states_updated = (
        estimate_satellites_initial_state(simulation_start_epoch, simulation_end_epoch,
                                          propagator_settings_estimation, bodies, force_estimate=True))

    ### CREATE DATA FOR VALIDATION ###
    # Define Keplerian elements of the Galilean moons as dependent variables
    dependent_variables_to_save = [
        propagation_setup.dependent_variable.keplerian_state('Io', 'Jupiter'),
        propagation_setup.dependent_variable.keplerian_state('Europa', 'Jupiter'),
        propagation_setup.dependent_variable.keplerian_state('Ganymede', 'Jupiter'),
        propagation_setup.dependent_variable.keplerian_state('Callisto', 'Jupiter')
    ]

    # Create propagator settings
    propagator_settings_space_based_astro = propagation_setup.propagator. \
        translational(central_bodies=central_bodies,
                      acceleration_models=acceleration_models,
                      bodies_to_integrate=bodies_to_propagate,
                      initial_states=initial_states_updated,
                      initial_time=simulation_start_epoch,
                      integrator_settings=integrator_settings_validation,
                      termination_settings=termination_settings_validation,
                      output_variables=dependent_variables_to_save)

    # Run propagation
    print('Running propagation...')
    with redirect_std():
        dynamics_simulator_laplace = create_dynamics_simulator(bodies, propagator_settings_space_based_astro)
    propagated_state_history = dynamics_simulator_laplace.state_history
    propagated_keplerian_states = dynamics_simulator_laplace.dependent_variable_history
    # Save propagated state history and Kepler elements to file
    save2txt(propagated_state_history, 'propagated_state_history.dat', dynamical_validation_output_path)
    save2txt(propagated_keplerian_states, 'propagated_keplerian_states.dat', dynamical_validation_output_path)

    ### Ephemeris Kepler elements ####
    # Initialize containers
    ephemeris_state_history = dict()
    ephemeris_keplerian_states = dict()
    jupiter_gravitational_parameter = bodies.get('Jupiter').gravitational_parameter
    # Loop over the propagated states and use the IMCEE ephemeris as benchmark solution
    for epoch in propagated_state_history.keys():
        io_from_ephemeris = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name='Io',
            observer_body_name='Jupiter',
            reference_frame_name='ECLIPJ2000',
            aberration_corrections='none',
            ephemeris_time=epoch)
        keplerian_state_io = element_conversion.cartesian_to_keplerian(io_from_ephemeris,
                                                                       jupiter_gravitational_parameter)

        europa_from_ephemeris = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name='Europa',
            observer_body_name='Jupiter',
            reference_frame_name='ECLIPJ2000',
            aberration_corrections='none',
            ephemeris_time=epoch)
        keplerian_state_europa = element_conversion.cartesian_to_keplerian(europa_from_ephemeris,
                                                                           jupiter_gravitational_parameter)

        ganymede_from_ephemeris = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name='Ganymede',
            observer_body_name='Jupiter',
            reference_frame_name='ECLIPJ2000',
            aberration_corrections='none',
            ephemeris_time=epoch)
        keplerian_state_ganymede = element_conversion.cartesian_to_keplerian(ganymede_from_ephemeris,
                                                                             jupiter_gravitational_parameter)

        callisto_from_ephemeris = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name='Callisto',
            observer_body_name='Jupiter',
            reference_frame_name='ECLIPJ2000',
            aberration_corrections='none',
            ephemeris_time=epoch)
        keplerian_state_callisto = element_conversion.cartesian_to_keplerian(callisto_from_ephemeris,
                                                                             jupiter_gravitational_parameter)

        ephemeris_state = np.concatenate((io_from_ephemeris, europa_from_ephemeris,
                                          ganymede_from_ephemeris, callisto_from_ephemeris))
        keplerian_state = np.concatenate((keplerian_state_io, keplerian_state_europa,
                                          keplerian_state_ganymede, keplerian_state_callisto))

        ephemeris_state_history[epoch] = ephemeris_state
        ephemeris_keplerian_states[epoch] = keplerian_state

    # Save ephemeris state history and Kepler elements to file
    save2txt(ephemeris_state_history, 'ephemeris_state_history.dat', dynamical_validation_output_path)
    save2txt(ephemeris_keplerian_states, 'ephemeris_keplerian_states.dat', dynamical_validation_output_path)

    if auto_plot:
        Util.plot_dynamical_model_validation()
        Util.plot_laplace_resonance_validation()

###########################################################################
# JUICE CLOSEST APPROACHES ################################################
###########################################################################

if get_juice_closest_approaches:
    # Define output-path for JUICE's closest approach validation_data
    closest_approach_output_path = os.path.join(current_dir, 'validation_data', 'juice_validation_data')
    if not os.path.exists(closest_approach_output_path):
        os.makedirs(closest_approach_output_path)

    # Define termination condition(s)
    termination_settings_estimation = get_termination_settings(simulation_start_epoch, simulation_duration)

    # Define integrator settings for estimation
    integrator_settings_estimation = get_integrator_settings()

    # Create propagator settings
    propagator_settings_estimation = propagation_setup.propagator. \
        translational(central_bodies=central_bodies,
                      acceleration_models=acceleration_models,
                      bodies_to_integrate=bodies_to_propagate,
                      initial_states=initial_states,
                      initial_time=simulation_start_epoch,
                      integrator_settings=integrator_settings_estimation,
                      termination_settings=termination_settings_estimation)

    ### ESTIMATE BETTER INITIAL STATES ###
    initial_states_updated = (estimate_satellites_initial_state(simulation_start_epoch, simulation_end_epoch,
                                                                propagator_settings_estimation, bodies))

    ### CREATE DATA FOR CLOSETS APPROACHES ###

    # Define integrator settings for estimation
    integrator_settings_approaches = get_integrator_settings()
    termination_settings_approaches = termination_settings_estimation
    # Create propagator settings
    propagator_settings_approaches = propagation_setup.propagator. \
        translational(central_bodies=central_bodies,
                      acceleration_models=acceleration_models,
                      bodies_to_integrate=bodies_to_propagate,
                      initial_states=initial_states_updated,
                      initial_time=simulation_start_epoch,
                      integrator_settings=integrator_settings_approaches,
                      termination_settings=termination_settings_approaches)

    # Run propagation
    print('Running propagation...')
    with redirect_std():
        dynamics_simulator_approaches = create_dynamics_simulator(bodies, propagator_settings_approaches)
    state_history_temp = dynamics_simulator_approaches.state_history

    # Create 8th order interpolator
    relative_states_interpolator = (interpolators.create_one_dimensional_vector_interpolator(
        state_history_temp, interpolators.lagrange_interpolation(8)))
    # Initialize difference dictionaries
    state_history = dict()
    relative_distances = dict()
    juice_state_history = dict()
    # Calculate the difference between the states and dependent variables in an iterative manner
    for epoch in np.arange(simulation_start_epoch, simulation_end_epoch, 60):
        # Interpolate moons' state history
        state_history[epoch] = relative_states_interpolator.interpolate(epoch)

        # Get JUICE state history from SPICE
        juice_from_spice = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name='JUICE',
            observer_body_name='Jupiter',
            reference_frame_name='ECLIPJ2000',
            aberration_corrections='none',
            ephemeris_time=epoch)
        juice_state_history[epoch] = juice_from_spice

        # Interpolate relative states
        relative_states_temp = (relative_states_interpolator.interpolate(epoch) -
                                np.tile(juice_from_spice, len(bodies_to_propagate)))
        # Define the size of each group
        position_state_size = 3
        # Calculate the number of groups
        num_groups = len(relative_states_temp) // position_state_size
        # Reshape the array to split it into groups of size 'group_size'
        arr_reshaped = relative_states_temp[:num_groups * position_state_size].reshape(-1, position_state_size)
        # Access every other group
        relative_states_temp = arr_reshaped[::2]

        # Compute (interpolated) relative distances
        relative_distances[epoch] = np.ravel(np.linalg.norm(relative_states_temp, axis=1))

    # Save propagated state history to file
    save2txt(state_history, 'satellites_closest_approaches_state_history.dat', closest_approach_output_path)
    save2txt(juice_state_history, 'juice_closest_approaches_state_history.dat', closest_approach_output_path)
    save2txt(relative_distances, 'relative_distances_juice_moons.dat', closest_approach_output_path)

    if auto_plot:
        Util.plot_juice_closest_approaches()

###########################################################################
# PREDICT SPACE-BASED ASTROMETRY ##########################################
###########################################################################

if run_space_based_astrometry:
    # Define output-path for JUICE's closest approach validation_data
    closest_approach_input_path = os.path.join(current_dir, 'validation_data/juice_validation_data',
                                               'relative_distances_juice_moons.dat')

    # Define overall output-path for all prediction-related validation_data
    prediction_output_path = os.path.join(current_dir, 'validation_data', 'space_based_astrometry_data')
    if not os.path.exists(prediction_output_path):
        os.makedirs(prediction_output_path)

    # Define termination condition(s)
    termination_settings = get_termination_settings(simulation_start_epoch, simulation_duration)

    # Define integrator settings for propagation
    integrator_settings = get_integrator_settings()

    # Create propagator settings
    propagator_settings = propagation_setup.propagator. \
        translational(central_bodies=central_bodies,
                      acceleration_models=acceleration_models,
                      bodies_to_integrate=bodies_to_propagate,
                      initial_states=initial_states,
                      initial_time=simulation_start_epoch,
                      integrator_settings=integrator_settings,
                      termination_settings=termination_settings)

    ### ESTIMATE BETTER INITIAL STATES ###
    initial_states_updated = (
        estimate_satellites_initial_state(bodies, simulation_start_epoch, simulation_end_epoch, propagator_settings))

    ### CREATE DATA FOR VALIDATION ###
    dependent_variables_to_save = [
        propagation_setup.dependent_variable.relative_position('JUICE', 'Io'),
        propagation_setup.dependent_variable.relative_position('JUICE', 'Sun'),
        propagation_setup.dependent_variable.relative_position('JUICE', 'Jupiter'),
        propagation_setup.dependent_variable.relative_position('Jupiter', 'Io'),
        propagation_setup.dependent_variable.relative_position('Io', 'Sun')
    ]

    # Create propagator settings
    propagator_settings_space_based_astro = propagation_setup.propagator. \
        translational(central_bodies=central_bodies,
                      acceleration_models=acceleration_models,
                      bodies_to_integrate=bodies_to_propagate,
                      initial_states=initial_states_updated,
                      initial_time=simulation_start_epoch,
                      integrator_settings=integrator_settings,
                      termination_settings=termination_settings,
                      output_variables=dependent_variables_to_save)

    # Run propagation
    print('Running propagation...')
    with redirect_std():
        dynamics_simulator = create_dynamics_simulator(bodies, propagator_settings_space_based_astro)
    dependent_variable_history = dynamics_simulator.dependent_variable_history

    # Save propagated dependent variable history to file
    save2txt(dependent_variable_history, 'constraints_dependent_variables.dat', prediction_output_path)

    epochs_with_errors = \
        space_based_astrometry_epochs_errors(dependent_variable_history, closest_approach_input_path)
    save2txt(epochs_with_errors, 'epochs_with_errors_list.dat', prediction_output_path)

    ### RELATIVE GEOMETRY - IO-FIXED FRAME ###
    juice_wrt_to_io = dict()
    for epoch in dependent_variable_history.keys():
        juice_from_spice = spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name='JUICE',
            observer_body_name='Io',
            reference_frame_name='IAU_Io',
            aberration_corrections='none',
            ephemeris_time=epoch)
        juice_wrt_to_io[epoch] = juice_from_spice

    # Save propagated state history to file
    save2txt(juice_wrt_to_io, 'relative_geometry_io_fixed_state_history.dat', prediction_output_path)

    if auto_plot:
        Util.plot_space_based_astrometry_verification()
        Util.plot_space_based_astrometry_arcs()
