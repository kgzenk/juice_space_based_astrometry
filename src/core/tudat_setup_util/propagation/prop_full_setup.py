
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os
import math
import numpy as np
# tudatpy imports
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import propagation_setup
# Problem-specific imports
from src.core.tudat_setup_util.env.env_full_setup import \
    create_simulation_bodies
from src.core.tudat_setup_util.propagation.prop_general_utilities import \
    get_integrator_settings, get_termination_settings

current_dir = os.path.dirname(__file__)

###########################################################################
# PROPAGATION SETTING UTILITIES ###########################################
###########################################################################

def get_acceleration_models_moons(bodies_to_propagate):
    """
    Creates the relevant settings for the accelerations of the Galilean satellites.
    Performs a routine check if the individual satellites are a member of bodies_to_propagate.

    Parameters
    ----------
    bodies_to_propagate : list[string]

    Returns
    -------
    acceleration_settings_moons : Dict[str, Dict[str, List[tudatpy.kernel.numerical_simulation.
                                                           propagation_setup.acceleration.AccelerationSettings]]]
        Key-value container, with key denoting the body undergoing the acceleration, and the value containing an
        additional key-value container, with the body exerting acceleration, and list of acceleration settings exerted
        by this body.
    """
    # Dirkx et al. (2016) - restricted to second degree
    love_number_moons = 0.3
    dissipation_parameter_moons = 0.015
    q_moons = love_number_moons / dissipation_parameter_moons
    # Lari (2018)
    mean_motion_io = 203.49 * (math.pi / 180) * 1 / constants.JULIAN_DAY
    mean_motion_europa = 101.37 * (math.pi / 180) * 1 / constants.JULIAN_DAY
    mean_motion_ganymede = 50.32 * (math.pi / 180) * 1 / constants.JULIAN_DAY
    mean_motion_callisto = 21.57 * (math.pi / 180) * 1 / constants.JULIAN_DAY

    # Dirkx et al. (2016) - restricted to second degree
    love_number_jupiter = 0.38
    dissipation_parameter_jupiter= 1.1E-5
    q_jupiter = love_number_jupiter / dissipation_parameter_jupiter

    # Lainey et al. (2009)
    tidal_frequency_io = 23.3  # rad.day-1
    spin_frequency_jupiter = math.pi/tidal_frequency_io + mean_motion_io

    acceleration_settings_moons = dict()

    if 'Io' in bodies_to_propagate:
        time_lag_io = 1 / mean_motion_io * np.arctan(1 / q_moons)
        time_lag_jupiter_io = 1/(spin_frequency_jupiter - mean_motion_io) * np.arctan(1 / q_jupiter)
        acceleration_settings_io = dict(
            Jupiter=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(12, 0, 2, 2, 12, 0),
                     propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_moons,
                                                                                          time_lag_io,
                                                                                          True, False),
                     propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_jupiter,
                                                                                          time_lag_jupiter_io,
                                                                                          True, True),
                     propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True)],
            Europa=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2, 12, 0)],
            Ganymede=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2, 12, 0)],
            Callisto=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2, 12, 0)],
            Sun=[propagation_setup.acceleration.point_mass_gravity()],
            Saturn=[propagation_setup.acceleration.point_mass_gravity()]
        )
        acceleration_settings_moons['Io'] = acceleration_settings_io

    if 'Europa' in bodies_to_propagate:
        time_lag_europa = 1 / mean_motion_europa * np.arctan(1 / q_moons)
        time_lag_jupiter_europa = 1 / (spin_frequency_jupiter - mean_motion_europa) * np.arctan(1 / q_jupiter)
        acceleration_settings_europa = dict(
            Jupiter=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(12, 0, 2, 2),
                     propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_moons,
                                                                                          time_lag_europa,
                                                                                          True, False),
                     propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_jupiter,
                                                                                          time_lag_jupiter_europa,
                                                                                          True, True),
                     propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True)],
            Io=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2, 12, 0)],
            Ganymede=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2, 12, 0)],
            Callisto=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2, 12, 0)],
            Sun=[propagation_setup.acceleration.point_mass_gravity()],
            Saturn=[propagation_setup.acceleration.point_mass_gravity()]
        )
        acceleration_settings_moons['Europa'] = acceleration_settings_europa

    if 'Ganymede' in bodies_to_propagate:
        time_lag_ganymede = 1 / mean_motion_ganymede * np.arctan(1 / q_moons)
        time_lag_jupiter_ganymede = 1 / (spin_frequency_jupiter - mean_motion_ganymede) * np.arctan(1 / q_jupiter)
        acceleration_settings_ganymede = dict(
            Jupiter=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(12, 0, 2, 2),
                     propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_moons,
                                                                                          time_lag_ganymede,
                                                                                          True, False),
                     propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_jupiter,
                                                                                          time_lag_jupiter_ganymede,
                                                                                          True, True),
                     propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True)],
            Io=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2, 12, 0)],
            Europa=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2, 12, 0)],
            Callisto=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2, 12, 0)],
            Sun=[propagation_setup.acceleration.point_mass_gravity()],
            Saturn=[propagation_setup.acceleration.point_mass_gravity()]
        )
        acceleration_settings_moons['Ganymede'] = acceleration_settings_ganymede

    if 'Callisto' in bodies_to_propagate:
        time_lag_callisto = 1 / mean_motion_callisto * np.arctan(1 / q_moons)
        time_lag_jupiter_callisto = 1 / (spin_frequency_jupiter - mean_motion_callisto) * np.arctan(1 / q_jupiter)
        acceleration_settings_callisto = dict(
            Jupiter=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(12, 0, 2, 2),
                     propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_moons,
                                                                                          time_lag_callisto,
                                                                                          True, False),
                     propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_jupiter,
                                                                                          time_lag_jupiter_callisto,
                                                                                          True, True),
                     propagation_setup.acceleration.relativistic_correction(use_schwarzschild=True)],
            Io=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2, 12, 0)],
            Europa=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2, 12, 0)],
            Ganymede=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2, 12, 0)],
            Sun=[propagation_setup.acceleration.point_mass_gravity()],
            Saturn=[propagation_setup.acceleration.point_mass_gravity()]
        )
        acceleration_settings_moons['Callisto'] = acceleration_settings_callisto

    return acceleration_settings_moons


def create_global_environment(simulation_start_epoch, simulation_end_epoch):
    ### CREATE ENVIRONMENT ###
    # Define bodies that are propagated, and their central bodies of propagation
    bodies_to_propagate = ['Io', 'Europa', 'Ganymede', 'Callisto']
    central_bodies = ['Jupiter', 'Jupiter', 'Jupiter', 'Jupiter']
    # Create standard bodies and associated settings
    body_settings, bodies = create_simulation_bodies(simulation_start_epoch, simulation_end_epoch, bodies_to_propagate)

    ### CREATE PROPAGATION SETTINGS ###
    # Get global accelerations dictionary
    acceleration_settings_moons = get_acceleration_models_moons(bodies_to_propagate)
    acceleration_settings = acceleration_settings_moons
    # Create acceleration models
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies)

    # Define (preliminary) initial state of Io
    initial_state = list()
    for body in bodies_to_propagate:
        initial_state.append(spice_interface.get_body_cartesian_state_at_epoch(
            target_body_name=body,
            observer_body_name='Jupiter',
            reference_frame_name='ECLIPJ2000',
            aberration_corrections='none',
            ephemeris_time=simulation_start_epoch))
    initial_state = np.concatenate(initial_state)

    # Define integrator settings for propagation
    integrator_settings = get_integrator_settings()

    # Define termination condition(s)
    simulation_duration = simulation_end_epoch - simulation_start_epoch
    termination_settings = get_termination_settings(simulation_start_epoch, simulation_duration)

    # Create propagator settings
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies=central_bodies,
        acceleration_models=acceleration_models,
        bodies_to_integrate=bodies_to_propagate,
        initial_states=initial_state,
        initial_time=simulation_start_epoch,
        integrator_settings=integrator_settings,
        termination_settings=termination_settings
    )

    return bodies, body_settings, propagator_settings
