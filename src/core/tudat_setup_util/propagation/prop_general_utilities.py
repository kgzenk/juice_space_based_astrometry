
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# tudatpy imports
from tudatpy.kernel.numerical_simulation import propagation_setup


###########################################################################
# INTEGRATOR UTILITIES ####################################################
###########################################################################

def get_integrator_settings(time_step=30.0):
    """
    Retrieves the integrator settings for an RKDP8 integrator with fixed step-size of 30.0 minutes.

    Parameters
    ----------
    time_step : float (default=30.0)
        Time-step in minutes

    Returns
    -------
    integrator_settings : tudatpy.kernel.numerical_simulation.propagation_setup.integrator.IntegratorSettings
        Integrator settings to be provided to the dynamics simulator.
    """
    # Create integrator settings
    time_step_sec = time_step * 60.0
    integrator_settings = propagation_setup.integrator. \
        runge_kutta_fixed_step_size(initial_time_step=time_step_sec,
                                    coefficient_set=propagation_setup.integrator.CoefficientSets.rkdp_87)

    return integrator_settings


###########################################################################
# TERMINATION SETTINGS UTILITIES ##########################################
###########################################################################

def get_termination_settings(simulation_start_epoch,
                             maximum_duration):
    """
    Get the termination settings for the simulation.

    Parameters
    ----------
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.
    maximum_duration : float
        Maximum duration of the simulation [s].

    Returns
    -------
    hybrid_termination_settings : tudatpy.kernel.numerical_simulation.propagation_setup.propagator.
                                  PropagationTerminationSettings
        Propagation termination settings object.
    """
    # Mission duration
    time_termination_settings = propagation_setup.propagator.time_termination(
        simulation_start_epoch + maximum_duration,
        terminate_exactly_on_final_condition=False)

    # Define list of termination settings
    termination_settings_list = [time_termination_settings]

    return propagation_setup.propagator.hybrid_termination(termination_settings_list, fulfill_single_condition=True)
