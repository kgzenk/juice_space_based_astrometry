
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import math
import numpy as np
# tudatpy imports
from tudatpy.kernel.astro import element_conversion

###########################################################################
# LAPLACE UTILITIES #######################################################
###########################################################################

def calculate_mean_longitude(kepler_elements: dict):
    # Calculate dictionary for moon-wise longitudes
    mean_longitude_dict = dict()
    # Loop over every moon of interest (Io, Europa, Ganymede)
    for moon in kepler_elements.keys():
        mean_anomaly_per_moon = list()
        kepler_elements_per_moon = kepler_elements[moon]
        # For every epoch get the mean anomaly of the moon
        for i in range(len(kepler_elements[moon])):
            mean_anomaly_per_moon.append(element_conversion.true_to_mean_anomaly(
                eccentricity=kepler_elements_per_moon[i, 1],
                true_anomaly=kepler_elements_per_moon[i, 5]))
        mean_anomaly_per_moon = np.array(mean_anomaly_per_moon)
        mean_anomaly_per_moon[mean_anomaly_per_moon < 0] = (mean_anomaly_per_moon[mean_anomaly_per_moon < 0]
                                                            + 2 * math.pi)
        # Calculate the mean longitude as
        # (longitude of the ascending node) + (argument of the pericenter) + (mean anomaly)
        longitude_of_the_ascending_node = kepler_elements_per_moon[:, 4]
        argument_of_the_pericenter = kepler_elements_per_moon[:, 3]

        mean_longitude_per_moon = longitude_of_the_ascending_node + argument_of_the_pericenter + mean_anomaly_per_moon
        # Include epoch-wise mean longitude in dictionary
        mean_longitude_per_moon = np.mod(mean_longitude_per_moon, 2 * math.pi)
        mean_longitude_dict[moon] = mean_longitude_per_moon

    return mean_longitude_dict

def resonance_theta(first_moon_index: int,
                    second_moon_index: int,
                    kepler_elements: dict):
    # Perform routine-check if moons match
    if first_moon_index > 3 or second_moon_index > 3:
        print('Moon index must lie between 1 and 3, no calculations were possible.')
        return 0
    if second_moon_index < first_moon_index:
        first_moon_index, second_moon_index = second_moon_index, first_moon_index
    if second_moon_index > first_moon_index + 1:
        print('Your chosen moons are not trapped in resonance, no calculations were possible.')
        return 0
    moons_verbose = ['Io', 'Europa', 'Ganymede']

    # Perform the calculations
    mean_longitude_dict = calculate_mean_longitude(kepler_elements)
    first_mean_longitude = mean_longitude_dict[moons_verbose[first_moon_index - 1]]
    second_mean_longitude = mean_longitude_dict[moons_verbose[first_moon_index]]

    argument_of_the_pericenter = kepler_elements[moons_verbose[second_moon_index - 1]][:, 3]
    longitude_of_the_ascending_node = kepler_elements[moons_verbose[second_moon_index - 1]][:, 4]
    longitude_of_the_pericenter = argument_of_the_pericenter + longitude_of_the_ascending_node

    laplace_theta = first_mean_longitude - 2 * second_mean_longitude + longitude_of_the_pericenter
    laplace_theta = np.mod(laplace_theta, 2 * math.pi)

    return laplace_theta

###########################################################################
# CLOSEST APPROACH UTILITIES ##############################################
###########################################################################

def determine_closest_approaches(closest_approach_input_path):
    ### LOAD DATA ###
    relative_distances = np.loadtxt(closest_approach_input_path)
    # Create dictionary to return
    flybys_dict = dict()
    # Check which spacecraft is dealt with
    epoch_list = relative_distances[:, 0]
    relative_distances_dict = {'Io': relative_distances[:, 1], 'Europa': relative_distances[:, 2],
                               'Ganymede': relative_distances[:, 3], 'Callisto': relative_distances[:, 4]}

    ### DETERMINE CENTRAL INSTANTS PER MOON AND FLYBY ###
    # Loop over all relative distances per moon
    for moon in relative_distances_dict.keys():
        # Get individual list of distances
        relative_distances_per_moon = relative_distances_dict[moon]
        flybys_per_moon_idx = list()
        central_instants_per_moon = list()
        # Start without flyby-phase activated
        flyby_initiated = False
        # Loop over all distances and determine start and end epochs
        for epoch_idx in range(len(relative_distances_per_moon)):
            if not flyby_initiated and relative_distances_per_moon[epoch_idx] < 20000E3:
                start_epoch_flyby = epoch_idx
                flyby_initiated = True
            elif flyby_initiated and relative_distances_per_moon[epoch_idx] > 20000E3:
                end_epoch_flyby = epoch_idx
                flybys_per_moon_idx.append((start_epoch_flyby, end_epoch_flyby))
                flyby_initiated = False
        # Per determined flyby, get central instant (i.e. epoch with minimal distance)
        for flyby in flybys_per_moon_idx:
            central_instant_idx = flyby[0] + np.argmin(relative_distances_per_moon[flyby[0]:flyby[1]])
            central_instants_per_moon.append((epoch_list[central_instant_idx]))
        flybys_dict[moon] = central_instants_per_moon

    print(flybys_dict)

    return flybys_dict

###########################################################################
# GEOMETRY - HELPER FUNCTIONS #############################################
###########################################################################

def unit_vector(vector):
    """
    Returns the unit vector of the vector.
    """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))