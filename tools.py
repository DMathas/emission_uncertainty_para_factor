## TNO tools for calculations - David Mathas - TNO
# import binas
import numpy as np
import numba

def ll_distance( lon1, lat1, lon2, lat2, radius=6371000) : # NOT MY OWN CODE

    """
    NOT MY CODE: TNO code 

    Calculates the distance between two points given their (lat, lon) co-ordinates.
    It uses the Spherical Law Of Cosines (http://en.wikipedia.org/wiki/Spherical_law_of_cosines):
    
    cos(c) = cos(a) * cos(b) + sin(a) * sin(b) * cos(C)                        (1)
    
    In this case:
    a = lat1 in radians, b = lat2 in radians, C = (lon2 - lon1) in radians
    and because the latitude range is  [-pi/2, pi/2] instead of [0, pi]
    and the longitude range is [-pi, pi] instead of [0, 2pi]
    (1) transforms into:
    
    x = cos(c) = sin(a) * sin(b) + cos(a) * cos(b) * cos(C)
    
    Finally the distance is arccos(x)
    
    (source: http://cyberpython.wordpress.com/2010/03/31/python-calculate-the-distance-between-2-points-given-their-coordinates/)
    """

    # modules:
    import math
    
    # quick fix ..
    if (lon1 == lon2) and (lat1 == lat2) :
    
        # same location ...
        distance = 0.0
        
    else :

        delta = lon2 - lon1
        a = np.radians(lat1)
        b = np.radians(lat2)
        C = np.radians(delta)
        x = np.sin(a) * np.sin(b) + np.cos(a) * np.cos(b) * np.cos(C)

        # distance in radians:
        phi = np.acos(x)   # radians

        # in m over the globe:
        distance = phi * radius
        
    #endif
    
    # ok
    return distance

##
@numba.jit(nopython=True)
def make_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """DOCSTRINGS
    """
    n_points = coords.shape[0]
    distance_matrix = np.zeros((n_points, n_points))

    for i in range(n_points):
        for j in range(i + 1, n_points): 
            distance = ll_distance(coords[i][1], coords[i][0], coords[j][1], coords[j][0])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # since it's a symmetric matrix

    return distance_matrix



# def make_distance_matrix_vec(coords): # made with help of Arjo function
#     """
#     DOCSTRINGS
#     """
#     # Earths' radius in meters:
#     radius = 6371000
    
#     # From degrees to radians:
#     lat_rad, lon_rad = np.radians(coords[:, 0]), np.radians(coords[:, 1])

#     lat_rad = lat_rad[:, np.newaxis]
#     lon_rad = lon_rad[:, np.newaxis]

#     # Using the spherical law of cosines:
#     delta_lon = lon_rad - lon_rad.T
#     #### TIME BORDER
#     # print(delta_)
#     cos_c = np.sin(lat_rad) * np.sin(lat_rad.T) + np.cos(lat_rad) * np.cos(lat_rad.T) * np.cos(delta_lon)
#     cos_c = np.clip(cos_c, -1, 1) # need this range for arccos!

#     distance_matrix = radius * np.arccos(cos_c)

#     return distance_matrix


def make_distance_matrix_vec(coords): 
    """
    NOT fully MY CODE: TNO code 

    Calculate the geodesic distance matrix for a set of geographical coordinates using the spherical law of cosines,
    using 32-bit floating point precision.
    """

    unique_coords, inv = np.unique(coords, axis=0, return_inverse=True)
    # Earth's radius in meters, as a 32-bit float:
    radius = np.float32(6371000)

    # Convert degrees to radians and cast to float32:
    lat_rad = np.radians(unique_coords[:, 1]).astype(np.float32)
    lon_rad = np.radians(unique_coords[:, 0]).astype(np.float32)

    lat_rad = lat_rad[:, np.newaxis]
    lon_rad = lon_rad[:, np.newaxis]

    # Calculate delta longitude:
    delta_lon = lon_rad - lon_rad.T

    # Calculate the spherical law of cosines distances:
    cos_d = np.sin(lat_rad) * np.sin(lat_rad.T) + np.cos(lat_rad) * np.cos(lat_rad.T) * np.cos(delta_lon)
    cos_d = np.clip(cos_d, -1, 1)  # Ensure the values are within the valid range for arccos

    # Compute UNIQUE distances using 32-bit floating point precision:
    unique_distance_matrix = radius * np.arccos(cos_d.astype(np.float32)) # distance in radians:

    distance_matrix = unique_distance_matrix[inv][:, inv]

    return distance_matrix

