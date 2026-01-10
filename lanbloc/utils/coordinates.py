"""
Coordinate Transformation Utilities

Functions for converting between different coordinate systems:
- Latitude/Longitude (geographic)
- XYZ (Cartesian, Earth-centered)
- Local ENU (East-North-Up)
"""

import math
import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass


# Earth's radius in kilometers
EARTH_RADIUS_KM = 6371.0
EARTH_RADIUS_M = 6371000.0


def latlon_to_xyz(
    lat: float,
    lon: float,
    radius: float = EARTH_RADIUS_KM
) -> Tuple[float, float, float]:
    """
    Convert latitude/longitude to Cartesian XYZ coordinates.
    
    Uses Earth-centered coordinate system where:
    - X axis passes through the equator at 0° longitude
    - Y axis passes through the equator at 90° longitude
    - Z axis passes through the North Pole
    
    Args:
        lat: Latitude in degrees (-90 to 90)
        lon: Longitude in degrees (-180 to 180)
        radius: Sphere radius (default: Earth radius in km)
        
    Returns:
        Tuple (x, y, z) in the same units as radius
    """
    # Convert to radians
    theta = math.pi / 2 - math.radians(lat)  # Zenith angle
    phi = math.radians(lon)  # Azimuthal angle
    
    # Spherical to Cartesian
    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(theta)
    
    return (x, y, z)


def xyz_to_latlon(
    x: float,
    y: float,
    z: float
) -> Tuple[float, float]:
    """
    Convert Cartesian XYZ coordinates to latitude/longitude.
    
    Args:
        x: X coordinate
        y: Y coordinate
        z: Z coordinate
        
    Returns:
        Tuple (latitude, longitude) in degrees
    """
    r = math.sqrt(x**2 + y**2 + z**2)
    
    # Avoid division by zero
    if r < 1e-10:
        return (0.0, 0.0)
    
    # Latitude from z
    lat = math.degrees(math.asin(z / r))
    
    # Longitude from x, y
    lon = math.degrees(math.atan2(y, x))
    
    return (lat, lon)


def latlon_to_enu(
    lat: float,
    lon: float,
    alt: float,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float = 0.0
) -> Tuple[float, float, float]:
    """
    Convert latitude/longitude/altitude to local ENU coordinates.
    
    ENU (East-North-Up) is a local tangent plane coordinate system.
    
    Args:
        lat: Point latitude in degrees
        lon: Point longitude in degrees
        alt: Point altitude in meters
        ref_lat: Reference point latitude
        ref_lon: Reference point longitude
        ref_alt: Reference point altitude
        
    Returns:
        Tuple (east, north, up) in meters
    """
    # Convert to ECEF first
    x, y, z = latlon_to_ecef(lat, lon, alt)
    ref_x, ref_y, ref_z = latlon_to_ecef(ref_lat, ref_lon, ref_alt)
    
    # Difference in ECEF
    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z
    
    # Rotation matrix from ECEF to ENU
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)
    
    sin_lat = math.sin(ref_lat_rad)
    cos_lat = math.cos(ref_lat_rad)
    sin_lon = math.sin(ref_lon_rad)
    cos_lon = math.cos(ref_lon_rad)
    
    # East
    east = -sin_lon * dx + cos_lon * dy
    
    # North
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    
    # Up
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    
    return (east, north, up)


def latlon_to_ecef(
    lat: float,
    lon: float,
    alt: float = 0.0
) -> Tuple[float, float, float]:
    """
    Convert latitude/longitude/altitude to ECEF coordinates.
    
    ECEF (Earth-Centered, Earth-Fixed) coordinates.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude above ellipsoid in meters
        
    Returns:
        Tuple (x, y, z) in meters
    """
    # WGS84 ellipsoid parameters
    a = 6378137.0  # Semi-major axis in meters
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f**2  # Eccentricity squared
    
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    
    # Radius of curvature in the prime vertical
    N = a / math.sqrt(1 - e2 * sin_lat**2)
    
    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1 - e2) + alt) * sin_lat
    
    return (x, y, z)


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    radius: float = EARTH_RADIUS_M
) -> float:
    """
    Calculate great-circle distance between two points using Haversine formula.
    
    Args:
        lat1, lon1: First point coordinates in degrees
        lat2, lon2: Second point coordinates in degrees
        radius: Sphere radius (default: Earth radius in meters)
        
    Returns:
        Distance in same units as radius
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat / 2)**2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return radius * c


def euclidean_distance(
    p1: Tuple[float, ...],
    p2: Tuple[float, ...]
) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1: First point (any dimension)
        p2: Second point (same dimension)
        
    Returns:
        Euclidean distance
    """
    return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))


@dataclass
class CoordinateTransformer:
    """
    Coordinate transformer for converting between reference frames.
    
    Supports conversions between:
    - Geographic (lat/lon)
    - ECEF
    - Local ENU
    - Custom XYZ (Earth-centered Cartesian)
    
    Attributes:
        reference_lat: Reference point latitude
        reference_lon: Reference point longitude
        reference_alt: Reference point altitude
        earth_radius: Earth radius for XYZ conversion
    """
    reference_lat: float
    reference_lon: float
    reference_alt: float = 0.0
    earth_radius: float = EARTH_RADIUS_KM
    
    def latlon_to_xyz(self, lat: float, lon: float) -> Tuple[float, float, float]:
        """Convert lat/lon to XYZ using configured earth radius."""
        return latlon_to_xyz(lat, lon, self.earth_radius)
    
    def xyz_to_latlon(self, x: float, y: float, z: float) -> Tuple[float, float]:
        """Convert XYZ to lat/lon."""
        return xyz_to_latlon(x, y, z)
    
    def latlon_to_local(
        self,
        lat: float,
        lon: float,
        alt: float = 0.0
    ) -> Tuple[float, float, float]:
        """Convert lat/lon/alt to local ENU coordinates."""
        return latlon_to_enu(
            lat, lon, alt,
            self.reference_lat, self.reference_lon, self.reference_alt
        )
    
    def batch_latlon_to_xyz(
        self,
        coordinates: List[Tuple[float, float]]
    ) -> List[Tuple[float, float, float]]:
        """Convert multiple lat/lon pairs to XYZ."""
        return [self.latlon_to_xyz(lat, lon) for lat, lon in coordinates]
    
    def batch_xyz_to_latlon(
        self,
        coordinates: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float]]:
        """Convert multiple XYZ points to lat/lon."""
        return [self.xyz_to_latlon(x, y, z) for x, y, z in coordinates]
    
    def distance(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        method: str = "haversine"
    ) -> float:
        """
        Calculate distance between two geographic points.
        
        Args:
            p1: First point (lat, lon) in degrees
            p2: Second point (lat, lon) in degrees
            method: Distance calculation method ("haversine" or "euclidean_xyz")
            
        Returns:
            Distance in meters
        """
        if method == "haversine":
            return haversine_distance(p1[0], p1[1], p2[0], p2[1])
        else:
            xyz1 = latlon_to_xyz(p1[0], p1[1], EARTH_RADIUS_M)
            xyz2 = latlon_to_xyz(p2[0], p2[1], EARTH_RADIUS_M)
            return euclidean_distance(xyz1, xyz2)


# Pre-computed landmark XYZ coordinates for the trilat dataset
# These are provided in the original data file
LANDMARK_XYZ = {
    'l1': (-155.46958665050204, -5021.052019397063, 3918.482719897511),
    'l2': (-155.5275225330039, -5021.067536189176, 3918.460537859467),
    'l3': (-155.50858521677614, -5021.096175968406, 3918.4245905209573),
    'l4': (-155.427796619701, -5021.099019535775, 3918.424152137542),
    'l5': (-155.51625603400018, -5021.1909069352505, 3918.302894139237),
    'l6': (-155.43124174129483, -5021.181976267134, 3918.3177117611117),
    'l7': (-155.4446365600995, -5021.217960998301, 3918.271066824341),
    'l8': (-155.50334129050444, -5021.17043872006, 3918.3296359764445),
    'l9': (-155.345696490337, -5021.099165492784, 3918.4272208208235),
    'l10': (-155.2337216990451, -5021.069306824268, 3918.469919205038),
    'l11': (-155.26568272057534, -5021.098698213657, 3918.4309909154235),
    'l12': (-155.29315955488224, -5021.059668642676, 3918.4799142684606),
    'l13': (-155.442960717081, -5021.231833584247, 3918.2533556884355),
    'l14': (-155.51444341560574, -5021.290991593178, 3918.174707135241),
    'l15': (-155.46123863929583, -5021.292365445306, 3918.17505785556),
    'l16': (-155.50364490508034, -5021.248222482582, 3918.229945349909),
    'l17': (-155.6624715916961, -5021.043306353907, 3918.4862269321043),
    'l18': (-155.57735854829974, -5021.028291043319, 3918.5088472593434),
    'l19': (-155.60928539431396, -5021.056518470882, 3918.471409697936),
    'l20': (-155.69365182574242, -5021.092698783091, 3918.421697189865),
    'l21': (-155.45757680085168, -5021.210923684809, 3918.279571659422),
    'l22': (-155.49922578969395, -5021.173508255323, 3918.3258658225127),
    'l23': (-155.42421592963404, -5021.127867546292, 3918.387327824125),
    'l24': (-155.41698950223196, -5021.106948907657, 3918.414420018038),
    'l25': (-155.37395626040703, -5021.102464846763, 3918.4218725433025),
    'l26': (-155.53128662566192, -5021.030471038011, 3918.507882828409),
    'l27': (-155.49234605733477, -5021.050493554795, 3918.483772008089),
    'l28': (-155.52946694816177, -5021.062344243297, 3918.467113570411),
    'l29': (-155.28038174914317, -5020.9670070273, 3918.599152425158),
    'l30': (-155.30926356995215, -5021.01880065921, 3918.531642857578),
    'l31': (-155.33827627113155, -5021.012429234593, 3918.5386568715703),
    'l32': (-155.36652816883068, -5021.09769988053, 3918.4282729404686),
    'l33': (-155.42196494585352, -5021.094821030207, 3918.4297634430054),
    'l34': (-155.36097192550096, -5021.06838169041, 3918.4660614571117)
}

NODE_XYZ = {
    'n35': (-155.48900838034146, -5021.075844982095, 3918.451419529179),
    'n36': (-155.48440948410598, -5021.0802980198505, 3918.445895918976),
    'n37': (-155.4694431034911, -5021.172035780967, 3918.3289345526246),
    'n38': (-155.47664447190914, -5021.177970641806, 3918.3210435293827),
    'n39': (-155.47456568172285, -5021.195824256157, 3918.298247185711),
    'n40': (-155.48535721297145, -5021.204384684454, 3918.2868489836173),
    'n41': (-155.47657124279831, -5021.263428768788, 3918.2115326643225),
    'n42': (-155.47596153849395, -5021.272068361994, 3918.2004850277053),
    'n43': (-155.63586840978962, -5021.079300426541, 3918.4411613921743),
    'n44': (-155.6505190787095, -5021.068035503408, 3918.455014257084),
    'n45': (-155.25330558310975, -5021.078554201358, 3918.457293839594),
    'n46': (-155.27397556947002, -5021.074699168972, 3918.4614146213135),
    'n47': (-155.27753667485024, -5021.067883605779, 3918.4700068811003),
    'n48': (-155.38389219578713, -5021.0720515040975, 3918.4604501832755),
    'n49': (-155.3914103373693, -5021.08256119688, 3918.4466850064377),
    'n50': (-155.59383184316783, -5021.041875840492, 3918.49078607422),
    'n51': (-155.6168492699034, -5021.034525438007, 3918.499290619156),
    'n52': (-155.60319636985224, -5021.024206047709, 3918.513055683552),
    'n53': (-155.40053092625925, -5021.1165036908515, 3918.402829132197),
    'n54': (-155.40101815189624, -5021.110139063996, 3918.4109655504612),
    'n55': (-155.53246402642, -5021.044694025854, 3918.4896112186752),
    'n56': (-155.52476341945388, -5021.048408458807, 3918.485157286755),
    'n57': (-155.52594315460618, -5021.04146800135, 3918.4940037745014),
    'n58': (-155.3077623701633, -5020.989835185148, 3918.568817044686),
    'n59': (-155.29698155777163, -5020.980178629677, 3918.581617540579),
    'n60': (-155.28187201175615, -5020.978038951442, 3918.5849579397727)
}
