from .coordinates import CartesianCoords, EulerSphericalCoords, SphericalCoords, CylindricalCoords,\
    BalancedSphericalCoords, DirectionalSphericalCoords, DirectionalBalancedSphericalCoords, \
    YinYangSphericalCoords, GenericSphericalCoords

coordinates_dict = {
    'xyz': CartesianCoords,
    'sphere': SphericalCoords,
    'balanced_sphere': BalancedSphericalCoords,
    'directional_sphere': DirectionalSphericalCoords,
    'directional_balanced_sphere': DirectionalBalancedSphericalCoords,
    'cylinder': CylindricalCoords,
    'euler_sphere': EulerSphericalCoords,
    'yinyang': YinYangSphericalCoords,
    'generic_sphere': GenericSphericalCoords
}
