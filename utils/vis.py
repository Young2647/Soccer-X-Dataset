import numpy as np
import torch

def axis_angle_to_quaternion_np(axis_angle: np.array) -> np.array:
    axis_angle_th = torch.from_numpy(axis_angle)
    return axis_angle_to_quaternion(axis_angle_th).numpy()

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def cope_tensor(message):
    return ','.join(['%g' % v for v in message])  + '#'
def cope_string(message):
    return message + '#'

def package_message(data):
    s = "@"
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            s = s + key + '&'
            s += cope_tensor(value)
        elif isinstance(value, np.ndarray):
            s = s + key + '&'
            s += cope_tensor(value)
        else:
            s = s + key + '&'
            s += cope_string(value)
    return s