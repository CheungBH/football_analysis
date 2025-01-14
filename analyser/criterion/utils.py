import numpy as np
import math


def check_speed_ls(location_ls):
    speed = 0
    for i in range(len(location_ls)-1):
        speed += abs(np.linalg.norm(np.array(location_ls[i]) - np.array(location_ls[i+1])))
    return speed/(len(location_ls)-1)


def calculate_vector(point1, point2):
    vector = (point2[0] - point1[0], point2[1] - point1[1])
    return vector


def calculate_angle_between_vectors(vector1, vector2):
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude_vector1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude_vector2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
    angle_radians = math.acos(dot_product / (magnitude_vector1 * magnitude_vector2))
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees


point1 = (0, 0)
point2 = (0, 1)
point3 = (1, 1)

vector1 = calculate_vector(point1, point2)
vector2 = calculate_vector(point2, point3)

angle_between_vectors = calculate_angle_between_vectors(vector1, vector2)

print("Vector between point1 and point2:", vector1)
print("Vector between point2 and point3:", vector2)
print("Angle between the two vectors:", angle_between_vectors, "degrees")
