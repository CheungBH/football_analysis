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

def calculate_distances(points_dict, ref_point, item):
    distances = {}
    for key, value in points_dict.items():
        point = value[item]
        distance = math.sqrt((point[0] - ref_point[0])**2 + (point[1] - ref_point[1])**2)
        distances[key] = distance
    return distances

def find_closest_player(players, ball_position, item):
    distances = calculate_distances(players, ball_position, item)
    min_key = min(distances, key=distances.get)
    return min_key, distances[min_key]

def check_ball_possession(lst,thre):
    count_1 = 0
    count_2 = 0 # 持球状态：0-交错，1-1持球，2-2持球
    prev_possession, possession = 0,0
    possession_list=[]
    possession_changed = False
    for i in range(len(lst)):
        if lst[i] == 1:
            count_1 += 1
            count_2 = 0
            if count_1 >= thre:
                possession = 1
                possession_list.append(possession)
        elif lst[i] == 2:
            count_2 += 1
            count_1 = 0
            if count_2 >= thre:
                possession = 2
                possession_list.append(possession)
        else:
            count_1 = 0
            count_2 = 0
            possession = 0
        # 如果 1 和 2 交错，判定为 0
        if i > 0 and lst[i] != lst[i - 1]:
            count_1 = 0
            count_2 = 0
            possession = 0
        if len(possession_list) > 1:
            if possession_list[-1] != possession_list[-2]:
                possession_changed = True

    return possession, possession_changed

'''
point1 = (0, 0)
point2 = (0, 1)
point3 = (1, 1)

vector1 = calculate_vector(point1, point2)
vector2 = calculate_vector(point2, point3)

angle_between_vectors = calculate_angle_between_vectors(vector1, vector2)

print("Vector between point1 and point2:", vector1)
print("Vector between point2 and point3:", vector2)
print("Angle between the two vectors:", angle_between_vectors, "degrees")

'''
def angle_between_vectors(v1, v2):
    """计算向量之间的夹角"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(cos_theta)
    return np.degrees(angle)

def is_vector_in_angle(A, B, C, B_motion_vector):
    """判断 B 的运动向量是否在角 ABC 内"""
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)

    angle_ABC = angle_between_vectors(BA, BC)
    angle_BM_BA = angle_between_vectors(B_motion_vector, BA)
    angle_BM_BC = angle_between_vectors(B_motion_vector, BC)

    # 判断 B 的运动向量是否在角 ABC 内
    return angle_BM_BA <= angle_ABC and angle_BM_BC <= angle_ABC

def compare_motion_direction(A, B, C):
    """比较 A 和 B 在第一帧的位置，并计算 B 的运动方向，判断是否在三角形 B 角内"""
    A_first, A_last = A[0], A[-1]
    B_first, B_last = B[0], B[-1]

    # 计算 B 的运动方向
    B_motion_vector = np.array(B_last) - np.array(B_first)

    # 判断 B 的运动方向是否在角 ABC 的范围内
    #return is_vector_in_angle(A_first, B_first, C, B_motion_vector)
    A_motion_vector = np.array(A_last) - np.array(A_first)
    return True if A_motion_vector[0]*B_motion_vector[0]>0 else False
