import numpy as np
import math

def calculate_speed(position):
    valid_position=[]
    valid_distance=[]
    for i in range(len(position)-1):
        if position[i] != [-1,-1] and position[i+1] != [-1,-1]:
            x1, y1 = position[i]
            x2, y2 = position[i+1]
            valid_position.append([position[i],position[i+1]])
            valid_distance.append(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    return valid_distance


def check_speed_distance(location_ls):
    speed = 0
    for i in range(len(location_ls)-1):
        speed += abs(np.linalg.norm(np.array(location_ls[i]) - np.array(location_ls[i+1])))
    return speed/(len(location_ls)-1)

def check_speed_displacement(location_ls):
    speed_x, speed_y = 0, 0
    for i in range(len(location_ls)-1):
        speed_x += np.array(location_ls[i+1][0]) - np.array(location_ls[i][0])
        speed_y += np.array(location_ls[i+1][1]) - np.array(location_ls[i][1])

    speed = abs(speed_x) + abs(speed_y)
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


def polygon_area(points):
    n = len(points)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0][0] * points[j][0][1]
        area -= points[j][0][0] * points[i][0][1]
    area = abs(area) / 2.0
    return area

def vector_angle(v1, v2):
    # 计算两个向量的夹角（以弧度为单位）
    dot_product = sum(a*b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a**2 for a in v1))
    magnitude_v2 = math.sqrt(sum(a**2 for a in v2))
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2) if magnitude_v1 * magnitude_v2 != 0 else 0
    angle = math.acos(cosine_angle)
    return math.degrees(angle)


def is_in_rectangle(value, rect):
    return rect[0][0] <= value[0] <= rect[1][0] and rect[0][1] <= value[1] <= rect[1][1]


def is_within_radius(a, b, radius=20):
    distance = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    return distance <= radius

def calculate_ratio(v1, v2, boundary):
    diff1 = abs(v1 - boundary)
    diff2 = abs(v2 - boundary)
    return min(diff1, diff2) / max(diff1, diff2)


def is_point_in_triangle(a, b, c, d):
    # 计算向量
    def vector(p, q):
        return q[0] - p[0], q[1] - p[1]

    # 计算叉积
    def cross_product(u, v):
        return u[0] * v[1] - u[1] * v[0]

    # 向量ba, ca, da
    ba = vector(b, a)
    bc = vector(b, c)
    bd = vector(b, d)
    ca = vector(c, a)
    cb = vector(c, b)
    cd = vector(c, d)
    da = vector(d, a)
    db = vector(d, b)
    dc = vector(d, c)

    # 计算叉积
    cross1 = cross_product(bc, ba)
    cross2 = cross_product(cd, ca)
    cross3 = cross_product(db, da)

    # 检查点a是否在三角形bcd内
    if (cross1 >= 0 and cross2 >= 0 and cross3 >= 0) or (cross1 <= 0 and cross2 <= 0 and cross3 <= 0):
        return True
    else:
        return False

def is_point_in_triangle(a, b, c, d):
    # 计算向量
    def vector(p, q):
        return q[0] - p[0], q[1] - p[1]

    # 计算叉积
    def cross_product(u, v):
        return u[0] * v[1] - u[1] * v[0]

    # 向量ba, ca, da
    ba = vector(b, a)
    bc = vector(b, c)
    bd = vector(b, d)
    ca = vector(c, a)
    cb = vector(c, b)
    cd = vector(c, d)
    da = vector(d, a)
    db = vector(d, b)
    dc = vector(d, c)

    # 计算叉积
    cross1 = cross_product(bc, ba)
    cross2 = cross_product(cd, ca)
    cross3 = cross_product(db, da)

    # 检查点a是否在三角形bcd内
    if (cross1 >= 0 and cross2 >= 0 and cross3 >= 0) or (cross1 <= 0 and cross2 <= 0 and cross3 <= 0):
        return True
    else:
        return False

