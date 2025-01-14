import numpy as np


def check_speed_ls(location_ls):
    speed = 0
    for i in range(len(location_ls)-1):
        speed += abs(np.linalg.norm(np.array(location_ls[i]) - np.array(location_ls[i+1])))
    return speed/(len(location_ls)-1)

