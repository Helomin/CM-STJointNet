import numpy as np
from scipy.ndimage import zoom


def interpolate(data, size=128):
    tem_list = []
    for i in range(data.shape[0]):
        tem_list.append(
            zoom(data[i], zoom=(size/data.shape[1], size/data.shape[2]), order=0))
    return np.array(tem_list)


def data_filtering(data, range=[0, 255]):
    index_to_delete = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.all(data[i, j] == range[0]) or np.all(data[i, j] == range[1]):
                index_to_delete.append(i)
    index_to_delete = np.unique(index_to_delete)
    data = np.delete(data, index_to_delete, axis=0)
    return data
