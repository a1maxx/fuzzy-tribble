import numpy as np


def depr_setRandomState(nSensors, cluster_size):
    dict_pris = {1: 0.1, 2: 0.3, 3: 0.6}
    dict_sensors = {i: [0, 0] for i in range(0, nSensors)}
    for i in range(0, nSensors):
        dict_sensors[i][0] = np.random.randint(low=1, high=3)
        dict_sensors[i][1] = dict_pris[dict_sensors[i][0]]

    return np.hstack((np.array([dict_sensors[i][1] for i in dict_sensors]), np.array([cluster_size])))


def depr_update_pris(cluster_size, dict_sensors):
    transition_name = [["s_11", "s_12", "s_13"], ["s_21", "s_22", "s_23"], ["s_31", "s_32", "s_33"]]
    transition_matrix = [[0.2, 0.6, 0.2], [0.1, 0.6, 0.3], [0.2, 0.7, 0.1]]
    dict_pris = {1: 0.1, 2: 0.3, 3: 0.6}
    n_sensors = 4
    for i in range(0, n_sensors):
        change = np.random.choice(transition_name[dict_sensors[i][0]], replace=True,
                                  p=transition_matrix[dict_sensors[i][0]])
        dict_sensors[i][0] = int(change[-1])
        dict_sensors[i][1] = dict_pris[dict_sensors[i][0]]

    return np.hstack((np.array([dict_sensors[i][1] for i in dict_sensors]), np.array([cluster_size])))


def setRandomState(nSensors, cluster_size):
    dict_pris = {1: 0.1, 2: 0.3, 3: 0.6}
    sensor_states = np.random.randint(low=0, high=3, size=nSensors, dtype="int16")

    return np.hstack((sensor_states, np.array([cluster_size])))


def transition_state(state, cluster_size):
    n_sensors = len(state) - 1
    transition_name = [["s_01", "s_01", "s_02"], ["s_10", "s_11", "s_12"], ["s_21", "s_21", "s_22"]]
    transition_matrix = [[0.2, 0.6, 0.2], [0.1, 0.6, 0.3], [0.2, 0.7, 0.1]]
    for i in range(0, n_sensors):
        change = np.random.choice(transition_name[state[i]], replace=True,
                                  p=transition_matrix[state[i]])
        state[i] = int(change[-1])

    return state
