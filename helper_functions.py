import numpy as np
import pyomo.common.errors
import pyomo.environ as pyo
from numpy import ndarray
from pyomo.environ import value, NonNegativeReals, TransformationFactory, Binary, Constraint, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
import random
from math import sqrt
from pyomo.gdp import Disjunction
from pyomo.common.timing import TicTocTimer


def setRandomState(nSensors, cluster_size):
    sensor_states = np.random.randint(low=0, high=3, size=nSensors, dtype="int16")

    return np.hstack((sensor_states, np.array([cluster_size])))


def transition_state(state, cluster_size):
    state[len(state) - 1] = cluster_size
    n_sensors = len(state) - 1
    transition_name = [["s_01", "s_00", "s_02"], ["s_10", "s_11", "s_12"], ["s_21", "s_21", "s_22"]]
    transition_matrix = [[0.2, 0.6, 0.2], [0.1, 0.6, 0.3], [0.2, 0.7, 0.1]]
    for i in range(0, n_sensors):
        change = np.random.choice(transition_name[state[i]], replace=True,
                                  p=transition_matrix[state[i]])
        state[i] = int(change[-1])

    return state


def create_param_set(state) -> dict:
    sensor_x = np.array([74, 56, 60, 37, 66, 132, 73, 142, 159, 166, 88, 64])
    sensor_y = np.array([179, 152, 122, 79, 128, 2, 43, 168, 101, 45, 105, 178])
    sensor_d = np.round(np.sqrt(np.add(sensor_x ** 2, sensor_y ** 2)), 0).astype(int)

    def db2pow(d):
        return 10 ** (d / 10)

    param_set = dict()
    dict_pris = {0: 0.1, 1: 0.3, 2: 0.6}
    # N = 10 ** 5
    # Pt = 30
    # pt = 10 ** -3 * db2pow(Pt)
    # No = -114
    # no = 10 ** -3 * db2pow(No)
    d, eta = sensor_d.astype(np.float32), np.random.uniform(3,4)

    h = np.sqrt(d ** -eta) * (np.random.normal(size=(1,)) + 1j * np.random.normal(size=(1,))) / sqrt(2)
    h = np.array(h).astype(dtype=np.complex_)
    g = np.abs(h) ** 2
    g: ndarray = np.sort(g)

    # max_value = np.max(np.abs(g), axis=0)
    mean_value = np.mean(np.abs(g), axis=0)
    param_set['gains'] = g[::-1] / mean_value
    param_set['scores'] = [dict_pris[i] for i in state[0:len(state) - 1:1]]
    param_set['bits'] = {(i, int(j)): np.random.uniform(90, 150) for i, j in
                         zip(range(1, len(sensor_x) + 1), np.ones(len(sensor_x)))}

    return param_set


# %%
def solve_noma(cluster_size, param_set):
    # Parameters
    model: ConcreteModel = pyo.ConcreteModel()
    model.NOFD = pyo.Param(initialize=len(param_set['scores']))  # Number of devices
    model.NOFTF = pyo.Param(initialize=1)  # Number of time frames
    model.NOFS = pyo.Param(initialize=5)  # Number of time slots
    model.C = pyo.Param(domain=pyo.NonNegativeIntegers, initialize=cluster_size)
    # model.C = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=cluster_size)
    model.M = pyo.RangeSet(1, model.NOFD)  # Range of devices
    model.K = pyo.RangeSet(1, model.NOFTF)  # Range of time frames
    model.S = pyo.RangeSet(1, model.NOFS)  # Range of time slots

    def N_init(m):
        return ((t, j) for t, j in (range(1, m.NOFS + 1) * m.K))  # Initialization of RBs (j,t)

    model.N = pyo.Set(initialize=N_init)  # Set of RBs (j,t)

    model.L = pyo.Param(model.M, model.K,
                        initialize=12)  # The length of packet in bits that device i needs to send at time t

    def A_init(m, i, j):
        return int(random.uniform(1, m.NOFS))

    model.A = pyo.Param(model.M, model.K, initialize=A_init,
                        within=pyo.NonNegativeIntegers)  # Packet arrival time of device i at time frame t

    def D_init(m, i, j):
        return int(random.uniform((m.A[i, 1]) + 1, m.NOFS + 1))

    model.D = pyo.Param(model.M, model.K, initialize=D_init)  # Packet due time of device i at time frame t
    model.E = pyo.Param(model.M, initialize=120)  # Units of energy stored in the battery of device i
    model.W = pyo.Param(model.N, initialize=round(300 / value(model.NOFS), 0))  # Bandwidth of RB (j,t)
    model.H = pyo.Param(model.M, model.N, initialize=1)  # The wireless channel between device i and BS using RB (j,t)

    # model.G = pyo.Param(model.M, model.N,
    #                     initialize=dict(
    #                         zip((model.M * model.N), np.array(
    #                             np.array([0.2, 0.1, 0.05, 0.01]).repeat(len(model.M) / 4 * (value(model.NOFS)))))))

    model.G = pyo.Param(model.M, model.N,
                        initialize=dict(
                            zip((model.M * model.N), np.array(
                                param_set['gains'].repeat(value(model.NOFS))))))

    model.PRI = pyo.Param(model.M, initialize=dict(zip(model.M, param_set['scores'])))

    model.x = pyo.Var(model.M, model.N, domain=Binary)
    model.p = pyo.Var(model.M, model.N, domain=NonNegativeReals)
    model.z = pyo.Var(model.M, model.K, domain=Binary)

    def objective_expression(m):
        return sum(m.PRI[i] * m.z[i, t] for i, t in m.M * m.K)

    model.o = pyo.Objective(rule=objective_expression, sense=pyo.maximize)

    def constraint_1(m, i, t):
        return sum(m.W[j, t] *
                   pyo.log(1 +
                           (m.p[i, j, t] * m.G[i, j, t]) /
                           (1 + sum(m.p[i_prime, j, t] * m.G[i_prime, j, t] for i_prime in range(i + 1, len(m.M) + 1))))
                   for j in range(1, m.NOFS + 1)) / pyo.log(2) >= m.L[i, t] * m.z[i, t]

    def constraint_2(m, i, j, t):
        return m.p[i, j, t] <= m.E[i] * m.x[i, j, t]

    def constraint_3(m, i):
        return sum(sum(m.p[i, j, t] for j in m.S) for t in m.K) <= m.E[i]

    def constraint_4(m, i, j, t):
        if j < m.A[i, t] or j > m.D[i, t]:
            return m.x[i, j, t] == 0
        else:
            return pyo.Constraint.Skip

    def constraint_5(m, i, j, t):
        return m.x[i, j, t] <= m.z[i, t]

    def constraint_6(m, j, t):
        return sum(m.x[i, j, t] for i in m.M) == m.C

    def constraint_7(m, i, t):
        return m.z[i, t] <= m.L[i, t]

    model.cons1 = Constraint(model.M, model.K, rule=constraint_1)
    model.cons2 = Constraint(model.M, model.N, rule=constraint_2)
    model.cons3 = Constraint(model.M, rule=constraint_3)
    model.cons4 = Constraint(model.M, model.N, rule=constraint_4)
    model.cons5 = Constraint(model.M, model.N, rule=constraint_5)
    model.cons6 = Constraint(model.N, rule=constraint_6)
    model.cons7 = Constraint(model.M, model.K, rule=constraint_7)

    model.name = "UplinkNoma"

    opt = pyo.SolverFactory(
        'scip')

    results = opt.solve(model, tee=False)

    # if (results.solver.status == SolverStatus.ok) and (
    #         results.solver.termination_condition == TerminationCondition.optimal):
    #     return sum([value(model.z[i, j]) for (i, j) in model.z.index_set()])

    if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal):
        return value(model.o)

    return -1


def solve_noma2(cluster_size, param_set):
    model: ConcreteModel = pyo.ConcreteModel()
    model.NOFD = pyo.Param(initialize=len(param_set['scores']))  # Number of devices
    model.NOFTF = pyo.Param(initialize=1)  # Number of time frames
    model.NOFS = pyo.Param(initialize=5)  # Number of time slots
    model.C = pyo.Param(domain=pyo.NonNegativeIntegers, initialize=cluster_size)
    # model.C = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=cluster_size)
    model.M = pyo.RangeSet(1, model.NOFD)  # Range of devices
    model.K = pyo.RangeSet(1, model.NOFTF)  # Range of time frames
    model.S = pyo.RangeSet(1, model.NOFS)  # Range of time slots

    def N_init(m):
        return ((t, j) for t, j in (range(1, m.NOFS + 1) * m.K))  # Initialization of RBs (j,t)

    model.N = pyo.Set(initialize=N_init)  # Set of RBs (j,t)

    # model.L = pyo.Param(model.M, model.K,
    #                     initialize=120)  # The length of packet in bits that device i needs to send at time t
    model.L = pyo.Param(model.M, model.K,
                        initialize={(i, int(j)): np.random.uniform(90, 150) for i, j in zip(range(1, 13), np.ones(12))})

    def A_init(m, i, j):
        return int(random.uniform(1, m.NOFS))

    model.A = pyo.Param(model.M, model.K, initialize=A_init,
                        within=pyo.NonNegativeIntegers)  # Packet arrival time of device i at time frame t

    def D_init(m, i, j):
        return int(random.uniform((m.A[i, 1]) + 1, m.NOFS + 1))

    model.D = pyo.Param(model.M, model.K, initialize=D_init)  # Packet due time of device i at time frame t
    model.E = pyo.Param(model.M, initialize=120)  # Units of energy stored in the battery of device i
    model.W = pyo.Param(model.N, initialize=round(300 / value(model.NOFS), 0))  # Bandwidth of RB (j,t)
    model.H = pyo.Param(model.M, model.N, initialize=1)  # The wireless channel between device i and BS using RB (j,t)

    model.G = pyo.Param(model.M, model.N,
                        initialize=dict(
                            zip((model.M * model.N), np.array(
                                param_set['gains'].repeat(value(model.NOFS))))))

    model.PRI = pyo.Param(model.M, initialize=dict(zip(model.M, param_set['scores'])))

    model.x = pyo.Var(model.M, model.N, domain=Binary)
    model.p = pyo.Var(model.M, model.N, domain=NonNegativeReals)
    model.z = pyo.Var(model.M, model.K, domain=Binary)

    def objective_expression(m):
        return sum(m.PRI[i] * m.z[i, t] for i, t in m.M * m.K)

    model.o = pyo.Objective(rule=objective_expression, sense=pyo.maximize)

    def constraint_1(m, i, t):
        return sum(m.W[j, t] *
                   (pyo.log(1 +
                            (m.p[i, j, t] * m.G[i, j, t]) /
                            (1 + sum(
                                m.p[i_prime, j, t] * m.G[i_prime, j, t] for i_prime in range(i + 1, len(m.M) + 1))))
                    / pyo.log(2)) for j in range(1, m.NOFS + 1)) >= m.L[i, t] * m.z[i, t]

    def constraint_2(m, i, j, t):
        return m.p[i, j, t] <= m.E[i] * m.x[i, j, t]

    def constraint_3(m, i):
        return sum(sum(m.p[i, j, t] for j in m.S) for t in m.K) <= m.E[i]

    def constraint_4(m, i, j, t):
        if j < m.A[i, t] or j > m.D[i, t]:
            return m.x[i, j, t] <= 0
        else:
            return pyo.Constraint.Skip

    # Non interrupted version could also be implemented

    def constraint_5(m, i, j, t):
        return m.x[i, j, t] <= m.z[i, t]

    def constraint_7(m, i, t):
        return m.z[i, t] <= m.L[i, t]

    model.disjunctions = Disjunction(model.N, rule=lambda model, j, t:
    [sum(model.x[_i, j, t] for _i in model.M) == model.C,
     sum(model.x[_i, j, t] for _i in model.M) == 0])

    model.cons1 = Constraint(model.M, model.K, rule=constraint_1)
    model.cons2 = Constraint(model.M, model.N, rule=constraint_2)
    model.cons3 = Constraint(model.M, rule=constraint_3)
    model.cons4 = Constraint(model.M, model.N, rule=constraint_4)
    model.cons5 = Constraint(model.M, model.N, rule=constraint_5)
    model.cons7 = Constraint(model.M, model.K, rule=constraint_7)

    model.name = "UplinkNoma"
    # TransformationFactory('gdp.hull').apply_to(model)
    TransformationFactory('gdp.bigm').apply_to(model)
    opt = pyo.SolverFactory(
        'scip')
    try:
        results = opt.solve(model, tee=False)
        if (results.solver.status == SolverStatus.ok) and (
                results.solver.termination_condition == TerminationCondition.optimal):
            return value(model.o)
        else:
            return 0
            print("Solver Status:", result.solver.status)
    except pyomo.common.errors.ApplicationError:
        return 0






