import numpy
import pandas as pd
import cmath
import pyomo.environ as pyo
import math
import logging
from pyomo.environ import value, NonNegativeReals, TransformationFactory, Binary, Constraint
import numpy as np
import random

random.seed(10)

## Parameters

model = pyo.ConcreteModel()
model.NOFD = pyo.Param(initialize=4)  ## Number of devices
model.NOFTF = pyo.Param(initialize=1)  ## Number of time frames
model.NOFS = pyo.Param(initialize=5)  ## Number of time slots
model.C = pyo.Param(initialize=2)  ## Cluster size
model.M = pyo.RangeSet(1, model.NOFD)  ## Range of devices
model.K = pyo.RangeSet(1, model.NOFTF)  ## Range of time frames
model.S = pyo.RangeSet(1, model.NOFS)  ## Range of time slots


def N_init(m):
    return ((t, j) for t, j in (range(1, m.NOFS + 1) * m.K))  ## Initialization of RBs (j,t)

model.N = pyo.Set(initialize=N_init)  ## Set of RBs (j,t)

model.L = pyo.Param(model.M, model.K, initialize=120)  ## The length of packet in bits that device i needs to send


# at time t
def A_init(m, i, j):
    return int(random.uniform(1, m.NOFS))


model.A = pyo.Param(model.M, model.K, initialize=A_init,
                    within=pyo.NonNegativeIntegers)  ## Packet arrival time of device i at time frame t


def D_init(m, i, j):
    return int(random.uniform((m.A[i, 1]) + 1, m.NOFS + 1))


model.D = pyo.Param(model.M, model.K, initialize=D_init)  ## Packet due time of device i at time frame t
model.E = pyo.Param(model.M, initialize=120)  ## Units of energy stored in the battery of device i
model.W = pyo.Param(model.N, initialize=round(300 / model.NOFS, 0))  ## Bandwidth of RB (j,t)
model.H = pyo.Param(model.M, model.N, initialize=1)  ## The wireless channel between device i and BS using RB (j,t)
# model.G = pyo.Param(model.M, initialize=dict(zip(model.M, [40, 32, 24, 16])))  ## Initial gains of UEs
model.G = pyo.Param(model.M, model.N,
                    initialize=dict(
                        zip((model.M * model.N), np.array([0.2, 0.1, 0.05, 0.01]).repeat(value(model.NOFS)))))
model.PRI = pyo.Param(model.M, initialize=dict(zip(model.M, [0.2, 0.8, 0.9, 0.9])))

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
    return sum(m.x[i, j, t] for i in m.M) <= m.C


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
# opt.options['acceptable_tol'] = 1e-2
# opt.options['max_iter'] = 100000

results = opt.solve(model, tee=True)

# %%

model.nSensors = pyo.Param(initialize=100)
model.rSensors = pyo.RangeSet(1, model.nSensors)
sensor_x = np.array([42., 88., 197., 144., 163., 179., 4., 109., 142., 180., 20.,
                     17., 131., 180., 129., 90., 135., 46., 65., 67., 169., 17.,
                     31., 151., 139., 190., 133., 51., 172., 111., 104., 189., 33.,
                     15., 150., 166., 43., 183., 69., 140., 187., 179., 6., 36.,
                     152., 136., 132., 81., 98., 199., 54., 104., 189., 1., 80.,
                     48., 97., 90., 170., 16., 119., 41., 105., 46., 45., 96.,
                     58., 123., 49., 121., 112., 101., 182., 21., 68., 135., 74.,
                     78., 137., 145., 90., 47., 169., 151., 185., 71., 81., 54.,
                     139., 137., 155., 179., 92., 162., 183., 63., 61., 148., 124.,
                     6.])
sensor_y = np.array([29., 94., 130., 109., 4., 140., 182., 107., 83., 72., 154.,
                     151., 81., 159., 183., 123., 136., 188., 113., 39., 196., 63.,
                     46., 33., 33., 162., 143., 7., 23., 21., 178., 176., 129.,
                     198., 117., 116., 56., 176., 199., 194., 24., 24., 177., 70.,
                     18., 72., 25., 131., 198., 3., 114., 83., 166., 56., 99.,
                     155., 14., 180., 124., 19., 163., 125., 17., 96., 177., 25.,
                     92., 131., 88., 137., 37., 75., 3., 186., 132., 25., 164.,
                     13., 28., 125., 43., 83., 78., 174., 76., 170., 15., 21.,
                     140., 30., 162., 151., 192., 185., 53., 159., 69., 181., 39.,
                     46.])
sensor_d = np.round(np.sqrt(np.add(sensor_x ** 2, sensor_y ** 2)), 0).astype(int)
10/sensor_d
model.sensorD = pyo.Param(model.rSensors, initialize=dict(enumerate(sensor_d, 1)))
np.add(np.array([1, 1]), np.array([2, 2]))

# %%
for parmobject in model.component_objects(pyo.Var, active=True):
    nametoprint = str(str(parmobject.name))
    print("Variable ", nametoprint)
    for index in parmobject:
        vtoprint = pyo.value(parmobject[index])
        print("   ", index, vtoprint)

n = 1
z = np.random.normal(loc=0, scale=1 / 2, size=(n, 2)).view(np.complex128)  ## CN(0,1) h_(km)


def squareNorm(c: np.complex128):
    return (c.real ** 2 + c.imag ** 2)


alpha = 1.5
100 ** -alpha

#%%

transition_frequencies = np.zeros(shape=(6,6))
transition_frequencies[0, 1] = 1418
transition_frequencies[1, 0] = 1798
transition_frequencies[3, 4] = 1633
transition_frequencies[4, 3] = 2625
transition_frequencies[1, 2] = 2199
transition_frequencies[2, 1] = 2031
transition_frequencies[4, 5] = 1268
transition_frequencies[5, 4] = 2191
transition_frequencies[2, 3] = 1835
transition_frequencies[3, 2] = 2249

n_10s = 52560
s_obs = np.sum(transition_frequencies)

for i,j in enumerate(np.arange(0,6),0):
    transition_frequencies[i,j] = int((n_10s - s_obs) / 6)

transition_probs = np.round(transition_frequencies / np.sum(transition_frequencies),2)


np.sum(transition_frequencies)






