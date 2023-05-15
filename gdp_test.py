import pyomo.environ as pyo
from pyomo.environ import Var, ConcreteModel, Binary
from pyomo.environ import Objective, Constraint, SolverFactory, maximize, TransformationFactory, value, RangeSet
from pyomo.environ import LogicalConstraint, Reference
from pyomo.gdp import Disjunct, Disjunction
import random

model = pyo.ConcreteModel()
model.I = pyo.RangeSet(5)
model.J = pyo.RangeSet(20)
model.x = Var(model.I, model.J, domain=Binary)


def A_init(m, i):
    return int(random.uniform(1, len(model.J)))


model.A = pyo.Param(model.I, initialize=A_init,
                    within=pyo.NonNegativeIntegers)


def D_init(m, i):
    return int(random.uniform((m.A[i]) + 1, len(model.J) + 1))


model.D = pyo.Param(model.R, initialize=D_init)

model.d = Disjunct(model.J)


model.x = Var(model.R, domain=Binary)
model.y = Var(model.R, domain=Binary)


def obj_expression(model):
    return model.x + model.y


model.OBJ = Objective(rule=obj_expression, sense=maximize)

model.fix_x = Disjunct()
model.fix_x.c1 = Constraint(expr=model.x == 10)

model.fix_y = Disjunct()
model.fix_y.c1 = Constraint(expr=model.y == 10)

model.d = Disjunction(expr=[model.fix_x, model.fix_y])

TransformationFactory('gdp.hull').apply_to(model)
results = SolverFactory('scip').solve(model, tee=True)
model.display()

model.x.pprint()
model.y.pprint()
value(model.OBJ)

# model.disjunctions = Disjunction(model.DISJUNCTIONS, rule=lambda model, j, k, m:
# [model.start[j, m] + model.dur[j, m] <= model.start[k, m],
#  model.start[k, m] + model.dur[k, m] <= model.start[j, m]])
#
# if ('Task_D' in TASKS.keys()):
#     model.dummy = Constraint(rule=lambda model: model.start['Task_D', 'Server_1'] == 15)
#     model.dummy2 = Constraint(rule=lambda model: model.start['Task_D2', 'Server_3'] == 4)

m = ConcreteModel()
m.s = RangeSet(4)
m.ds = RangeSet(2)
m.d = Disjunct(m.s)
m.djn = Disjunction(m.ds)
m.djn[1] = [m.d[1], m.d[2]]
m.djn[2] = [m.d[3], m.d[4]]
m.x = Var(bounds=(-2, 10))
m.d[1].c = Constraint(expr=m.x >= 2)
m.d[2].c = Constraint(expr=m.x >= 3)
m.d[3].c = Constraint(expr=m.x <= 8)
m.d[4].c = Constraint(expr=m.x == 2.5)
m.o = Objective(expr=m.x)
m.p = LogicalConstraint(
    expr=m.d[1].indicator_var.implies(m.d[4].indicator_var))

TransformationFactory('gdp.bigm').apply_to(m)

Reference(m.d[:].indicator_var).display()

run_data = SolverFactory('glpk').solve(m)
Reference(m.d[:].indicator_var).display()
