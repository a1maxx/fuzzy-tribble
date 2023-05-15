# Import necessary libraries
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

# Define the Bayesian network structure
model = BayesianModel([('A', 'B'), ('C', 'B')])

# Define the conditional probability distributions (CPDs) for each variable
cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.7], [0.3]])
cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.4], [0.6]])
cpd_b = TabularCPD(variable='B', variable_card=2,
                    evidence=['A', 'C'],
                    evidence_card=[2, 2],
                    values=[[0.9, 0.6, 0.8, 0.1],
                            [0.1, 0.4, 0.2, 0.9]])

# Add the CPDs to the model
model.add_cpds(cpd_a, cpd_c, cpd_b)

# Check if the model is valid
model.check_model()

# Predict the probability of B=1 given A=0 and C=1
from pgmpy.inference import VariableElimination

infer = VariableElimination(model)
query = infer.query(['B'], evidence={'A': 0, 'C': 1})
print(query)


# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

