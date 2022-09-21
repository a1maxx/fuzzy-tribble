from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from keras.optimizers import Adam


def build_model(states, actions):
    model = Sequential()
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model
