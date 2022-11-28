from tensorflow import keras
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def build_model(states, actions):
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, activation='relu', input_shape=states))
    model.add(keras.layers.Dense(24, activation='relu'))
    model.add(keras.layers.Dense(actions, activation='linear'))
    return model


def build_model2(states, actions):
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, activation='relu', input_shape=states))
    model.add(keras.layers.Dense(24, activation='relu'))
    model.add(keras.layers.Dense(24, activation='relu'))
    model.add(keras.layers.Dense(actions, activation='linear'))
    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn


def flatten(actions):
    # This function flattens any actions passed somewhat like so -:
    # INPUT -: [[1, 2, 3], 4, 5]
    # OUTPUT -: [1, 2, 3, 4, 5]

    new_actions = []  # Initializing the new flattened list of actions.
    for action in actions:
        # Loop through the actions
        if type(action) == list:
            # If any actions is a pair of actions i.e. a list e.g. [1, 1] then
            # add it's elements to the new_actions list.
            new_actions += action
        elif type(action) == int:
            # If the action is an integer then append it directly to the new_actions
            # list.
            new_actions.append(action)

    # Returns the new_actions list generated.
    return new_actions


def get_actions(possible_actions):
    # This functions recieves as input the possibilities of actions for every dimension
    # and returns all possible dimensional combinations for the same.
    # Like so -:
    # INPUT-: [[1, 2, 3, 4], [1, 2, 3, 4]] # Example for 2 dimensions but can be scaled for any.
    # OUTPUT-: [[1, 1], [1, 2], [1, 3] ... [4, 1] ... [4, 4]]
    if len(possible_actions) == 1:
        # If there is only one possible list of actions then it itself is the
        # list containing all possible combinations and thus is returned.
        return possible_actions
    pairs = []  # Initializing a list to contain all pairs of actions generated.
    for action in possible_actions[0]:
        # Now we loop over the first set of possibilities of actions i.e. index 0
        # and we make pairs of it with the second set i.e. index 1, appending each pair
        # to the pairs list.
        # NOTE: Incase the function is recursively called the first set of possibilities
        # of actions may contain vectors and thus the newly formed pair has to be flattened.
        # i.e. If a pair has already been made in previous generation like so -:
        # [[[1, 1], [2, 2], [3, 3] ... ], [1, 2, 3, 4]]
        # Then the pair formed will be this -: [[[1, 1], 1], [[1, 1], 2] ... ]
        # But we want them to be flattened like so -: [[1, 1, 1], [1, 1, 2] ... ]
        for action2 in possible_actions[1]:
            pairs.append(flatten([action, action2]))

    # Now we create a new list of all possible set of actions by combining the
    # newly generated pairs and the sets of possibilities of actions that have not
    # been paired i.e. sets other than the first and the second.
    # NOTE: When we made pairs we did so only for the first two indexes and not for
    # all thus to do so we make a new list with the sets that remained unpaired
    # and the paired set. i.e.
    # BEFORE PAIRING -: [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    # AFTER PAIRING -: [[[1, 1], [1, 2] ... ], [1, 2, 3, 4]] # Notice how the third set
    # i.e. the index 2 is still unpaired and first two sets have been paired.
    new_possible_actions = [pairs] + possible_actions[2:]
    # Now we recurse the function and call it within itself to make pairs for the
    # left out sets, Note that since the first two sets were combined to form a paired
    # first set now this set will be paired with the third set.
    # This recursion will keep happening until all the sets have been paired to form
    # a single set with all possible combinations.
    possible_action_vectors = get_actions(new_possible_actions)
    # Finally the result of the recursion is returned.
    # NOTE: Only the first index is returned since now the first index contains the
    # paired set of actions.
    return possible_action_vectors[0]