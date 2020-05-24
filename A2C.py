# Import: ----------------------
# Keras backend for sum and log:
# Keras imports:
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model, model_from_json
from keras.optimizers import Adam
import numpy as np


class Agent(object):
    # Lunar lander has 4 possible actions
    # The actor and the critic have diff. learning rates (alpha, beta)
    # Gamma is the coefficient for future rewards
    # Lunar lander has observation vector -> no need for CNN then
    def __init__(self, alpha, beta, gamma=0.99, num_actions=4, hid1_size=1024,
                 hid2_size=720, hid3_size=512, hid4_size=256, input_dims=8):
        # Saving the variables in class
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.n_actions = num_actions
        # Network layers.:
        self.fc1_dims = hid1_size
        self.fc2_dims = hid2_size
        self.fc3_dims = hid3_size
        self.fc4_dims = hid4_size
        # Networks:
        self.actor, self.critic, self.policy = self.build_a2c()
        # List for action space
        self.action_space = [i for i in range(self.n_actions)]

    def build_a2c(self):
        # Shape indicates that batch size will be defined
        input = Input(shape=(self.input_dims,))
        delta = Input(shape=[1])
        # Shared part of a2c:
        dense1 = Dense(self.fc1_dims, activation='relu')(input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        # dense3 = Dense(self.fc3_dims, activation='relu')(dense2)
        # dense4 = Dense(self.fc4_dims, activation='relu')(dense3)
        # Different heads of the network for actor / critic:
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        # Defining the custom loss function:
        # y_true is the action the agent took, and the pred is the outp. of a2c:
        def custom_loss(y_true, y_pred):
            # Must not be 0 or 1 as we take log
            out = K.clip(y_pred, 1e-8, 1 - 1e-8)
            # y_pred is in one-hot
            log_lik = y_true * K.log(out)

            return K.sum(-log_lik * delta)
        # Actor:
        actor = Model(input=[input, delta], output=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)
        # Critic:
        critic = Model(input=[input], output=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')

        policy = Model(input=[input], output=[probs])

        return actor, critic, policy

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def learn(self, state, action, reward, state_, done):

        state = state[np.newaxis, :]
        state_ = state_[np.newaxis, :]

        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)

        target = reward + self.gamma * critic_value_ * (1 - int(done))
        delta = target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1.0
        self.actor.fit([state, delta], actions, verbose=0)
        self.critic.fit(state, target, verbose=0)

    def save_weights(self):
        self.actor.save_weights(filepath='actor_more1000.h5')
        self.critic.save_weights(filepath='critic_more1000.h5')
        print("Saved weights to h5 file")

    def load_weights(self, filename1, filename2):
        self.actor.load_weights(filename1)
        self.critic.load_weights(filename2)
        print("Loaded weights from h5 file")
