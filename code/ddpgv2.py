import random

import keras
from keras import layers, Model
import tensorflow as tf
import numpy as np
from keras.src.layers import ReLU, Add


def res_block(x, trainable):
    # Shortcut connection (skip connection)
    x_shortcut = x

    # First convolution layer
    conv1 = keras.layers.Conv2D(32, (1, 1), padding='same',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.15),
                                   trainable=trainable)(x)
    norm1 = keras.layers.BatchNormalization()(conv1)
    x = ReLU()(norm1)

    # Second convolution layer
    conv2 = keras.layers.Conv2D(32, (1, 1), padding='same',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.15),
                                   trainable=trainable)(x)
    norm2 = keras.layers.BatchNormalization()(conv2)

    # Add the shortcut (skip connection)
    x = Add()([norm2, x_shortcut])

    # Apply activation function (ReLU)
    x = ReLU()(norm1)

    return x


def build_predictor(inputs, predictor, nes_num, scope, trainable):
    with tf.name_scope(scope):
        if predictor == 'CNN':
            # Get the shape of the input tensor
            L = int(inputs.shape[2])
            N = int(inputs.shape[3])

            # Corrected Conv2D layer
            conv1 = keras.layers.Conv2D(32, (L, N), strides=(1, 1), padding='valid',
                                        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.15),
                                        trainable=trainable)(inputs)
            norm1 = keras.layers.BatchNormalization()(conv1)
            x = ReLU()(norm1)

            # Add residual blocks (assuming `res_block` is defined)
            for _ in range(nes_num):
                x = res_block(x, trainable)

            # Corrected final Conv2D layer
            conv3 = keras.layers.Conv2D(32, (1, 1), strides=(1, 1), padding='valid',
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.15),
                                        trainable=trainable)(x)
            norm3 = keras.layers.BatchNormalization()(conv3)
            net = ReLU()(norm3)

            # Flatten the output
            net = keras.layers.Flatten()(net)

            return net


def variables_summaries(var, name):
    # Compute mean and standard deviation
    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

    # Log the mean, stddev, and histogram to TensorBoard
    with tf.name_scope(name):
        tf.summary.scalar(f'{name}_mean', mean)
        tf.summary.scalar(f'{name}_stddev', stddev)
        tf.summary.histogram(f'{name}_histogram', var)



class StockActor(keras.Model):
    def __init__(self, predictor, M, L, N, batch_size):
        super(StockActor, self).__init__()

        # Initial hyperparameters
        self.tau = 1e-3
        self.learning_rate = 1e-2
        self.gamma = 0.99
        self.batch_size = batch_size

        # Input shapes
        self.M = M
        self.L = L
        self.N = N

        self.actor = self.build_actor(predictor, "actor", trainable=True)
        self.target_actor = self.build_actor(predictor, "actor", trainable=False)

        # Debug: Print number of weights
        print(f"Online Critic Weights: {len(self.actor.get_weights())}")
        print(f"Target Critic Weights: {len(self.target_actor.get_weights())}")

        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

    def build_actor(self, predictor, scope, trainable):
        # Input layer using keras.Input
        inputs = keras.Input(shape=(self.M, self.L, self.N), name='input')

        # Build predictor (assuming it's a custom function)
        x = build_predictor(inputs, predictor, 5, scope, trainable=trainable)

        # First fully connected layer
        t1_w = keras.layers.Dense(64, activation=None, use_bias=True,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.15))(x)
        net = keras.layers.BatchNormalization()(t1_w)
        net = ReLU()(net)

        # Dropout layer
        net = keras.layers.Dropout(0.5)(net)

        # Final output layer
        out = keras.layers.Dense(self.M, activation='softmax',
                                    kernel_initializer=tf.random_uniform_initializer(-0.003, 0.003))(net)

        # Create and return the model
        model = keras.Model(inputs=inputs, outputs=out, name=scope)
        return model

    def train(self, inputs, a_gradient):
        with tf.GradientTape() as tape:
            # Fix shape issue
            inputs = tf.squeeze(inputs, axis=1)  # ✅ Remove extra dimension
            # Get actor output (predicted actions)
            action_pred = self.actor(inputs)  # ✅ Use self.actor instead of self.out

            # Compute the loss and gradients
            loss = self.compute_actor_loss(a_gradient, action_pred)

        # Compute gradients and apply them
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    def compute_actor_loss(self, a_gradient, action_pred):
        # Compute the loss using the gradients and predictions
        return -tf.reduce_mean(a_gradient * action_pred)  # Example loss function (adjust based on your needs)

    def pre_train(self, s, a):
        # Compute pre-training loss and apply the gradients
        with tf.GradientTape() as tape:
            pre_loss = self.pre_loss(s, a)  # Call pre_loss method to calculate pre-training loss

        grads = tape.gradient(pre_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return pre_loss

    def pre_loss(self, s, a):
        # Calculate pre-training loss (mean squared error)
        pred = self.out(s)
        return tf.reduce_sum(tf.square(pred - a))

    def predict(self, inputs):
        return self.actor(inputs)  # ✅ self.actor is a Keras model, so it's callable

    def predict_target(self, inputs):
        return self.target_actor(inputs)  # ✅ Same fix

    def update_target_network(self):
        # Update target network using soft update (tau scaling)
        for target_var, var in zip(self.target_variables, self.trainable_variables):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)

class StockCritic(keras.Model):
    def __init__(self, predictor, M, L, N):
        super(StockCritic, self).__init__()

        # Initial hyperparameters
        self.tau = 1e-2
        self.learning_rate = 1e-3
        self.gamma = 0.99

        # Input shapes
        self.M = M
        self.L = L
        self.N = N

        # Ensure both critic networks have identical architectures
        self.online_critic = self.build_critic(predictor, "online_critic")
        self.target_critic = self.build_critic(predictor, "target_critic")

        # Debug: Print number of weights
        print(f"Online Critic Weights: {len(self.online_critic.get_weights())}")
        print(f"Target Critic Weights: {len(self.target_critic.get_weights())}")

        # Ensure target critic has the same initial weights as the online critic
        self.target_critic.set_weights(self.online_critic.get_weights())  # ✅ Force identical weights

        # Optimizer
        self.optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)

        # Extra layers (separate from the critic models)
        self.dense1 = keras.layers.Dense(64, activation='relu')
        self.dense2 = keras.layers.Dense(32, activation='relu')
        self.out = keras.layers.Dense(1)  # This is the output layer for Q-value
        def predict(self, inputs, actions):
            # Prediction for given inputs and actions
            return self.online_critic([inputs, actions])

        def predict_target(self, inputs, actions):
            # Prediction for target network
            return self.target_critic([inputs, actions])

        def update_target_network(self):
            # Soft update for target network
            for target_var, online_var in zip(self.target_critic.variables, self.online_critic.variables):
                target_var.assign(self.tau * online_var + (1 - self.tau) * target_var)

        def action_gradients(self, inputs, actions):
            # Use GradientTape to calculate action gradients
            with tf.GradientTape() as tape:
                tape.watch(actions)
                q_values = self.online_critic([inputs, actions], training=True)
            action_grads = tape.gradient(q_values, actions)
            return action_grads

    def build_critic(self, predictor, scope):
        # Define input layers
        states = keras.Input(shape=(self.M, self.L, self.N), name="states")
        actions = keras.Input(shape=(self.M,), name="actions")

        # Build predictor network
        net = build_predictor(states, predictor, 5, scope, trainable)

        # Define dense layers
        t1 = layers.Dense(64, activation=None, kernel_initializer=tf.random_normal_initializer(0., 0.15))(net)
        t2 = layers.Dense(64, activation=None, kernel_initializer=tf.random_normal_initializer(0., 0.15))(actions)

        # Merge state and action paths
        net = layers.Add()([t1, t2])
        net = layers.BatchNormalization()(net)
        net = layers.ReLU()(net)
        net = layers.Dropout(0.5)(net)

        # More dense layers
        net = layers.Dense(64, activation=None, kernel_initializer=tf.random_normal_initializer(0., 0.15))(net)
        net = layers.ReLU()(net)
        net = layers.Dropout(0.5)(net)

        # Output layer
        out = layers.Dense(1, kernel_initializer=tf.random_uniform_initializer(-0.003, 0.003))(net)

        # Create the Keras model
        critic_model = keras.Model(inputs=[states, actions], outputs=out, name=scope)

        return critic_model  # ✅ Return the actual model, not (states, actions, out)

    def train(self, inputs, actions, targets):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        targets = tf.convert_to_tensor(targets, dtype=tf.float32)

        # ✅ Fix: Ensure `inputs` and `targets` have the correct batch size
        inputs = tf.squeeze(inputs, axis=1)  # Remove the extra dim at axis=1
        targets = tf.reshape(targets, [-1, 1])  # Ensure targets is a 2D tensor with shape (batch_size, 1)

        with tf.GradientTape() as tape:
            tape.watch(inputs)
            predicted_q = self.online_critic([inputs, actions], training=True)  # ✅ Use self.online_critic

            # ✅ Fix: Slice `targets` to match `predicted_q`'s batch size
            targets = targets[:tf.shape(predicted_q)[0]]
            # Compute loss
            loss = keras.losses.MeanSquaredError()(targets, predicted_q)

        # Compute gradients and apply optimizer
        gradients = tape.gradient(loss, self.online_critic.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.online_critic.trainable_variables))

        return loss, tf.reduce_mean(predicted_q)

    def predict(self, inputs, actions):
        # Ensure actions has the shape [64, 5] -> [64, 5, 1]
        actions = tf.expand_dims(actions, axis=-1)  # [64, 5] -> [64, 5, 1]

        # Repeat the actions to match the spatial dimensions of inputs: [64, 5, 5, 1]
        actions = tf.repeat(actions, repeats=5, axis=2)  # [64, 5, 1] -> [64, 5, 5, 1]
        # **Fix:** Expand actions to add a channel dimension -> [64, 5, 5, 1] -> [64, 5, 5, 1]
        actions = tf.expand_dims(actions, axis=-1)
        # Cast both inputs and actions to float32 before concatenation
        inputs = tf.cast(inputs, tf.float32)  # Convert inputs to float32
        actions = tf.cast(actions, tf.float32)
        # Now, actions has the shape [64, 5, 5, 1], and we can concatenate it with inputs
        # inputs has the shape [64, 5, 5, 4], so the concatenated shape will be [64, 5, 5, 5]
        x = tf.concat([inputs, actions], axis=-1)

        # Pass through the neural network layers
        x = self.dense1(x)
        x = self.dense2(x)
        return self.out(x)

    def predict_target(self, inputs, actions):
        return self.target_out(inputs, actions)

    def update_target_network(self):
        for target_param, param in zip(self.target_network.trainable_variables,
                                       self.online_network.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)

    def action_gradients(self, inputs, actions):
        # Ensure inputs and actions are TensorFlow tensors
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        print("Inputs shape:", len(inputs))
        inputs = tf.squeeze(inputs, axis=1)
        # if(inputs.)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(actions)
            q_values = self.online_critic([inputs, actions], training=True)  # ✅ Use self.online_critic
        return tape.gradient(q_values, actions)


def build_summaries():
    critic_loss = tf.Variable(0.0, dtype=tf.float32)
    reward = tf.Variable(0.0, dtype=tf.float32)
    ep_ave_max_q = tf.Variable(0.0, dtype=tf.float32)
    actor_loss = tf.Variable(0.0, dtype=tf.float32)

    summary_vars = [critic_loss, reward, ep_ave_max_q, actor_loss]

    return summary_vars


class DDPG:

    def __init__(self, predictor, M, L, N, name, load_weights, trainable):
        # Experience replay buffer
        self.buffer = []
        self.buffer_size = 50000  # Increased for stability
        self.batch_size = 64  # Increased batch size for better training

        self.name = name
        self.gamma = 0.99
        self.tau = 0.005  # Lower tau for smoother target updates

        # Define Actor and Critic models
        self.actor = StockActor(predictor, M, L, N, self.batch_size)
        self.critic = StockCritic(predictor, M, L, N)

        # Define target networks
        self.target_actor = StockActor(predictor, M, L, N, self.batch_size)
        self.target_critic = StockCritic(predictor, M, L, N)

        # Copy initial weights to target networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        # Checkpointing
        self.checkpoint = tf.train.Checkpoint(actor=self.actor, critic=self.critic)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.name, max_to_keep=10)

        if load_weights:
            print("Loading Model")
            if self.checkpoint_manager.latest_checkpoint:
                self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
                print("Successfully loaded:", self.checkpoint_manager.latest_checkpoint)
            else:
                print("Could not find old network weights")
        else:
            print("Initializing new model weights")

        if trainable:
            self.summary_writer = tf.summary.create_file_writer("../summary/DDPG")
            print("Done")

    def save_transition(self, s, w, r, not_terminal, s_next, action_precise):
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((s, w[0], r, not_terminal, s_next, action_precise))

    def train(self, method, epoch):
        if len(self.buffer) < self.batch_size:
            return {"critic_loss": 0, "q_value": 0, "actor_loss": 0}

        # Sample minibatch from buffer
        s, a, r, not_terminal, s_next, a_precise = self.get_transition_batch()

        # Compute target Q-value
        s_next = tf.squeeze(s_next, axis=1)
        target_q = self.target_critic.predict(s_next, self.target_actor.predict(s_next))
        y_i = np.array([r[i] + not_terminal[i] * self.gamma * target_q[i] for i in range(self.batch_size)])

        # Train critic
        critic_loss, q_value = self.critic.train(s, a, tf.reshape(y_i, (-1, 1)))
        actor_loss = 0
        # Train actor
        if method == 'model_free':
            a_outs =  self.actor.predict(tf.squeeze(s, axis=1))
            grads = self.critic.action_gradients(s, a_outs)
            self.actor.train(s, grads)

        elif method == 'model_based':
            if epoch <= 100:
                actor_loss = self.actor.pre_train(s, a_precise)
            else:
                a_outs = self.actor.predict(s)
                grads = self.critic.action_gradients(s, a_outs)
                self.actor.train(s, grads)
        else:
            actor_loss = 0

        # Soft update target networks
        self.update_target_network(self.target_actor, self.target_critic)

        return {"critic_loss": critic_loss.numpy(), "q_value": np.amax(q_value), "actor_loss": actor_loss}

    def update_target_network(self, target_actor, target_critic):
        # Soft update target networks using tau
        # new_actor_weights = [
        #     self.tau * aw + (1 - self.tau) * tw
        #     for aw, tw in zip(self.actor.get_weights(), self.target_actor.get_weights())
        # ]
        # new_critic_weights = [
        #     self.tau * cw + (1 - self.tau) * tw
        #     for cw, tw in zip(self.critic.get_weights(), self.target_critic.get_weights())
        # ]

        self.target_actor.set_weights(target_actor.get_weights())
        self.target_critic.set_weights(target_critic.get_weights())

    def get_transition_batch(self):
        minibatch = random.sample(self.buffer, self.batch_size)
        s = np.array([data[0] for data in minibatch])
        a = np.array([data[1] for data in minibatch])
        r = np.array([data[2] for data in minibatch])
        not_terminal = np.array([data[3] for data in minibatch])
        s_next = np.array([data[4] for data in minibatch])
        action_precise = np.array([data[5] for data in minibatch])

        return s, a, r, not_terminal, s_next, action_precise

    def save_model(self, epoch):
        save_path = self.checkpoint_manager.save(checkpoint_number=epoch)
        print(f"Model saved at {save_path}")
    # def __init__(self, predictor, M, L, N, name, load_weights, trainable):
    #     # Initial buffer
    #     self.buffer = []
    #     self.buffer_size = 10000
    #     self.batch_size = 32
    #     self.name = name
    #
    #     # Set up models
    #     self.actor = StockActor(predictor, M, L, N, self.batch_size)
    #     self.critic = StockCritic(predictor, M, L, N)
    #
    #     # Initial Hyperparameters
    #     self.gamma = 0.99
    #
    #     # Initialize checkpoint manager
    #     self.checkpoint = tf.train.Checkpoint(actor=self.actor, critic=self.critic)
    #     self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.name, max_to_keep=10)
    #
    #     if load_weights == 'True':
    #         print("Loading Model")
    #         if self.checkpoint_manager.latest_checkpoint:
    #             self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
    #             print("Successfully loaded:", self.checkpoint_manager.latest_checkpoint)
    #         else:
    #             print("Could not find old network weights")
    #     else:
    #         print("Initializing new model weights")
    #
    #     if trainable:
    #         # Initial summary
    #         self.summary_writer = tf.summary.create_file_writer('./summary/DDPG')

    def log_summaries(self, step, summaries):
        with self.summary_writer.as_default():
            for name, value in summaries.items():
                tf.summary.scalar(name, value, step=step)
            self.summary_writer.flush()

    #online actor
    def predict(self,s):
        return self.actor.predict(s)

    #target actor
    def test_predict(self,s):
        return self.actor.predict_target(s)

    # def save_transition(self,s,w,r,not_terminal,s_next,action_precise):
    #     if len(self.buffer)>self.buffer_size:
    #         self.buffer.pop(0)
    #     self.buffer.append((s,w[0],r,not_terminal,s_next,action_precise))


    # def train(self, method, epoch):
    #     info = dict()
    #
    #     # Ensure the buffer has enough data before training
    #     if len(self.buffer) < self.buffer_size:
    #         info["critic_loss"], info["q_value"], info["actor_loss"] = 0, 0, 0
    #         return info
    #
    #     # Sample batch from buffer
    #     s, a, r, not_terminal, s_next, a_precise = self.get_transition_batch()
    #
    #     # Compute target Q-values
    #     target_q = self.critic.predict_target(s_next, self.actor.predict_target(s_next))
    #     y_i = tf.convert_to_tensor([r[i] + not_terminal[i] * self.gamma * target_q[i] for i in range(self.batch_size)],
    #                                dtype=tf.float32)
    #
    #     # Train critic
    #     critic_loss, q_value = self.critic.train(s, a, tf.reshape(y_i, (-1, 1)))
    #     info["critic_loss"] = critic_loss
    #     info["q_value"] = np.amax(q_value)
    #
    #     # Train actor
    #     if method == 'model_free':
    #         a_outs = self.actor.predict(s)
    #         grads = self.critic.action_gradients(s, a_outs)
    #         self.actor.train(s, grads[0])  # Applying gradients
    #
    #     elif method == 'model_based':
    #         if epoch <= 100:
    #             actor_loss = self.actor.pre_train(s, a_precise)
    #             info["actor_loss"] = actor_loss
    #         else:
    #             a_outs = self.actor.predict(s)
    #             grads = self.critic.action_gradients(s, a_outs)
    #             self.actor.train(s, grads[0])  # Applying gradients
    #
    #     # Update target networks
    #     self.actor.update_target_network()
    #     self.critic.update_target_network()
    #
    #     return info

    # def save_model(self, epoch):
    #     save_path = self.checkpoint_manager.save(checkpoint_number=epoch)
    #     print(f"Model saved at {save_path}")
    #
    # def get_transition_batch(self):
    #     minibatch =random.sample(self.buffer, self.batch_size)
    #     s = [data[0][0] for data in minibatch]
    #     a = [data[1] for data in minibatch]
    #     r = [data[2] for data in minibatch]
    #     not_terminal = [data[3] for data in minibatch]
    #     s_next = [data[4][0] for data in minibatch]
    #     action_precise=[data[5][0] for data in minibatch]
    #     return s, a, r, not_terminal, s_next,action_precise

    def write_summary(self, Loss, reward, ep_ave_max_q, actor_loss, epoch):
        with self.summary_writer.as_default():
            tf.summary.scalar("Loss", Loss, step=epoch)
            tf.summary.scalar("Reward", reward, step=epoch)
            tf.summary.scalar("Ep_Ave_Max_Q", ep_ave_max_q, step=epoch)
            tf.summary.scalar("Actor_Loss", actor_loss, step=epoch)
            self.summary_writer.flush()


# predictor = "CNN"  # Type of predictor (Assuming CNN is implemented)
# M = 10  # Number of assets or stocks (Adjust based on your use case)
# L = 5   # Length of input time series (Adjust based on data)
# N = 3   # Number of features per asset (Adjust based on data)
# name = "DDPG"  # Name for model saving
# load_weights = False  # Set to True if you have saved weights to load
# trainable = True
# ddpg_agent = DDPG(predictor, M, L, N, name, load_weights, trainable)

predictor = "CNN"  # Type of predictor (Assuming CNN is implemented)
M = 4  # Number of assets or stocks (Adjust based on your use case)
L = 5   # Length of input time series (Adjust based on data)
N = 4   # Number of features per asset (Adjust based on data)
name = "DDPG"  # Name for model saving
load_weights = False  # Set to True if you have saved weights to load
trainable = True
ddpg_agent = DDPG(predictor, M, L, N, name, load_weights, trainable)
