import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, BatchNormalization
from keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import SGD
import numpy as np


class PPO(tf.keras.Model):
    def __init__(self, M, L, N, name, load_weights=False, trainable=True):
        super(PPO, self).__init__()
        self.M, self.L, self.N = M, L, N
        self.gamma = 0.99
        self.name = name

        # Critic network
        self.critic_conv = Conv2D(2, (1, 3), padding='same', activation='relu', trainable=trainable)
        self.critic_bn = BatchNormalization(trainable=trainable)
        self.critic_flatten = Flatten()
        self.critic_dense = Dense(1, activation=None, trainable=trainable)

        # Actor network
        self.actor_conv = Conv2D(2, (1, 3), padding='same', activation='relu', trainable=trainable)
        self.actor_bn = BatchNormalization(trainable=trainable)
        self.actor_flatten = Flatten()
        self.actor_mu = Dense(M, activation='tanh', trainable=trainable)
        self.actor_sigma = Dense(M, activation='softplus', trainable=trainable)

        # Optimizers with learning rate decay
        self.global_step = tf.Variable(0, trainable=False)
        self.critic_lr = ExponentialDecay(initial_learning_rate=1e-3, decay_steps=2000, decay_rate=0.9)
        self.actor_lr = ExponentialDecay(initial_learning_rate=1e-3, decay_steps=2000, decay_rate=0.9)

        self.optimizer_critic = SGD(learning_rate=self.critic_lr)
        self.optimizer_actor = SGD(learning_rate=self.actor_lr)

        # Buffer for training data
        self.buffer = []

        # Load weights if specified
        if load_weights:
            self.load_model()

    def call(self, state):
        # Critic Forward Pass
        x_c = self.critic_conv(state)
        x_c = self.critic_bn(x_c)
        x_c = self.critic_flatten(x_c)
        v = self.critic_dense(x_c)

        # Actor Forward Pass
        x_a = self.actor_conv(state)
        x_a = self.actor_bn(x_a)
        x_a = self.actor_flatten(x_a)
        mu = self.actor_mu(x_a)
        sigma = self.actor_sigma(x_a)
        return v, mu, sigma

    def sample_action(self, state):
        _, mu, sigma = self(state)
        sigma = tf.clip_by_value(sigma, 1e-2, 1.0)
        norm_dist = tf.random.normal(shape=mu.shape, mean=mu, stddev=sigma)
        return tf.nn.softmax(norm_dist, axis=-1)

    @tf.function
    def train_step(self, states, actions, rewards):
        with tf.GradientTape() as tape_critic:
            v, _, _ = self(states)
            advantage = rewards - v
            critic_loss = tf.reduce_mean(tf.square(advantage))

        critic_grads = tape_critic.gradient(critic_loss, self.critic_variables())
        self.optimizer_critic.apply_gradients(zip(critic_grads, self.critic_variables()))

        with tf.GradientTape() as tape_actor:
            _, mu, sigma = self(states)
            norm_dist = tf.random.normal(shape=mu.shape, mean=mu, stddev=sigma)
            prob = tf.exp(-tf.square((actions - mu) / (sigma + 1e-8)) / 2) / (sigma * tf.sqrt(2 * np.pi))
            old_prob = tf.stop_gradient(prob)
            ratio = prob / (old_prob + 1e-8)
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - 0.2, 1.0 + 0.2)
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantage, clipped_ratio * advantage))

        actor_grads = tape_actor.gradient(actor_loss, self.actor_variables())
        self.optimizer_actor.apply_gradients(zip(actor_grads, self.actor_variables()))

        return critic_loss, actor_loss

    def critic_variables(self):
        return self.critic_conv.trainable_variables + self.critic_bn.trainable_variables + \
            self.critic_flatten.trainable_variables + self.critic_dense.trainable_variables

    def actor_variables(self):
        return self.actor_conv.trainable_variables + self.actor_bn.trainable_variables + \
            self.actor_flatten.trainable_variables + self.actor_mu.trainable_variables + \
            self.actor_sigma.trainable_variables

    def save_model(self, path="ppo_model.keras"):
        self.save_weights(path)

    def load_model(self, path="ppo_model.keras"):
        try:
            # Build the model by specifying the input shape
            dummy_input = tf.random.normal((1, self.M, self.L, self.N))
            self(dummy_input)
            print("Model loaded successfully.")
        except Exception as e:
            print("Failed to load model:", e)

# """
# A simple version of Proximal Policy Optimization (PPO) using single thread.
#
# """
#
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
#
# from tensorflow.python.layers.normalization import BatchNormalization
#
# import numpy as np
# import json
# import time
# import math
# import pandas as pd
# from argparse import ArgumentParser
#
# EP_MAX = 1000
# EP_LEN = 200
# GAMMA = 0.9
# A_LR = 10e-4
# C_LR = 10e-4
# BATCH = 32
# A_UPDATE_STEPS = 10
# C_UPDATE_STEPS = 10
# METHOD = [
#     dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
#     dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
# ][1]        # choose the method for optimization
#
# def con2d(x,scope,trainable):
#     with tf.variable_scope(scope):
#        # First convolution layer
#         con_W_1 = tf.Variable(tf.truncated_normal([1, 3, int(x.shape[3]), 2], stddev=0.5), trainable=trainable)
#         layer = tf.nn.conv2d(x, con_W_1, padding='SAME', strides=[1, 1, 1, 1])  # Use 'SAME' padding
#         norm = BatchNormalization(layer, training=trainable)
#         x = tf.nn.relu(norm)
#
#         # Second convolution layer
#         con_W_2 = tf.Variable(tf.truncated_normal([1, 3, int(x.shape[3]), 48], stddev=0.5), trainable=trainable)
#         layer = tf.nn.conv2d(x, con_W_2, padding='SAME', strides=[1, 1, 1, 1])  # Use 'SAME' padding
#         norm = BatchNormalization(layer, training=trainable)
#         x = tf.nn.relu(norm)
#
#         # Third convolution layer
#         con_W_3 = tf.Variable(tf.truncated_normal([1, 3, 48, 1], stddev=0.5), trainable=trainable)
#         layer = tf.nn.conv2d(x, con_W_3, padding='SAME', strides=[1, 1, 1, 1])  # Use 'SAME' padding
#         norm = BatchNormalization(layer, training=trainable)
#         out = tf.nn.relu(norm)
#
#
#     return out
#
# def dense(x,out_dim,activation,scope,trainable):
#     with tf.variable_scope(scope):
#         t1_w = tf.Variable(tf.truncated_normal([int(x.shape[1]), out_dim], stddev=0.1),trainable=trainable)
#         t1_b = tf.Variable(tf.constant(0.1, shape=[out_dim]),trainable=trainable)
#         out = tf.matmul(x, t1_w) + t1_b
#
#         if activation=='relu':
#             out=tf.nn.relu(out)
#         elif activation=='tanh':
#             out=tf.nn.tanh(out)
#         elif activation=='softplus':
#             out=tf.nn.softplus(out)
#         elif activation=='sigmoid':
#             out=tf.nn.sigmoid(out)
#         else:
#             print('fail to build up')
#     return out
#
# def build_summaries():
#     critic_loss=tf.Variable(0.)
#     reward=tf.Variable(0.)
#     ep_ave_max_q=tf.Variable(0.)
#     actor_loss=tf.Variable(0.)
#     tf.summary.scalar('Critic_loss',critic_loss)
#     tf.summary.scalar('Reward',reward)
#     tf.summary.scalar('Ep_ave_max_q',ep_ave_max_q)
#     tf.summary.scalar('Actor_loss',actor_loss)
#
#
#     summary_vars=[critic_loss,reward,ep_ave_max_q,actor_loss]
#     summary_ops=tf.summary.merge_all()
#     return summary_ops,summary_vars
#
#
# class PPO(object):
#
#     def __init__(self,predictor,M,L,N,name,load_weights,trainable):
#         self.sess = tf.Session()
#         self.tfs = tf.placeholder(tf.float32, [None, M,L,N], 'state')
#         self.name=name
#
#         self.M=M
#         self.L=L
#         self.N=N
#
#         self.gamma=0.99
#
#         # critic
#         with tf.variable_scope('critic'):
#             l1 = con2d(self.tfs,'critic',True)[:,:,0,0]
#             self.v = dense(l1,1,'relu','critic',True)
#             self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
#             self.advantage = self.tfdc_r - self.v
#             self.closs = tf.reduce_mean(tf.square(self.advantage))
#             # Optimization Op
#             global_step = tf.Variable(0, trainable=False)
#             C_learning_rate = tf.train.exponential_decay(C_LR, global_step,
#                                                        decay_steps=2000,
#                                                        decay_rate=0.9, staircase=False)
#             self.ctrain_op = tf.train.GradientDescentOptimizer(C_learning_rate).minimize(self.closs,global_step=global_step)
#
#         # actor
#         pi, pi_params = self._build_anet('pi', trainable=True)
#         oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
#         with tf.variable_scope('sample_action'):
#             self.sample_op =pi.sample(1)[0]#tf.squeeze(pi.sample(1),axis=[1,3])       # choosing action
#         with tf.variable_scope('update_oldpi'):
#             self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]
#
#         self.tfa = tf.placeholder(tf.float32, [None, self.M], 'action')
#         self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
#         with tf.variable_scope('loss'):
#             with tf.variable_scope('surrogate'):
#                 # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
#                 ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
#                 surr = ratio * self.tfadv
#             if METHOD['name'] == 'kl_pen':
#                 self.tflam = tf.placeholder(tf.float32, None, 'lambda')
#                 kl = tf.distributions.kl_divergence(oldpi, pi)
#                 self.kl_mean = tf.reduce_mean(kl)
#                 self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
#             else:   # clipping method, find this is better
#                 self.aloss = -tf.reduce_mean(tf.minimum(
#                     surr,
#                     tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))
#
#         with tf.variable_scope('atrain'):
#             A_learning_rate = tf.train.exponential_decay(A_LR, global_step,
#                                                          decay_steps=2000,
#                                                          decay_rate=0.9, staircase=False)
#             self.atrain_op = tf.train.GradientDescentOptimizer(A_learning_rate).minimize(self.aloss)
#
#         # Initial saver
#         self.saver = tf.train.Saver(max_to_keep=3)
#         if load_weights=="True":
#             print("Loading Model")
#             try:
#                 checkpoint = tf.train.get_checkpoint_state(self.result_save_path)
#                 if checkpoint and checkpoint.model_checkpoint_path:
#                     self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
#                     print("Successfully loaded:", checkpoint.model_checkpoint_path)
#                 else:
#                     print("Could not find old network weights")
#             except:
#                 print("Could not find old network weights")
#                 self.sess.run(tf.global_variables_initializer())
#         else:
#             self.sess.run(tf.global_variables_initializer())
#
#         if trainable:
#             self.summary_writer = tf.summary.FileWriter("./summary/PPO", self.sess.graph)
#             self.summary_ops, self.summary_vars = build_summaries()
#
#         #Initial buffer
#         self.buffer=[]
#
#     def update(self, s, a, r):
#         self.sess.run(self.update_oldpi_op)
#         adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
#
#         # update actor # clipping method, find this is better (OpenAI's paper)
#         [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]
#
#         # update critic
#         critic_loss=0
#         for _ in range(C_UPDATE_STEPS):
#             closs,_=self.sess.run([self.closs, self.ctrain_op], {self.tfs: s, self.tfdc_r: r})
#             print('*--------------------*',closs)
#             critic_loss+=closs
#         return critic_loss
#
#     def _build_anet(self, name, trainable):
#         with tf.variable_scope(name):
#             input=con2d(self.tfs,'critic',trainable)[:,:,0,0]
#             l1 = dense(input, 100,'relu', 'critic',trainable)
#             mu = dense(l1, self.M, 'tanh', 'critic',trainable)
#             sigma =dense(l1, self.M, 'softplus','critic',trainable)
#             norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
#         params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
#         return norm_dist, params
#
#     def predict(self, s):
#         a = self.sess.run(self.sample_op, {self.tfs:s})[0]
#         a=np.exp(a)
#         a=a/np.sum(a)
#         a=a[np.newaxis,:]
#         return a
#
#     def get_v(self, s):
#         if s.ndim < 2: s = s[np.newaxis, :]
#         return self.sess.run(self.v, {self.tfs: s})[0]
#
#     def write_summary(self, Loss, reward, ep_ave_max_q, actor_loss, epoch):
#         summary_str = self.sess.run(self.summary_ops, feed_dict={
#             self.summary_vars[0]: Loss,
#             self.summary_vars[1]: reward,
#             self.summary_vars[2]: ep_ave_max_q,
#             self.summary_vars[3]: actor_loss
#         })
#         self.summary_writer.add_summary(summary_str, epoch)
#
#     def save_model(self,epoch):
#         self.saver.save(self.sess, './saved_network/PPO/'+self.name,global_step=epoch)
#
#     def save_transition(self,s,w,r,contin,s_next,action_precise):
#         self.buffer.append([s,w,r])
#
#     def train(self,method,epoch):
#         info=dict()
#         v=self.get_v(self.buffer[-1][0])
#         discounted_r=[]
#         rs=[transition[2] for transition in self.buffer[::-1]]
#         for r in rs:
#             v=r+self.gamma*v
#             discounted_r.append(v)
#         discounted_r.reverse()
#         discounted_r=np.array(discounted_r)
#         mini_batch_s = np.vstack([transition[0] for transition in self.buffer[::-1]])
#         mini_batch_a = np.vstack([transition[1] for transition in self.buffer[::-1]])
#         critic_loss=self.update(mini_batch_s,mini_batch_a,discounted_r)
#         self.buffer=[]
#         info["critic_loss"]=critic_loss
#         info["q_value"]=0
#         info["actor_loss"]=0
#         return info
