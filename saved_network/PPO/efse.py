def __init__(self, predictor, M, L, N, name, load_weights, trainable):
    self.sess = tf.Session()
    self.tfs = tf.placeholder(tf.float32, [None, M, L, N], 'state')
    self.name = name

    self.M = M
    self.L = L
    self.N = N

    self.gamma = 0.99

    # critic
    with tf.variable_scope('critic'):
        l1 = con2d(self.tfs, 'critic', True)[:, :, 0, 0]
        self.v = dense(l1, 1, 'relu', 'critic', True)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        # Optimization Op
        global_step = tf.Variable(0, trainable=False)
        C_learning_rate = tf.train.exponential_decay(C_LR, global_step,
                                                     decay_steps=2000,
                                                     decay_rate=0.9, staircase=False)
        self.ctrain_op = tf.train.GradientDescentOptimizer(C_learning_rate).minimize(self.closs,
                                                                                     global_step=global_step)

    # actor
    pi, pi_params = self._build_anet('pi', trainable=True)
    oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
    with tf.variable_scope('sample_action'):
        self.sample_op = pi.sample(1)[0]  # tf.squeeze(pi.sample(1),axis=[1,3])       # choosing action
    with tf.variable_scope('update_oldpi'):
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

    self.tfa = tf.placeholder(tf.float32, [None, self.M], 'action')
    self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
    with tf.variable_scope('loss'):
        with tf.variable_scope('surrogate'):
            # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
            surr = ratio * self.tfadv
        if METHOD['name'] == 'kl_pen':
            self.tflam = tf.placeholder(tf.float32, None, 'lambda')
            kl = tfp.distributions.kl_divergence(oldpi, pi)
            self.kl_mean = tf.reduce_mean(kl)
            self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
        else:  # clipping method, find this is better
            self.aloss = -tf.reduce_mean(tf.minimum(
                surr,
                tf.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * self.tfadv))

    with tf.variable_scope('atrain'):
        A_learning_rate = tf.train.exponential_decay(A_LR, global_step,
                                                     decay_steps=2000,
                                                     decay_rate=0.9, staircase=False)
        self.atrain_op = tf.train.GradientDescentOptimizer(A_learning_rate).minimize(self.aloss)

    # Initial saver
    self.saver = tf.train.Saver(max_to_keep=3)
    if load_weights == "True":
        print("Loading Model")
        try:
            checkpoint = tf.train.get_checkpoint_state(self.result_save_path)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                print("Could not find old network weights")
        except:
            print("Could not find old network weights")
            self.sess.run(tf.global_variables_initializer())
    else:
        self.sess.run(tf.global_variables_initializer())

    if trainable:
        self.summary_writer = tf.summary.FileWriter("./summary/PPO", self.sess.graph)
        self.summary_ops, self.summary_vars = build_summaries()

    # Initial buffer
    self.buffer = []
