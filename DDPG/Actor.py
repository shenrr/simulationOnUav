import numpy as np
import tensorflow as tf


class Actor:
    def __init__(self, sess, action_bound, action_dim, state_shape, lr=1e-4, tau=0.001):
        self.sess = sess
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.state_shape = state_shape
        self.tau = tau
        
        self.state = tf.placeholder(tf.float32, [None, state_shape])
        self.img = tf.placeholder(tf.float32, [None, 64, 64, 1])
        self.post_state = tf.placeholder(tf.float32, [None, state_shape])
        self.post_img = tf.placeholder(tf.float32, [None, 64, 64, 1])
        self.Q_gradient =  tf.placeholder(tf.float32, [None, action_dim])
        
        with tf.variable_scope("actor"):
            self.eval_net = self._build_network(self.state, self.img, "eval_net")
            # target net is used to predict action for critic
            self.target_net = self._build_network(self.post_state, self.post_img, "target_net")
        
        self.eval_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor/eval_net")
        self.target_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor/target_net")
        self.update_opsc2 = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.update_opsc = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # use negative Q gradient to guide gradient ascent
        self.policy_gradient = tf.gradients(ys=self.eval_net, xs=self.eval_param, grad_ys=-self.Q_gradient)
        self.train_step = tf.train.AdamOptimizer(lr).apply_gradients(zip(self.policy_gradient, self.eval_param))

        self.update_ops = self._update_target_net_op()


        
    def _build_network(self, X, image, scope):

        with tf.variable_scope(scope):
            init_w1 = tf.truncated_normal_initializer(0., 3e-4)
            init_w2 = tf.random_uniform_initializer(-0.05, 0.05)

            conv1 = tf.layers.conv2d(image, 32, [5,5], strides=[2,2], padding="same", kernel_initializer=init_w1, activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
            conv2 = tf.layers.conv2d(pool1, 32, [5,5], strides=[1,1], padding="same", kernel_initializer=init_w1, activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
            conv3 = tf.layers.conv2d(pool2, 32, [5,5], strides=[1,1], padding="same", kernel_initializer=init_w1, activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2)
            flatten = tf.layers.flatten(pool3) # shape(None, 4*4*32)
            concat = tf.concat([flatten, X], 1)

            fc1 = tf.layers.dense(inputs=concat, units=200, activation=tf.nn.leaky_relu, kernel_initializer=init_w2)
            #fc1 = tf.layers.dropout(fc1, 0.2)
            fc1 = tf.layers.batch_normalization(fc1, training=True)
            fc1=tf.nn.leaky_relu(fc1)
            fc2 = tf.layers.dense(inputs=fc1, units=160,  kernel_initializer=init_w2)
            # dropout为了防止过拟合，可以删掉不影响
            #fc2 = tf.layers.dropout(fc2, 0.2)
            fc2 = tf.layers.batch_normalization(fc2, training=True)
            fc2=tf.nn.relu(fc2)
            fc3 = tf.layers.dense(inputs=fc2, units=120, activation=tf.nn.relu, kernel_initializer=init_w2)
            #fc3 = tf.layers.dropout(fc3, 0.2)
            #正则化，可能没实际效用，可以加上
            fc3=tf.layers.batch_normalization(fc3, training=True)
            fc3=tf.nn.relu(fc3)
            action_normal = tf.layers.dense(inputs=fc3, units=self.action_dim, kernel_initializer=init_w2)
            action_normal = tf.layers.batch_normalization(action_normal, training=True)
            action_normal=tf.nn.tanh(action_normal)
            action = tf.multiply(action_normal, self.action_bound)
        return action
        
    def act(self, state):
        img, dstate = state
        img = np.reshape(img, [1, 64, 64, 1])
        #img=np.add(img,test)
        dstate = np.reshape(dstate, [1, self.state_shape])
        action = self.sess.run(self.eval_net, feed_dict={self.state:dstate, self.img:img})[0]
        return action

    def actTarget(self, state):
        img, dstate = state

        test=np.zeros([1,64,64,1])
        img = np.reshape(img, [1, 64, 64, 1])
        #img=np.add(img,test)
        dstate = np.reshape(dstate, [1, self.state_shape])
        action = self.sess.run(self.target_net, feed_dict={self.post_state:dstate, self.post_img:img})[0]
        return action
        
    def predict_action(self, states):
        imgs, dstates = self._seperate_image(states)
        pred_actions = self.sess.run(self.eval_net, feed_dict={self.state:dstates, self.img:imgs})
        return pred_actions
        
    def target_action(self, post_states):
        imgs, dstates = self._seperate_image(post_states)
        actions = self.sess.run(self.target_net, feed_dict={self.post_state:dstates, self.post_img:imgs})
        return actions
        
    def train(self, Q_gradient, states):
        imgs, dstates = self._seperate_image(states)
        with tf.control_dependencies(self.update_opsc):
            self.sess.run(self.train_step, feed_dict={self.state:dstates, self.img:imgs, self.Q_gradient:Q_gradient})
        
    def _update_target_net_op(self):

        ops = [tf.assign(dest_var, (1-self.tau) * dest_var + self.tau * src_var)
            for dest_var, src_var in zip(self.target_param, self.eval_param)]
        return ops

    def _seperate_image(self, states):
        images = np.array([state[0] for state in states])
        dstates = np.array([state[1] for state in states])
        return images, dstates


