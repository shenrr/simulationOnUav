import os

import numpy as np
import tensorflow as tf

from DDPG.Actor import Actor
from DDPG.Critic import Critic
from DDPG.OUNoise import OrnsteinUhlenbeckActionNoise
from DDPG.ReplayMemory import ReplayMemory


class DDPG_agent:
    def __init__(self, sess, state_shape, action_bound, action_dim,
                 memory_size=100000, minibatch_size=128, gamma=0.99, tau=0.001, train_after=200):#train_after猜测是训练延迟
        self.actor = Actor(sess, action_bound, action_dim, state_shape,lr = 0.0001, tau=tau)
        self.critic = Critic(sess, state_shape, action_dim, minibatch_size,lr = 0.001, tau=tau)
        self.state_shape = state_shape
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.memory_size = memory_size
        self.replay_memory = ReplayMemory(self.memory_size)
        self.sess = sess
        self.minibatch_size = minibatch_size
        self.action_bound = action_bound
        self.gamma = gamma
        self.train_after = max(minibatch_size, train_after)
        self.num_action_taken = 0
        self.action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(action_dim))
        self.countit=0
        self.lastQ=100
        self.countreplay=1

    def observe(self, state, action, reward, post_state, terminal):
        self.replay_memory.append(state, action, reward, post_state, terminal)

    def act(self, state, noise=True):

        action = self.actor.act(state)
        action2 = self.actor.actTarget(state)

        if noise:
            noise = self.action_noise()
            showAction=action
            judge=np.random.rand()
            goodaction=np.zeros((2))
            output2 = self.critic.eval_net_evalInit(state, action)
            if judge>=0.1 and output2>5:# and output2>30 :
                length=np.random.rand()
                if length<0.4:
                    length=0.4
                if abs(state[1][0])>abs(state[1][1]):
                    action[0]=length*state[1][0]/abs(state[1][0])
                    action[1]=action[0]*state[1][1]/state[1][0]
                else:
                    action[1]=1*state[1][1]/abs(state[1][1])
                    action[0]=action[1]*state[1][0]/state[1][1]
                goodaction=action
            elif output2<20 and judge<=0.5:
                action = np.clip(noise + action, -self.action_bound, self.action_bound)
            #action=np.clip(noise, -self.action_bound, self.action_bound)
            action = np.clip(noise+action, -self.action_bound, self.action_bound)
            #print(noise)
            state[1][0]=action[0]
            state[1][1] = action[1]
            output = self.critic.target_net_evalInit(state, action)
            output2=self.critic.eval_net_evalInit(state, action)
            print("noise: {}".format(noise).ljust(20, " "),"for good: {}".format(judge).ljust(20, " "),"forgood: {}".format(goodaction*2).ljust(20, " "),"initAction: {}".format(showAction*2).ljust(20, " "),"TargetAction: {}".format(action2*2).ljust(20, " "),"eval Q: {}".format(output2).ljust(20, " "),"target Q: {}".format(output).ljust(20, " "))
        else:
            action = np.clip(action, -self.action_bound, self.action_bound)
            output = self.critic.target_net_evalInit(state, action)
            print(
                "eval Q: {}".format(output).ljust(20, " "),"target Q: {}".format(output).ljust(20, " "))
        self.num_action_taken += 1

        return action

    def update_target_nets(self):
        # update target net for both actor and critic
        self.sess.run([self.actor.update_ops, self.critic.update_ops])

    def train(self,times = 1):
        if self.num_action_taken >= self.train_after:
            for i in range(times):
                #print ("training:{} / {}".format(i,times),end = '\r')
               # print("action")
               # print(self.num_action_taken)
                # 1 sample random minibatch from replay memory
                states, actions, rewards, post_states, terminals = \
                    self.replay_memory.sample(self.minibatch_size)

                # 2 use actor's target net to select action for Si+1, denote as mu(S_i+1)
                mu_post_states = self.actor.target_action(post_states)

                # 3 use critic's target net to evaluate Q(S_i+1, a_i+1) and calculate td target
                Q_target = self.critic.target_net_eval(post_states, mu_post_states)
               # print( "target Q: {}".format(Q_target[0]).ljust(20, " "))
                rewards = rewards.reshape([self.minibatch_size, 1])
                terminals = terminals.reshape([self.minibatch_size, 1])
                td_target = rewards + self.gamma * Q_target * (1 - terminals)

                # 4 update critic's online network
                self.critic.train(states, actions, td_target)

                # 5 predict action using actors online network and calculate the sampled gradients
                pred_actions = self.actor.predict_action(states)
                Q_gradients = self.critic.action_gradient(states, pred_actions) / self.minibatch_size

                # 6 update actor's online network
                self.actor.train(Q_gradients, states)

                # 7 apply soft replacement for both target networks
                self.update_target_nets()

    def save(self, saver, dir,dir2):


        path = os.path.join(dir, 'model')#路径连接
        p = self.countreplay % 2

        if self.countit%10==0:
            self.countreplay+=1
            self.replay_memory.save(dir2+"\\"+"replay"+str(p))
        #if self.countit % 5 == 0:
        saver.save(self.sess, path)
        self.countit+=1


    def load(self, saver, dir):
        path = os.path.join(dir, 'checkpoint')
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(path))
        #要接着replaymemory继续训练就打开下行
        #self.replay_memory=self.replay_memory.load(dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)#这是取么
            return True
        return False
