# -*- coding: UTF-8 -*-
'''
最简单的policy gradient的pytorch版本
'''
from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
import gym
from gym.spaces import Discrete, Box
import pdb

class net(nn.Module):
    def __init__(self, layer_sizes, activation=nn.Tanh, output_activation=None):
        # layer_sizes是包含[输入层，隐藏层，输出层]
        super(net, self).__init__()
        fc = []
        for l in range(len(layer_sizes) - 1):
            fc.append(nn.Linear(layer_sizes[l], layer_sizes[l + 1]))
            if l != len(layer_sizes) - 2: #不是最后一层
                fc.append(activation())
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        y = self.fc(x)
        return y


def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0] # observations的长度
    n_acts = env.action_space.n # actions的数目

    # make core of policy network
    layers = [obs_dim] + hidden_sizes + [n_acts]
    policy_net = net(layers)

    # make train op
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        nb_trajectories = 0

        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_logits = []
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if not(finished_rendering_this_epoch):
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            obs = np.reshape(obs, (1,-1))
            obs = torch.from_numpy(obs)
            obs = obs.type(torch.FloatTensor)

            logits = policy_net(obs)

            act = torch.exp(logits) # act的概率
            act = torch.squeeze(torch.multinomial(act, num_samples=1))
            act = act.data.numpy()

            obs, rew, done, _ = env.step(act)

            # save logits, action, reward
            batch_logits.append(logits)
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                nb_trajectories += 1  # 统计进入done的次数，表示一个batch里面有多少个trajectories

                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau) 整个轨迹有一个 R(tau)，从状态s0 => done这个完整过程
                # 是有一个 R(tau), 下面batch_weights这个list会不断地加长的，每一个轨迹tau后面加长一段。
                # [4,4,4,3,3,3,3,3,2,2,2,2,2,2,2,2] 里面有三段轨迹，其中[4,4,4]是tau_1， [3,3,3,3,3]是轨迹tau_2
                # [2,2,2,2,2,2,2,2]是轨迹tau_3这种感觉
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        print('nb_trajectories: ', nb_trajectories)
        # take a single policy gradient update step
        # make loss function whose gradient, for the right data, is policy gradient
        batch_logits = torch.cat(batch_logits, dim=0)

        batch_acts = np.array(batch_acts)
        batch_acts = np.expand_dims(batch_acts, 1)
        batch_acts = torch.from_numpy(batch_acts)
        batch_acts = batch_acts.type(torch.LongTensor)

        action_masks = torch.zeros(batch_acts.shape[0], n_acts).scatter_(1, batch_acts, 1)
        log_probs = torch.sum(action_masks * nn.LogSoftmax()(batch_logits), dim=1)
        batch_loss = -torch.sum(torch.from_numpy(np.array(batch_weights)).type(torch.FloatTensor) * log_probs) / nb_trajectories

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
