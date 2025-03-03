############################################################################################################
#                                           IMPORTING LIBRARIES
# ##########################################################################################################

import warnings
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import gym
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)

standard = StandardScaler()
minmax = MinMaxScaler()
dir_path = os.getcwd()

############################################################################################################
# **********************************************************************************************************
#                                           ENVIRONMENT MODELING
# **********************************************************************************************************
# ##########################################################################################################


############################################################################################################
#                                           Alarm Management
# ##########################################################################################################
import matlab
import matlab.engine

from itertools import chain, repeat, islice


def pad_infinite(iterable, padding=None):
    return chain(iterable, repeat(padding))


def pad(iterable, size, padding=None):
    return islice(pad_infinite(iterable, padding), size)


''' run  this command in Matlab '''
# matlab.engine.shareEngine()

# eng = matlab.engine.start_matlab()
name_eng = matlab.engine.find_matlab()
eng1 = matlab.engine.connect_matlab(name_eng[0])

Y_Thresh = pd.read_csv(r".\PV_thresh.csv", header=0, delimiter=';')
Y_Thresh = Y_Thresh[:40]
U_Thresh = pd.read_csv(r".\CV_thresh.csv", header=0, delimiter=',')

init_corr = [0.0]
eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/PID-DRPRL', 'sw', str(1), nargout=0)
eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/xmv_corr', 'Value', str(init_corr), nargout=0)


def get_reward(state):

    return -(abs(state - Y_Thresh['Normal'].to_numpy()[0]))


upper_bound = 1.0
scale_factor = 100.0


class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, test=False, reset_orig=True, run_orig=True, sim_time=10):
        self.dist_stop = None
        self.dist_start = None
        self.reset_orig = reset_orig
        self.run_orig = run_orig
        self.num_alarms = []
        self.true_num_alarms = []
        self.control_action = []
        self.alarms = []
        self.alarms_seq = []
        self.true_alarms = []
        self.errorDRL = []
        self.error = []
        self.test = test
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'SimulationCommand', 'stop', nargout=0)
        self.disturbance = [0, 0, 0, 0, 0, 0, 0, 0]
        self.disturbances = self.disturbance + ([0] * 20)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/Disturbances', 'Value', str(self.disturbances), nargout=0)
        self.sim_time = sim_time
        if self.test:
            self.sim_time = sim_time
        self.sim_step = 0.00
        self.step_size = 0.01
        self.step_step_size = 0.01
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'StopTime', str(self.sim_time), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'EnablePauseTimes', 'on', nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'PauseTimes', str(self.sim_step), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/PID-DRPRL', 'sw', str(1), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/xmv_corr', 'Value', str(init_corr), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'SimulationCommand', 'start', 'SimulationCommand', 'pause',
                       nargout=0)
        self.process_history = 2
        self.state = []
        self.pv = []
        self.mv = []
        self.rp = []
        mv = 1
        pv = 1
        dev = 0
        self.feat = mv + pv + dev
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=(self.feat,))
        self.action_space = gym.spaces.Box(low=0,
                                           high=upper_bound,
                                           shape=(mv,))
        self.reward = 0
        self.c_f = -10000
        self.prev_reward = 0
        self.T = 0
        self.done = False
        self.max_episode_steps = int(self.sim_time / self.step_size)

    def step(self, control, disturbance=False, dist_mag=None):
        mu = control * scale_factor
        control = matlab.double(mu.tolist())
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/xmv_corr', 'Value', str(control), nargout=0)

        self.sim_step += self.step_size
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'PauseTimes', str(round(self.sim_step, 2)), nargout=0)
        eng1.set_param("MultiLoop_mode1_DRL_SingleAgent", 'SimulationCommand', 'continue', nargout=0)

        if disturbance:
            if self.sim_time * self.dist_start < self.sim_step < self.sim_time * self.dist_stop:
                disturbance = [0, 0, 0, 0, 0, dist_mag, 0, 0]
                disturbances = disturbance + ([0] * 20)
                eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/Disturbances', 'Value', str(disturbances), nargout=0)
            else:
                disturbance = [0, 0, 0, 0, 0, 0, 0, 0]
                disturbances = disturbance + ([0] * 20)
                eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/Disturbances', 'Value', str(disturbances), nargout=0)

        if self.sim_step >= self.sim_time:
            self.reward = 0
            self.done = True
            self.alarms.append(np.mean(self.num_alarms))
            self.alarms_seq.append(self.num_alarms)
            self.true_alarms.append(np.mean(self.true_num_alarms))
            self.run_orig = False
        elif eng1.get_param('MultiLoop_mode1_DRL_SingleAgent', 'SimulationStatus') == ('stopped' or 'terminating'):
            eng1.set_param("MultiLoop_mode1_DRL_SingleAgent", 'SimulationCommand', 'stop', nargout=0)
            self.reward = self.c_f
            self.done = True
            self.alarms.append(np.mean(self.num_alarms))
            self.alarms_seq.append(self.num_alarms)
            self.true_alarms.append(np.mean(self.true_num_alarms))
            self.run_orig = False
        else:
            self.mv = [eng1.workspace['drprl'] / 100.0]
            self.pv = [eng1.workspace['output']._data.tolist()[0]]
            curr_dev = [eng1.workspace['output']._data.tolist()[0] - Y_Thresh['Normal'].to_numpy()[0]]
            self.state = self.state[self.feat:] + self.mv + self.pv
            self.reward = get_reward((np.delete(eng1.workspace['output']._data, 18)).tolist()[0])

        control_room = ~pd.Series(np.delete(eng1.workspace['output']._data, 18).tolist()).between(
            Y_Thresh['LO-Alarm'].tolist(), Y_Thresh['HI-Alarm'].tolist())
        num_alarms = sum(control_room)
        op_screen = list(pad(control_room[control_room].index.values, 38, 0))
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/true_alarms', 'Value', str(op_screen), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/num_alarms', 'Value', str(num_alarms), nargout=0)
        self.num_alarms.append(num_alarms)
        self.control_action.append(control[0])
        self.errorDRL.append(
            -(abs((np.delete(eng1.workspace['output']._data, 18)).tolist()[0] - Y_Thresh['Normal'].to_numpy()[0])))
        info = {}
        return self.state, self.reward, self.done, info

    def reset(self):
        self.num_alarms = []
        self.true_num_alarms = []
        self.control_action = []
        self.state = []
        self.pv = []
        self.mv = []
        self.rp = []
        self.dist_start = np.random.uniform(0.0, 0.5)
        self.dist_stop = np.random.uniform(0.6, 1.0)

        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'StopTime', str(self.sim_time), nargout=0)
        self.disturbance = [0, 0, 0, 0, 0, 0, 0, 0]
        self.disturbances = self.disturbance + ([0] * 20)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/Disturbances', 'Value', str(self.disturbances), nargout=0)
        self.sim_step = 0.00
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'PauseTimes', str(self.sim_step), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/PID-DRPRL', 'sw', str(1), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/xmv_corr', 'Value', str(init_corr), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'SimulationCommand', 'start', 'SimulationCommand', 'pause',
                       nargout=0)
        
        for i in range(self.process_history):
            self.mv = [eng1.workspace['input']._data.tolist()[2] / 100.0]
            self.pv = [eng1.workspace['output']._data.tolist()[0]]
            self.state = self.state + self.mv + self.pv

            eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/PID-DRPRL', 'sw', str(1), nargout=0)
            control = np.array([0.0])
            mu = control
            control = matlab.double(mu.tolist())
            eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/xmv_corr', 'Value', str(control), nargout=0)

            self.sim_step += self.step_size
            eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'PauseTimes', str(round(self.sim_step, 2)), nargout=0)
            eng1.set_param("MultiLoop_mode1_DRL_SingleAgent", 'SimulationCommand', 'continue', nargout=0)

        self.done = False
        self.reset_orig = False
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/PID-DRPRL', 'sw', str(0), nargout=0)
        return self.state, self.done


env = CustomEnv(sim_time=5)
eval_env = CustomEnv(test=True, reset_orig=True, run_orig=True, sim_time=5)

############################################################################################################
# **********************************************************************************************************
#                                   DEEP REINFORCEMENT LEARNING ARCHITECTURES
# **********************************************************************************************************
# ##########################################################################################################


############################################################################################################
# **********************************************************************************************************
#                                              Actor-Critic
# ##########################################################################################################


##************************** Twin Delayed Deep Deterministic Policy Gradients (TD3) ************************

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, env.process_history, env.feat))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, env.process_history, env.feat))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class extract_tensor(nn.Module):
    def forward(self, x):
        tensor, _ = x
        return tensor


class keep_sequence(nn.Module):
    def forward(self, x):
        tensor, _ = x
        return tensor[:, -1, :]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.LSTM(env.feat, 64, batch_first=True)
        self.l2 = nn.LSTM(64, 32, batch_first=True)
        self.l3 = nn.Linear(32, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = self.l1(state)[0]
        a = self.l2(a)[0][:, -1, :]
        af = self.l3(a)
        return self.max_action * torch.sigmoid(af)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(env.process_history * state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 1)

        self.l4 = nn.Linear(env.process_history * state_dim + action_dim, 64)
        self.l5 = nn.Linear(64, 32)
        self.l6 = nn.Linear(32, 1)

    def forward(self, state, action):
        sa = torch.cat([torch.reshape(state, (-1, env.process_history * env.feat)), action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([torch.reshape(state, (-1, env.process_history * env.feat)), action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            min_action,
            batch_size,
            replay_buffer,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):

        """def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight, -0.003, 0.003)
                m.bias.data.fill_(0.001)"""

        self.max_action = max_action
        self.min_action = min_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_freq = policy_freq

        self.total_it = 0
        self.burn_in = 0
        self.warm_up = 10000
        self.burn_period = 10000
        self.actor_bc_loss_hist = []
        self.critic_bc_loss_hist = []

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.bc_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-3)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-3, weight_decay=1e-2)
        self.actor_target = self.train_actor_bc(replay_buffer, batch_size=batch_size)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, weight_decay=1e-2)
        self.critic_target = self.train_critic_bc(replay_buffer, batch_size=batch_size)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train_actor_bc(self, replay_buffer, batch_size=None):
        xmv = np.expand_dims(
            np.load(r".\XMV_RPL_big.npy")[:, 2] / 100.0, 1)
        xmeas = np.expand_dims(
            np.load(r".\XMEAS_RPL_big.npy")[:, 0], 1)

        error_signal = []
        for i in range(len(xmeas)):
            error_signal.append(xmeas[i] - Y_Thresh['Normal'].to_numpy()[0])
        error_signal = np.array(error_signal)

        samples = list()
        length = env.process_history
        n = len(xmeas)
        for i in range(0, n):
            if i == n - length:
                break
            sample = np.concatenate((xmv[i:i + length], xmeas[i:i + length]), 1)
            samples.append(sample)
        samples = np.array(samples)

        expert_traj = xmv[length:]

        for i, (state, action) in enumerate(zip(samples, expert_traj)):
            if i >= len(samples) - 1:
                break
            next_state = samples[i + 1]
            reward = get_reward(xmeas[length + i])
            done_bool = 0
            replay_buffer.add(state, action, next_state, reward, done_bool)

        print("----------------------- Behaviour Cloning -------------------------")
        for i in range(self.warm_up):
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            bc_loss = F.mse_loss(self.actor(state), action)
            self.bc_optimizer.zero_grad()
            bc_loss.backward()
            self.bc_optimizer.step()
            self.actor_bc_loss_hist.append(bc_loss.item())
        print("\nactor_loss", round(bc_loss.item(), 5))
        return copy.deepcopy(self.actor)

    def train_critic_bc(self, replay_buffer, batch_size=None):
        for i in range(self.burn_period):
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            with torch.no_grad():
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(self.min_action, self.max_action)

                target_Q1, target_Q2 = self.critic(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = torch.layer_norm(reward, reward.shape) + (not_done * self.discount * target_Q)

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            self.critic_bc_loss_hist.append(critic_loss.item())
        print("\ncritic_loss", round(critic_loss.item(), 5))
        print("\n-------------------------------------------------------------------")
        return copy.deepcopy(self.critic)

    def train(self, replay_buffer, batch_size=None):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(self.min_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = torch.layer_norm(reward, reward.shape) + (not_done * self.discount * target_Q)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



############################################################################################################
# **********************************************************************************************************
#                                               TRAINING
# **********************************************************************************************************
# ##########################################################################################################


############################################################################################################
# **********************************************************************************************************
#                                          Alarm Management
# ##########################################################################################################


# ********************************** TD3 *************************************
eval_reward = []
def eval_policy(policy, eval_episodes=1, dist_mag=None):
    eval_env.reset_orig = True
    eval_env.run_orig = True
    eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/Scope 5', 'commented', 'on', nargout=0)
    eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/Scope 6', 'commented', 'on', nargout=0)
    avg_reward = 0.
    for i in range(eval_episodes):
        eval_DRL_action = []
        state, done = eval_env.reset()
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/PID-DRPRL', 'sw', str(0), nargout=0)
        disturbance = True
        eval_timesteps = 0
        while not done:
            eval_timesteps += 1
            PID = eng1.workspace['input']._data[2] / 100.0
            DRPRL = policy.select_action(np.reshape(state, (1, env.process_history, env.feat)))
            action_mix = DRPRL + PID
            action = DRPRL
            eval_DRL_action.append(action)
            state, reward, done, _ = eval_env.step(action_mix, disturbance=disturbance, dist_mag=dist_mag)
            if eval_timesteps > 70:
                avg_reward += reward
    avg_reward /= eval_episodes
    eval_reward.append(avg_reward)
    print("--------------------------------------------------------------------")
    print("eval_reward:", round(avg_reward, 2), "\t\t\t\t eval_alarms:", round(eval_env.alarms[-1], 2))
    print("--------------------------------------------------------------------")
    eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/Scope 5', 'commented', 'on', nargout=0)
    eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/Scope 6', 'commented', 'on', nargout=0)
    return round(avg_reward, 2), round(eval_env.alarms[-1], 2)


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_action = float(env.action_space.low[0])

total_warmup = 1e5
expl_noise = 0.01
batch_size = 128
epsilon_action = 0.2
alpha_action = 0.8
beta_mix = 0.5
gamma_mix = 0.3
replay_buffer = ReplayBuffer(state_dim, action_dim)
policy = TD3(state_dim,
             action_dim,
             max_action,
             min_action,
             batch_size,
             replay_buffer,
             discount=0.99,
             tau=0.005,
             policy_noise=0.02,
             noise_clip=0.05,
             policy_freq=2)

state, done = env.reset()
episode_reward = 0
episode_timesteps = 0
episode_num = 0
episode_rewards = []
c = 0
for t in range(int(env.max_episode_steps * 1e7)):
    episode_timesteps += 1
    disturbance = True

    z = np.random.uniform(0, 1)
    if z < epsilon_action:
        action_mix = np.array([np.random.uniform(0, 1)])
        action = action_mix
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/PID-DRPRL', 'sw', str(0), nargout=0)
    else:
        PID = eng1.workspace['input']._data[2] / 100
        DRPRL = (policy.select_action(np.reshape(state, (1, env.process_history, env.feat))) +
                 np.random.normal(0, max_action * expl_noise, size=action_dim)).clip(min_action, max_action)
        action_mix = DRPRL + PID
        action = DRPRL
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/PID-DRPRL', 'sw', str(0), nargout=0)

    next_state, reward, done, _ = env.step(action_mix, disturbance=disturbance, dist_mag=0.65)
    done_bool = float(done) if episode_timesteps < env.max_episode_steps else 0

    replay_buffer.add(np.reshape(state, (env.process_history, env.feat)), action,
                      np.reshape(next_state, (env.process_history, env.feat)), reward, done_bool)

    state = next_state
    if episode_timesteps > 70:
        episode_reward += reward

    policy.train(replay_buffer, batch_size=batch_size)

    if done:
        episode_rewards.append(episode_reward)
        print("episode:", episode_num,
              "\t\t episode reward:", round(episode_rewards[-1], 2),
              "\t\t alarms:", round(env.alarms[-1], 2))
        if episode_num % 5 == 0:
            evaluations = eval_policy(policy, dist_mag=0.65, eval_episodes=1)
            if evaluations[1] == 0.0:
                pass
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
        state, done = env.reset()

plt.plot(pd.DataFrame(episode_rewards).rolling(50).mean())
plt.plot(pd.DataFrame(env.alarms).rolling(50).mean())

print("Training finished.\n")

############################################################################################################
# **********************************************************************************************************
#                                              EVALUATION
# **********************************************************************************************************
# ##########################################################################################################


# ********************************* TD3 **************************************

evaluations = eval_policy(policy)

eng1.quit()
# eng2.quit()
env.close()
print("done")
