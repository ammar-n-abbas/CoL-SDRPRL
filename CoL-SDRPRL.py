############################################################################################################
#                                           IMPORTING LIBRARIES
# ##########################################################################################################

import warnings
import time

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
standard_err = StandardScaler()
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
import ast
import matlab
import matlab.engine

from itertools import chain, repeat, islice
from sklearn.decomposition import PCA


def pad_infinite(iterable, padding=None):
    return chain(iterable, repeat(padding))


def pad(iterable, size, padding=None):
    return islice(pad_infinite(iterable, padding), size)


''' run  this command in Matlab '''
# matlab.engine.shareEngine()

# eng = matlab.engine.start_matlab()
name_eng = matlab.engine.find_matlab()
eng1 = matlab.engine.connect_matlab(name_eng[0])
eng2 = matlab.engine.connect_matlab(name_eng[1])

Y_Thresh = pd.read_csv(r".\PV_thresh.csv", header=0, delimiter=';')
Y_Thresh = Y_Thresh[:40]
U_Thresh = pd.read_csv(r".\CV_thresh.csv", header=0, delimiter=',')

xmv = np.load(r".\xmv_full_1000000.npy") / 100.0
xmv = np.expand_dims(np.delete(xmv, [4, 8, 11], 1)[:, 2], 1)
xmeas = np.load(r".\xmeas_full_1000000.npy")
xmeas = np.delete(xmeas, 18, 1)

xmv_IOHMM = np.load(r".\xmv_IOHMM.npy") / 100.0
xmv_IOHMM = np.expand_dims(np.delete(xmv_IOHMM, [4, 8, 11], 1)[:, 2], 1)
xmeas_IOHMM = np.load(r".\xmeas_IOHMM.npy")
xmeas_IOHMM = np.delete(xmeas_IOHMM, 18, 1)

n_dim = 20
pca = PCA(n_components=n_dim)
pca2 = PCA(n_components=n_dim)

xmeas_std = standard.fit_transform(xmeas)
xmeas_PCA = pca.fit_transform(xmeas_std)
xmeas_IOHMM_std = standard.transform(xmeas_IOHMM)
xmeas_IOHMM_PCA = pca.transform(xmeas_IOHMM_std)

init_corr = [0.0] * 12
eng1.set_param('MultiLoop_mode1_DRL/TE Plant/PID-DRPRL', 'sw', str(1), nargout=0)
eng1.set_param('MultiLoop_mode1_DRL/TE Plant/xmv_corr', 'Value', str(init_corr), nargout=0)

##################################################### IOHMM #####################################################

from IOHMM import UnSupervisedIOHMM
from IOHMM import OLS, CrossEntropyMNL

df_obs = pd.DataFrame(xmeas_IOHMM)
outputs = [[m] for m in df_obs.head()]

num_states = 4
IOHMM = UnSupervisedIOHMM(num_states=num_states, EM_tol=1e-2)
IOHMM.set_models(
    model_emissions=[OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                     OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                     OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                     OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                     OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                     OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                     OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                     OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                     OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                     OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), ],
    model_transition=CrossEntropyMNL(solver='lbfgs'),
    model_initial=CrossEntropyMNL(solver='lbfgs'))
IOHMM.set_inputs(covariates_initial=[], covariates_transition=[], covariates_emissions=[[]] * len(outputs))
IOHMM.set_outputs(outputs)
IOHMM.set_data([df_obs])
IOHMM.train()

state_pred_seq = np.argmax(np.exp(IOHMM.log_gammas[0]), 1)

fig, ax = plt.subplots()
ax.plot(xmeas_IOHMM[:, 0], color='red')
ax.tick_params(axis='y', labelcolor='red')
ax2 = ax.twinx()
ax2.plot(state_pred_seq, color='green')
ax2.tick_params(axis='y', labelcolor='green')
plt.show()

print("IOHMM Training complete")


def get_reward(state):
    return -np.sum(abs(state - Y_Thresh['Normal'].to_numpy()))


upper_bound = 1.0
scale_factor = 100.0


class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, test=False, reset_orig=False, run_orig=False, sim_time=10):
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
        eng1.set_param('MultiLoop_mode1_DRL', 'SimulationCommand', 'stop', nargout=0)
        eng2.set_param('Copy_of_MultiLoop_mode1', 'SimulationCommand', 'stop', nargout=0)
        self.disturbance = [0, 0, 0, 0, 0, 0, 0, 0]
        self.disturbances = self.disturbance + ([0] * 20)
        eng1.set_param('MultiLoop_mode1_DRL/Disturbances', 'Value', str(self.disturbances), nargout=0)
        eng2.set_param('Copy_of_MultiLoop_mode1/Disturbances', 'Value', str(self.disturbances), nargout=0)
        self.sim_time = sim_time
        if self.test:
            self.sim_time = sim_time
        self.sim_step = 0.00
        self.step_size = 0.01
        self.step_step_size = 0.01
        eng1.set_param('MultiLoop_mode1_DRL', 'StopTime', str(self.sim_time), nargout=0)
        eng2.set_param('Copy_of_MultiLoop_mode1', 'StopTime', str(self.sim_time), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL', 'EnablePauseTimes', 'on', nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL', 'PauseTimes', str(self.sim_step), nargout=0)
        eng2.set_param('Copy_of_MultiLoop_mode1', 'EnablePauseTimes', 'on', nargout=0)
        eng2.set_param('Copy_of_MultiLoop_mode1', 'PauseTimes', str(self.sim_step), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL/TE Plant/PID-DRPRL', 'sw', str(1), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL/TE Plant/xmv_corr', 'Value', str(init_corr), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL', 'SimulationCommand', 'start', 'SimulationCommand', 'pause', nargout=0)
        eng2.set_param('Copy_of_MultiLoop_mode1', 'SimulationCommand', 'start', 'SimulationCommand', 'pause', nargout=0)
        self.process_history = 2
        self.state = []
        self.pv = []
        self.mv = []
        self.rp = []
        mv = 1
        pv = n_dim
        dev = 0
        self.feat = mv + pv + dev
        self.dim = 3
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=(self.feat,))
        self.action_space = gym.spaces.Box(low=0,
                                           high=upper_bound,
                                           shape=(mv,))
        self.reward = 0
        self.c_f = -100000
        self.prev_reward = [0]
        self.prev_state = (np.delete(eng1.workspace['output']._data, 18)).tolist()
        self.T = 0
        self.done = False
        self.max_episode_steps = int(self.sim_time / self.step_size)
        self.dist_bool = False
        self.state_seq = []

    def step(self, control, disturbance=False, dist_mag=None):
        mu = control * scale_factor
        control = np.insert(mu, [4, 7], 0, axis=0)
        control = np.insert(control, 11, 100, axis=0)
        control = matlab.double(control.tolist())
        eng1.set_param('MultiLoop_mode1_DRL/TE Plant/xmv_corr', 'Value', str(control), nargout=0)

        self.sim_step += self.step_size
        eng1.set_param('MultiLoop_mode1_DRL', 'PauseTimes', str(round(self.sim_step, 2)), nargout=0)
        # if self.run_orig:
        eng2.set_param('Copy_of_MultiLoop_mode1', 'PauseTimes', str(round(self.sim_step, 2)), nargout=0)
        eng1.set_param("MultiLoop_mode1_DRL", 'SimulationCommand', 'continue', nargout=0)
        # if self.run_orig:
        eng2.set_param("Copy_of_MultiLoop_mode1", 'SimulationCommand', 'continue', nargout=0)

        if disturbance:
            if self.sim_time * self.dist_start < self.sim_step < self.sim_time * self.dist_stop:
                disturbance = [0, 0, 0, 0, 0, dist_mag, 0, 0]
                disturbances = disturbance + ([0] * 20)
                eng1.set_param('MultiLoop_mode1_DRL/Disturbances', 'Value', str(disturbances), nargout=0)
                self.dist_bool = True
                # if self.run_orig:
                eng2.set_param('Copy_of_MultiLoop_mode1/Disturbances', 'Value', str(disturbances), nargout=0)
            else:
                disturbance = [0, 0, 0, 0, 0, 0, 0, 0]
                disturbances = disturbance + ([0] * 20)
                eng1.set_param('MultiLoop_mode1_DRL/Disturbances', 'Value', str(disturbances), nargout=0)
                self.dist_bool = False
                # if self.run_orig:
                eng2.set_param('Copy_of_MultiLoop_mode1/Disturbances', 'Value', str(disturbances), nargout=0)

        if self.sim_step >= self.sim_time:
            self.reward = 0
            self.done = True
            self.alarms.append(np.mean(self.num_alarms))
            self.alarms_seq.append(self.num_alarms)
            self.true_alarms.append(np.mean(self.true_num_alarms))
            self.run_orig = False
        elif eng1.get_param('MultiLoop_mode1_DRL', 'SimulationStatus') == ('stopped' or 'terminating'):
            eng1.set_param("MultiLoop_mode1_DRL", 'SimulationCommand', 'stop', nargout=0)
            self.reward = self.c_f
            self.done = True
            self.alarms.append(np.mean(self.num_alarms))
            self.alarms_seq.append(self.num_alarms)
            self.true_alarms.append(np.mean(self.true_num_alarms))
            self.run_orig = False
        else:
            self.mv = [eng1.workspace['drprl']._data[2] / 100.0]
            self.pv = np.squeeze(
                standard.transform(np.delete(eng1.workspace['output']._data, 18).reshape(1, -1))).tolist()
            self.pv = np.squeeze(pca.transform(np.array(self.pv).reshape(1, -1))).tolist()
            self.state = self.state[self.feat:] + self.mv + self.pv
            self.state_seq.append(np.delete(eng2.workspace['copy_of_output']._data, 18))
            self.reward = [get_reward((np.delete(eng1.workspace['output']._data, 18)).tolist())]
            self.prev_reward = self.reward
            self.prev_state = (np.delete(eng1.workspace['output']._data, 18)).tolist()

        # if self.run_orig:
        true_obs = eng2.workspace['copy_of_output']
        true_obs = np.delete(np.array(true_obs), 18)
        true_control_room = ~pd.Series(true_obs).between(Y_Thresh['LO-Alarm'].tolist(),
                                                         Y_Thresh['HI-Alarm'].tolist())
        true_num_alarms = sum(true_control_room)
        eng2.set_param('Copy_of_MultiLoop_mode1/TE Plant/num_alarms', 'Value', str(true_num_alarms), nargout=0)
        self.error.append(-(
            abs((np.delete(eng2.workspace['copy_of_output']._data, 18)).tolist() - Y_Thresh['Normal'].to_numpy())))
        self.true_num_alarms.append(true_num_alarms)

        control_room = ~pd.Series(np.delete(eng1.workspace['output']._data, 18).tolist()).between(
            Y_Thresh['LO-Alarm'].tolist(), Y_Thresh['HI-Alarm'].tolist())
        num_alarms = sum(control_room)
        op_screen = list(pad(control_room[control_room].index.values, 38, 0))
        eng1.set_param('MultiLoop_mode1_DRL/TE Plant/true_alarms', 'Value', str(op_screen), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL/TE Plant/num_alarms', 'Value', str(num_alarms), nargout=0)
        self.num_alarms.append(num_alarms)
        self.control_action.append(control[0])
        self.errorDRL.append(
            -(abs((np.delete(eng1.workspace['output']._data, 18)).tolist() - Y_Thresh['Normal'].to_numpy())))

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
        self.dist_start = np.random.uniform(0.3, 0.6)
        self.dist_stop = np.random.uniform(0.6, 0.8)

        eng1.set_param('MultiLoop_mode1_DRL', 'StopTime', str(self.sim_time), nargout=0)
        # if self.reset_orig:
        eng2.set_param('Copy_of_MultiLoop_mode1', 'StopTime', str(self.sim_time), nargout=0)
        self.disturbance = [0, 0, 0, 0, 0, 0, 0, 0]
        self.disturbances = self.disturbance + ([0] * 20)
        eng1.set_param('MultiLoop_mode1_DRL/Disturbances', 'Value', str(self.disturbances), nargout=0)
        # if self.reset_orig:
        eng2.set_param('Copy_of_MultiLoop_mode1/Disturbances', 'Value', str(self.disturbances), nargout=0)
        self.sim_step = 0.00
        eng1.set_param('MultiLoop_mode1_DRL', 'PauseTimes', str(self.sim_step), nargout=0)
        # if self.reset_orig:
        eng2.set_param('Copy_of_MultiLoop_mode1', 'PauseTimes', str(self.sim_step), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL/TE Plant/PID-DRPRL', 'sw', str(1), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL/TE Plant/xmv_corr', 'Value', str(init_corr), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL', 'SimulationCommand', 'start', 'SimulationCommand', 'pause', nargout=0)
        # if self.reset_orig:
        eng2.set_param('Copy_of_MultiLoop_mode1', 'SimulationCommand', 'start', 'SimulationCommand', 'pause', nargout=0)

        for i in range(self.process_history):
            self.mv = [np.delete([var / 100.0 for var in eng1.workspace['input']._data], [4, 8, 11]).tolist()[2]]
            self.pv = np.squeeze(
                standard.transform(np.delete(eng1.workspace['output']._data, 18).reshape(1, -1))).tolist()
            self.pv = np.squeeze(pca.transform(np.array(self.pv).reshape(1, -1))).tolist()
            self.state = self.state + self.mv + self.pv
            self.state_seq.append(np.delete(eng2.workspace['copy_of_output']._data, 18))

            eng1.set_param('MultiLoop_mode1_DRL/TE Plant/PID-DRPRL', 'sw', str(1), nargout=0)
            control = np.array([0.0] * 12)
            mu = control
            control = matlab.double(mu.tolist())
            eng1.set_param('MultiLoop_mode1_DRL/TE Plant/xmv_corr', 'Value', str(control), nargout=0)

            self.sim_step += self.step_size
            eng1.set_param('MultiLoop_mode1_DRL', 'PauseTimes', str(round(self.sim_step, 2)), nargout=0)
            # if self.reset_orig:
            eng2.set_param('Copy_of_MultiLoop_mode1', 'PauseTimes', str(round(self.sim_step, 2)), nargout=0)
            eng1.set_param("MultiLoop_mode1_DRL", 'SimulationCommand', 'continue', nargout=0)
            # if self.reset_orig:
            eng2.set_param("Copy_of_MultiLoop_mode1", 'SimulationCommand', 'continue', nargout=0)

        self.done = False
        self.reset_orig = False
        self.prev_state = (np.delete(eng1.workspace['output']._data, 18)).tolist()
        eng1.set_param('MultiLoop_mode1_DRL/TE Plant/PID-DRPRL', 'sw', str(0), nargout=0)
        return self.state, self.done


env = CustomEnv(sim_time=5)
eval_env = CustomEnv(test=True, reset_orig=False, run_orig=False, sim_time=5)

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

        self.state = np.zeros((max_size, env.process_history, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, env.process_history, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = np.array(reward)
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
        self.l0 = nn.LayerNorm(state_dim)
        self.l1 = nn.LSTM(state_dim, 64, batch_first=True)
        self.l2 = nn.LSTM(64, 32, batch_first=True)
        self.l3 = nn.Linear(32, 1)

        self.max_action = max_action

    def forward(self, state):
        a = self.l1(state)[0]
        a = self.l2(a)[0][:, -1, :]
        af = self.l3(a)
        return self.max_action * torch.sigmoid(af)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l0 = nn.LayerNorm(env.process_history * state_dim + action_dim)
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

class Value(nn.Module):
    def __init__(self, state_dim):
        super(Value, self).__init__()
        self.l0 = nn.LayerNorm(state_dim * 2)
        self.l1 = nn.Linear(state_dim * 2, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, state):
        sa = self.l0(torch.flatten(state, 1, -1))

        v = F.relu(self.l1(sa))
        v = F.relu(self.l2(v))
        v = self.l3(v)
        return v


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

        self.max_action = max_action
        self.min_action = min_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_freq = policy_freq

        self.total_it = 0
        self.burn_in = 0
        self.warm_up = 1000
        self.burn_period = 0
        self.critic_warmup_online = 1000
        self.actor_bc_loss_hist = []
        self.critic_bc_loss_hist = []
        self.value_bc_loss_hist = []

        self.actor = [Actor(state_dim, action_dim, max_action).to(device) for _ in range(9)]
        self.bc_optimizer = [torch.optim.Adam(actor.parameters(), lr=3e-3) for actor in self.actor]
        self.actor_optimizer = [torch.optim.Adam(actor.parameters(), lr=3e-3, weight_decay=1e-2) for actor in
                                self.actor]
        self.actor_target = self.train_actor_bc(replay_buffer, batch_size=batch_size)

        self.critic = [Critic(state_dim, action_dim).to(device) for _ in range(9)]
        self.critic_optimizer = [torch.optim.Adam(critic.parameters(), lr=3e-4, weight_decay=1e-2) for critic in self.critic]
        # self.critic_target = self.train_critic_bc(replay_buffer, batch_size=batch_size)
        self.critic_target = copy.deepcopy(self.critic)

        self.value = [Value(state_dim).to(device) for _ in range(9)]
        self.value_optimizer = [torch.optim.Adam(value.parameters(), lr=3e-4, weight_decay=1e-2) for value in self.value]
        self.value_target = self.train_value_bc(replay_buffer, batch_size=batch_size)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return [actor(state).cpu().data.numpy().flatten() for actor in self.actor]

    def train_actor_bc(self, replay_buffer, batch_size=None):
        xmeas_s = np.expand_dims(xmeas[:, 0], 1)

        samples = list()
        length = env.process_history
        n = len(xmeas)
        for i in range(0, n):
            if i == n - length:
                break
            sample = np.concatenate((xmv[i:i + length], xmeas_PCA[i:i + length]), 1)
            samples.append(sample)
        samples = np.array(samples)

        expert_traj = xmv[length:]
        prev_reward = 0
        for i, (state, action) in enumerate(zip(samples, expert_traj)):
            if i >= len(samples) - 1:
                break
            next_state = samples[i + 1]
            reward = [get_reward(xmeas[length + i])]
            prev_reward = reward
            done_bool = 0
            replay_buffer.add(state, action, next_state, reward, done_bool)

        print("---------------------------- Behaviour Cloning ------------------------------")
        for i in range(self.warm_up):
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            for agent_id, actor in enumerate(self.actor):
                bc_loss = F.mse_loss(actor(state), action)
                self.bc_optimizer[agent_id].zero_grad()
                bc_loss.backward()
                self.bc_optimizer[agent_id].step()
                self.actor_bc_loss_hist.append(bc_loss.item())
        print("\nactor_loss", round(np.average(self.actor_bc_loss_hist[-50:]), 5))
        return copy.deepcopy(self.actor)


    def train_value_bc(self, replay_buffer, batch_size=None):
        for i in range(self.burn_period):
            state, _, next_state, reward, not_done = replay_buffer.sample(batch_size)

            with torch.no_grad():
                target = [value(next_state) for value in self.value]
                target_V = [torch.layer_norm(reward, reward.shape) + (not_done * self.discount * t_V) for t_V in target]

            current_V = [value(state) for value in self.value]
            value_loss = [F.mse_loss(c_V, t_V) for c_V, t_V in zip(current_V, target_V)]
            [value_optimizer.zero_grad() for value_optimizer in self.value_optimizer]
            [loss.backward() for loss in value_loss]
            [value_optimizer.step() for value_optimizer in self.value_optimizer]
            [self.value_bc_loss_hist.append(loss.item()) for loss in value_loss]
        print("\nvalue_loss", round(np.average(self.value_bc_loss_hist[-50:]), 5))
        print("\n-------------------------------------------------------------------")
        return copy.deepcopy(self.value)

    def train(self, replay_buffer_e, replay_buffer, batch_size=None):
        self.total_it += 1
        # _, action_e, _, _, _ = replay_buffer_e.sample(batch_size)
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = [(actor_target(next_state) + noise).clamp(self.min_action, self.max_action) for actor_target
                           in self.actor_target]

            target = [critic_target(next_state, next_action[i]) for i, critic_target in enumerate(self.critic_target)]
            target_Q1 = [t_Q[0] for t_Q in target]
            target_Q2 = [t_Q[1] for t_Q in target]
            target_Q = [torch.min(t_Q1, t_Q2) for t_Q1, t_Q2 in zip(target_Q1, target_Q2)]
            target_Q = [torch.layer_norm(reward, reward.shape) + (not_done * self.discount * t_Q) for t_Q in target_Q]

        current = [critic(state, action) for i, critic in enumerate(self.critic)]
        current_Q1 = [Q[0] for Q in current]
        current_Q2 = [Q[1] for Q in current]
        critic_loss = [F.mse_loss(Q1, Q) + F.mse_loss(Q2, Q) for Q, Q1, Q2 in zip(target_Q, current_Q1, current_Q2)]
        [critic_optimizer.zero_grad() for critic_optimizer in self.critic_optimizer]
        [loss.backward() for loss in critic_loss]
        [critic_optimizer.step() for critic_optimizer in self.critic_optimizer]

        if self.total_it > self.critic_warmup_online:
            if self.total_it % self.policy_freq == 0:
                actor_loss = [-critic.Q1(state, actor(state)).mean()
                              for actor, critic in zip(self.actor, self.critic)]
                [actor_optimizer.zero_grad() for actor_optimizer in self.actor_optimizer]
                [loss.backward() for loss in actor_loss]
                [actor_optimizer.step() for actor_optimizer in self.actor_optimizer]

                for agent_id in range(len(self.actor)):
                    for param, target_param in zip(self.critic[agent_id].parameters(),
                                                   self.critic_target[agent_id].parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    for param, target_param in zip(self.actor[agent_id].parameters(),
                                                   self.actor_target[agent_id].parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

############################################################################################################
# **********************************************************************************************************
#                                               TRAINING
# **********************************************************************************************************
# ##########################################################################################################

# Define and Train the agent
print('#############################################################################')
print("                                 Training                                    ")
print('#############################################################################', '\n')


############################################################################################################
# **********************************************************************************************************
#                                          Alarm Management
# ##########################################################################################################

# ********************************** TD3 *************************************

def eval_policy(policy, eval_episodes=1, dist_mag=None):
    eval_env.reset_orig = False
    eval_env.run_orig = False
    avg_reward = 0.
    eng1.set_param('MultiLoop_mode1_DRL/TE Plant/Scope 5', 'commented', 'on', nargout=0)
    eng1.set_param('MultiLoop_mode1_DRL/TE Plant/Scope 6', 'commented', 'on', nargout=0)
    state, done = eval_env.reset()
    eng1.set_param('MultiLoop_mode1_DRL/TE Plant/PID-DRPRL', 'sw', str(0), nargout=0)
    disturbance = True
    dist = False
    while not done:
        IOHMM.set_data([pd.DataFrame(eval_env.state_seq)])
        IOHMM.E_step()
        state_pred = np.argmax(np.exp(IOHMM.log_gammas[0]), 1)[-1]

        PID = np.delete(np.array(eng1.workspace['input']._data.tolist()) / 100.0, [4, 8, 11])
        DRPRL = np.concatenate(policy.select_action(np.reshape(state, (1, env.process_history, env.feat))))
        if state_pred == 3:
            dist = True
        if dist:
            PID[2] = DRPRL[2]
        action = PID
        state, reward, done, _ = eval_env.step(action, disturbance=disturbance, dist_mag=dist_mag)
        # avg_reward += reward
    avg_reward /= eval_episodes
    print("-----------------------------------------------------------------------------")
    print("eval_reward:", round(avg_reward, 2), "\t\t eval_alarms:", round(eval_env.alarms[-1], 2),
          "\t\t eval_runTime:", round(eval_env.sim_step, 2))
    print("-----------------------------------------------------------------------------")
    eng1.set_param('MultiLoop_mode1_DRL/TE Plant/Scope 5', 'commented', 'on', nargout=0)
    eng1.set_param('MultiLoop_mode1_DRL/TE Plant/Scope 6', 'commented', 'on', nargout=0)
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
replay_buffer_e = ReplayBuffer(state_dim, action_dim)
replay_buffer = ReplayBuffer(state_dim, action_dim)
policy = TD3(state_dim,
             action_dim,
             max_action,
             min_action,
             batch_size,
             replay_buffer_e,
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
dist = False
for t in range(int(env.max_episode_steps * 1e7)):
    episode_timesteps += 1
    disturbance = True

    IOHMM.set_data([pd.DataFrame(env.state_seq)])
    IOHMM.E_step()
    state_pred = np.argmax(np.exp(IOHMM.log_gammas[0]), 1)[-1]

    z = np.random.uniform(0, 1)
    if z < epsilon_action:
        PID = np.delete(np.array(eng1.workspace['input']._data.tolist()) / 100.0, [4, 8, 11])
        DRPRL = np.random.uniform(0, 1, size=9)
        if state_pred == 3:
            dist = True
        if dist:
            PID[2] = DRPRL[2]
    else:
        PID = np.delete(np.array(eng1.workspace['input']._data.tolist()) / 100.0, [4, 8, 11])
        DRPRL = (np.concatenate(policy.select_action(np.reshape(state, (1, env.process_history, env.feat)))) +
                 np.random.normal(0, max_action * expl_noise, size=action_dim)).clip(min_action, max_action)
        if state_pred == 3:
            dist = True
        if dist:
            PID[2] = DRPRL[2]
    action_mix = PID
    if dist:
        action = DRPRL
    else:
        action = PID
    eng1.set_param('MultiLoop_mode1_DRL/TE Plant/PID-DRPRL', 'sw', str(0), nargout=0)

    next_state, reward, done, _ = env.step(action_mix, disturbance=disturbance, dist_mag=0.65)
    done_bool = float(done)

    if dist:
        replay_buffer.add(np.reshape(state, (env.process_history, env.feat)),
                          action[2],
                          np.reshape(next_state, (env.process_history, env.feat)),
                          reward,
                          done_bool)

        state = next_state
        # episode_reward += reward
        policy.train(replay_buffer_e, replay_buffer, batch_size=batch_size)

    if done:
        episode_rewards.append(episode_reward)
        print("episode:", episode_num,
              "\t\t episode reward:", round(episode_rewards[-1], 2),
              "\t\t alarms:", round(env.alarms[-1], 2))
        if episode_num % 5 == 0:
            evaluations = eval_policy(policy, dist_mag=0.65)
            if evaluations[1] == 0.0:
                pass
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
        state, done = env.reset()
        dist = False

plt.plot(pd.DataFrame(episode_rewards).rolling(50).mean())
plt.plot(pd.DataFrame(env.alarms).rolling(50).mean())

print("Training finished.\n")


eng1.quit()
env.close()
print("done")
