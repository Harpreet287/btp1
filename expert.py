import airsim
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# === Expert data generation ===
def get_current_state(client, num_uavs=3):
    """
    Queries AirSim and returns a flattened state vector:
    [x,y,z,vx,vy,vz] for each UAV.
    """
    states = []
    for i in range(num_uavs):
        s = client.getMultirotorState(vehicle_name=f"UAV{i}")
        p = s.kinematics_estimated.position
        v = s.kinematics_estimated.linear_velocity
        states += [p.x_val, p.y_val, p.z_val, v.x_val, v.y_val, v.z_val]
    return np.array(states, dtype=np.float32)

# Placeholder for your VO-based planner
# You should replace this with your actual VO implementation

def expert_action(state, num_uavs=3):
    """
    Given the concatenated state, compute acceleration commands per UAV
    using your Velocity Obstacle planner.
    Returns a (num_uavs*3,) numpy array.
    """
    actions = np.zeros(num_uavs * 3, dtype=np.float32)
    # Example stub: hover in place (zero acceleration)
    # Replace with: actions = your_vo_planner(state)
    return actions

if __name__ == '__main__':
    # --- Generate and save expert demonstrations ---
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.reset()
    num_uavs = 3
    N = 5000  # number of timesteps to record
    obs_list, act_list = [], []
    state = get_current_state(client, num_uavs)
    for t in range(N):
        a = expert_action(state, num_uavs)
        obs_list.append(state)
        act_list.append(a)
        # apply actions
        for i in range(num_uavs):
            cmd = airsim.Vector3r(*a[3*i:3*(i+1)])
            client.moveByVelocityAsync(cmd.x_val, cmd.y_val, cmd.z_val,
                                       duration=0.1, vehicle_name=f"UAV{i}")
        client.simPause(False)
        airsim.time.sleep(0.1)
        client.simPause(True)
        state = get_current_state(client, num_uavs)
    obs_arr = np.stack(obs_list)
    act_arr = np.stack(act_list)
    np.savez('expert.npz', obs=obs_arr, acts=act_arr)

# === Environment wrapper ===
class MultiUAVEnv(gym.Env):
    def __init__(self, num_uavs=3):
        super().__init__()
        self.num_uavs = num_uavs
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        obs_dim = num_uavs * 6  # [x,y,z,vx,vy,vz]
        act_dim = num_uavs * 3  # accel commands
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
            shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0,
            shape=(act_dim,), dtype=np.float32)

    def reset(self):
        self.client.reset()
        return self._get_obs()

    def step(self, action):
        for i in range(self.num_uavs):
            idx = slice(3*i, 3*(i+1))
            cmd = airsim.Vector3r(*action[idx])
            self.client.moveByVelocityAsync(cmd.x_val,
                cmd.y_val, cmd.z_val,
                duration=0.1, vehicle_name=f"UAV{i}")
        self.client.simPause(False)
        airsim.time.sleep(0.1)
        self.client.simPause(True)
        return self._get_obs(), 0.0, False, {}

    def _get_obs(self):
        return get_current_state(self.client, self.num_uavs)

# === Networks (Policy with value head + Discriminator) ===
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=[256,256]):
        super().__init__()
        layers, prev = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.net = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(prev, act_dim)
        self.logstd = nn.Parameter(torch.zeros(act_dim))
        self.value_layer = nn.Linear(prev, 1)

    def forward(self, x):
        h = self.net(x)
        return (self.mu_layer(h), torch.exp(self.logstd),
                self.value_layer(h).squeeze(-1))

class Discriminator(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=[256,256]):
        super().__init__()
        layers, prev = [], obs_dim + act_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], -1))

# === Training imports ===
import torch.nn.utils as nn_utils
from torch.optim.lr_scheduler import StepLR

# === Load expert dataset ===
expert = np.load('expert.npz')
expert_obs = torch.tensor(expert['obs'], dtype=torch.float32)
expert_act = torch.tensor(expert['acts'], dtype=torch.float32)

# === Initialize ===
env = MultiUAVEnv(num_uavs=3)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
policy = PolicyNet(obs_dim, act_dim)
disc = Discriminator(obs_dim, act_dim)
policy_opt = optim.Adam(policy.parameters(), lr=3e-4)
disc_opt = optim.Adam(disc.parameters(), lr=3e-4)

# === (rest of GAIL+PPO training loop as before) ===
# ...
