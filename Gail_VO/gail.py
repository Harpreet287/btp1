# gail.py
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import airsim
from torch.distributions import Normal

class MultiUAVEnv(gym.Env):
    def __init__(self, num_uavs=3):
        super().__init__()
        self.num_uavs = num_uavs
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        # each UAV: x,y,z,vx,vy,vz + goal x,y,z
        obs_dim = num_uavs*(6 + 3)
        act_dim = num_uavs*3
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1,1,shape=(act_dim,),dtype=np.float32)
        self.goals = [np.array([10,10,-2]), np.array([0,10,-2]), np.array([5,15,-2])]

    def reset(self):
        self.client.reset()
        st = self._get_state()
        return np.concatenate([st] + [g for g in self.goals])

    def step(self, action):
        for i in range(self.num_uavs):
            vx,vy,vz = action[3*i:3*i+3]
            self.client.moveByVelocityAsync(vx,vy,vz,duration=0.1, vehicle_name=f"UAV{i}")
        self.client.simPause(False)
        airsim.time.sleep(0.1)
        self.client.simPause(True)
        st = self._get_state()
        obs = np.concatenate([st] + [g for g in self.goals])
        return obs, 0.0, False, {}

    def _get_state(self):
        s=[]
        for i in range(self.num_uavs):
            m = self.client.getMultirotorState(vehicle_name=f"UAV{i}")
            p = m.kinematics_estimated.position
            v = m.kinematics_estimated.linear_velocity
            s += [p.x_val, p.y_val, p.z_val, v.x_val, v.y_val, v.z_val]
        return np.array(s, dtype=np.float32)

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLU(),
        )
        self.mu = nn.Linear(256,act_dim)
        self.logstd = nn.Parameter(torch.zeros(act_dim))
        self.v = nn.Linear(256,1)
    def forward(self,x):
        h=self.net(x)
        return self.mu(h), torch.exp(self.logstd), self.v(h).squeeze(-1)

class Discriminator(nn.Module):
    def __init__(self, obs_dim,act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim+act_dim,256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLU(),
            nn.Linear(256,1), nn.Sigmoid()
        )
    def forward(self,o,a): return self.net(torch.cat([o,a],-1))

# load expert
data = np.load('expert.npz')
expert_obs = torch.tensor(data['obs'],dtype=torch.float32)
expert_act = torch.tensor(data['acts'],dtype=torch.float32)

# init
env = MultiUAVEnv()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
policy = PolicyNet(obs_dim,act_dim)
disc = Discriminator(obs_dim,act_dim)
policy_opt, disc_opt = optim.Adam(policy.parameters(),3e-4), optim.Adam(disc.parameters(),3e-4)

# === PPO hyperparameters ===
gamma = 0.99
lam = 0.95
ppo_epochs = 4
batch_size = 64
clip_eps = 0.2
vf_coef = 0.5
ent_coef = 0.0
max_grad_norm = 0.5

# === Trajectory collection ===
def collect_trajectories(env, policy, steps=2048):
    buf = {k:[] for k in ['obs','acts','logps','vals','rews','dones']}
    obs = env.reset()
    print(steps)
    for _ in range(steps):
        print(_)
        obs_t = torch.tensor(obs, dtype=torch.float32)
        mu, std, val = policy(obs_t)
        dist = Normal(mu, std)
        act = dist.sample()
        logp = dist.log_prob(act).sum()
        obs2, _, done, _ = env.step(act.numpy())
        buf['obs'].append(obs)
        buf['acts'].append(act.numpy())
        buf['logps'].append(logp.item())
        buf['vals'].append(val.item())
        buf['dones'].append(done)
        obs = obs2 if not done else env.reset()
    for k in buf:
        buf[k] = np.array(buf[k])
    return buf

# === GAE advantage calculation ===
def compute_gae(buf, disc):
    obs = torch.tensor(buf['obs'], dtype=torch.float32)
    acts = torch.tensor(buf['acts'], dtype=torch.float32)
    with torch.no_grad():
        D = disc(obs, acts).squeeze(-1)
        # imitation reward = log D(s,a)
        rewards = torch.log(D + 1e-8).numpy()
    values = buf['vals']
    dones = buf['dones']
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        nextval = values[t+1] if t < T-1 else 0
        delta = rewards[t] + gamma*nextval*nonterm - values[t]
        adv[t] = lastgaelam = delta + gamma*lam*nonterm*lastgaelam
    returns = adv + values
    return adv, returns

# === GAIL + PPO training loop ===
for iteration in range(100):
    print('iteration')
    print(iteration)
    # 1) Collect trajectories
    buf = collect_trajectories(env, policy)
    adv, returns = compute_gae(buf, disc)

    # 2) Discriminator updates
    obs_buf = torch.tensor(buf['obs'], dtype=torch.float32)
    act_buf = torch.tensor(buf['acts'], dtype=torch.float32)
    for _ in range(5):
        print(_)
        idx_p = np.random.randint(0, len(obs_buf), size=batch_size)
        idx_e = np.random.randint(0, len(expert_obs), size=batch_size)
        d_p = disc(obs_buf[idx_p], act_buf[idx_p])
        d_e = disc(expert_obs[idx_e], expert_act[idx_e])
        loss_d = - (torch.log(d_e + 1e-8).mean() + torch.log(1 - d_p + 1e-8).mean())
        disc_opt.zero_grad()
        loss_d.backward()
        disc_opt.step()

    # 3) PPO policy updates
    obs_t = torch.tensor(buf['obs'], dtype=torch.float32)
    acts_t = torch.tensor(buf['acts'], dtype=torch.float32)
    old_logp = torch.tensor(buf['logps'], dtype=torch.float32)
    adv_t = torch.tensor(adv, dtype=torch.float32)
    ret_t = torch.tensor(returns, dtype=torch.float32)
    # normalize advantage
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    for _ in range(ppo_epochs):
        idxs = np.random.permutation(len(adv_t))
        print(batch_size)
        print(len(adv_t))
        for start in range(0, len(adv_t), batch_size):
            print(start)
            mb = idxs[start:start+batch_size]
            mb_obs = obs_t[mb]
            mb_acts = acts_t[mb]
            mb_oldlogp = old_logp[mb]
            mb_adv = adv_t[mb]
            mb_ret = ret_t[mb]

            mu, std, val = policy(mb_obs)
            dist = Normal(mu, std)
            logp = dist.log_prob(mb_acts).sum(dim=1)
            ratio = torch.exp(logp - mb_oldlogp)
            # clipped surrogate
            s1 = ratio * mb_adv
            s2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * mb_adv
            policy_loss = -torch.min(s1, s2).mean()
            value_loss = (val - mb_ret).pow(2).mean()
            entropy = dist.entropy().sum(dim=1).mean()
            loss_p = policy_loss + vf_coef*value_loss - ent_coef*entropy

            policy_opt.zero_grad()
            loss_p.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            policy_opt.step()

    print(f"Iter {iteration} | D_loss {loss_d.item():.3f} | P_loss {policy_loss.item():.3f}")

torch.save(policy.state_dict(),'policy.pth')
torch.save(disc.state_dict(),'disc.pth')




# {
#   "SettingsVersion": 1.2,
#   "SimMode": "Multirotor",
#   "RpcPort": 41451,
#   "LocalHostIp": "192.168.244.207",
#   "ViewMode": "Manual",
#   "RpcEnabled": true,
#   "Vehicles": {
#     "UAV0": {"VehicleType": "SimpleFlight","AutoCreate": true,"DefaultVehicleState": "Inactive","X": 0, "Y": 0, "Z": -2,"EnableCollisionPassthrough": false},
#     "UAV1": {"VehicleType": "SimpleFlight","AutoCreate": true,"DefaultVehicleState": "Inactive","X": 10, "Y": 0, "Z": -2,"EnableCollisionPassthrough": false},
#     "UAV2": {"VehicleType": "SimpleFlight","AutoCreate": true,"DefaultVehicleState": "Inactive","X": 5, "Y": 5, "Z": -300,"EnableCollisionPassthrough": false}
#   }
# }