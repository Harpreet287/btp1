import airsim
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# === Environment wrapper ===
class MultiUAVEnv(gym.Env):
    def __init__(self, num_uavs=3):
        super().__init__()
        self.num_uavs = num_uavs
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        obs_dim = num_uavs * 6  # [x,y,z,vx,vy,vz] per UAV
        act_dim = num_uavs * 3  # acceleration commands per UAV
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)

    def reset(self):
        self.client.reset()
        return self._get_obs()

    def step(self, action):
        for i in range(self.num_uavs):
            idx = slice(3*i, 3*(i+1))
            cmd = airsim.Vector3r(*action[idx])
            self.client.moveByVelocityAsync(cmd.x_val, cmd.y_val, cmd.z_val,
                                            duration=0.1, vehicle_name=f"UAV{i}")
        self.client.simPause(False)
        airsim.time.sleep(0.1)
        self.client.simPause(True)
        obs = self._get_obs()
        # reward is provided by discriminator externally
        return obs, 0.0, False, {}

    def _get_obs(self):
        states = []
        for i in range(self.num_uavs):
            s = self.client.getMultirotorState(vehicle_name=f"UAV{i}")
            p = s.kinematics_estimated.position
            v = s.kinematics_estimated.linear_velocity
            states += [p.x_val, p.y_val, p.z_val, v.x_val, v.y_val, v.z_val]
        return np.array(states, dtype=np.float32)

# === Networks with value head for PPO ===
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=[256,256]):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.net = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(prev, act_dim)
        self.logstd = nn.Parameter(torch.zeros(act_dim))
        self.value_layer = nn.Linear(prev, 1)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_layer(h)
        std = torch.exp(self.logstd)
        value = self.value_layer(h).squeeze(-1)
        return mu, std, value

class Discriminator(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=[256,256]):
        super().__init__()
        layers = []
        prev = obs_dim + act_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)

# === Load expert dataset ===
data = np.load('expert.npz')
expert_obs = torch.tensor(data['obs'], dtype=torch.float32)
expert_act = torch.tensor(data['acts'], dtype=torch.float32)

# === Initialize environment, models, and optimizers ===
env = MultiUAVEnv(num_uavs=3)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
policy = PolicyNet(obs_dim, act_dim)
disc = Discriminator(obs_dim, act_dim)
policy_opt = optim.Adam(policy.parameters(), lr=3e-4)
disc_opt   = optim.Adam(disc.parameters(), lr=3e-4)

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
    for _ in range(steps):
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
for iteration in range(1000):
    # 1) Collect trajectories
    buf = collect_trajectories(env, policy)
    adv, returns = compute_gae(buf, disc)

    # 2) Discriminator updates
    obs_buf = torch.tensor(buf['obs'], dtype=torch.float32)
    act_buf = torch.tensor(buf['acts'], dtype=torch.float32)
    for _ in range(5):
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
        for start in range(0, len(ad
        vs_t), batch_size):
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

# Save models
torch.save(policy.state_dict(), 'policy.pth')
torch.save(disc.state_dict(), 'disc.pth')
