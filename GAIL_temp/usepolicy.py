import airsim
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import time

# === Environment ===
class MultiUAVEnv(gym.Env):
    def __init__(self, num_uavs=3):
        super().__init__()
        self.num_uavs = num_uavs
        self.client = airsim.MultirotorClient(ip="192.168.1.9", port=41451)
        self.client.confirmConnection()
        self.client.simPause(True)
        obs_dim = num_uavs * 6
        act_dim = num_uavs * 3
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)

    # def reset(self):
    #     self.client.reset()
    #     for i in range(self.num_uavs):
    #         self.client.enableApiControl(True, vehicle_name=f"UAV{i}")
    #         self.client.armDisarm(True, vehicle_name=f"UAV{i}")

    #         state = self.client.getMultirotorState(vehicle_name=f"UAV{i}")
    #         z = state.kinematics_estimated.position.z_val
    #         altitude = -z  # Make it positive for human readability
    #         print(f"UAV{i} Altitude: {altitude:.2f} meters")

    #         self.client.takeoffAsync(vehicle_name=f"UAV{i}").join()
    #         state = self.client.getMultirotorState(vehicle_name=f"UAV0")
    #         z = state.kinematics_estimated.position.z_val
    #         altitude = -z  # Make it positive for human readability
    #         print(f"UAV{i} Altitude: {altitude:.2f} meters")
    #         print(i)
    #     return self._get_obs()


    def reset(self):
        self.client.reset()
        self.client.simPause(False)  # ✅ UNPAUSE before takeoff
    
        for i in range(self.num_uavs):
            vehicle_name = f"UAV{i}"
            self.client.enableApiControl(True, vehicle_name=vehicle_name)
            self.client.armDisarm(True, vehicle_name=vehicle_name)
    
            # Print pre-takeoff altitude
            state = self.client.getMultirotorState(vehicle_name=vehicle_name)
            z = state.kinematics_estimated.position.z_val
            print(f"{vehicle_name} Altitude before takeoff: {-z:.2f} meters")
    
            # Issue takeoff and wait for it to complete
            print(f"{vehicle_name}: Initiating takeoff...")
            self.client.takeoffAsync(vehicle_name=vehicle_name).join()
    
            # Monitor post-takeoff altitude
            max_wait = 10
            interval = 0.5
            waited = 0
            while waited < max_wait:
                state = self.client.getMultirotorState(vehicle_name=vehicle_name)
                altitude = -state.kinematics_estimated.position.z_val
                print(f"{vehicle_name} current altitude: {altitude:.2f} meters")
                if altitude > 1.0:
                    print(f"{vehicle_name}: Takeoff successful.")
                    break
                time.sleep(interval)
                waited += interval
            else:
                print(f"{vehicle_name}: Takeoff timeout or failure.")
            
            print(i)
    
        self.client.simPause(True)  # ✅ Re-pause after takeoffs
        return self._get_obs()

    def step(self, action):
        for i in range(self.num_uavs):
            idx = slice(3*i, 3*(i+1))
            vx, vy, vz = map(float, action[idx])
            self.client.moveByVelocityAsync(vx, vy, vz, duration=0.1, vehicle_name=f"UAV{i}")
        self.client.simPause(False)
        time.sleep(0.1)
        self.client.simPause(True)
        obs = self._get_obs()
        return obs, 0.0, False, {}
    
    def _get_obs(self):
        states = []
        for i in range(self.num_uavs):
            s = self.client.getMultirotorState(vehicle_name=f"UAV{i}")
            p = s.kinematics_estimated.position
            v = s.kinematics_estimated.linear_velocity
            states += [p.x_val, p.y_val, p.z_val, v.x_val, v.y_val, v.z_val]
        return np.array(states, dtype=np.float32)

# === Policy Network (same as training) ===
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=[256, 256]):
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

# === Load trained policy and run ===
if __name__ == "__main__":
    num_uavs = 3
    env = MultiUAVEnv(num_uavs=num_uavs)
    state = env.client.getMultirotorState(vehicle_name="UAV1")
    z = state.kinematics_estimated.position.z_val
    altitude = -z  # Make it positive for human readability
    print(f"UAV0 Altitude: {altitude:.2f} meters")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    policy = PolicyNet(obs_dim, act_dim)
    policy.load_state_dict(torch.load("policy.pth"))
    policy.eval()
    print("here")
    obs = env.reset()
    positions = []

    for t in range(500):  # Adjust as needed
        print(t)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            mu, _, _ = policy(obs_tensor)
        action = mu.numpy()
        obs, _, _, _ = env.step(action)
        positions.append(obs)

    np.savetxt("trajectory.csv", np.array(positions), delimiter=",")
    print("Evaluation complete. Trajectory saved to trajectory.csv")
