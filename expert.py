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

