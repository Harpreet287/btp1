# import torch
# from gail import MultiUAVEnv, PolicyNet
# import numpy as np
# import airsim

# def test(ep=10):
#     env = MultiUAVEnv()
#     obs_dim = env.observation_space.shape[0]
#     act_dim = env.action_space.shape[0]
#     policy = PolicyNet(obs_dim,act_dim)
#     policy.load_state_dict(torch.load('policy.pth'))
#     policy.eval()
#     for e in range(ep):
#         obs = env.reset()
#         done=False
#         trajs=[]
#         for t in range(200):
#             x = torch.tensor(obs,dtype=torch.float32)
#             mu, std, _ = policy(x)
#             act = mu.detach().numpy()
#             obs, _, _, _ = env.step(act)
#             trajs.append(obs)
#         print(f"Episode {e} done, trajectory length {len(trajs)}")

# if __name__=='__main__':
#     test()

import torch
from gail import MultiUAVEnv, PolicyNet
import numpy as np
import airsim

def test(ep=10):
    env = MultiUAVEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = PolicyNet(obs_dim,act_dim)
    policy.load_state_dict(torch.load('policy.pth'))
    policy.eval()
    for e in range(ep):
        obs = env.reset()
        done=False
        trajs=[]
        for t in range(200):
            x = torch.tensor(obs,dtype=torch.float32)
            mu, std, _ = policy(x)
            act = mu.detach().numpy()        # now 3D velocities per UAV
            obs, _, _, _ = env.step(act)
            trajs.append(obs)                # obs includes x,y,z,vx,vy,vz for each UAV + 3D goals
        print(f"Episode {e} done, trajectory length {len(trajs)}")

if __name__=='__main__':
    test()
