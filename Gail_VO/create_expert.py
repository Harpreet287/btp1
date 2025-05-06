# import airsim
# import numpy as np
# from VO import DroneExpert

# # --- Expert data generation ---
# def get_state(client, num_uavs):
#     states = []
#     for i in range(num_uavs):
#         s = client.getMultirotorState(vehicle_name=f"UAV{i}")
#         p = s.kinematics_estimated.position
#         v = s.kinematics_estimated.linear_velocity
#         states += [p.x_val, p.y_val, v.x_val, v.y_val]
#     return np.array(states, dtype=np.float32)

# def main():
#     client = airsim.MultirotorClient(ip="192.168.244.207",port=41451)
#     client.confirmConnection()
#     client.reset()
#     num_uavs = 3
#     # Define fixed goals per UAV
#     goals = [np.array([100,100]), np.array([0,10]), np.array([5,15])]
#     obstacles = [ (np.array([4,4]),1.5), (np.array([7,7]),2.0) ]
#     experts = [DroneExpert([0,0], goals[0]),
#                DroneExpert([10,0], goals[1]),
#                DroneExpert([5,0], goals[2])]
#     for ex in experts: ex.plan(obstacles)

#     obs_buf = []
#     act_buf = []
#     steps = 1000
#     for t in range(steps):
#         print(t)
#         st = get_state(client, num_uavs)
#         # record state + goals
#         obs = np.concatenate([st] + [g for g in goals])
#         # compute expert actions
#         acts = []
#         for ex in experts:
#             a = ex.compute_action(experts, obstacles)
#             acts.append(a)
#         act = np.concatenate(acts)
#         obs_buf.append(obs)
#         act_buf.append(act)
#         # send to AirSim
#         for i in range(num_uavs):
#             vx, vy, vz = act[3*i:3*i+3]
#             client.moveByVelocityAsync(float(vx), float(vy), float(vz), duration=0.1, vehicle_name=f"UAV{i}")
#         client.simPause(False)
#         airsim.time.sleep(0.1)
#         client.simPause(True)

#     np.savez('expert1.npz', obs=np.stack(obs_buf), acts=np.stack(act_buf))

# if __name__=='__main__':
#     main()


import airsim
import numpy as np
from VO import DroneExpert

# --- Expert data generation ---
def get_state(client, num_uavs):
    states = []
    for i in range(num_uavs):
        s = client.getMultirotorState(vehicle_name=f"UAV{i}")
        p = s.kinematics_estimated.position
        v = s.kinematics_estimated.linear_velocity
        # include x, y, z and vx, vy, vz
        states += [p.x_val, p.y_val, p.z_val, v.x_val, v.y_val, v.z_val]
    return np.array(states, dtype=np.float32)

def main():
    client = airsim.MultirotorClient(ip="192.168.244.207",port=41451)
    client.confirmConnection()
    client.reset()
    num_uavs = 3
    # Define fixed 3D goals per UAV
    goals = [np.array([10,10,-2]), np.array([0,10,-2]), np.array([5,15,-2])]
    obstacles = [ (np.array([4,4,0]),1.5), (np.array([7,7,0]),2.0) ]
    experts = [DroneExpert([0,0,-2], goals[0]),
               DroneExpert([10,0,-2], goals[1]),
               DroneExpert([5,5,-2], goals[2])]
    for ex in experts: ex.plan(obstacles)

    obs_buf = []
    act_buf = []
    steps = 1000
    for t in range(steps):
        print(t)
        st = get_state(client, num_uavs)
        obs = np.concatenate([st] + [g for g in goals])
        acts = [ex.compute_action(experts, obstacles) for ex in experts]
        act = np.concatenate(acts)
        obs_buf.append(obs)
        act_buf.append(act)
        # send to AirSim
        for i in range(num_uavs):
            vx, vy, vz = act[3*i:3*i+3]
            client.moveByVelocityAsync(float(vx), float(vy), float(vz), duration=0.1, vehicle_name=f"UAV{i}")
        client.simPause(False)
        airsim.time.sleep(0.1)
        client.simPause(True)

    print(obs_buf)
    print(act_buf)
    np.savez('expert.npz', obs=np.stack(obs_buf), acts=np.stack(act_buf))

if __name__=='__main__':
    main()