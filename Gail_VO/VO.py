# import numpy as np
# import random

# class RRTNode:
#     def __init__(self, position):
#         self.position = np.array(position)
#         self.parent = None
#         self.cost = 0.0

# class RRT:
#     def __init__(self, start, goal, obstacles, bounds,
#                  step_size=0.5, max_iterations=2000, goal_sample_rate=0.1):
#         self.start = np.array(start)
#         self.goal = np.array(goal)
#         self.obstacles = obstacles
#         self.bounds = bounds  # [x_min, x_max, y_min, y_max]
#         self.step_size = step_size
#         self.max_iterations = max_iterations
#         self.goal_sample_rate = goal_sample_rate
#         self.nodes = []
#         self.path = []
#         self.safety_margin = 0.8

#     def build_tree(self):
#         self.nodes = [RRTNode(self.start)]
#         for i in range(self.max_iterations):
#             if random.random() < self.goal_sample_rate:
#                 rnd = self.goal
#             else:
#                 rnd = self.random_config()
#             nearest = self.nearest_node(rnd)
#             new_pt = self.steer(nearest.position, rnd)
#             if not self.collision_free(nearest.position, new_pt):
#                 continue
#             node = RRTNode(new_pt)
#             node.parent = nearest
#             node.cost = nearest.cost + np.linalg.norm(new_pt - nearest.position)
#             self.nodes.append(node)
#             if np.linalg.norm(new_pt - self.goal) < self.step_size:
#                 if self.collision_free(new_pt, self.goal):
#                     goal_node = RRTNode(self.goal)
#                     goal_node.parent = node
#                     self.nodes.append(goal_node)
#                     self.extract_path(goal_node)
#                     return True
#         # fallback to closest node
#         dists = [np.linalg.norm(n.position - self.goal) for n in self.nodes]
#         best = self.nodes[np.argmin(dists)]
#         self.extract_path(best)
#         return True

#     def random_config(self):
#         x = random.uniform(self.bounds[0], self.bounds[1])
#         y = random.uniform(self.bounds[2], self.bounds[3])
#         return np.array([x, y])

#     def nearest_node(self, pt):
#         dists = [np.linalg.norm(n.position - pt) for n in self.nodes]
#         return self.nodes[np.argmin(dists)]

#     def steer(self, p, q):
#         d = q - p
#         dist = np.linalg.norm(d)
#         if dist < self.step_size:
#             return q
#         return p + d / dist * self.step_size

#     def collision_free(self, p, q):
#         checks = max(10, int(np.linalg.norm(q-p)/0.1))
#         for i in range(checks+1):
#             t = i / checks
#             pt = p*(1-t) + q*t
#             for center, radius in self.obstacles:
#                 if np.linalg.norm(pt-center) <= radius + self.safety_margin:
#                     return False
#         return True

#     def extract_path(self, node):
#         pts = []
#         cur = node
#         while cur:
#             pts.append(cur.position)
#             cur = cur.parent
#         self.path = pts[::-1]
#         self.smooth_path()

#     def smooth_path(self):
#         if len(self.path) <= 2:
#             return
#         sm = [self.path[0]]
#         i = 0
#         while i < len(self.path)-1:
#             for j in range(len(self.path)-1, i, -1):
#                 if self.collision_free(self.path[i], self.path[j]):
#                     sm.append(self.path[j])
#                     i = j
#                     break
#         self.path = sm

# class DroneExpert:
#     def __init__(self, start, goal, radius=0.5, max_speed=1.0):
#         self.pos = np.array(start, dtype=np.float32)
#         self.goal = np.array(goal, dtype=np.float32)
#         self.radius = radius
#         self.max_speed = max_speed
#         self.planned = []
#         self.idx = 0

#     def plan(self, obstacles):
#         bounds = [-5,15,-5,15]
#         rrt = RRT(self.pos, self.goal, obstacles, bounds,
#                   step_size=1.0, max_iterations=5000, goal_sample_rate=0.1)
#         rrt.build_tree()
#         self.planned = rrt.path
#         self.idx = 0

#     def compute_action(self, others, obstacles):
#         # Preferred velocity to next waypoint
#         if self.idx < len(self.planned):
#             tgt = self.planned[self.idx]
#             d = tgt - self.pos
#             dist = np.linalg.norm(d)
#             if dist < 0.3:
#                 self.idx += 1
#             v_pref = d/dist*self.max_speed if dist>0 else np.zeros(2)
#         else:
#             d = self.goal - self.pos
#             dist = np.linalg.norm(d)
#             v_pref = d/dist*self.max_speed if dist>0 else np.zeros(2)
#         # VO avoidance: simple repulsion
#         avoid = np.zeros(2)
#         for o in others:
#             if o is self: continue
#             off = o.pos - self.pos
#             dist = np.linalg.norm(off)
#             if dist < self.radius*3:
#                 avoid -= off/(dist**2+1e-6)
#         vel = v_pref + 0.5*avoid
#         norm = np.linalg.norm(vel)
#         if norm>self.max_speed:
#             vel = vel/norm*self.max_speed
#         self.pos += vel*0.1
#         return np.array([vel[0], vel[1], 0.0], dtype=np.float32)


import numpy as np
import random

class RRTNode:
    def __init__(self, position):
        self.position = np.array(position)
        self.parent = None
        self.cost = 0.0

class RRT:
    def __init__(self, start, goal, obstacles, bounds,
                 step_size=0.5, max_iterations=2000, goal_sample_rate=0.1):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles  # list of (center(3,), radius)
        self.bounds = bounds        # [x_min,x_max,y_min,y_max,z_min,z_max]
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate
        self.nodes = []
        self.path = []
        self.safety_margin = 0.8

    def build_tree(self):
        self.nodes = [RRTNode(self.start)]
        for i in range(self.max_iterations):
            rnd = self.goal if random.random() < self.goal_sample_rate else self.random_config()
            nearest = self.nearest_node(rnd)
            new_pt = self.steer(nearest.position, rnd)
            if not self.collision_free(nearest.position, new_pt):
                continue
            node = RRTNode(new_pt)
            node.parent = nearest
            node.cost = nearest.cost + np.linalg.norm(new_pt - nearest.position)
            self.nodes.append(node)
            if np.linalg.norm(new_pt - self.goal) < self.step_size:
                if self.collision_free(new_pt, self.goal):
                    goal_node = RRTNode(self.goal)
                    goal_node.parent = node
                    self.nodes.append(goal_node)
                    self.extract_path(goal_node)
                    return True
        # fallback to closest node
        dists = [np.linalg.norm(n.position - self.goal) for n in self.nodes]
        best = self.nodes[np.argmin(dists)]
        self.extract_path(best)
        return True

    def random_config(self):
        x = random.uniform(self.bounds[0], self.bounds[1])
        y = random.uniform(self.bounds[2], self.bounds[3])
        z = random.uniform(self.bounds[4], self.bounds[5])
        return np.array([x, y, z])

    def nearest_node(self, pt):
        dists = [np.linalg.norm(n.position - pt) for n in self.nodes]
        return self.nodes[np.argmin(dists)]

    def steer(self, p, q):
        d = q - p
        dist = np.linalg.norm(d)
        if dist < self.step_size:
            return q
        return p + d / dist * self.step_size

    def collision_free(self, p, q):
        checks = max(10, int(np.linalg.norm(q - p) / 0.1))
        for i in range(checks + 1):
            t = i / checks
            pt = p * (1 - t) + q * t
            for center, radius in self.obstacles:
                if np.linalg.norm(pt - center) <= radius + self.safety_margin:
                    return False
        return True

    def extract_path(self, node):
        pts = []
        cur = node
        while cur:
            pts.append(cur.position)
            cur = cur.parent
        self.path = pts[::-1]
        self.smooth_path()

    def smooth_path(self):
        if len(self.path) <= 2:
            return
        sm = [self.path[0]]
        i = 0
        while i < len(self.path)-1:
            for j in range(len(self.path)-1, i, -1):
                if self.collision_free(self.path[i], self.path[j]):
                    sm.append(self.path[j])
                    i = j
                    break
        self.path = sm

class DroneExpert:
    def __init__(self, start, goal, radius=0.5, max_speed=1.0):
        self.pos = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.radius = radius
        self.max_speed = max_speed
        self.planned = []
        self.idx = 0

    def plan(self, obstacles):
        # Define 3D bounds matching environment
        bounds = [-5, 15, -5, 15, -5, 15]
        rrt = RRT(self.pos, self.goal, obstacles, bounds,
                  step_size=1.0, max_iterations=5000, goal_sample_rate=0.1)
        rrt.build_tree()
        self.planned = rrt.path
        self.idx = 0

    def compute_action(self, others, obstacles):
        # Preferred velocity to next waypoint
        if self.idx < len(self.planned):
            tgt = self.planned[self.idx]
            d = tgt - self.pos
            dist = np.linalg.norm(d)
            if dist < 0.3:
                self.idx += 1
            v_pref = (d/dist*self.max_speed) if dist>0 else np.zeros(3)
        else:
            d = self.goal - self.pos
            dist = np.linalg.norm(d)
            v_pref = (d/dist*self.max_speed) if dist>0 else np.zeros(3)
        # VO avoidance: simple repulsion in 3D
        avoid = np.zeros(3)
        for o in others:
            if o is self: continue
            off = o.pos - self.pos
            dist = np.linalg.norm(off)
            if dist < self.radius*3:
                avoid -= off/(dist**2+1e-6)
        vel = v_pref + 0.5*avoid
        norm = np.linalg.norm(vel)
        if norm > self.max_speed:
            vel = vel/norm*self.max_speed
        # update position
        self.pos += vel*0.1
        return vel.astype(np.float32)