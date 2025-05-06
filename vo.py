import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Drone:
    def __init__(self, start, goal, radius=15, max_speed=1.0):
        self.pos = np.array(start, dtype=np.float64)
        self.goal = np.array(goal, dtype=np.float64)
        self.vel = np.zeros(2)
        self.radius = radius
        self.max_speed = max_speed
        self.path = [self.pos.copy()]

    def preferred_velocity(self):
        to_goal = self.goal - self.pos
        dist = np.linalg.norm(to_goal)
        return to_goal / dist * self.max_speed if dist > 0 else np.zeros(2)

    def avoid_others(self, others):
        v_pref = self.preferred_velocity()
        avoid_force = np.zeros(2)
        for other in others:
            if other is self:
                continue
            offset = other.pos - self.pos
            dist = np.linalg.norm(offset)
            if dist < self.radius * 3:
                avoid_force -= offset / (dist**2 + 1e-6)
        return v_pref + avoid_force

    def step(self, others, dt=0.1):
        self.vel = self.avoid_others(others)
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = self.vel / speed * self.max_speed
        self.pos += self.vel * dt
        self.path.append(self.pos.copy())

    def reached_goal(self, threshold=0.5):
        return np.linalg.norm(self.goal - self.pos) < threshold


# Initialize drones
drones = [
    Drone(start=[0, 0], goal=[8, 8]),
    Drone(start=[8, 0], goal=[0, 8]),
    Drone(start=[4, -2], goal=[4, 10]),
    Drone(start=[-2, 5], goal=[10, 5]),
    Drone(start=[8, 8], goal=[0, 0])
]

# Set up figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-3, 12)
ax.set_ylim(-3, 12)
ax.set_title("Multi-UAV Collision Avoidance (Live Animation)")
ax.grid(True)

# Plot elements
dots = [ax.plot([], [], 'o')[0] for _ in drones]
goals = [ax.plot(drone.goal[0], drone.goal[1], 'x')[0] for drone in drones]
trails = [ax.plot([], [], '-', lw=1)[0] for _ in drones]

# Animation update function
def update(frame):
    for i, drone in enumerate(drones):
        if not drone.reached_goal():
            drone.step(drones)
        pos = drone.pos
        # dots[i].set_data(pos[0], pos[1])
        dots[i].set_data([pos[0]], [pos[1]])

        path = np.array(drone.path)
        trails[i].set_data(path[:, 0], path[:, 1])
    return dots + trails

# Animate!
anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
plt.show()
