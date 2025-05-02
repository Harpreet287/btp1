import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import math


class RRTNode:
    def __init__(self, position):
        self.position = np.array(position)
        self.parent = None
        self.cost = 0.0


class RRT:
    def __init__(self, start, goal, obstacles, bounds, step_size=0.5, max_iterations=2000, goal_sample_rate=0.1):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.bounds = bounds  # [x_min, x_max, y_min, y_max]
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate
        self.nodes = []
        self.path = []
        self.safety_margin = 0.8  # Add safety margin to obstacle collision detection

    def build_tree(self):
        start_node = RRTNode(self.start)
        self.nodes.append(start_node)

        for i in range(self.max_iterations):
            # Sample random point
            if random.random() < self.goal_sample_rate:
                random_point = self.goal
            else:
                random_point = self.random_config()

            # Find nearest node
            nearest_node = self.nearest_node(random_point)

            # Steer towards the random point
            new_point = self.steer(nearest_node.position, random_point)

            # Check for collision
            if not self.collision_free(nearest_node.position, new_point):
                continue

            # Add new node
            new_node = RRTNode(new_point)
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + np.linalg.norm(new_point - nearest_node.position)
            self.nodes.append(new_node)

            # Check if goal reached
            dist_to_goal = np.linalg.norm(new_point - self.goal)
            if dist_to_goal < self.step_size:
                # First check if direct path to goal is collision free
                if self.collision_free(new_point, self.goal):
                    goal_node = RRTNode(self.goal)
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + dist_to_goal
                    self.nodes.append(goal_node)
                    self.extract_path(goal_node)
                    return True

        # If max iterations reached without finding path to goal
        # Find node closest to goal and use that
        if len(self.nodes) > 1:
            distances = [np.linalg.norm(node.position - self.goal) for node in self.nodes]
            closest_node = self.nodes[np.argmin(distances)]
            self.extract_path(closest_node)
            return True

        return False

    def random_config(self):
        x = random.uniform(self.bounds[0], self.bounds[1])
        y = random.uniform(self.bounds[2], self.bounds[3])
        return np.array([x, y])

    def nearest_node(self, point):
        distances = [np.linalg.norm(node.position - point) for node in self.nodes]
        return self.nodes[np.argmin(distances)]

    def steer(self, from_point, to_point):
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        if distance < self.step_size:
            return to_point
        else:
            return from_point + direction / distance * self.step_size

    def collision_free(self, from_point, to_point):
        # Check multiple points along the line segment
        num_checks = max(10, int(np.linalg.norm(to_point - from_point) / 0.1))

        for i in range(num_checks + 1):
            t = i / num_checks
            point = from_point * (1 - t) + to_point * t

            # Check if point is within any obstacle (plus safety margin)
            for center, radius in self.obstacles:
                if np.linalg.norm(point - center) <= radius + self.safety_margin:
                    return False

        return True

    def extract_path(self, goal_node):
        path = []
        current = goal_node
        while current is not None:
            path.append(current.position)
            current = current.parent
        self.path = path[::-1]  # Reverse to get path from start to goal

        # Path smoothing
        self.smooth_path()

    def smooth_path(self):
        """Apply path smoothing to remove unnecessary waypoints"""
        if len(self.path) <= 2:
            return

        smoothed_path = [self.path[0]]
        i = 0

        while i < len(self.path) - 1:
            current = self.path[i]

            # Try to connect with furthest possible node
            for j in range(len(self.path) - 1, i, -1):
                if self.collision_free(current, self.path[j]):
                    smoothed_path.append(self.path[j])
                    i = j
                    break
            i += 1

        self.path = smoothed_path


class Drone:
    def __init__(self, start, goal, radius=0.5, max_speed=1.0, color=None):
        self.pos = np.array(start, dtype=np.float64)
        self.goal = np.array(goal, dtype=np.float64)
        self.vel = np.zeros(2)
        self.radius = radius
        self.max_speed = max_speed
        self.path = [self.pos.copy()]
        self.planned_path = []
        self.path_index = 0
        self.color = color
        self.reached_end = False

    def plan_path(self, obstacles):
        # Create RRT planner
        bounds = [-5, 15, -5, 15]  # [x_min, x_max, y_min, y_max]
        rrt = RRT(self.pos, self.goal, obstacles, bounds,
                  step_size=1,  # Smaller step size for more precision
                  max_iterations=10000,  # More iterations for better coverage
                  goal_sample_rate=0.15)  # Slightly higher goal bias

        success = rrt.build_tree()
        if success:
            self.planned_path = rrt.path
            return True
        return False

    def preferred_velocity(self):
        if len(self.planned_path) > 0 and self.path_index < len(self.planned_path):
            # Follow the planned path
            target = self.planned_path[self.path_index]
            to_target = target - self.pos
            dist = np.linalg.norm(to_target)

            # If close to current waypoint, move to next one
            if dist < 0.3 and self.path_index < len(self.planned_path) - 1:
                self.path_index += 1
                target = self.planned_path[self.path_index]
                to_target = target - self.pos
                dist = np.linalg.norm(to_target)

            return to_target / dist * self.max_speed if dist > 0 else np.zeros(2)
        else:
            # Direct to goal if no path available
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
                avoid_force -= offset / (dist ** 2 + 1e-6)
        return v_pref + avoid_force * 0.5  # Reduce collision avoidance weight to follow path better

    def step(self, others, obstacles, dt=0.1):
        if self.reached_goal():
            self.reached_end = True
            return

        # Check for replanning if obstacle is in the way
        if not self.reached_end and len(self.planned_path) > 0:
            # Check if current path segment passes through any obstacle
            if self.path_index < len(self.planned_path) - 1:
                current_pos = self.pos
                next_waypoint = self.planned_path[self.path_index + 1]

                # Simple collision check
                for center, radius in obstacles:
                    if self.line_circle_collision(current_pos, next_waypoint, center, radius):
                        # Replan
                        self.plan_path(obstacles)
                        self.path_index = 0
                        break

        self.vel = self.avoid_others(others)
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = self.vel / speed * self.max_speed
        self.pos += self.vel * dt
        self.path.append(self.pos.copy())

    def line_circle_collision(self, line_start, line_end, circle_center, circle_radius):
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len if line_len > 0 else np.zeros(2)

        start_to_center = circle_center - line_start
        proj_len = np.dot(start_to_center, line_unitvec)

        if proj_len < 0:
            closest_point = line_start
        elif proj_len > line_len:
            closest_point = line_end
        else:
            closest_point = line_start + line_unitvec * proj_len

        dist_to_center = np.linalg.norm(circle_center - closest_point)
        return dist_to_center <= circle_radius + 0.2  # Add small safety margin

    def reached_goal(self, threshold=0.3):
        return np.linalg.norm(self.goal - self.pos) < threshold


# Define obstacles - each is (center, radius)
obstacles = [
    (np.array([4, 4]), 1.5),
    (np.array([7, 7]), 2.0),  # Larger central obstacle
    (np.array([3, 8]), 1.3),
]

# Initialize drones with different colors
colors = ['blue', 'orange', 'green', 'purple', 'red', 'cyan']
drones = [
    Drone(start=[0, 0], goal=[12, 10], color=colors[0]),
    Drone(start=[10, 0], goal=[2, 10], color=colors[1]),
    Drone(start=[6, 0], goal=[6, 12], color=colors[2]),
    Drone(start=[0, 6], goal=[12, 6], color=colors[3]),
    Drone(start=[12, 12], goal=[0, 0], color=colors[4]),
    Drone(start=[0, 12], goal=[12, 2], color=colors[5]),
]

# Plan paths for all drones
for i, drone in enumerate(drones):
    print(f"Planning path for drone {i + 1}...")
    drone.plan_path(obstacles)

# Set up figure and axis
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-2, 14)
ax.set_ylim(-2, 14)
ax.set_title("Multi-UAV Navigation with RRT Path Planning")
ax.grid(True)

# Plot obstacles
for center, radius in obstacles:
    circle = plt.Circle(center, radius, color='red', alpha=0.5)
    ax.add_patch(circle)

# Plot elements - make dots and lines match drone colors
dots = [ax.plot([], [], 'o', color=drone.color)[0] for drone in drones]
goals = [ax.plot(drone.goal[0], drone.goal[1], 'x', color=drone.color)[0] for drone in drones]
trails = [ax.plot([], [], '-', lw=1.5, color=drone.color)[0] for drone in drones]
planned_paths = [ax.plot([], [], '--', lw=1, color=drone.color, alpha=0.6)[0] for drone in drones]

# Plot starting positions
for i, drone in enumerate(drones):
    ax.plot(drone.pos[0], drone.pos[1], '*', color=drone.color, markersize=10)

# Plot planned paths
for i, drone in enumerate(drones):
    if len(drone.planned_path) > 0:
        path = np.array(drone.planned_path)
        planned_paths[i].set_data(path[:, 0], path[:, 1])


# Animation update function
def update(frame):
    # Run multiple steps per frame for smoother animation
    for _ in range(5):
        for i, drone in enumerate(drones):
            if not drone.reached_goal():
                drone.step(drones, obstacles, dt=0.02)

    for i, drone in enumerate(drones):
        pos = drone.pos
        dots[i].set_data([pos[0]], [pos[1]])

        path = np.array(drone.path)
        trails[i].set_data(path[:, 0], path[:, 1])

    return dots + trails + planned_paths


# Create animation
anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
plt.show()