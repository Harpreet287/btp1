import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import heapq
from torch.distributions import Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Environment Parameters
GRID_SIZE = 8
NUM_UAVS = 2
ACTION_SIZE = 5  # Up, Down, Left, Right, Stay
OBSERVATION_SIZE = GRID_SIZE * GRID_SIZE * 3  # Channels: UAVs, Goals, Obstacles

# Hyperparameters
HIDDEN_DIM = 512
GAMMA = 0.99
LAMBDA = 0.95
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4
EPISODES = 2000
MAX_STEPS = 50


class MultiUAVEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.num_uavs = NUM_UAVS
        self.obstacles = torch.tensor([[2, 2], [2, 5], [5, 2], [5, 5]], device=device)
        self.goals = torch.tensor([[0, 7], [7, 0]], device=device)
        self.reset()

    def reset(self):
        self.positions = torch.tensor([[0, 0], [7, 7]], dtype=torch.float32, device=device)
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        state = torch.zeros(3, GRID_SIZE, GRID_SIZE, device=device)
        for i in range(NUM_UAVS):
            x, y = self.positions[i].int()
            state[0, y, x] = 1  # UAV channel
        for goal in self.goals:
            gx, gy = goal.int()
            state[1, gy, gx] = 1  # goal channel
        for obs in self.obstacles:
            ox, oy = obs.int()
            state[2, oy, ox] = 1  # obstacle channel
        return state.flatten()

    def step(self, actions):
        rewards = torch.zeros(NUM_UAVS, device=device)
        new_positions = self.positions.clone()

        for i in range(NUM_UAVS):
            action = actions[i]
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)][action]
            new_pos = new_positions[i] + torch.tensor([dx, dy], dtype=torch.float32, device=device)

            if self._valid_move(new_pos, i):
                new_positions[i] = new_pos
                rewards[i] += 0.1

        self.positions = new_positions
        self.steps += 1

        done = self.steps >= MAX_STEPS
        for i in range(NUM_UAVS):
            if torch.all(torch.eq(new_positions[i], self.goals[i])):
                rewards[i] += 10
                done = True

        if self._check_collisions():
            rewards -= 5

        return self._get_state(), rewards, done, {}

    def _valid_move(self, pos, uav_id):
        if (pos < 0).any() or (pos >= GRID_SIZE).any():
            return False
        if any(torch.all(torch.eq(pos.int(), obs)) for obs in self.obstacles):
            return False
        for i in range(NUM_UAVS):
            if i != uav_id and torch.all(torch.eq(pos, self.positions[i])):
                return False
        return True

    def _check_collisions(self):
        for i in range(NUM_UAVS):
            for j in range(i + 1, NUM_UAVS):
                if torch.all(torch.eq(self.positions[i], self.positions[j])):
                    return True
        return False


class MultiAgentPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, HIDDEN_DIM),
            nn.ReLU()
        )
        self.actor_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, ACTION_SIZE)
            ) for _ in range(NUM_UAVS)
        ])
        self.to(device)

    def forward(self, x):
        shared = self.shared_net(x)
        return [Categorical(logits=head(shared)) for head in self.actor_heads]


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE + NUM_UAVS * ACTION_SIZE, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
            nn.Sigmoid()
        )
        self.to(device)

    def forward(self, state, actions):
        actions_onehot = torch.zeros(NUM_UAVS * ACTION_SIZE, device=state.device)
        for i, a in enumerate(actions):
            actions_onehot[i * ACTION_SIZE + a] = 1
        return self.net(torch.cat([state, actions_onehot]))


def astar(start, goal, obstacles, other_positions, grid_size):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    start = (start[0], start[1])
    goal = (goal[0], goal[1])
    open_set = []
    heapq.heappush(open_set, (0, start, []))
    visited = set()
    blocked = set(obstacles + other_positions)

    while open_set:
        current_cost, current_pos, path_actions = heapq.heappop(open_set)

        if current_pos in visited:
            continue
        visited.add(current_pos)

        if current_pos == goal:
            return path_actions[0] if path_actions else 4

        for action, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
            nx, ny = current_pos[0] + dx, current_pos[1] + dy
            neighbor = (nx, ny)
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                if neighbor not in blocked and neighbor != start:
                    new_actions = path_actions + [action]
                    priority = current_cost + 1 + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, neighbor, new_actions))

    return 4


import heapq

# Directions: up, down, left, right, stay
ACTIONS = {
    0: (0, -1),  # up
    1: (0, 1),  # down
    2: (-1, 0),  # left
    3: (1, 0),  # right
    4: (0, 0),  # stay
}


def heuristic(a, b):
    """Manhattan distance for grid A*"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def plan_with_reservations(start, goal, static_obs, reservations, grid_size, max_time=200):
    """
    Time-expanded A* that respects static obstacles and existing reservations.
    States: (x, y, t), moves in 4 directions or stay.
    """
    start_state = (start[0], start[1], 0)
    goal_xy = (goal[0], goal[1])

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start_state, []))
    visited = set()

    while open_set:
        f, (x, y, t), path = heapq.heappop(open_set)
        if (x, y, t) in visited:
            continue
        visited.add((x, y, t))

        # if reached goal
        if (x, y) == goal_xy:
            return [(px, py) for px, py, pt in path] + [(x, y)]
        if t >= max_time:
            continue

        for a, (dx, dy) in ACTIONS.items():
            nx, ny, nt = x + dx, y + dy, t + 1
            # bounds & static obstacles
            if not (0 <= nx < grid_size and 0 <= ny < grid_size):
                continue
            if (nx, ny) in static_obs:
                continue
            # reservation conflicts
            if (nx, ny, nt) in reservations:
                continue
            if (x, y, nt) in reservations:
                continue

            new_path = path + [(x, y, t)]
            heapq.heappush(open_set, (nt + heuristic((nx, ny), goal_xy), (nx, ny, nt), new_path))

    return None


def plan_all_agents(positions, goals, static_obs, grid_size, max_time=200):
    """
    Plans all UAV paths with prioritized reservations. Avoids both static and dynamic conflicts.
    Returns list of waypoint lists for each agent.
    """
    num_agents = len(positions)
    reservations = set()
    paths = [None] * num_agents

    for i in range(num_agents):
        path = plan_with_reservations(
            start=positions[i], goal=goals[i], static_obs=static_obs,
            reservations=reservations, grid_size=grid_size, max_time=max_time
        )
        if path is None:
            raise RuntimeError(f"No path found for agent {i}")
        paths[i] = path
        # reserve each step
        for t, (x, y) in enumerate(path):
            reservations.add((x, y, t))
        # reserve goal for all remaining times to block others
        last_x, last_y = path[-1]
        for t in range(len(path), max_time + 1):
            reservations.add((last_x, last_y, t))

    return paths


def follow_paths(env, paths):
    """
    Follows precomputed paths exactly; no additional collision handling needed.
    Records (states, actions) until all UAVs reach goals.
    """
    state = env._get_state()
    done = False
    states, actions_log = [], []
    t = 0

    while not done:
        current_actions = []
        for i, path in enumerate(paths):
            # if there is a next waypoint, move; else stay
            if t + 1 < len(path):
                dx = path[t + 1][0] - path[t][0]
                dy = path[t + 1][1] - path[t][1]
                # map delta to action
                for a, (adx, ady) in ACTIONS.items():
                    if (dx, dy) == (adx, ady):
                        current_actions.append(a)
                        break
            else:
                current_actions.append(4)

        states.append(state.clone())
        actions_log.append(list(current_actions))
        state, _, done, _ = env.step(current_actions)
        t += 1

    return states, actions_log


def generate_expert_trajectories(env, num_trajectories=50, verbose=True):
    expert_data = []
    for ep in range(num_trajectories):
        env.reset()
        state = env._get_state()

        static_obs = {tuple(obs.cpu().numpy()) for obs in env.obstacles}
        positions = [tuple(pos.cpu().numpy().astype(int)) for pos in env.positions]
        goals = [tuple(goal.cpu().numpy().astype(int)) for goal in env.goals]

        # Plan once per episode
        paths = plan_all_agents(
            positions, goals, static_obs, env.grid_size, max_time=env.grid_size * 4
        )

        # Print source and destination, or full path
        if verbose:
            for i, path in enumerate(paths):
                # convert numpy ints to Python ints
                start_py = (int(path[0][0]), int(path[0][1]))
                goal_py = (int(path[-1][0]), int(path[-1][1]))
                path_py = [(int(x), int(y)) for x, y in path]
                print(f"Drone {i}: Start={start_py}, Goal={goal_py}")
                print(f"Full path {i}: {path_py}\n")

        # Execute trajectories
        states, actions = follow_paths(env, paths)
        # Append only states and actions for training unpack consistency
        expert_data.append((states, actions))
    return expert_data


import math


def train():
    env = MultiUAVEnv()
    expert_data = generate_expert_trajectories(env)

    policy = MultiAgentPolicy().to(device)
    discriminator = Discriminator().to(device)
    optimizer_p = optim.Adam(policy.parameters(), lr=LR_ACTOR)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LR_CRITIC)

    best_loss = math.inf
    best_episode = -1

    for episode in range(EPISODES):
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                probs = policy(state)
                action = [p.sample().item() for p in probs]
            next_state, r, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(r)
            state = next_state

        with torch.no_grad():
            disc_rewards = []
            for s, a in zip(states, actions):
                d = discriminator(s, a)
                disc_rewards.append(-torch.log(1 - d + 1e-8))
            disc_rewards = torch.stack(disc_rewards)

        optimizer_p.zero_grad()
        policy_loss = 0
        for s, a, dr in zip(states, actions, disc_rewards):
            probs = policy(s)
            for i, p in enumerate(probs):
                log_prob = p.log_prob(torch.tensor(a[i], device=device))
                policy_loss -= log_prob * dr
        policy_loss = policy_loss / len(states)
        policy_loss.backward()
        optimizer_p.step()

        if policy_loss.item() < best_loss:
            best_loss = policy_loss.item()
            best_episode = episode
            torch.save(policy.state_dict(), "multi_uav_policy_best.pt")

        expert_batch = expert_data[np.random.randint(0, len(expert_data))]
        expert_states, expert_actions = expert_batch
        learner_inputs = list(zip(states, actions))
        expert_inputs = list(zip(expert_states, expert_actions))

        for _ in range(5):
            optimizer_d.zero_grad()
            # learner (fake)
            s, a = learner_inputs[np.random.randint(0, len(learner_inputs))]
            real_output = discriminator(s, a)
            d_loss = -torch.log(1 - real_output + 1e-8).mean()
            # expert (real)
            s, a = expert_inputs[np.random.randint(0, len(expert_inputs))]
            expert_output = discriminator(s, a)
            d_loss += -torch.log(expert_output + 1e-8).mean()

            d_loss.backward()
            optimizer_d.step()

        if episode % 50 == 0:
            print(f"Episode {episode}, Policy Loss: {policy_loss.item():.4f}, "
                  f"D Loss: {d_loss.item():.4f}, Best Loss so far: {best_loss:.4f} (ep {best_episode})")

    print(f"Training complete. Best policy saved at episode {best_episode} with loss {best_loss:.4f}")


def evaluate(policy_path="multi_uav_policy_best.pt"):
    env = MultiUAVEnv()
    policy = MultiAgentPolicy().to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))

    state = env.reset()
    done = False

    while not done:
        with torch.no_grad():
            probs = policy(state)
            action = [p.probs.argmax().item() for p in probs]
        state, _, done, _ = env.step(action)
        grid = torch.zeros(GRID_SIZE, GRID_SIZE)
        for obs in env.obstacles:
            grid[obs[0], obs[1]] = 8
        for i, pos in enumerate(env.positions):
            grid[int(pos[0]), int(pos[1])] = i + 1
        print("Current state:")
        print(grid.cpu().numpy())
    print("Final positions:", env.positions.cpu().numpy())


if __name__ == "__main__":
    train()
    evaluate()