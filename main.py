import torch
from torch import nn
import gymnasium as gym
import pygame
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

sqrt2 = 2 ** 0.5

def cosine_similarity(vec1, vec2):
  return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


###################
### ENVIRONMENT ###
###################
class SlipperyEnv(gym.Env):
  def __init__(
      self,
      size: int = 500,
      agent_size: int = 20,
      friction: float = 0.95,
      accel = 0.05,
      min_target_dist_coeff: float = 0.4,
      goal_reward: float = 20,
      time_penalty: float = -1/60,
      direction_reward_scale: float = 0.5,
      edge_penalty: float = -20,
      max_steps: int = 1800,
      target_reset_interval: int = 600,
      render_mode = 'human',
  ):
    self.metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    self.size = size                                     # Size of space in pixels
    self.agent_size = agent_size                         # Size of agent's circular collision shape
    self.friction = friction                             # Coefficient of friction
    self.accel = accel                                   # Acceleration of movement, measured in px/frame^2
    self.min_target_dist_coeff = min_target_dist_coeff   # Relates environment size to the min distance for agent to travel
    self.goal_reward = goal_reward                       # Reward for reaching goal
    self.time_penalty = time_penalty                     # Reward lost with each frame
    self.direction_reward_scale = direction_reward_scale # Scale of the reward depending on what direction the agent is moving
    self.edge_penalty = edge_penalty                     # Reward lost for hitting edge
    self.max_steps = max_steps                           # max number of steps before an episode is truncated; passed to the TimeLimit wrapper
    self.target_reset_interval = target_reset_interval   # steps interval after which target automatically resets to a random position

    assert self.friction < 1 and self.friction > 0

    if (self.min_target_dist_coeff >= 0.5):
      raise ValueError('min_target_dist_coeff must be less than 0.5')

    self.min_target_dist = self.size * self.min_target_dist_coeff
    self.max_v = self.accel / (1 - self.friction)

    self._agent_position = np.array([0, 0], dtype=float)
    self._target_position = np.array([0, 0], dtype=float)

    self._agent_v = np.array([0, 0], dtype=float)

    # Continuous, bounded coordinate space for both the agent and the target
    self.observation_space = gym.spaces.Dict({
      'agent': gym.spaces.Box(0, size - 1, shape=(2,), dtype=float),
      'target': gym.spaces.Box(0, size - 1, shape=(2,), dtype=float)
    })

    self.action_space = gym.spaces.Discrete(8)

    self._action_to_dv = {
      0: np.array([0, self.accel]),
      1: np.array([self.accel / sqrt2, self.accel / sqrt2]),
      2: np.array([self.accel, 0]),
      3: np.array([self.accel / sqrt2, -self.accel / sqrt2]),
      4: np.array([0, -self.accel]),
      5: np.array([-self.accel / sqrt2, -self.accel / sqrt2]),
      6: np.array([-self.accel, 0]),
      7: np.array([-self.accel / sqrt2, self.accel / sqrt2]),
    }

    self.curr_step = 0

    # ensure a valid render_mode value is being passed
    assert render_mode is None or render_mode in self.metadata['render_modes']
    self.render_mode = render_mode

    # used with human render mode
    self.window = None
    self.clock = None

    self.window_size = self.size


  def _get_obs(self):
    """Gets observation for .reset() and .step()
    Returns:
      dict: observation of current environment (agent and target)
    """
    return {'agent': self._agent_position, 'target': self._target_position, 'v': self._agent_v}
  
  def normalize_obs(self, obs):
    """Normalizes values of obs based on environment size and max possible speed
    Args:
      obs: an observation dict returned by .reset() or .step()
    Returns:
      list[float]: a list of normalized observation values
    """
    return [
      obs['agent'][0] / self.size,
      obs['agent'][1] / self.size,
      obs['target'][0] / self.size,
      obs['target'][1] / self.size,
      obs['v'][0] / self.max_v,
      obs['v'][1] / self.max_v,
    ]

  def _get_info(self):
    return None


  def _get_dist_to_target(self):
    """Returns distance between agent and target
      Returns:
        float: distance
    """
    return np.linalg.norm(self._agent_position - self._target_position)


  def _reset_target_position(self):
    """Moves target to a random valid position based on the position of agent
    Returns:
      np.array: new target location
    """
    while True:
      self._target_position = self.np_random.random(size=(2,)) * self.size
      if self._get_dist_to_target() >= self.min_target_dist:
        break
      
    return self._target_position
  
  def _respawn_agent(self):
    """Moves agent to a random valid position based on the position of target and sets its velocity to zero
    Returns:
      np.array: new agent location
    """
    while True:
      self._agent_position = self.np_random.random(size=(2,)) * self.size
      if self._get_dist_to_target() >= self.min_target_dist:
        break
    # set velo to 0
    self._agent_v[:] = 0.0

    return self._agent_position


  def reset(self, seed: int | None = None, options: dict | None = None):
    """Starts a new episode.
    Args:
      seed: RNG seed for reproducible episodes
      options: additional config
    Returns:
      tuple: (observation, info) for initial state
    """

    # seed RNG
    super().reset(seed=seed)

    self._agent_position = self.np_random.random(size=(2,)) * self.size
    self._reset_target_position()

    self._agent_v = np.array([0, 0], dtype=float)

    observation = self._get_obs()
    info = self._get_info()

    self.curr_step = 0
    self.target_reset_timer = self.target_reset_interval

    return observation, info


  def step(self, action):
    """Execute an action within the environment.
    Args:
      action: The action to take
    
    Returns:
      tuple: (observation, reward, terminated, truncated, info)
    """

    # accelerate the agent according to the action
    dv = self._action_to_dv[action]
    self._agent_v += dv
    self._agent_v *= self.friction

    # update the agent's position based on its velocity
    self._agent_position += self._agent_v

    # reward
    reward = 0
    if self._get_dist_to_target() < self.agent_size / 2:
      reward += self.goal_reward
      # reset the target position and timer
      self.target_reset_timer = self.target_reset_interval
      self._reset_target_position()
    else:
      c_sim = cosine_similarity(
        self._target_position - self._agent_position,
        self._agent_v
      )

      reward += c_sim * self.direction_reward_scale * (np.linalg.norm(self._agent_v) / self.max_v) ** 2
    
    pos = self._agent_position
    if (
      pos[0] < 0 or
      pos[0] > self.size or
      pos[1] < 0 or
      pos[1] > self.size
    ):
      # penalty for "falling off the edge"
      reward += self.edge_penalty
      # respawn agent
      self._respawn_agent()
    

    self.curr_step += 1

    # target reset timer
    self.target_reset_timer -= 1
    if self.target_reset_timer == 0:
      self.target_reset_timer = self.target_reset_interval
      self._reset_target_position()

    observation = self._get_obs()
    info = self._get_info()

    if (self.curr_step == self.max_steps):
      return observation, reward, False, True, info
    return observation, reward, False, False, info
  
  
  def setup_pygame(self):
    if self.window is None and self.render_mode == 'human':
      pygame.init()
      pygame.display.init()
      self.window = pygame.display.set_mode(
        (self.window_size, self.window_size)
      )
    if self.clock is None and self.render_mode == 'human':
      self.clock = pygame.time.Clock()

  def cleanup_pygame(self):
    if (self.window is not None):
      pygame.display.quit()
      self.window = None
    if (self.clock is not None):
      self.clock = None


  def _render_canvas(self, render_to_window = True):
    if render_to_window:
      self.setup_pygame()
    else:
      self.cleanup_pygame()

    canvas = pygame.Surface((self.window_size, self.window_size))
    canvas.fill((255, 255, 255))
  
    # ice
    pygame.draw.rect(
      canvas,
      (175, 221, 240),
      pygame.Rect(0, 0, self.size, self.size)
    )

    # target
    pygame.draw.circle(
      canvas,
      (0, 170, 0),
      self._target_position,
      5
    )
    # agent
    pygame.draw.circle(
      canvas,
      (0, 0, 0),
      self._agent_position,
      self.agent_size / 2
    )

    if self.render_mode == 'human' and self.window is not None:
      self.window.blit(canvas, canvas.get_rect())
      pygame.event.pump()
      pygame.display.update()

      self.clock.tick(self.metadata['render_fps'])
    else:
      # swap the width and height axes with transpose
      # pixels3d returns shape (width, height, 3)
      return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
  
  def render(self, render_to_window = True):
    self._render_canvas(render_to_window)


#####################
### AGENT NETWORK ###
#####################
class SlipperyAgentNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(6, 32),
      nn.ReLU(),
      nn.Linear(32, 32),
      nn.ReLU(),
      nn.Linear(32, 8),
    )

  def forward(self, x):
    logits = self.linear_relu_stack(x)
    return logits


#################################
### REWARD ESTIMATION NETWORK ###
#################################
class SlipperyRewardEstNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(6, 32),
      nn.ReLU(),
      nn.Linear(32, 32),
      nn.ReLU(),
      nn.Linear(32, 1),
    )

  def forward(self, x):
    logits = self.linear_relu_stack(x)
    return logits


#########################################
### ENVIRONMENT AND TRAINING SETTINGS ###
#########################################

# Always use CPU (small network means the CPU-GPU overhead is not worth it)
device = 'cpu'

env = SlipperyEnv(
  size=500,
  agent_size=40,
  min_target_dist_coeff=0.1,

  friction=0.98,
  accel=0.08,

  goal_reward=30,
  direction_reward_scale=1,
  edge_penalty=-30,

  max_steps=2400,
  target_reset_interval=600,
)

# learning rates for policy and baseline networks
a_theta = 0.001
a_b = 0.001

# reward / return settings
gamma = 0.995
return_scale = 0.1

episode_count = 10000

# rendering and graphing settings
render_during_training = False
episode_render_interval = 500

# how often to print episode information
episode_print_interval = 10

# moving average sizes for graphing
avg_window_1 = 20
avg_window_2 = 100


#########################
# NETWORK INITALIZATION #
#########################

policy = SlipperyAgentNN()
policy_device = 'cpu'
policy.to(policy_device)
policy_optim = torch.optim.AdamW(policy.parameters(), lr=a_theta)

baseline = SlipperyRewardEstNN()
baseline_device = 'cpu'
baseline.to(baseline_device)
baseline_optim = torch.optim.AdamW(baseline.parameters(), lr=a_b)


########################################
# MATPLOTLIB / GRAPHING INITIALIZATION #
########################################

plt.ion()
fig, ax = plt.subplots()

reward_graph, = ax.plot([], [], 'c-', label='reward/step')
reward_avg_graph_1, = ax.plot([], [], 'b-', label=f'avg. of {avg_window_1}')
reward_avg_graph_2, = ax.plot([], [], 'r-', label=f'avg. of {avg_window_2}')
ax.legend()
ax.set_xlabel('episode')
ax.set_ylabel('reward/step')
ax.set_title('Average Agent Reward Per Episode')

plt.show()

reward_history = np.array([])
reward_avg_1 = []
reward_avg_2 = []

episode = 0
start_time = time.perf_counter()


#####################
# MAIN EPISODE LOOP #
#####################

while episode <= episode_count:
  total_reward = 0
  observation, info = env.reset()

  # history for states, actions, rewards, discounted rewards, and the probability of the chosen action
  s = []
  a = []
  r = []
  dr = []
  log_probs = []

  episode_length = 0
  should_render = render_during_training and episode % episode_render_interval == 0
  
  ###################
  # RUN THE EPISODE #
  ###################
  while True:
    norm_obs = torch.tensor(env.normalize_obs(observation), dtype=torch.float32, device=policy_device)
    # get action
    action = None
    
    # sample an action from the agent network
    policy_logits = policy.forward(norm_obs)
    action_probs = torch.softmax(policy_logits, dim=-1)
    dist = torch.distributions.Categorical(action_probs)
    action = dist.sample().item()

    s.append(norm_obs)
    a.append(action)
    log_probs.append(torch.log(action_probs[action]))
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    r.append(reward)
    total_reward += reward

    episode_length += 1

    if (should_render):
      env.render()

    # check if the episode is over
    if terminated or truncated:
      # clean up pygame if this episode was rendered
      if (should_render):
        env.cleanup_pygame()
      break
  
  # convert lists to tensors and move to GPU for baseline network training
  s = torch.stack(s, dim=0).to(device=baseline_device)
  log_probs = torch.stack(log_probs, dim=0).to(device=baseline_device)


  ############################
  # NETWORK TRAINING UPDATES #
  ############################

  # calculate discounted rewards and train baseline network
  dr = torch.zeros(episode_length, dtype=torch.float32, device=baseline_device)
  dr[episode_length - 1] = r[-1]
  for i in range(episode_length - 2, -1, -1):
    dr[i] = r[i] + gamma * dr[i + 1]
  # scale all returns
  dr *= return_scale

  b_values = baseline.forward(s)

  b_loss = torch.mean((b_values - dr) ** 2)

  baseline_optim.zero_grad()
  b_loss.backward()
  baseline_optim.step()
  
  # train policy
  b_values = b_values.detach() # detach b_values from autograd graph
  policy_loss = torch.mean(-log_probs * (dr - b_values))

  policy_optim.zero_grad()
  policy_loss.backward()
  policy_optim.step()


  # reward history and average
  reward_history = np.append(reward_history, np.mean(r))
  
  if (episode >= avg_window_1):
    reward_avg_1.append(reward_history[-avg_window_1:].mean())
  if (episode >= avg_window_2):
    reward_avg_2.append(reward_history[-avg_window_2:].mean())

  if episode % episode_print_interval == 0:
    print(str(datetime.timedelta(seconds=time.perf_counter() - start_time)), episode, np.mean(r), b_loss.item())
  episode += 1



# final graph render
reward_graph.set_data(np.arange(episode), reward_history)
reward_avg_graph_1.set_data(np.arange(avg_window_1, episode), reward_avg_1)
reward_avg_graph_2.set_data(np.arange(avg_window_2, episode), reward_avg_2)

plt.xlim(0, episode - 1)
plt.ylim(-0.1, 0.7)
plt.show()
plt.pause(0.01)

# Final demo episode (always rendered)
env.max_steps = 36000 # 10 minutes

observation, info = env.reset()

while True:
  norm_obs = torch.tensor(env.normalize_obs(observation), dtype=torch.float32, device=policy_device)
  # get action
  action = None
  
  policy_logits = policy.forward(norm_obs)
  action_probs = torch.softmax(policy_logits, dim=-1)
  dist = torch.distributions.Categorical(action_probs)
  # sample an action from the action probability distribution
  action = dist.sample().item()
  observation, reward, terminated, truncated, info = env.step(action)

  env.render()

  # check if the episode is over
  if terminated or truncated:
    env.cleanup_pygame()
    break
