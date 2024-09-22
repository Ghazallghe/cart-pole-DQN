from config import *
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque

import matplotlib
matplotlib.use('TKAgg')


class ReplayMemory:
    def __init__(self, capacity):
        """
        Experience Replay Memory defined by deques to store transitions/agent experiences
        """

        self.capacity = capacity

        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

    def store(self, state, action, next_state, reward, done):
        """
        Append (store) the transitions to their respective deques
        """

        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample(self, batch_size):
        """
        Randomly sample transitions from memory, then convert sampled transitions
        to tensors and move to device (CPU or GPU).
        """

        indices = np.random.choice(len(self), size=batch_size, replace=False)

        states = torch.stack([torch.as_tensor(
            self.states[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        actions = torch.as_tensor(
            [self.actions[i] for i in indices], dtype=torch.long, device=device)
        next_states = torch.stack([torch.as_tensor(
            self.next_states[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        rewards = torch.as_tensor(
            [self.rewards[i] for i in indices], dtype=torch.float32, device=device)
        dones = torch.as_tensor(
            [self.dones[i] for i in indices], dtype=torch.bool, device=device)

        return states, actions, next_states, rewards, dones

    def __len__(self):
        """
        To check how many samples are stored in the memory. self.dones deque
        represents the length of the entire memory.
        """

        return len(self.dones)


class DQN_Network(nn.Module):
    """
    The Deep Q-Network (DQN) model for reinforcement learning.
    This network consists of Fully Connected (FC) layers with ReLU activation functions.
    """

    def __init__(self, num_actions, input_dim):
        """
        Initialize the DQN network.

        Parameters:
            num_actions (int): The number of possible actions in the environment.
            input_dim (int): The dimensionality of the input state space.
        """

        super(DQN_Network, self).__init__()

        self.FC = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, num_actions),
        )

        # Initialize FC layer weights using He initialization
        for layer in [self.FC]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(
                        module.weight, nonlinearity='relu')

    def forward(self, x):
        """
        Forward pass of the network to find the Q-values of the actions.

        Parameters:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            Q (torch.Tensor): Tensor containing Q-values for each action.
        """
        Q = self.FC(x)
        return Q


class DQN_Agent:
    """
    DQN Agent Class. This class defines some key elements of the DQN algorithm,
    such as the learning method, hard update, and action selection based on the
    Q-value of actions or the epsilon-greedy policy.
    """

    def __init__(self, env, e_greedy, epsilon_max, epsilon_min, epsilon_decay,
                 temp_max, temp_min, temp_decay, clip_grad_norm, learning_rate, discount, memory_capacity):

        # To save the history of network loss
        self.loss_history = []
        self.running_loss = 0
        self.learned_counts = 0

        # RL hyperparameters
        self.e_greedy = e_greedy
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.temp_max = temp_max
        self.temp_min = temp_min
        self.temp_decay = temp_decay

        self.discount = discount

        self.action_space = env.action_space
        # Set the seed to get reproducible results when sampling the action space
        self.action_space.seed(seed)
        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(memory_capacity)

        self.input_dim = self.observation_space.shape[0]
        self.output_dim = self.action_space.n

        # Initiate the network models
        self.main_network = DQN_Network(
            num_actions=self.output_dim, input_dim=self.input_dim).to(device)
        self.target_network = DQN_Network(
            num_actions=self.output_dim, input_dim=self.input_dim).to(device).eval()
        self.target_network.load_state_dict(self.main_network.state_dict())

        # For clipping exploding gradients caused by high reward value
        self.clip_grad_norm = clip_grad_norm
        self.critertion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.main_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        if self.e_greedy:
            return self.epsilon_greedy(state=state)
        return self.boltzmann(state=state)

    def epsilon_greedy(self, state):
        """
        Selects an action using epsilon-greedy strategy OR based on the Q-values.

        Parameters:
            state (torch.Tensor): Input tensor representing the state.

        Returns:
            action (int): The selected action.
        """

        # Exploration: epsilon-greedy
        if np.random.random() < self.epsilon_max:
            return self.action_space.sample()

        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=device)

        # Exploitation: the action is selected based on the Q-values.
        with torch.no_grad():
            Q_values = self.main_network(state)
            action = torch.argmax(Q_values).item()

        return action

    def boltzmann(self, state):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=device)

        with torch.no_grad():
            Q_values = self.main_network(state)

        if self.temp_max != -1:
            # Subtract the maximum Q-value for stability
            max_Q = torch.max(Q_values)
            exp_values = torch.exp((Q_values - max_Q) / self.temp_max)
            probabilities = exp_values / torch.sum(exp_values)
            action = torch.multinomial(probabilities, 1).item()
        else:
            action = torch.argmax(Q_values).item()

        return action

    def learn(self, batch_size, done):
        """
        Train the main network using a batch of experiences sampled from the replay memory.

        Parameters:
            batch_size (int): The number of experiences to sample from the replay memory.
            done (bool): Indicates whether the episode is done or not. If done,
            calculate the loss of the episode and append it in a list for plot.
        """

        # Sample a batch of experiences from the replay memory
        states, actions, next_states, rewards, dones = self.replay_memory.sample(
            batch_size)

        # # Preprocess the data for training
        # states        = states.unsqueeze(1)
        # next_states   = next_states.unsqueeze(1)
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        # forward pass through the main network to find the Q-values of the states
        predicted_q = self.main_network(states)
        # selecting the Q-values of the actions that were actually taken
        predicted_q = predicted_q.gather(dim=1, index=actions)

        # Compute the maximum Q-value for the next states using the target network
        with torch.no_grad():
            # not argmax (cause we want the maxmimum q-value, not the action that maximize it)
            next_target_q_value = self.target_network(
                next_states).max(dim=1, keepdim=True)[0]

        # Set the Q-value for terminal states to zero
        next_target_q_value[dones] = 0
        # Compute the target Q-values
        y_js = rewards + (self.discount * next_target_q_value)
        loss = self.critertion(predicted_q, y_js)  # Compute the loss

        # Update the running loss and learned counts for logging and plotting
        self.running_loss += loss.item()
        self.learned_counts += 1

        if done:
            # The average loss for the episode
            episode_loss = self.running_loss / self.learned_counts
            # Append the episode loss to the loss history for plotting
            self.loss_history.append(episode_loss)
            # Reset the running loss and learned counts
            self.running_loss = 0
            self.learned_counts = 0

        self.optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Perform backward pass and update the gradients

        # # Uncomment the following two lines to find the best value for clipping gradient (Comment torch.nn.utils.clip_grad_norm_ while uncommenting the following two lines)
        # grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
        # self.main_network.parameters(), float('inf'))
        # print("Gradient norm before clipping:", grad_norm_before_clip)

        # Clip the gradients to prevent exploding gradients
        # torch.nn.utils.clip_grad_norm_(
        # self.main_network.parameters(), self.clip_grad_norm)

        self.optimizer.step()  # Update the parameters of the main network using the optimizer

    def hard_update(self):
        """
        Navie update: Update the target network parameters by directly copying
        the parameters from the main network.
        """

        self.target_network.load_state_dict(self.main_network.state_dict())

    def update_epsilon(self):
        """
        Update the value of epsilon for epsilon-greedy exploration.

        This method decreases epsilon over time according to a decay factor, ensuring
        that the agent becomes less exploratory and more exploitative as training progresses.
        """

        self.epsilon_max = max(
            self.epsilon_min, self.epsilon_max * self.epsilon_decay)

    def update_temperature(self):
        """
        Update the value of temperature for Boltzmann policy.

        This method decreases temperature over time according to a decay factor, ensuring
        that the agent becomes less exploratory and more exploitative as training progresses.
        """
        self.temp_max = max(
            self.temp_min, self.temp_max * self.temp_decay)

    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extention.

        """
        torch.save(self.main_network.state_dict(), path)
