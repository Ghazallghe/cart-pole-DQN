import torch
import pygame
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


from DQN import *
from config import *


class step_wrapper(gym.Wrapper):
    """
    A wrapper class for modifying the state function of the 
    CartPole-v1 environment.
    """

    def __init__(self, env):
        """
        Initializes the StepWrapper. This is the main class for wrapping the environment with it.

        Args:
            env (gym.Env): The environment to be wrapped.

        Attributes:
            observation_wrapper (Inherit from ObservationWrapper): 
                An instance of the ObservationWrapper class for modifying observations.
        """
        super().__init__(env)  # We give the env here to initialize the gym.Wrapper superclass (inherited).

        self.observation_wrapper = observation_wrapper(env)

    def step(self, action):
        """
        Executes a step in the environment with the provided action.The reason 
        behind using this method is to have access to the state and reward functions return.

        Args:
            action (int): The action to be taken.
        """

        state, reward, done, truncation, info = self.env.step(
            action)  # Same as before as usual

        # Give the state to another Wrapper, which returns a modified version of state
        modified_state = self.observation_wrapper.observation(state)

        # The same returns as usual but with modified versions of the state and reward functions
        return modified_state, reward, done, truncation, info

    def reset(self, seed):
        state, info = self.env.reset(seed=seed)  # Same as before as usual
        # Give the state to another Wrapper, which returns a modified version of state
        modified_state = self.observation_wrapper.observation(state)

        # Same as before as usual but with returning the modified version of the state
        return modified_state, info


class observation_wrapper(gym.ObservationWrapper):
    """
    Wrapper class for modifying observations in the CartPole environment.

    Args:
        env (gym.Env): The environment to wrap.

    Attributes:
        min_value (numpy.ndarray): Array of minimum observation values.
        max_value (numpy.ndarray): Array of maximum observation values.
    """

    def __init__(self, env):
        super().__init__(env)

        self.min_value = env.observation_space.low
        self.max_value = env.observation_space.high

    def observation(self, state):
        """
        Modifies the observation by clipping the values and normalizing it.

        Args:
            state (numpy.ndarray): The original observation from the environment.

        Returns:
            numpy.ndarray: The modified and normalized observation.
        """

        # Min-max normalization
        normalized_state = (state - self.min_value) / \
            (self.max_value - self.min_value)

        return normalized_state


class Model_TrainTest:
    def __init__(self, hyperparams):

        # Define RL Hyperparameters
        self.train_mode = hyperparams["train_mode"]
        self.RL_load_path = hyperparams["RL_load_path"]
        self.save_path = hyperparams["save_path"]
        self.save_interval = hyperparams["save_interval"]

        # self.clip_grad_norm = hyperparams["clip_grad_norm"]
        self.clip_grad_norm = 10
        self.learning_rate = hyperparams["learning_rate"]
        self.discount_factor = hyperparams["discount_factor"]
        self.batch_size = hyperparams["batch_size"]
        self.update_frequency = hyperparams["update_frequency"]
        self.max_episodes = hyperparams["max_episodes"]
        self.max_steps = hyperparams["max_steps"]
        self.render = hyperparams["render"]

        self.e_greedy = hyperparams["e_greedy"]

        self.epsilon_max = hyperparams["epsilon_max"]
        self.epsilon_min = hyperparams["epsilon_min"]
        self.epsilon_decay = hyperparams["epsilon_decay"]

        self.temp_max = hyperparams['temp_max']
        self.temp_min = hyperparams['temp_min']
        self.temp_decay = hyperparams['temp_decay']

        self.memory_capacity = hyperparams["memory_capacity"]

        self.render_fps = hyperparams["render_fps"]

        # Define Env
        self.env = gym.make(
            'CartPole-v1', render_mode="human" if self.render else None)
        # For max frame rate make it 0
        self.env.metadata['render_fps'] = self.render_fps

        # Define the agent class
        self.agent = DQN_Agent(env=self.env,
                               e_greedy=self.e_greedy,
                               epsilon_max=self.epsilon_max,
                               epsilon_min=self.epsilon_min,
                               epsilon_decay=self.epsilon_decay,
                               temp_max=self.temp_max,
                               temp_min=self.temp_min,
                               temp_decay=self.temp_decay,
                               clip_grad_norm=self.clip_grad_norm,
                               learning_rate=self.learning_rate,
                               discount=self.discount_factor,
                               memory_capacity=self.memory_capacity)

    def train(self):
        """                
        Reinforcement learning training loop.
        """

        total_steps = 0
        self.reward_history = []
        self.policy_param_history = []

        # Training loop over episodes
        for episode in range(1, self.max_episodes+1):
            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation:
                action = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)

                self.agent.replay_memory.store(
                    state, action, next_state, reward, done)

                if len(self.agent.replay_memory) > self.batch_size:
                    self.agent.learn(self.batch_size, (done or truncation))

                    # Update target-network weights
                    if total_steps % self.update_frequency == 0:
                        self.agent.hard_update()

                state = next_state
                episode_reward += reward
                step_size += 1

            # Appends for tracking history
            self.reward_history.append(episode_reward)  # episode reward
            total_steps += step_size

            policy = 'Epsilon'
            val = self.agent.epsilon_max
            if self.e_greedy:
                self.policy_param_history.append(self.agent.epsilon_max)
                # Decay epsilon at the end of each episode
                self.agent.update_epsilon()
            else:
                self.policy_param_history.append(self.agent.temp_max)
                # Decay temperature at the end of each episode
                self.agent.update_temperature()

                policy = 'Temprature'
                val = self.agent.temp_max

            # -- based on interval
            if episode % self.save_interval == 0:
                self.agent.save(self.save_path + '_' + f'{episode}' + '.pth')
                if episode != self.max_episodes:
                    self.plot_training(episode, policy)
                print('\n~~~~~~Interval Save: Model saved.\n')

            result = (f"Episode: {episode}, "
                      f"Total Steps: {total_steps}, "
                      f"Ep Step: {step_size}, "
                      f"Raw Reward: {episode_reward:.2f}, "
                      f"{policy}: {val:.2f}")
            print(result)
        self.plot_training(episode, policy)

    def test(self, max_episodes):
        """                
        Reinforcement learning policy evaluation.
        """

        # Load the weights of the test_network
        self.agent.main_network.load_state_dict(torch.load(self.RL_load_path))
        self.agent.main_network.eval()

        # Testing loop over episodes
        for episode in range(1, max_episodes+1):
            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation:
                action = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)

                state = next_state
                episode_reward += reward
                step_size += 1

            # Print log
            result = (f"Episode: {episode}, "
                      f"Steps: {step_size:}, "
                      f"Reward: {episode_reward:.2f}, ")
            print(result)

        pygame.quit()  # close the rendering window

    def plot_training(self, episode, policy):
        # Calculate the Simple Moving Average (SMA) with a window size of 50
        sma = np.convolve(self.reward_history, np.ones(50)/50, mode='valid')

        plt.figure()
        plt.title("Rewards")
        plt.plot(self.reward_history, label='Raw Reward',
                 color='#F6CE3B', alpha=1)
        plt.plot(sma, label='SMA 50', color='#385DAA')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()

        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('./reward_plot.png', format='png',
                        dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()

        plt.figure()
        plt.title("Loss")
        plt.plot(self.agent.loss_history, label='Loss',
                 color='#CB291A', alpha=1)
        plt.xlabel("Episode")
        plt.ylabel("Loss")

        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('./Loss_plot.png', format='png',
                        dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.title(policy)
        plt.plot(self.policy_param_history, label=f'Max {policy}',
                 color='#8A2BE2', alpha=1)
        plt.xlabel("Episode")
        plt.ylabel(policy)

        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig(f'./{policy}_plot.png', format='png',
                        dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    # Parameters:
    train_mode = True
    render = not train_mode
    RL_hyperparams = {
        "train_mode": train_mode,
        "RL_load_path": f'./final_weights' + '_' + '600' + '.pth',
        "save_path": f'./final_weights_test',
        "save_interval": 200,  # To save the model weights before an episode ends

        "learning_rate": 6e-4,
        "discount_factor": 0.99,
        "batch_size": 32,
        "update_frequency": 10,
        "max_episodes": 600 if train_mode else 5,
        "max_steps": 500,
        "render": render,

        "epsilon_max": 0.999 if train_mode else -1,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.996,

        "temp_max": 2 if train_mode else -1,
        "temp_min": 0.01,
        "temp_decay": 0.997,


        "memory_capacity": 10_000 if train_mode else 0,

        "render_fps": 6,

        "e_greedy": True,
    }

    # Run
    DRL = Model_TrainTest(RL_hyperparams)  # Define the instance
    # Train
    if train_mode:
        DRL.train()
    else:
        # Test
        DRL.test(max_episodes=RL_hyperparams['max_episodes'])
