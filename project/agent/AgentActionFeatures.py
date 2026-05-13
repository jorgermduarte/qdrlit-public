import random
import csv
import signal
from typing import List, Tuple
from db_env.DatabaseEnvironment import DatabaseEnvironment
from shared_utils.consts import PROJECT_DIR
from shared_utils.utils import create_logger
from .QDQNM import QDQN
#from .QDQNB import QDQN
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import torch
import time

AGENT_CSV_FILE = f'{PROJECT_DIR}/data/agent_history.csv'
WEIGHTS_FILE = f'{PROJECT_DIR}/data/weights.csv'

class Agent:
    def __init__(self, env: DatabaseEnvironment):
        random.seed(2)
        np.random.seed(2)
        self._log = create_logger('agent')
        self._env = env

        self.state_size = self._env.observation_space.n
        self.action_space_size = self._env.action_space.n

        # DQN parameters
        self.exploration_probability = 0.9
        self.exploration_probability_discount = 0.9
        self.learning_rate = 0.001
        self.discount_factor = 0.8

        # Experience replay configuration
        self._experience_memory_max_size = np.inf # Maximum size of experience memory
        self._experience_replay_count = 32  # Number of samples to use for experience replay
        self.target_update_frequency = 50 # Frequency of target network updates

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Neural networks (PyTorch)
        self.main_network = QDQN(
            input_size=self.state_size,
            output_size=self.action_space_size
        ).to(self.device)
        self.target_network = QDQN(
            input_size=self.state_size,
            output_size=self.action_space_size
        ).to(self.device)

        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)
        # Copy weights initially
        self.target_network.load_state_dict(self.main_network.state_dict())

        self._experience_memory: List[Tuple[List[int], int, float, List[int]]] = []
        self.dict_info = {
            'episode': int,
            'step': int,
            'state': List[bool],
            'action': int,
            'reward': float,
            'next_state': List[bool],
            'q': float,
            'max_a': int,
            'max_q': float,
            'td_target': float,
            'td_error': float,
            'total_reward': float,
            'exploration_probability': float,
            'random_action': bool,
            'initial_state_reward': float,

            'step_execution_time_seconds': float,
            'power_queries_exec_time_sum_sec': float,
            'power_refresh_function_exec_time_sum_sec': float,
            'throughput_total_exec_time_sec': float,
            'benchmark_metrics_raw_data': dict
        }

        self._pause_request = False

        def signal_handler(sig, frame):
            self._log.info('CTRL+C pressed - pausing training requested')
            self._pause_request = True

        signal.signal(signal.SIGINT, signal_handler)


    def _update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def train(self, episode_count: int, steps_per_episode: int):
        with open(AGENT_CSV_FILE, 'w', newline='') as file:
            wr = csv.writer(file)
            wr.writerow(self.dict_info.keys())

        for episode in range(episode_count):

            state = self._env.reset()
            total_reward = 0.0

            for step in range(steps_per_episode):
                start_time = time.time()
                self._log.info(f'EPISODE {episode} - STEP {step} '
                               f'({(episode_count - episode) * steps_per_episode - step - 1} more steps to go)')

                action = self._choose_action(state)
                next_state, reward, _, info = self._env.step(action)
                total_reward += reward

                # DQN Adaptation
                self._experience_append(state, action, reward, next_state)
                self._experience_replay() # Update network before saving
                self._save_agent_information(episode, step, state, next_state, action, reward, total_reward, info, start_time)
                #self._save_agent_weights()
                state = next_state

                if self._pause_request:
                    return

                # DQN Update target network periodically
                if step % self.target_update_frequency == 0:
                    self._update_target_network()
                
                # step message execution time
                step_execution_time = time.time() - start_time
                # in minutes and seconds
                minutes, seconds = divmod(step_execution_time, 60)
                print(f"::::::::::::::::::::: - [{step}] step execution time: {int(minutes)}m {seconds:.2f}s")

            self._reduce_exploration_probability()

    def _experience_append(self, state, action, reward, next_state):
        self._experience_memory.append((state, action, reward, next_state))

        if len(self._experience_memory) > self._experience_memory_max_size:
            self._experience_memory.pop(0)

    def _experience_replay(self):
        if len(self._experience_memory) < self._experience_replay_count:
            return

        print("Using experience replay...")
        # Sample batch
        batch = random.sample(self._experience_memory, self._experience_replay_count)
        states, actions, rewards, next_states = zip(*batch)  # Add done flags if available

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        # Current Q-values
        current_q = self.main_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target calculation
        with torch.no_grad():
            # Get valid actions for next states using MAIN network
            main_next_q = self.main_network(next_states)
            valid_actions_mask = [self._possible_actions(s) for s in next_states.cpu().numpy()]
            main_next_q[~self._create_action_mask_tensor(valid_actions_mask)] = -float('inf')
            next_actions = main_next_q.argmax(1)

            # Get Q-values from TARGET network using these actions
            target_next_q = self.target_network(next_states)
            max_next_q = target_next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # TD target
            targets = rewards + self.discount_factor * max_next_q  # Simplified

        # Compute loss
        loss = F.mse_loss(current_q, targets)

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 1.0)
        self.optimizer.step()

    def _create_action_mask_tensor(self, valid_actions_list):
        mask = torch.zeros((len(valid_actions_list), self.action_space_size),
                        dtype=torch.bool).to(self.device)
        for i, valid_actions in enumerate(valid_actions_list):
            mask[i, valid_actions] = True
        return mask

    def _get_max_action(self, state):
        # Convert and pad state
        state_array = np.array(state, dtype=np.float32)
        state_tensor = torch.FloatTensor(state_array).to(self.device)

        with torch.no_grad():
            q_values = self.main_network(state_tensor.unsqueeze(0))

        valid_actions = self._possible_actions(state)
        valid_q_values = q_values.cpu().numpy()[valid_actions]
        max_q = np.max(valid_q_values)
        max_action = valid_actions[np.argmax(valid_q_values)]
        return max_action, max_q

    def _calculate_q_value(self, state, action):
        # Convert state to tensor with proper typing
        state_tensor = torch.FloatTensor(np.array(state, dtype=np.float32)).to(self.device)
        # Ensure action is valid
        if action not in self._possible_actions(state):
            return float('-inf')

        # Get Q-value with gradient tracking disabled
        with torch.no_grad():
            q_values = self.main_network(state_tensor)

        return q_values[action].item()  # Convert tensor to Python float

    def _possible_actions(self, state) -> List[int]:
        """EXACTLY replicate original action generation logic"""
        return [i * 2 + (not is_indexed) for i, is_indexed in enumerate(state)]

    def _choose_action(self, state):
        """
        :param state: current environment state
        :return: random action with probability epsilon otherwise best action with probability 1-epsilon
        """
        self.dict_info['random_action'] = False
        self.dict_info['exploration_probability'] = self.exploration_probability

        if random.random() < self.exploration_probability:
            self.dict_info['random_action'] = True
            return random.choice(self._possible_actions(state))

        max_action, _ = self._get_max_action(state)
        return max_action

    def _reduce_exploration_probability(self):
        self.exploration_probability = self.exploration_probability_discount * self.exploration_probability

    def _save_agent_information(self, episode, step, state, next_state, action, reward, total_reward, info, start_time):
        # Convert state to tensor for calculation
        state_tensor = torch.FloatTensor(np.array(state, dtype=np.float32)).to(self.device)
    
        # Calculate Q-values properly
        with torch.no_grad():
            current_q = self.main_network(state_tensor)[action].item()
            max_next_action, max_next_q = self._get_max_action(next_state)
            
            # Calculate TD target and error using target network
            td_target = reward + self.discount_factor * max_next_q
            td_error = td_target - current_q

            # Update dictionary with DQN-specific values
            self.dict_info.update({
                'episode': episode,
                'step': step,
                'state': state,
                'next_state': next_state,
                'action': action,
                'reward': reward,
                'total_reward': total_reward,
                'q': current_q,
                'max_a': max_next_action,
                'max_q': max_next_q,
                'td_target': td_target,
                'td_error': td_error,
                'initial_state_reward': info['initial_state_reward'],
                'exploration_probability': self.exploration_probability,
                'random_action': self.dict_info['random_action']  # Preserve this flag
            })

            self.dict_info['step_execution_time_seconds'] = time.time() - start_time
            self.dict_info['power_queries_exec_time_sum_sec'] = info['power_queries_exec_time_sum_sec']
            self.dict_info['power_refresh_function_exec_time_sum_sec'] = info['power_refresh_function_exec_time_sum_sec']
            self.dict_info['throughput_total_exec_time_sec'] = info['throughput_total_exec_time_sec']
            self.dict_info['benchmark_metrics_raw_data'] = info['benchmark_metrics_raw_data']

            with open(AGENT_CSV_FILE, 'a', newline='') as file:
                wr = csv.writer(file)
                wr.writerow(self.dict_info.values())

    def _save_agent_weights(self):
        torch.save(self.main_network.state_dict(), WEIGHTS_FILE)
