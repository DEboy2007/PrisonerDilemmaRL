import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class OpponentHistory:
    """
    Data structure to track opponent's complete history of moves and outcomes.
    Creates a 5-dimensional continuous state vector with recency-weighted bins.
    Each bin covers 1/5th of the entire history, weighted by recency (5x to 1x).
    """
    def __init__(self):
        self.history = []  # Stores all moves (0 for share, 1 for steal)

    def add_round(self, opponent_move, my_reward, opponent_reward):
        """Add a round's outcome to history. Convert move to binary (steal=1, share=0)."""
        move_binary = 1 if opponent_move == 'steal' else 0
        self.history.append({
            'move': opponent_move,
            'move_binary': move_binary,
            'my_reward': my_reward,
            'opponent_reward': opponent_reward
        })

    def get_state_vector(self):
        """
        Converts history into a 5-dimensional weighted state vector.

        Divides entire history into 5 equal bins (each 1/5th of total history).
        Calculates steal rate in each bin, then applies recency weighting:
        - Bin 0 (most recent 1/5th): steal_rate x 5
        - Bin 1 (next 1/5th): steal_rate x 4
        - Bin 2 (middle 1/5th): steal_rate x 3
        - Bin 3 (older 1/5th): steal_rate x 2
        - Bin 4 (oldest 1/5th): steal_rate x 1

        Returns: numpy array of shape (5,) with weighted values
        """
        state = np.zeros(5, dtype=np.float32)

        if len(self.history) == 0:
            return state  # All zeros if no history

        total_moves = len(self.history)
        bin_size = total_moves / 5.0

        # Process each of the 5 bins (from most recent to oldest)
        for bin_idx in range(5):
            # Calculate the range for this bin
            # bin_idx=0 is most recent, bin_idx=4 is oldest
            start_idx = int(total_moves - (bin_idx + 1) * bin_size)
            end_idx = int(total_moves - bin_idx * bin_size)

            # Ensure valid indices
            start_idx = max(0, start_idx)
            end_idx = min(total_moves, end_idx)

            # Get moves in this bin
            if end_idx > start_idx:
                moves_in_bin = [self.history[i]['move_binary'] for i in range(start_idx, end_idx)]
                steal_rate = sum(moves_in_bin) / len(moves_in_bin)

                # Apply recency weight: most recent gets 5x, oldest gets 1x
                weight = 5 - bin_idx
                state[bin_idx] = steal_rate * weight
            else:
                state[bin_idx] = 0.0

        return state

    def analyze_opponent_pattern(self):
        """
        Analyzes opponent's behavior patterns from history.
        Returns a dictionary with statistics.
        """
        if len(self.history) == 0:
            return {
                'share_rate': 0.5,
                'steal_rate': 0.5,
                'avg_my_reward': 0,
                'avg_opp_reward': 0,
                'total_rounds': 0
            }

        shares = sum(1 for r in self.history if r['move'] == 'share')
        steals = sum(1 for r in self.history if r['move'] == 'steal')
        total = len(self.history)

        avg_my_reward = sum(r['my_reward'] for r in self.history) / total
        avg_opp_reward = sum(r['opponent_reward'] for r in self.history) / total

        return {
            'share_rate': shares / total if total > 0 else 0.5,
            'steal_rate': steals / total if total > 0 else 0.5,
            'avg_my_reward': avg_my_reward,
            'avg_opp_reward': avg_opp_reward,
            'total_rounds': total
        }


class DQNetwork(nn.Module):
    """
    Deep Q-Network for approximating Q(s, a).
    Input: 5-dimensional state vector
    Output: 2 Q-values (one for 'share', one for 'steal')
    """
    def __init__(self, state_dim=5, action_dim=2, hidden_dim=64):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """
    Experience replay buffer to store and sample transitions.
    Helps stabilize DQN training by breaking correlations between consecutive samples.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNPrisoner:
    """
    Prisoner agent using Deep Q-Network for learning.
    """
    def __init__(self, learning_rate=0.001, discount_factor=0.9, initial_epsilon=0.5):
        self.net_money = 0
        self.discount_factor = discount_factor
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.episode_count = 0

        # Neural network
        self.q_network = DQNetwork(state_dim=5, action_dim=2, hidden_dim=64)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 32

        # Track opponent's history
        self.opponent_history = OpponentHistory()

        # Action mapping
        self.actions = ['share', 'steal']

        # For tracking last state/action
        self.last_state = None
        self.last_action = None
        self.last_action_idx = None

        # Curiosity-driven exploration parameters
        self.recent_losses = deque(maxlen=20)  # Track last 20 losses
        self.epsilon_min = 0.05
        self.epsilon_max = 0.8
        self.epsilon_increase = 0.05  # How much to increase when plateau detected
        self.epsilon_decrease = 0.02  # How much to decrease when learning
        self.plateau_threshold = 0.01  # If loss change < this, it's a plateau

    def update_epsilon_curiosity(self):
        """
        Update epsilon based on learning progress (curiosity-driven).

        If learning is plateauing (loss not changing much), increase epsilon to explore more.
        If actively learning (loss improving), decrease epsilon to exploit more.
        """
        if len(self.recent_losses) < 10:
            # Not enough data yet, keep current epsilon
            return

        # Calculate trend in recent losses
        recent_10 = list(self.recent_losses)[-10:]
        recent_5 = list(self.recent_losses)[-5:]

        avg_older = sum(recent_10[:5]) / 5
        avg_recent = sum(recent_5) / 5

        # Check if we're plateauing or learning
        loss_change = abs(avg_older - avg_recent)

        if loss_change < self.plateau_threshold:
            # Plateau detected - increase exploration
            self.epsilon = min(self.epsilon_max, self.epsilon + self.epsilon_increase)
        else:
            # Actively learning - decrease exploration
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decrease)

    def update_epsilon(self, episode):
        """Update epsilon based on current episode (for compatibility)."""
        self.episode_count = episode
        # Curiosity-driven epsilon is updated in _train_network() based on loss

    def make_choice(self):
        """
        Makes a choice using epsilon-greedy policy with DQN.
        Returns: 'share' or 'steal'
        """
        state = self.opponent_history.get_state_vector()

        # Epsilon-greedy: explore vs exploit
        if random.random() < self.epsilon:
            # Explore: random choice
            action_idx = random.randint(0, 1)
        else:
            # Exploit: choose action with highest Q-value from network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()

        action = self.actions[action_idx]

        # Store for learning
        self.last_state = state
        self.last_action = action
        self.last_action_idx = action_idx

        return action

    def update_q_value(self, my_reward, opponent_reward, opponent_move):
        """
        Updates Q-network using experience replay and gradient descent.
        Also updates opponent history.
        """
        # Add to opponent history first
        self.opponent_history.add_round(opponent_move, my_reward, opponent_reward)

        if self.last_action is None:
            return

        # Get next state after adding to history
        next_state = self.opponent_history.get_state_vector()

        # Store transition in replay buffer
        # done is always False since this is a continuing game
        self.replay_buffer.push(self.last_state, self.last_action_idx, my_reward, next_state, False)

        # Train network if we have enough samples
        if len(self.replay_buffer) >= self.batch_size:
            self._train_network()

    def _train_network(self):
        """Train the Q-network using a batch from replay buffer."""
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)

        # Current Q-values
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + γ * max(Q(s', a'))
        with torch.no_grad():
            next_q_values = self.q_network(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.discount_factor * next_q_values

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Track loss for curiosity-driven exploration
        self.recent_losses.append(loss.item())

        # Update epsilon based on learning progress
        self.update_epsilon_curiosity()

    def get_opponent_analysis(self):
        """Returns analysis of opponent's behavior."""
        return self.opponent_history.analyze_opponent_pattern()


def play_round(prisoner1, prisoner2, choice1, choice2):
    """
    Plays one round of the Prisoner's Dilemma and updates both prisoners' money.

    Args:
        prisoner1: First Prisoner object
        prisoner2: Second Prisoner object
        choice1: Choice of prisoner1 ('share' or 'steal')
        choice2: Choice of prisoner2 ('share' or 'steal')

    Returns:
        Tuple of (reward1, reward2)

    Payoff matrix:
        Both share: each gets 3
        Both steal: each gets 1
        One steals, one shares: stealer gets 5, sharer gets 0
    """
    if choice1 == 'share' and choice2 == 'share':
        reward1, reward2 = 3, 3
    elif choice1 == 'steal' and choice2 == 'steal':
        reward1, reward2 = 1, 1
    elif choice1 == 'steal' and choice2 == 'share':
        reward1, reward2 = 5, 0
    else:  # choice1 == 'share' and choice2 == 'steal'
        reward1, reward2 = 0, 5

    prisoner1.net_money += reward1
    prisoner2.net_money += reward2

    return reward1, reward2


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Deep Q-Network (DQN) Prisoner's Dilemma Simulation")
    print("=" * 70)

    # Create two learning prisoners with DQN
    p1 = DQNPrisoner(learning_rate=0.001, discount_factor=0.9, initial_epsilon=0.5)
    p2 = DQNPrisoner(learning_rate=0.001, discount_factor=0.9, initial_epsilon=0.5)

    # Training phase
    num_episodes = 1000
    print(f"\nTraining prisoners with Deep Q-Network...")
    print(f"Network: 5-dim input -> 64 hidden -> 64 hidden -> 2 output")
    print(f"Learning rate: 0.001, Batch size: 32, Replay buffer: 10000")
    print(f"Initial epsilon: 0.5, Exploration: CURIOSITY-DRIVEN")
    print(f"  - Epsilon increases when learning plateaus (loss change < 0.01)")
    print(f"  - Epsilon decreases when actively learning")
    print(f"  - Range: [0.05, 0.8]")
    print(f"State representation: 5 recency-weighted bins covering entire history")
    print(f"Weighting: [Most recent 1/5 x 5, Next 1/5 x 4, Middle 1/5 x 3, Older 1/5 x 2, Oldest 1/5 x 1]\n")

    for episode in range(num_episodes):
        # Update epsilon for both prisoners
        p1.update_epsilon(episode)
        p2.update_epsilon(episode)

        # Make choices
        c1 = p1.make_choice()
        c2 = p2.make_choice()

        # Play round and get rewards
        r1, r2 = play_round(p1, p2, c1, c2)

        # Update Q-values with received rewards and opponent's move
        p1.update_q_value(r1, r2, c2)
        p2.update_q_value(r2, r1, c1)

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            p1_state = p1.opponent_history.get_state_vector()
            p1_avg_loss = sum(p1.recent_losses) / len(p1.recent_losses) if p1.recent_losses else 0
            print(f"Episode {episode + 1}: P1 money: {p1.net_money}, P2 money: {p2.net_money}, "
                  f"Epsilon: {p1.epsilon:.4f}, Avg Loss: {p1_avg_loss:.4f}")
            print(f"  P1 state vector: {p1_state}")

    print(f"\n{'=' * 70}")
    print(f"Final Results after {num_episodes} episodes:")
    print(f"{'=' * 70}")
    print(f"P1 total money: {p1.net_money}")
    print(f"P2 total money: {p2.net_money}")

    # Analyze opponent patterns
    print(f"\nP1's analysis of P2:")
    p1_analysis = p1.get_opponent_analysis()
    for key, value in p1_analysis.items():
        print(f"  {key}: {value:.3f}")

    print(f"\nP2's analysis of P1:")
    p2_analysis = p2.get_opponent_analysis()
    for key, value in p2_analysis.items():
        print(f"  {key}: {value:.3f}")

    # Show state vectors
    print(f"\nP1's current state vector (P2's behavior pattern):")
    p1_state = p1.opponent_history.get_state_vector()
    print(f"  {p1_state}")
    print(f"  (Each value = proportion of 'steal' in that time slot)")

    print(f"\nP2's current state vector (P1's behavior pattern):")
    p2_state = p2.opponent_history.get_state_vector()
    print(f"  {p2_state}")

    # Test phase with learned strategy (no exploration)
    print(f"\n{'=' * 70}")
    print("Testing learned strategies (20 rounds, no exploration)")
    print(f"{'=' * 70}")
    p1.epsilon = 0
    p2.epsilon = 0

    for i in range(20):
        # Get state vectors before making choices
        p1_state = p1.opponent_history.get_state_vector()
        p2_state = p2.opponent_history.get_state_vector()

        c1 = p1.make_choice()
        c2 = p2.make_choice()
        r1, r2 = play_round(p1, p2, c1, c2)

        print(f"Round {i+1}: P1 chose {c1}, P2 chose {c2} | Rewards: P1={r1}, P2={r2}")
        print(f"  P1's state (P2's pattern): {p1_state}")
        print(f"  P2's state (P1's pattern): {p2_state}")

        # Still update history during testing
        p1.update_q_value(r1, r2, c2)
        p2.update_q_value(r2, r1, c1)

    print(f"\n{'=' * 70}")
    print("Simulation Complete!")
    print(f"{'=' * 70}")
