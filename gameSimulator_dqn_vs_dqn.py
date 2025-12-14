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
            return {'share_rate': 0, 'steal_rate': 0, 'avg_my_reward': 0, 'avg_opp_reward': 0, 'total_rounds': 0}

        shares = sum(1 for r in self.history if r['move'] == 'share')
        steals = sum(1 for r in self.history if r['move'] == 'steal')
        total = len(self.history)

        avg_my_reward = sum(r['my_reward'] for r in self.history) / total
        avg_opp_reward = sum(r['opponent_reward'] for r in self.history) / total

        return {
            'share_rate': shares / total,
            'steal_rate': steals / total,
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
    def __init__(self, capacity=1000000):
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
    Uses lookahead rewards: each decision's reward is the sum of the next N rounds.
    """
    def __init__(self, learning_rate=0.001, discount_factor=0.9, initial_epsilon=0.5, lookahead_rounds=5):
        self.net_money = 0
        self.discount_factor = discount_factor
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.episode_count = 0
        self.lookahead_rounds = lookahead_rounds

        # Neural network
        self.q_network = DQNetwork(state_dim=5, action_dim=2, hidden_dim=64)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.batch_size = 32

        # Track opponent's history
        self.opponent_history = OpponentHistory()

        # Action mapping
        self.actions = ['share', 'steal']

        # For tracking last state/action
        self.last_state = None
        self.last_action = None
        self.last_action_idx = None

        # Lookahead reward tracking
        self.reward_buffer = deque(maxlen=lookahead_rounds)  # Store recent rewards
        self.state_action_buffer = deque(maxlen=lookahead_rounds)  # Store (state, action) pairs

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
        Updates Q-network using lookahead rewards.
        Each decision's reward = sum of next N rounds, showing consequences of choices.
        Also updates opponent history.
        """
        # Add to opponent history first
        self.opponent_history.add_round(opponent_move, my_reward, opponent_reward)

        # Store current reward
        self.reward_buffer.append(my_reward)

        # Store current state and action if we made a choice
        if self.last_state is not None:
            current_state = self.last_state.copy()
            current_action = self.last_action_idx
            self.state_action_buffer.append((current_state, current_action))

            # Once we have lookahead_rounds of data, create training example
            if len(self.reward_buffer) >= self.lookahead_rounds:
                # Get the state/action from lookahead_rounds ago
                old_state, old_action = self.state_action_buffer[0]

                # Calculate lookahead reward: sum of next lookahead_rounds rewards
                lookahead_reward = sum(self.reward_buffer)

                # Get next state (current state)
                next_state = self.opponent_history.get_state_vector()

                # Store in replay buffer with lookahead reward
                self.replay_buffer.push(old_state, old_action, lookahead_reward, next_state, False)

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


# Main game
if __name__ == "__main__":
    print("=" * 70)
    print("DQN vs DQN: Both with 5-Round Lookahead")
    print("=" * 70)
    print("\nExperiment: Two DQN agents both using 5-round lookahead rewards")
    print("Question: Can mutual lookahead enable cooperation between learning agents?")
    print("\n5-round lookahead: Each decision's reward = sum of next 5 rounds")
    print("This teaches agents that current choices affect future outcomes.")

    # Create two DQN agents with 5-round lookahead
    dqn1 = DQNPrisoner(learning_rate=0.001, discount_factor=0.9, initial_epsilon=1.0, lookahead_rounds=5)
    dqn2 = DQNPrisoner(learning_rate=0.001, discount_factor=0.9, initial_epsilon=1.0, lookahead_rounds=5)

    # Phase 1: Random exploration (no learning)
    exploration_rounds = 100
    print(f"\n" + "=" * 70)
    print(f"PHASE 1: RANDOM EXPLORATION ({exploration_rounds} rounds)")
    print("=" * 70)
    print("Building experience buffer without training...\n")

    for round_num in range(exploration_rounds):
        # Force random exploration
        dqn1.epsilon = 1.0
        dqn2.epsilon = 1.0

        # Get choices
        choice1 = dqn1.make_choice()
        choice2 = dqn2.make_choice()

        # Play round
        r1, r2 = play_round(dqn1, dqn2, choice1, choice2)

        # Add to history and replay buffer without training
        dqn1.opponent_history.add_round(choice2, r1, r2)
        dqn2.opponent_history.add_round(choice1, r2, r1)

        # Store in replay buffers without training
        if dqn1.last_state is not None:
            next_state1 = dqn1.opponent_history.get_state_vector()
            dqn1.replay_buffer.push(dqn1.last_state, dqn1.last_action_idx, r1, next_state1, False)

        if dqn2.last_state is not None:
            next_state2 = dqn2.opponent_history.get_state_vector()
            dqn2.replay_buffer.push(dqn2.last_state, dqn2.last_action_idx, r2, next_state2, False)

        # Update last state/action for next iteration
        state1 = dqn1.opponent_history.get_state_vector()
        dqn1.last_state = state1.copy()
        dqn1.last_action_idx = 0 if choice1 == 'share' else 1

        state2 = dqn2.opponent_history.get_state_vector()
        dqn2.last_state = state2.copy()
        dqn2.last_action_idx = 0 if choice2 == 'share' else 1

        # Print progress
        if (round_num + 1) % 25 == 0:
            print(f"  Round {round_num + 1}/{exploration_rounds}: DQN1={dqn1.net_money}, DQN2={dqn2.net_money}")

    print(f"\nExploration complete!")
    print(f"Buffer sizes: DQN1={len(dqn1.replay_buffer)}, DQN2={len(dqn2.replay_buffer)}")
    print(f"Scores after exploration: DQN1={dqn1.net_money}, DQN2={dqn2.net_money}")

    # Phase 2: Training phase
    training_rounds = 40000
    print(f"\n" + "=" * 70)
    print(f"PHASE 2: TRAINING ({training_rounds} rounds)")
    print("=" * 70)
    print("Now both agents learn from experience with 5-round lookahead!")
    print("Hypothesis: Mutual lookahead may enable cooperation\n")

    # Reset epsilon for learning
    dqn1.epsilon = 0.3
    dqn2.epsilon = 0.3

    for round_num in range(training_rounds):
        # Get choices
        choice1 = dqn1.make_choice()
        choice2 = dqn2.make_choice()

        # Play round
        r1, r2 = play_round(dqn1, dqn2, choice1, choice2)

        # Update with learning enabled
        dqn1.update_q_value(r1, r2, choice2)
        dqn2.update_q_value(r2, r1, choice1)

        # Print progress
        if (round_num + 1) % 1000 == 0:
            analysis1 = dqn1.get_opponent_analysis()
            analysis2 = dqn2.get_opponent_analysis()
            avg_loss1 = sum(dqn1.recent_losses) / len(dqn1.recent_losses) if dqn1.recent_losses else 0
            avg_loss2 = sum(dqn2.recent_losses) / len(dqn2.recent_losses) if dqn2.recent_losses else 0
            print(f"  Round {round_num + 1}/{training_rounds}: "
                  f"DQN1={dqn1.net_money} (share={analysis2['share_rate']*100:.1f}%, ε={dqn1.epsilon:.3f}), "
                  f"DQN2={dqn2.net_money} (share={analysis1['share_rate']*100:.1f}%, ε={dqn2.epsilon:.3f})")

    # Final results
    total_rounds = exploration_rounds + training_rounds
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nScores after {total_rounds} rounds ({exploration_rounds} exploration + {training_rounds} training):")
    print(f"  DQN Agent 1:   {dqn1.net_money} points ({dqn1.net_money/total_rounds:.2f} per round)")
    print(f"  DQN Agent 2:   {dqn2.net_money} points ({dqn2.net_money/total_rounds:.2f} per round)")

    difference = abs(dqn1.net_money - dqn2.net_money)
    if dqn1.net_money > dqn2.net_money:
        print(f"\nDQN1 leads by {difference} points")
    elif dqn2.net_money > dqn1.net_money:
        print(f"\nDQN2 leads by {difference} points")
    else:
        print(f"\nPerfect tie!")

    # Show detailed statistics
    print(f"\n" + "=" * 70)
    print("DETAILED STATISTICS")
    print("=" * 70)

    analysis1 = dqn1.get_opponent_analysis()
    analysis2 = dqn2.get_opponent_analysis()

    print(f"\nDQN Agent 1's behavior:")
    print(f"  Share rate: {analysis2['share_rate']*100:.1f}%")
    print(f"  Steal rate: {analysis2['steal_rate']*100:.1f}%")
    print(f"  Avg points per round: {analysis2['avg_opp_reward']:.2f}")

    print(f"\nDQN Agent 2's behavior:")
    print(f"  Share rate: {analysis1['share_rate']*100:.1f}%")
    print(f"  Steal rate: {analysis1['steal_rate']*100:.1f}%")
    print(f"  Avg points per round: {analysis1['avg_opp_reward']:.2f}")

    print(f"\nDQN1's state vector (understanding of DQN2):")
    print(f"  {dqn1.opponent_history.get_state_vector()}")

    print(f"\nDQN2's state vector (understanding of DQN1):")
    print(f"  {dqn2.opponent_history.get_state_vector()}")

    # Interpretation
    print(f"\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    avg_share = (analysis1['share_rate'] + analysis2['share_rate']) / 2
    avg_points = (dqn1.net_money + dqn2.net_money) / (2 * total_rounds)

    if avg_share > 0.7:
        print(f"\n✓ COOPERATION EMERGED! Both agents learned that sharing is beneficial.")
        print(f"  Average share rate: {avg_share*100:.1f}%")
        print(f"  5-round lookahead successfully taught long-term value of cooperation.")
    elif avg_share > 0.4:
        print(f"\n≈ MIXED STRATEGY. Agents found a balance between cooperation and exploitation.")
        print(f"  Average share rate: {avg_share*100:.1f}%")
        print(f"  Neither full cooperation nor mutual defection - a middle ground.")
    else:
        print(f"\n✗ MUTUAL DEFECTION. Agents converged to stealing strategy.")
        print(f"  Average share rate: {avg_share*100:.1f}%")
        print(f"  5-round lookahead didn't overcome the defection equilibrium.")

    mutual_coop_score = total_rounds * 3  # If both always cooperated
    mutual_defect_score = total_rounds * 1  # If both always defected

    print(f"\nReference scores (for {total_rounds} rounds):")
    print(f"  Perfect cooperation (both always share): {mutual_coop_score} each")
    print(f"  Mutual defection (both always steal):    {mutual_defect_score} each")
    print(f"  Actual performance: {avg_points:.2f} per round ({avg_points/3*100:.1f}% of optimal)")

    print("\n" + "=" * 70)
