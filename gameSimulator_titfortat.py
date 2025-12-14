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
    Default is 2 rounds (current + next) to teach immediate consequences.
    """
    def __init__(self, learning_rate=0.001, discount_factor=0.9, initial_epsilon=0.5, lookahead_rounds=1):
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


class TitForTatPrisoner:
    """
    Tit-for-Tat strategy with constant forgiveness: Start with cooperation, then copy opponent's last move.
    Always applies 10% forgiveness (randomly flips decision).
    This is one of the most successful strategies in iterated prisoner's dilemma.
    """
    def __init__(self, forgiveness_rate=0.1):
        self.net_money = 0
        self.opponent_history = OpponentHistory()
        self.last_opponent_move = None
        self.forgiveness_rate = forgiveness_rate
        self.my_last_move = None

    def _detect_alternating_loop(self):
        """
        Detects if we're in a share-steal-share-steal loop.
        Checks last 4 moves: opponent alternates between share/steal, we alternate opposite.
        """
        if len(self.opponent_history.history) < 3:
            return False

        # Get last 3 opponent moves
        recent = self.opponent_history.history[-3:]
        moves = [r['move'] for r in recent]

        # Check if alternating pattern: share, steal, share OR steal, share, steal
        pattern1 = moves == ['share', 'steal', 'share']
        pattern2 = moves == ['steal', 'share', 'steal']

        return pattern1 or pattern2

    def make_choice(self):
        """
        Makes a choice using Tit-for-Tat strategy with constant 10% forgiveness.
        - First move: always share (cooperate)
        - Subsequent moves: copy opponent's last move
        - 10% chance to flip the decision (forgiveness mechanism)
        """
        if self.last_opponent_move is None:
            # First move: cooperate
            choice = 'share'
        else:
            # Copy opponent's last move
            choice = self.last_opponent_move

            # Apply 10% forgiveness randomly (always, not conditionally)
            if random.random() < self.forgiveness_rate:
                choice = 'share' if choice == 'steal' else 'steal'

        self.my_last_move = choice
        return choice

    def update_q_value(self, my_reward, opponent_reward, opponent_move):
        """Updates history and remembers opponent's last move."""
        self.opponent_history.add_round(opponent_move, my_reward, opponent_reward)
        self.last_opponent_move = opponent_move

    def update_epsilon(self, episode):
        """No-op for Tit-for-Tat."""
        pass

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
    print("DQN vs Tit-for-Tat Strategy")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("PHASE 1: AI SELF-TRAINING")
    print("=" * 70)
    print("\nTraining DQN agent against itself for 1000 episodes...")

    # Train the AI against itself
    ai_trainer_1 = DQNPrisoner(learning_rate=0.001, discount_factor=0.9, initial_epsilon=0.5)
    ai_trainer_2 = DQNPrisoner(learning_rate=0.001, discount_factor=0.9, initial_epsilon=0.5)

    num_training_episodes = 1000
    for episode in range(num_training_episodes):
        ai_trainer_1.update_epsilon(episode)
        ai_trainer_2.update_epsilon(episode)

        c1 = ai_trainer_1.make_choice()
        c2 = ai_trainer_2.make_choice()

        r1, r2 = play_round(ai_trainer_1, ai_trainer_2, c1, c2)

        ai_trainer_1.update_q_value(r1, r2, c2)
        ai_trainer_2.update_q_value(r2, r1, c1)

        if (episode + 1) % 250 == 0:
            print(f"  Episode {episode + 1}/{num_training_episodes}: AI1={ai_trainer_1.net_money}, AI2={ai_trainer_2.net_money}")

    print(f"\nSelf-training complete!")
    ai1_analysis = ai_trainer_1.get_opponent_analysis()
    print(f"Training result: Agents learned to steal ~{ai1_analysis['steal_rate']*100:.1f}% of the time")

    # Now test against Tit-for-Tat
    print("\n" + "=" * 70)
    print("PHASE 2: DQN vs TIT-FOR-TAT (with Forgiveness)")
    print("=" * 70)

    print("\nTit-for-Tat with Constant Forgiveness Strategy:")
    print("  - First move: Always cooperate (share)")
    print("  - Every other move: Copy opponent's last move")
    print("  - 10% chance to flip the decision (constant forgiveness)")
    print("\nThis forgiveness helps break cycles of mutual defection!")

    # Create new DQN agent from scratch (not using pre-trained weights)
    # Using no lookahead (standard 1-round rewards)
    dqn_agent = DQNPrisoner(learning_rate=0.001, discount_factor=0.9, initial_epsilon=1.0, lookahead_rounds=1)
    tft_agent = TitForTatPrisoner()

    # Phase 2a: Random exploration (no learning)
    exploration_rounds = 1000
    print(f"\nPhase 2a: Random exploration for {exploration_rounds} rounds (no learning)...")
    print("Building experience buffer without training...\n")

    for round_num in range(exploration_rounds):
        # Force random exploration
        dqn_agent.epsilon = 1.0  # Always random

        # Get choices
        dqn_choice = dqn_agent.make_choice()
        tft_choice = tft_agent.make_choice()

        # Play round
        r_dqn, r_tft = play_round(dqn_agent, tft_agent, dqn_choice, tft_choice)

        # Add to history and replay buffer, but DON'T train
        dqn_agent.opponent_history.add_round(tft_choice, r_dqn, r_tft)
        tft_agent.update_q_value(r_tft, r_dqn, dqn_choice)

        # Store in replay buffer without training
        if dqn_agent.last_state is not None:
            next_state = dqn_agent.opponent_history.get_state_vector()
            dqn_agent.replay_buffer.push(dqn_agent.last_state, dqn_agent.last_action_idx, r_dqn, next_state, False)

        # Update last state/action for next iteration
        state = dqn_agent.opponent_history.get_state_vector()
        dqn_agent.last_state = state.copy()
        dqn_agent.last_action_idx = 0 if dqn_choice == 'share' else 1

        # Print progress
        if (round_num + 1) % 250 == 0:
            print(f"  Round {round_num + 1}/{exploration_rounds}: DQN={dqn_agent.net_money}, TFT={tft_agent.net_money}, Buffer size={len(dqn_agent.replay_buffer)}")

    print(f"\nExploration complete! Buffer has {len(dqn_agent.replay_buffer)} experiences.")
    print(f"Scores after exploration: DQN={dqn_agent.net_money}, TFT={tft_agent.net_money}")

    # Phase 2b: Training phase
    training_rounds = 40000
    print(f"\nPhase 2b: Training for {training_rounds} rounds...")
    print("Now the DQN will learn from experience!")
    print("**NO LOOKAHEAD:** Each decision's reward = current round only")
    print("This teaches the agent immediate consequences of their choices!\n")

    # Reset epsilon for learning with exploration
    dqn_agent.epsilon = 0.3  # Start with some exploration

    for round_num in range(training_rounds):
        # Get choices
        dqn_choice = dqn_agent.make_choice()
        tft_choice = tft_agent.make_choice()

        # Play round
        r_dqn, r_tft = play_round(dqn_agent, tft_agent, dqn_choice, tft_choice)

        # Update with learning enabled
        dqn_agent.update_q_value(r_dqn, r_tft, tft_choice)
        tft_agent.update_q_value(r_tft, r_dqn, dqn_choice)

        # Print progress
        if (round_num + 1) % 500 == 0:
            analysis = dqn_agent.get_opponent_analysis()
            avg_loss = sum(dqn_agent.recent_losses) / len(dqn_agent.recent_losses) if dqn_agent.recent_losses else 0
            print(f"  Round {round_num + 1}/{training_rounds}: DQN={dqn_agent.net_money}, TFT={tft_agent.net_money}, "
                  f"Epsilon={dqn_agent.epsilon:.3f}, Loss={avg_loss:.2f}, DQN share%={analysis['share_rate']*100:.1f}%")

    # Final results
    total_rounds = exploration_rounds + training_rounds
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nScores after {total_rounds} rounds ({exploration_rounds} exploration + {training_rounds} training):")
    print(f"  DQN Agent:     {dqn_agent.net_money} points ({dqn_agent.net_money/total_rounds:.2f} per round)")
    print(f"  Tit-for-Tat:   {tft_agent.net_money} points ({tft_agent.net_money/total_rounds:.2f} per round)")

    difference = dqn_agent.net_money - tft_agent.net_money
    if difference > 0:
        print(f"\n🤖 DQN WINS by {difference} points!")
    elif difference < 0:
        print(f"\n🔄 TIT-FOR-TAT WINS by {-difference} points!")
    else:
        print(f"\n🤝 It's a TIE!")

    # Show detailed statistics
    print(f"\n" + "=" * 70)
    print("DETAILED STATISTICS")
    print("=" * 70)

    dqn_analysis = dqn_agent.get_opponent_analysis()
    tft_analysis = tft_agent.get_opponent_analysis()

    print(f"\nDQN Agent's behavior:")
    print(f"  Share rate: {tft_analysis['share_rate']*100:.1f}%")
    print(f"  Steal rate: {tft_analysis['steal_rate']*100:.1f}%")
    print(f"  Avg points per round: {tft_analysis['avg_opp_reward']:.2f}")

    print(f"\nTit-for-Tat's behavior:")
    print(f"  Share rate: {dqn_analysis['share_rate']*100:.1f}%")
    print(f"  Steal rate: {dqn_analysis['steal_rate']*100:.1f}%")
    print(f"  Avg points per round: {dqn_analysis['avg_opp_reward']:.2f}")

    print(f"\nDQN's state vector (understanding of TFT):")
    print(f"  {dqn_agent.opponent_history.get_state_vector()}")

    # Interpretation
    print(f"\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if tft_analysis['share_rate'] > 0.7:
        print("\n✓ DQN learned to cooperate! When facing Tit-for-Tat's conditional")
        print("  cooperation, the DQN agent discovered that sharing is more profitable.")
    elif tft_analysis['steal_rate'] > 0.7:
        print("\n✗ DQN exploited Tit-for-Tat. The agent tried to steal frequently,")
        print("  but TFT retaliated by copying the stealing behavior.")
    else:
        print("\n≈ Mixed strategy. DQN showed some cooperation but also attempted")
        print("  exploitation, leading to a mixed outcome.")

    mutual_coop_score = total_rounds * 3  # If both always cooperated
    mutual_defect_score = total_rounds * 1  # If both always defected

    print(f"\nReference scores (for {total_rounds} rounds):")
    print(f"  Perfect cooperation (both always share): {mutual_coop_score} each")
    print(f"  Mutual defection (both always steal):    {mutual_defect_score} each")
    print(f"  Maximum exploitation (always steal vs always share): {total_rounds*5} vs 0")

    print("\n" + "=" * 70)
