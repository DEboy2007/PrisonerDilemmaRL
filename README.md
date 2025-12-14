# Reinforcement Learning: Prisoner's Dilemma Experiments

A comprehensive exploration of Deep Q-Network (DQN) agents learning to navigate the classic prisoner's dilemma game through various experimental setups.

## Table of Contents

- [Overview](#overview)
- [Technical Architecture](#technical-architecture)
- [Experiments and Results](#experiments--results)
- [Key Findings](#key-findings)
- [Comparative Summary](#comparative-summary)
- [Technical Deep Dive](#technical-deep-dive)
- [Lessons Learned](#lessons-learned)
- [Potential Improvements](#potential-improvements)

## Overview

This project implements a Deep Q-Network (DQN) reinforcement learning agent to learn optimal strategies in the iterated prisoner's dilemma. We explored how different initial conditions, opponent strategies, and learning parameters affect the emergence of cooperation vs. defection.

### The Prisoner's Dilemma

A fundamental game theory problem where two players must decide whether to cooperate (share) or defect (steal):

**Payoff Matrix:**
| P1 \ P2 | Share | Steal |
|---------|-------|-------|
| Share   | 3, 3  | 0, 5  |
| Steal   | 5, 0  | 1, 1  |

**Nash Equilibrium:** Both players steal (1, 1)

**Pareto Optimal:** Both players share (3, 3)

The dilemma: Individual rationality leads to mutual defection, but cooperation yields better outcomes.

---

## Technical Architecture

### State Representation
**5-dimensional continuous state vector** with time-weighted bins:
- Divides opponent's complete history into 5 equal time bins
- Each bin covers 1/5th of total history
- Recency weighting: Most recent bin × 5, oldest bin × 1
- Values represent weighted steal rates in each time period

Example state vector: `[4.75, 3.6, 3.0, 2.0, 0.95]` means opponent recently steals ~95%, historically less.

### Deep Q-Network Architecture
```
Input: 5D state vector
├─ FC1: 5 → 64 (ReLU)
├─ FC2: 64 → 64 (ReLU)
└─ FC3: 64 → 2 (Q-values for share/steal)
```

**Training Features:**
- Experience replay buffer (capacity: 10,000)
- Batch size: 32
- Learning rate: 0.001
- Discount factor (γ): 0.9
- Curiosity-driven epsilon (adaptive exploration)
  - Increases when learning plateaus
  - Decreases when actively learning
  - Range: [0.05, 0.8]

---

## Experiments & Results

### Experiment 1: Neutral Self-Play (Two DQN Agents)
**File:** `gameSimulator.py`

**Setup:**
- Both agents start with neutral 50% share belief
- 1000 training episodes
- No prior assumptions

**Results:**
- **P1:** 1224 points (8.4% share, 91.6% steal)
- **P2:** 1284 points (9.6% share, 90.4% steal)
- **Average:** 1.25 points/round

**Conclusion:** ✗ Converged to Nash equilibrium (mutual defection). Agents learned that stealing is the dominant strategy when facing another learning agent.

---

### Experiment 2: Symmetric Optimistic Prior
**File:** `gameSimulator_optimistic.py`

**Setup:**
- Both agents start believing opponent shares 80% of time
- 1000 training episodes
- Optimistic initialization for both

**Results:**
- **P1:** 1091 points (12.6% share, 87.4% steal)
- **P2:** 1431 points (5.8% share, 94.2% steal)
- **Average:** 1.26 points/round

**Conclusion:** ✗ **Worse than neutral!** P2 exploited P1's optimism early, gaining 340-point advantage.

---

### Experiment 3: Asymmetric Trust
**File:** `gameSimulator_asymmetric.py`

**Setup:**
- P1: Optimistic (80% share belief)
- P2: Neutral (50% share belief)
- 1000 training episodes

**Results:**
- **P1:** 1389 points (14.3% share, 85.7% steal)
- **P2:** 1394 points (14.2% share, 85.8% steal)
- **Average:** 1.39 points/round

**Conclusion:** Near-parity despite trust asymmetry. Being the only trusting agent created limited cooperation window, then both converged to balanced mixed strategy.

### Experiment 4: DQN vs Standard Tit-for-Tat (NO Forgiveness, NO Lookahead, 40k rounds)
**File:** `gameSimulator_tft_standard.py`

**Setup:**
- **Phase 1:** 1000 rounds pure random exploration (no learning)
  - Builds diverse experience buffer
- **Phase 2:** 40,000 rounds of training against Standard TFT
  - **Total: 41,000 rounds**
- **Standard Tit-for-Tat**: NO forgiveness mechanism (pure TFT from Axelrod's tournaments)
- **DQN Agent**: Standard 1-round rewards (NO lookahead)
- **Rationale for 40k rounds:** The agent needs extensive experience to observe when cooperation works. With no forgiveness mechanism to create escape routes, the agent must learn purely from exploration and long-term pattern recognition. More episodes = more opportunities to discover that sustained cooperation is more profitable than mutual defection.

**Results:**
- **DQN:** 104,427 points (75.2% share)
- **Standard TFT:** 104,427 points (75.2% share)
- **Average:** 2.55 points/round

**Share Rate Progression:**
- Round 500: 40.2%
- Round 5000: 49.5%
- Round 10,000: 48.0%
- Round 20,000: 61.9%
- Round 30,000: 70.7%
- Round 40,000: 75.2%

**Conclusion:** **Cooperation emerged WITHOUT forgiveness.** With sufficient training time (40k rounds), the DQN learned to cooperate even against unforgiving Tit-for-Tat.

**Key Insights:**
1. **No forgiveness needed with enough training:** 75.2% cooperation achieved through pure learning
2. **Slow but steady convergence:** Share rate gradually increased from 40% → 75% over 40k rounds
3. **Training time compensates for lack of forgiveness:** The agent eventually learned that cooperation is more profitable
4. **85% of theoretical maximum:** Achieved 2.55 points/round vs 3.0 optimal

---

### Experiment 5: DQN vs Standard Tit-for-Tat (NO Forgiveness, 5-Round Lookahead, 40k rounds)
**File:** `gameSimulator_tft_5round.py`

**Setup:**
- **Phase 1:** 1000 rounds pure random exploration (no learning)
- **Phase 2:** 40,000 rounds of training against Standard TFT
  - **Total: 41,000 rounds**
- **Standard Tit-for-Tat**: NO forgiveness mechanism (pure TFT)
- **DQN Agent**: 5-round lookahead rewards (sum of next 5 rounds)
- **Rationale:** Tests if medium-term lookahead alone can enable cooperation without forgiveness

**Results:**
- **DQN:** 119,944 points (94.6% share!)
- **Standard TFT:** 119,944 points (94.6% share)
- **Average:** 2.93 points/round

**Share Rate Progression:**
- Round 500: 60.7%
- Round 1000: 68.4%
- Round 2000: 76.5%
- Round 5000: 85.9%
- Round 10,000: 89.9%
- Round 20,000: 92.8%
- Round 30,000: 94.0%
- Round 40,000: 94.6%

**Conclusion:** **Best result without forgiveness** The 5-round lookahead achieved near-perfect cooperation (94.6%) against unforgiving TFT.

**Key Insights:**
1. **Lookahead dramatically accelerates learning:** Reached 60% cooperation by round 500 (vs 40% for 1-round)
2. **Faster convergence:** Achieved 85% cooperation by round 5,000 (vs 49.5% for 1-round)
3. **Higher ceiling:** Final 94.6% cooperation vs 75.2% for 1-round
4. **97.7% of theoretical maximum:** Achieved 2.93 points/round vs 3.0 optimal

**Why 5-Round Lookahead Works So Well:**
1. **Medium-term consequences visible:** Agent sees that stealing triggers 5 rounds of retaliation
2. **Clear cost-benefit:** 5 rounds of mutual defection (5 points) vs 5 rounds of cooperation (15 points)

**Comparison: 1-Round vs 5-Round Lookahead (both NO forgiveness):**

| Lookahead | Final Share % | Points/Round | Round 5k Share % | Convergence Speed |
|-----------|---------------|--------------|------------------|-------------------|
| **1-round** | 75.2% | 2.55 | 49.5% | Slow |
| **5-round** | **94.6%** | **2.93** | **85.9%** | **Fast** |

**19.4% improvement in cooperation!** The 5-round lookahead provided enough temporal context to understand cooperation value without needing forgiveness to create escape routes.

---

### Experiment 6: DQN vs Tit-for-Tat with Constant Forgiveness (40k rounds)
**File:** `gameSimulator_titfortat.py` (with 10% constant forgiveness)

**Setup:**
- **Phase 1:** 1000 rounds pure random exploration (no learning)
- **Phase 2:** 40,000 rounds of training
  - **Total: 41,000 rounds**
- **Tit-for-Tat**: 10% constant forgiveness (randomly flips decision every round)
- **DQN Agent**: Standard 1-round rewards (NO lookahead)
- **Rationale:** Tests if forgiveness mechanism can enable cooperation without lookahead

**Results:**
- **DQN:** \~117,000 points (~90% share)
- **Forgiving TFT:** \~117,000 points (~90% share)
- **Average:** \~2.87 points/round

**Conclusion:** **Forgiveness enables cooperation!** The 10% forgiveness rate created escape routes from mutual defection, allowing the agent to learn cooperation even with standard 1-round rewards.

---

### Experiment 7: DQN vs DQN with 5-Round Lookahead
**File:** `gameSimulator_dqn_vs_dqn.py`

**Setup:**
- Both agents are DQN with 5-round lookahead
- 100 exploration rounds + 40,000 training rounds
- Tests if mutual lookahead enables cooperation between two learning agents

**Hypothesis:** If lookahead helps DQN learn cooperation against TFT, maybe two DQNs with lookahead can cooperate with each other?

**Results:**
- **DQN1:** 44,374 points (3.5% share, 96.5% steal)
- **DQN2:** 44,274 points (3.6% share, 96.4% steal)
- **Average:** 1.11 points/round

**Conclusion:** **FAILURE - Worse than Baseline!**

Even with 5-round lookahead, two learning agents converged to 96.5% defection - the worst result across all experiments except locked TFT defection.

**Why This Failed:**
1. **Co-evolution trap:** Both agents learning simultaneously means no stable strategy to learn from
2. **Exploitation reinforcement:** Early successful exploitations got reinforced through lookahead
3. **No forgiveness:** Neither agent had built-in forgiveness to break cycles
4. **Moving target problem:** Each agent's optimal strategy depends on opponent, but opponent keeps changing

Two learning agents with lookahead but no cooperation mechanisms simply optimize toward mutual defection faster.

---

## Key Findings

### 1. **Trust is Risky in Competitive Settings**
Symmetric optimism (Exp 2) led to exploitation. One agent gained 340-point advantage by betraying early trust. In zero-sum learning environments, cooperation is fragile.

### 2. **Training Duration is Critical**
**The most important finding:** With 40,000 training rounds, cooperation emerges even without forgiveness or lookahead mechanisms:
- **1-round, NO forgiveness (Exp 4):** 75.2% cooperation, 2.55 points/round
- **5-round, NO forgiveness (Exp 5):** 94.6% cooperation, 2.93 points/round
- **1-round, WITH forgiveness (Exp 6):** ~90% cooperation, 2.87 points/round

The key insight: **Agents need extensive experience to observe when cooperation works.** More episodes = more opportunities to discover that sustained cooperation is more profitable than mutual defection.

### 3. **Multiple Paths to Cooperation**
Three different mechanisms can achieve ~90-95% cooperation with 40k training:
1. **5-round lookahead, NO forgiveness:** 94.6% cooperation (best)
2. **1-round, WITH 10% forgiveness:** ~90% cooperation
3. **1-round, NO forgiveness:** 75.2% cooperation (slowest but still works)

This demonstrates cooperation is achievable through multiple approaches - the key is sufficient training time.

### 4. **Nash Equilibrium ≠ Optimal Strategy**
Against Tit-for-Tat, pure cooperation (always share) is optimal, but agents often converge to mixed strategies or mutual defection instead without sufficient training.

### 5. **Co-Evolution Trap: Learning Agents Need Stable Opponents**
When both agents learn simultaneously (Exp 7: DQN vs DQN), even with 5-round lookahead, they converged to 96.5% defection - worse than any other experiment. The moving target problem prevents cooperation emergence. **Stable opponent strategies (like TFT) are essential for learning cooperation.**

---

## Comparative Summary

### Points Per Round (Higher = Better)

| Experiment | Avg Points/Round | Key Feature |
|------------|------------------|-------------|
| **DQN vs DQN (5-round lookahead)** | **1.11** | **Co-evolution trap** |
| Neutral Self-Play | 1.25 | Baseline - mutual defection |
| Symmetric Optimism | 1.26 | Exploitation punishes trust |
| **Asymmetric Trust** | **1.39** | Best competitive outcome |
| **1-Round, NO forgiveness, 40k** | **2.55** | **Slow but steady learning** |
| **1-Round, WITH forgiveness, 40k** | **~2.87** | **Forgiveness creates escape routes** |
| **5-Round, NO forgiveness, 40k** | **2.93** | **BEST: Lookahead reveals cooperation value** |
| **Theoretical Maximum** | **3.00** | Perfect cooperation |

### Share Rate (Higher = More Cooperative)

| Experiment | DQN Share % | Opponent Share % | Mechanism |
|------------|-------------|------------------|-----------|
| **DQN vs DQN (5-round lookahead)** | **3.5%** | **3.6%** | Co-evolution trap |
| Neutral | 8.4% | 9.6% | Baseline |
| Symmetric Optimism | 12.6% | 5.8% | Exploitation |
| Asymmetric | 14.3% | 14.2% | Asymmetric trust |
| **1-Round, NO forgiveness, 40k** | **75.2%** | **75.2%** | **Pure learning** |
| **1-Round, WITH forgiveness, 40k** | **~90%** | **~90%** | **Forgiveness** |
| **5-Round, NO forgiveness, 40k** | **94.6%** | **94.6%** | **Lookahead** |

### Convergence Speed Comparison (40k rounds, NO forgiveness)

| Lookahead | Round 500 | Round 5k | Round 20k | Final (40k) |
|-----------|-----------|----------|-----------|-------------|
| **1-round** | 40.2% | 49.5% | 61.9% | **75.2%** |
| **5-round** | 60.7% | 85.9% | 92.8% | **94.6%** |

---

## Technical Deep Dive

### Why Deep Q-Network?
- **Continuous State Space:** Our 5D weighted state vector can't use traditional Q-tables

**Why recency weighting?**
- Recent behavior more predictive than distant history
- Helps agent react to opponent strategy shifts

### Curiosity-Driven Exploration
Traditional epsilon decay:
```python
epsilon = max(epsilon_min, epsilon * decay_rate)
```

Our adaptive approach:
```python
if loss_change < threshold:
    epsilon ↑  # Plateau detected - explore more
else:
    epsilon ↓  # Learning - exploit more
```

**Benefits:**
- Automatically increases exploration when stuck
- Reduces exploration when discovering new patterns
- Avoids premature convergence

---

## Lessons Learned

### For Cooperation to Emerge:
1. ✓ **Sufficient training time** (40,000 rounds) - Essential foundation (Exp 4-6)
2. ✓ **Stable opponent strategy** (TFT) - Prevents co-evolution trap (Exp 7)
3. ✓ **Lookahead rewards** - Dramatically accelerates learning (Exp 5: 94.6% cooperation)
4. ✓ **Forgiveness mechanisms** - Creates escape routes (Exp 6: ~90% cooperation)
5. ✓ **Curiosity-driven epsilon** - Enables exploration when plateaued (Exp 4)
6. ✗ **Optimistic priors alone** aren't enough (Exp 2)
7. ✗ **Two learning agents** won't cooperate (Exp 7: co-evolution trap)

### Why Pure Cooperation is Hard to Learn:
1. **Early exploitation pays off:** Stealing once → +5 vs +3
2. **Immediate punishment:** TFT retaliates instantly
3. **Risk aversion:** Learned to avoid being the "sucker" (0 points)
4. **Local optima:** Mixed strategies are stable but suboptimal
5. **Exploration bias:** Random exploration generates many defection examples
6. **Credit assignment:** Hard to connect current action to long-term outcome without lookahead

### Key Design Choices That Enabled Success:
1. **40,000 training rounds:** Enough experience to observe cooperation working
2. **Curiosity-driven epsilon:** Increased exploration when learning plateaued, creating brief cooperation windows
3. **Recency-weighted state:** Recent opponent behavior more predictive than distant history
4. **5-round lookahead:** Optimal horizon - shows retaliation consequences without obscuring signal

## Potential Improvements:
1. We are considering a multi-agent environment and modifying the state vector to only include the last few decisions to see if the TFT strategy emerges.
2. Considering how to change the state vector into something more reflective of this situation. Some options are a single value to record previous moves, a vector recording only the last 10 moves, or other ways to record the history of the opponent and the player.

---

*Experiment conducted: December 2025*

*Authors:* Dhanush Ekollu, Yash Shah

*Framework: PyTorch, Python 3.x*

*Inspiration: Axelrod's tournaments, game theory, multi-agent RL*
