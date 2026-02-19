# Blackjack Reinforcement Learning

A research project comparing Deep Q-Network (DQN) and tabular Q-learning agents trained to play multi-deck blackjack with card counting, split decisions, and adaptive betting strategies.

## Overview

This project investigates whether reinforcement learning agents can learn optimal blackjack strategy from scratch — and whether integrating card counting signals into the state representation yields a measurable advantage. Three agent types are trained and evaluated against mathematically optimal basic strategy across all hard totals, soft totals, and pair splits.

**Key research questions:**
- Can a Dueling DQN agent converge to basic strategy without supervised labels?
- Do card counting systems (Hi-Lo, Zen, Uston APC, Ten Count) provide a learnable advantage when included in the state?
- How does tabular Q-learning compare to deep RL at this task?
- Can a separate neural network learn to size bets proportionally to card counting advantage?

## Architecture

### Agents

| Agent | File | Description |
|---|---|---|
| `DQNAgent` | `agents/DQNAgent.py` | Dueling DQN with policy/target network pair and action masking |
| `QStateAgent` | `agents/QStateAgent.py` | Tabular Q-learning with a target Q-table for stable updates |
| `BasicStrategyAgent` | `agents/BasicStrategyAgent.py` | Rule-based baseline implementing mathematically optimal basic strategy |
| `BettingRLAgent` | `agents/BettingAgent.py` | DQN-based agent for discrete bet sizing conditioned on count and bankroll |
| `BettingNN` | `agents/BettingModel.py` | Supervised neural network trained on a precomputed count-to-reward dataset |

### Neural Network Architecture (DQN)

The DQN agent uses a **Dueling DQN** architecture, which separates value and advantage estimation:

```
Input (state)
    └── Shared Feature Layers (Linear 64 → 64 → 64, ReLU)
            ├── Value Stream    (64 → 32 → 16 → 1)
            └── Advantage Stream (64 → 32 → 16 → num_actions)
                        └── Q(s, a) = V(s) + A(s,a) − mean(A(s,·))
```

Actions: **Hit (0), Stand (1), Double (2), Split (3)**

### Environment

`environment/split_environment.py` implements a 6-deck blackjack shoe with:
- Automatic reshuffling at 25% penetration with count resets
- Split hands tracked via `utils/SplitTracker.py` across a binary tree of sub-hands
- Double-down support with proper reward scaling
- Decaying reward bonuses to guide early exploration of doubling and splitting

`environment/environment.py` is a legacy version retained for reference (no split support).

### Card Counting Systems

Each system assigns a running count to dealt cards, normalized by decks remaining (true count). Supported systems:

| System | Low Cards (2–6) | Mid Cards (7–9) | High Cards (10–A) |
|---|---|---|---|
| Hi-Lo | +1 | 0 | −1 |
| Zen Count | +1 to +2 | 0 to +1 | −1 to −2 |
| Uston APC | +1 to +3 | −1 to +2 | −3 |
| Ten Count | +4 | +4 | −9 |

An `empty` mode trains without count information; `full` mode includes per-card-value deck composition percentages (10-dimensional).

### State Space

| Count Mode | State Dimensions | Contents |
|---|---|---|
| `empty` | 5 | player sum, dealer upcard, usable ace, can double, can split |
| `hi_lo` / `zen` / `uston_apc` / `ten_count` | 6 | basic state + true count (normalized) |
| `full` | 15 | basic state + 10 per-card-value percentages |

### Betting System

A separate `BettingRLAgent` (DQN) and supervised `BettingNN` learn to size bets based on the card count. The supervised model is trained on a precomputed dictionary of ~10 million game simulations mapping count states to average outcomes (`data/count_reward_dict_10mil.pkl`).

## Installation

```bash
git clone https://github.com/Hanlin2005/blackjack_661_project.git
cd blackjack_661_project
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch, NumPy, Matplotlib

## Usage

All commands should be run from the project root directory.

### Train a new DQN agent

```bash
# Train with Hi-Lo counting for 500,000 episodes
python main.py --count_type hi_lo --episodes 500000

# Train with Zen counting
python main.py --count_type zen --episodes 500000

# Train without card counting
python main.py --count_type empty --episodes 500000
```

Supported `--count_type` values: `empty`, `hi_lo`, `zen`, `uston_apc`, `ten_count`, `full`

### Load and evaluate an existing model

```bash
python main.py --load checkpoints/dqn/blackjack_agent_hi_lo_FINAL.pth --count_type hi_lo
```

### Continue training a saved model

```bash
python main.py \
  --load checkpoints/dqn/blackjack_agent_hi_lo_FINAL.pth \
  --count_type hi_lo \
  --continue_training \
  --episodes 100000
```

### Train a Q-learning agent

```bash
python training/train_q_agent.py
```

### Run hyperparameter grid search

```bash
python scripts/grid_search.py
python scripts/betting_grid_search.py
```

### Cross-agent performance comparison

```bash
python evaluation/plot_model_comparisons.py
```

### Interactive environment testing

```bash
python tests/env_testing.py
```

After training, evaluation automatically saves:
- **`results/figures/blackjack_strategy_comparison.png`** — heatmap grid showing agent vs. basic strategy match rate across hard totals, soft totals, and pairs
- **`results/figures/action_preferences.png`** — bar charts of action Q-value distribution for specific hand scenarios

## Results

Agents are benchmarked by comparing their action selection against mathematically optimal basic strategy across all 320+ hand/dealer combinations:

| Metric | Description |
|---|---|
| Hard Totals Accuracy | % match vs. basic strategy on hard hands (8–21) |
| Soft Totals Accuracy | % match vs. basic strategy on soft hands (A,2–A,9) |
| Pairs Accuracy | % match vs. basic strategy on all pair splits |
| Overall Accuracy | Unweighted average of the three categories |

## Project Structure

```
blackjack_661_project/
│
├── main.py                        # Entry point: train, load, and evaluate agents
├── README.md
├── requirements.txt
│
├── agents/                        # Agent implementations
│   ├── DQNAgent.py                # Dueling DQN with policy/target networks and action masking
│   ├── QStateAgent.py             # Tabular Q-learning with target Q-table
│   ├── BasicStrategyAgent.py      # Rule-based optimal baseline
│   ├── BettingAgent.py            # DQN-based discrete bet sizing agent
│   ├── BettingModel.py            # Supervised betting neural network + training loop
│   └── Supervised_betting.py      # Data collection and evaluation for supervised betting
│
├── environment/                   # Blackjack simulation
│   ├── split_environment.py       # 6-deck env with split, double, and card counting (primary)
│   ├── environment.py             # Legacy env without split support (retained for reference)
│   └── deck_classes.py            # Card and Deck data structures
│
├── training/                      # Training pipelines
│   ├── split_train_agent.py       # Main training loop (DQN + Q-table, with splits)
│   ├── train_agent.py             # Legacy training loop without split support
│   ├── train_q_agent.py           # Standalone Q-table training script
│   ├── train_qstate_agent.py      # Alternate Q-state training script
│   └── train_betting_agent.py     # Betting agent training and data collection
│
├── evaluation/                    # Evaluation and visualization
│   ├── modeling.py                # DQN strategy comparison vs. basic strategy + heatmaps
│   ├── q_state_modeling.py        # Q-table strategy comparison vs. basic strategy
│   └── plot_model_comparisons.py  # Cross-agent cumulative reward comparison plots
│
├── utils/                         # Shared utilities
│   ├── hyperparameters.py         # Centralized hyperparameter configuration
│   ├── replay_buffer.py           # Experience replay buffer
│   ├── exponential_decay.py       # Exponential decay scheduler
│   ├── epsilon_decayer.py         # Multi-mode epsilon decay (linear, exponential, RBED)
│   ├── RewardBonus.py             # Decaying per-action exploration bonus system
│   └── SplitTracker.py            # Binary tree tracking split sub-hand states
│
├── scripts/                       # Standalone experiment scripts
│   ├── grid_search.py             # Hyperparameter grid search for DQN agent
│   └── betting_grid_search.py     # Hyperparameter grid search for betting model
│
├── tests/                         # Testing and debugging scripts
│   ├── env_testing.py             # Interactive environment step-through tester
│   └── betting_agent_test.py      # Betting agent integration test
│
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── plotting.ipynb             # Result visualization and analysis
│   └── train_bet_test.ipynb       # Betting model training experiments
│
├── checkpoints/                   # Saved model weights (gitignored)
│   ├── dqn/                       # DQN model checkpoints (.pth)
│   ├── betting/                   # Betting model checkpoints (.pt, .pth)
│   └── qstate/                    # Q-table checkpoints (.json)
│
├── data/                          # Precomputed datasets (gitignored)
│   ├── count_reward_dict.pkl      # Count-to-reward dataset (1M games)
│   └── count_reward_dict_10mil.pkl # Count-to-reward dataset (10M games)
│
└── results/                       # Generated outputs (gitignored)
    ├── figures/                   # Strategy heatmaps and action preference charts
    └── graphs/                    # Training curve and evaluation reward plots
```

## Technologies

- **Python 3.8+**
- **PyTorch** — Dueling DQN, supervised betting neural network, model checkpointing
- **NumPy** — State representation, Q-table operations, array manipulation
- **Matplotlib** — Training curve visualization, strategy heatmaps, comparison plots
- **Jupyter** — Exploratory analysis and betting model experiments
