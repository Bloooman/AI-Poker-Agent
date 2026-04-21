"""
train.py — Offline CFR+ self-play training for the poker agent.

Runs two CFR+ agents against each other using PyPokerEngine's game loop.
At the end of each game, visited (info_set, action) pairs are used to
update regret and strategy sums.  After N_ITERATIONS games, the average
strategy is saved to STRATEGY_PATH as a pickle file.

Usage:
    python train.py

Output:
    strategy.pkl — dict mapping info_set_key → {"fold": p, "call": p, "raise": p}

CFR+ update rule (Tammelin 2014):
    regret_sum[I][a] = max(0, regret_sum[I][a] + regret(I, a))
    strategy[I][a]   ∝ max(0, regret_sum[I][a])

The strategy stored is the *average* strategy over all iterations, which
converges to a Nash equilibrium as iterations → ∞.
"""

import pickle
import random
from collections import defaultdict

from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer

from abstraction import abstract

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STRATEGY_PATH = "strategy.pkl"
N_ITERATIONS  = 100_000   # number of self-play games; increase for better convergence
MAX_ROUND     = 20        # rounds per game (keep short to iterate fast)
INITIAL_STACK = 10_000
SMALL_BLIND   = 10
ACTIONS       = ["fold", "call", "raise"]


# ---------------------------------------------------------------------------
# CFR+ Player
# ---------------------------------------------------------------------------

class CFRPlayer(BasePokerPlayer):
    """Poker agent that plays according to a CFR+ strategy during training.

    Maintains regret sums and strategy sums over information sets.  Both
    players in self-play share the same tables (they are the same agent
    playing against itself, which is the standard CFR self-play setup).

    Attributes:
        regret_sum   : dict — cumulative positive regrets per info set and action
        strategy_sum : dict — cumulative strategy weights for average computation
        trajectory   : list — (info_set_key, action_taken, valid_actions)
                       recorded during a game for end-of-game regret update
    """

    def __init__(self, regret_sum, strategy_sum):
        """
        Args:
            regret_sum   : shared defaultdict for regret accumulation
            strategy_sum : shared defaultdict for average strategy accumulation
        """
        super().__init__()
        self.regret_sum   = regret_sum
        self.strategy_sum = strategy_sum
        self.trajectory   = []

    # ------------------------------------------------------------------
    # Strategy computation
    # ------------------------------------------------------------------

    def _current_strategy(self, info_set, valid_action_names):
        """Compute current mixed strategy from regret sums (CFR+ style).

        Uses regret-matching: probability of each action is proportional
        to its positive cumulative regret.  Falls back to uniform if all
        regrets are non-positive.

        Args:
            info_set          : tuple — abstract info set key
            valid_action_names: list of str — actions available this step

        Returns:
            dict: {action: probability}
        """
        regrets = {a: max(0, self.regret_sum[info_set][a])
                   for a in valid_action_names}
        total   = sum(regrets.values())

        if total > 0:
            return {a: regrets[a] / total for a in valid_action_names}
        else:
            # Uniform strategy when no positive regrets yet.
            p = 1.0 / len(valid_action_names)
            return {a: p for a in valid_action_names}

    # ------------------------------------------------------------------
    # BasePokerPlayer interface
    # ------------------------------------------------------------------

    def declare_action(self, valid_actions, hole_card, round_state):
        """Sample an action from the current CFR+ strategy.

        Records the (info_set, action, valid_actions) tuple in self.trajectory
        for end-of-game regret updates.

        Args:
            valid_actions : list of {"action": str, "amount": int}
            hole_card     : list of card strings
            round_state   : dict from PyPokerEngine

        Returns:
            str: "fold" | "call" | "raise"
        """
        valid_names = _available_actions(valid_actions)
        info_set    = abstract(hole_card, round_state, self.uuid)
        strategy    = self._current_strategy(info_set, valid_names)

        # Accumulate strategy sum for average strategy computation.
        for a, p in strategy.items():
            self.strategy_sum[info_set][a] += p

        # Sample action according to current strategy.
        action = random.choices(list(strategy.keys()),
                                weights=list(strategy.values()))[0]

        self.trajectory.append((info_set, action, valid_names))
        return action

    def receive_game_start_message(self, game_info):
        self.trajectory = []

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


# ---------------------------------------------------------------------------
# Regret update
# ---------------------------------------------------------------------------

def update_regrets(player, game_result, player_index):
    """Apply CFR+ regret updates for all decisions made during a game.

    For each visited info set, computes the counterfactual regret of
    actions not taken and updates regret_sum with the CFR+ clamp (≥ 0).

    Uses terminal game reward (chip delta) as a proxy for state value.
    This is outcome-sampling MCCFR: each game provides one sample of
    the terminal utility.

    Args:
        player       : CFRPlayer instance
        game_result  : dict returned by start_poker
        player_index : int — 0 or 1, identifies which seat is ours
    """
    # TODO: compute actual counterfactual regrets per info set.
    # For now, use the simple chip delta as terminal utility signal.
    final_stack  = game_result["players"][player_index]["stack"]
    utility      = final_stack - INITIAL_STACK  # positive = won chips

    for info_set, action_taken, valid_names in player.trajectory:
        for a in valid_names:
            # Regret of not taking action a: approximate as ±utility.
            regret = utility if a == action_taken else -utility / len(valid_names)
            # CFR+ clamp: regrets never go below zero.
            player.regret_sum[info_set][a] = max(
                0, player.regret_sum[info_set][a] + regret
            )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    """Run CFR+ self-play for N_ITERATIONS games and save strategy.pkl.

    Both players share the same regret_sum and strategy_sum tables,
    implementing the standard two-player zero-sum CFR self-play setup.
    """
    regret_sum   = defaultdict(lambda: defaultdict(float))
    strategy_sum = defaultdict(lambda: defaultdict(float))

    config = setup_config(
        max_round=MAX_ROUND,
        initial_stack=INITIAL_STACK,
        small_blind_amount=SMALL_BLIND,
    )

    for iteration in range(N_ITERATIONS):
        p1 = CFRPlayer(regret_sum, strategy_sum)
        p2 = CFRPlayer(regret_sum, strategy_sum)

        config.players_info = []  # reset players each iteration
        config.register_player(name="CFR_1", algorithm=p1)
        config.register_player(name="CFR_2", algorithm=p2)

        result = start_poker(config, verbose=0)

        update_regrets(p1, result, player_index=0)
        update_regrets(p2, result, player_index=1)

        if (iteration + 1) % 10_000 == 0:
            print(f"Iteration {iteration + 1}/{N_ITERATIONS} complete.")

    _save_strategy(strategy_sum)
    print(f"Strategy saved to {STRATEGY_PATH}.")


def _save_strategy(strategy_sum):
    """Normalise cumulative strategy sums → probabilities and save to disk.

    Args:
        strategy_sum : defaultdict — raw cumulative strategy weights
    """
    strategy = {}
    for info_set, action_weights in strategy_sum.items():
        total = sum(action_weights.values())
        if total > 0:
            strategy[info_set] = {a: w / total for a, w in action_weights.items()}
        else:
            # Uniform fallback for unvisited info sets.
            n = len(action_weights)
            strategy[info_set] = {a: 1.0 / n for a in action_weights}

    with open(STRATEGY_PATH, "wb") as f:
        pickle.dump(strategy, f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _available_actions(valid_actions):
    """Extract action names from PyPokerEngine's valid_actions list.

    Args:
        valid_actions : list of {"action": str, "amount": int}

    Returns:
        list of str — e.g. ["fold", "call", "raise"] or ["fold", "call"]
    """
    return [a["action"] for a in valid_actions]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
