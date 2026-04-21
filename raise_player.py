"""
raise_player.py — CFR+ poker agent for COMPSCI 683 term project.

At init time, loads a pre-trained strategy table (strategy.pkl) produced
by train.py.  During play, every call to declare_action does three things:
    1. Map current game state → abstract info set key (via abstraction.py)
    2. Look up the CFR+ strategy for that key
    3. Optionally adjust probabilities using a lightweight online opponent
       model, then sample and return an action.

Only declare_action (and the receive_* hooks used by the opponent model)
are submitted; the rest of the scaffold is untouched.
"""
from time import sleep
import pprint
import pickle
import random

from pypokerengine.players import BasePokerPlayer

from abstraction import abstract

STRATEGY_PATH = "strategy.pkl"
ACTIONS       = ["fold", "call", "raise"]


class RaisedPlayer(BasePokerPlayer):
    """CFR+ poker agent using a pre-trained strategy lookup table.

    Attributes:
        strategy       : dict — info_set_key → {action: probability}
        opponent_stats : dict — per-street aggression tracking for the
                         lightweight online opponent model
    """

    def __init__(self):
        super().__init__()
        self.strategy = self._load_strategy()
        # Opponent model: track raises vs. total actions per street.
        self.opponent_stats = {
            street: {"raises": 0, "total": 0}
            for street in ["preflop", "flop", "turn", "river"]
        }

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    def declare_action(self, valid_actions, hole_card, round_state):
        """Select an action using the CFR+ strategy table.

        Steps:
            1. Abstract the state to an info set key.
            2. Look up base probabilities from the pre-trained table.
               If the key is missing (should not happen after full training),
               fall back to uniform over available actions.
            3. Apply opponent model adjustment.
            4. Sample and return an action.

        Args:
            valid_actions : list of {"action": str, "amount": int}
            hole_card     : list of card strings, e.g. ["CA", "D3"]
            round_state   : dict from PyPokerEngine

        Returns:
            str: "fold" | "call" | "raise"
        """
        valid_names = [a["action"] for a in valid_actions]
        info_set    = abstract(hole_card, round_state, self.uuid)

        probs = self._lookup(info_set, valid_names)
        probs = self._apply_opponent_model(probs, round_state["street"])

        action = random.choices(list(probs.keys()),
                                weights=list(probs.values()))[0]
        return action

    # ------------------------------------------------------------------
    # Strategy lookup
    # ------------------------------------------------------------------

    def _load_strategy(self):
        """Load the pre-trained CFR+ strategy table from disk.

        Returns:
            dict: info_set_key → {action: probability}
                  Empty dict if file not found (agent will play uniformly).
        """
        try:
            with open(STRATEGY_PATH, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: {STRATEGY_PATH} not found. Playing uniformly.")
            return {}

    def _lookup(self, info_set, valid_names):
        """Return action probabilities for the given info set.

        Filters the stored distribution to only include currently valid
        actions (raise may be unavailable when the raise cap is reached),
        then renormalises.

        Args:
            info_set    : tuple — abstract info set key
            valid_names : list of str — actions available this step

        Returns:
            dict: {action: probability}, sums to 1.0
        """
        if info_set in self.strategy:
            raw = {a: self.strategy[info_set].get(a, 0.0) for a in valid_names}
        else:
            # Fallback: uniform over valid actions.
            raw = {a: 1.0 for a in valid_names}

        total = sum(raw.values())
        return {a: p / total for a, p in raw.items()}

    # ------------------------------------------------------------------
    # Opponent model
    # ------------------------------------------------------------------

    def _apply_opponent_model(self, probs, street):
        """Nudge action probabilities based on observed opponent aggression.

        If the opponent raises frequently on this street, tighten: reduce
        our raise probability and increase fold/call slightly.
        If the opponent is passive, loosen: increase raise probability.

        This is a small multiplicative adjustment, not a replacement for
        the CFR strategy.  The CFR strategy already plays near-Nash, so
        adjustments should be conservative.

        Args:
            probs  : dict — base {action: probability} from CFR lookup
            street : str — current street

        Returns:
            dict: adjusted and renormalised {action: probability}
        """
        # TODO: implement adjustment based on self.opponent_stats[street]
        return probs

    # ------------------------------------------------------------------
    # Receive hooks — used to update opponent model
    # ------------------------------------------------------------------

    def receive_game_start_message(self, game_info):
        """Reset opponent stats at the start of each game."""
        for street in self.opponent_stats:
            self.opponent_stats[street] = {"raises": 0, "total": 0}

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        """Track opponent actions to update the aggression model.

        Args:
            new_action  : dict — {"player_uuid": str, "action": str, "amount": int}
            round_state : dict from PyPokerEngine
        """
        if new_action["player_uuid"] == self.uuid:
            return  # ignore our own actions

        street = round_state["street"]
        self.opponent_stats[street]["total"] += 1
        if new_action["action"] == "raise":
            self.opponent_stats[street]["raises"] += 1

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return RaisedPlayer()
