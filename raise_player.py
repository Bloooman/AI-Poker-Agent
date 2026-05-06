"""
Runtime poker agent.

Primary path:
    1. Convert the PyPokerEngine round_state into the shared abstract key.
    2. Look up that key in strategy.pkl, which is produced offline by train.py.
    3. Filter the strategy to currently legal actions and sample an action.

Backup path:
    If the strategy table does not contain a usable probability distribution for
    this exact state, use a light heuristic policy. Preflop uses an embedded
    starting-hand strength table. Postflop uses PyPokerEngine's HandEvaluator
    through the already-computed abstraction bucket.
"""

import os
import pickle
import random

from pypokerengine.players import BasePokerPlayer

from abstraction import abstract


STRATEGY_PATH = "strategy.pkl"
ACTIONS = ("fold", "call", "raise")
STREETS = ("preflop", "flop", "turn", "river")
RANK_ORDER = "23456789TJQKA"
RANK_VALUE = {rank: index + 2 for index, rank in enumerate(RANK_ORDER)}


# Complete offline preflop backup table: all 169 Texas Hold'em starting hands.
# Copied from BrenoCPimenta/Poker-preflop-hand-rank-scraping-to-csv
# preFlop-rank.csv, where rank 1 is strongest and rank 169 is weakest.
PREFLOP_RANK_DATA = """
1,AA,p 2,KK,p 3,QQ,p 4,AK,s 5,JJ,p 6,AQ,s 7,KQ,s 8,AJ,s
9,KJ,s 10,TT,p 11,AK,o 12,AT,s 13,QJ,s 14,KT,s 15,QT,s 16,JT,s
17,99,p 18,AQ,o 19,A9,s 20,KQ,o 21,88,p 22,K9,s 23,T9,s 24,A8,s
25,Q9,s 26,J9,s 27,AJ,o 28,A5,s 29,77,p 30,A7,s 31,KJ,o 32,A4,s
33,A3,s 34,A6,s 35,QJ,o 36,66,p 37,K8,s 38,T8,s 39,A2,s 40,98,s
41,J8,s 42,AT,o 43,Q8,s 44,K7,s 45,KT,o 46,55,p 47,JT,o 48,87,s
49,QT,o 50,44,p 51,33,p 52,22,p 53,K6,s 54,97,s 55,K5,s 56,76,s
57,T7,s 58,K4,s 59,K3,s 60,K2,s 61,Q7,s 62,86,s 63,65,s 64,J7,s
65,54,s 66,Q6,s 67,75,s 68,96,s 69,Q5,s 70,64,s 71,Q4,s 72,Q3,s
73,T9,o 74,T6,s 75,Q2,s 76,A9,o 77,53,s 78,85,s 79,J6,s 80,J9,o
81,K9,o 82,J5,s 83,Q9,o 84,43,s 85,74,s 86,J4,s 87,J3,s 88,95,s
89,J2,s 90,63,s 91,A8,o 92,52,s 93,T5,s 94,84,s 95,T4,s 96,T3,s
97,42,s 98,T2,s 99,98,o 100,T8,o 101,A5,o 102,A7,o 103,73,s
104,A4,o 105,32,s 106,94,s 107,93,s 108,J8,o 109,A3,o 110,62,s
111,92,s 112,K8,o 113,A6,o 114,87,o 115,Q8,o 116,83,s 117,A2,o
118,82,s 119,97,o 120,72,s 121,76,o 122,K7,o 123,65,o 124,T7,o
125,K6,o 126,86,o 127,54,o 128,K5,o 129,J7,o 130,75,o 131,Q7,o
132,K4,o 133,K3,o 134,96,o 135,K2,o 136,64,o 137,Q6,o 138,53,o
139,85,o 140,T6,o 141,Q5,o 142,43,o 143,Q4,o 144,Q3,o 145,74,o
146,Q2,o 147,J6,o 148,63,o 149,J5,o 150,95,o 151,52,o 152,J4,o
153,J3,o 154,42,o 155,J2,o 156,84,o 157,T5,o 158,T4,o 159,32,o
160,T3,o 161,73,o 162,T2,o 163,62,o 164,94,o 165,93,o 166,92,o
167,83,o 168,82,o 169,72,o
"""


def _build_preflop_hand_ranks():
    ranks = {}
    for token in PREFLOP_RANK_DATA.split():
        rank_str, cards, suitedness = token.split(",")
        if suitedness == "p":
            key = cards
        else:
            key = cards + suitedness
        ranks[key] = int(rank_str)
    return ranks


PREFLOP_HAND_RANKS = _build_preflop_hand_ranks()
MAX_PREFLOP_RANK = 169


class RaisedPlayer(BasePokerPlayer):
    """CFR strategy-table player with a deterministic local fallback policy."""

    def __init__(self):
        super().__init__()
        self.strategy = self._load_strategy()
        self.opponent_stats = {
            street: {"raises": 0, "total": 0}
            for street in STREETS
        }

    # ------------------------------------------------------------------
    # Core PyPokerEngine interface
    # ------------------------------------------------------------------

    def declare_action(self, valid_actions, hole_card, round_state):
        """Return one legal action: "fold", "call", or "raise"."""
        valid_names = [action_info["action"] for action_info in valid_actions]
        info_set = abstract(hole_card, round_state, self.uuid)

        probs = self._strategy_probs(info_set, valid_names)
        if probs is None:
            probs = self._backup_probs(info_set, hole_card, round_state, valid_names)

        probs = self._apply_opponent_model(probs, round_state["street"], info_set)
        probs = self._filter_and_normalize(probs, valid_names)

        return self._sample_action(probs)

    def receive_game_start_message(self, game_info):
        for street in self.opponent_stats:
            self.opponent_stats[street] = {"raises": 0, "total": 0}

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        if new_action.get("player_uuid") == self.uuid:
            return

        street = round_state.get("street")
        if street not in self.opponent_stats:
            return

        self.opponent_stats[street]["total"] += 1
        if new_action.get("action") == "raise":
            self.opponent_stats[street]["raises"] += 1

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    # ------------------------------------------------------------------
    # Strategy table path
    # ------------------------------------------------------------------

    def _load_strategy(self):
        """Load strategy.pkl from the same directory as this file."""
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, STRATEGY_PATH)

        try:
            with open(path, "rb") as strategy_file:
                return pickle.load(strategy_file)
        except FileNotFoundError:
            return {}

    def _strategy_probs(self, info_set, valid_names):
        """Return normalized table probabilities, or None if unusable."""
        if info_set not in self.strategy:
            return None

        raw = {
            action: float(self.strategy[info_set].get(action, 0.0))
            for action in valid_names
        }
        return self._normalize(raw)

    # ------------------------------------------------------------------
    # Backup policy
    # ------------------------------------------------------------------

    def _backup_probs(self, info_set, hole_card, round_state, valid_names):
        """Build a conservative fallback distribution from local hand strength."""
        street, hand_bucket, pot_bucket, raise_count, position = info_set

        if street == 0:
            strength = self._preflop_strength(hole_card)
        else:
            strength = hand_bucket / 7.0

        probs = self._base_probs_from_strength(strength)
        probs = self._adjust_for_pressure(probs, strength, raise_count, pot_bucket)
        probs = self._adjust_for_position(probs, strength, position)
        probs = self._adjust_for_free_check(probs, round_state)

        return self._filter_and_normalize(probs, valid_names)

    def _preflop_strength(self, hole_card):
        """Return a 0.0-1.0 strength score from the full 169-hand rank table."""
        hand_code = self._preflop_hand_code(hole_card)
        rank = PREFLOP_HAND_RANKS.get(hand_code)
        if rank is None:
            return self._default_preflop_strength(hand_code)
        return (MAX_PREFLOP_RANK - rank + 1) / MAX_PREFLOP_RANK

    def _preflop_hand_code(self, hole_card):
        """Canonicalize ["CA", "DK"] to "AKo", ["CA", "CK"] to "AKs"."""
        c1, c2 = hole_card
        suit1, rank1 = c1[0], c1[1]
        suit2, rank2 = c2[0], c2[1]

        if RANK_VALUE[rank2] > RANK_VALUE[rank1]:
            rank1, rank2 = rank2, rank1
            suit1, suit2 = suit2, suit1

        if rank1 == rank2:
            return rank1 + rank2

        suited = "s" if suit1 == suit2 else "o"
        return rank1 + rank2 + suited

    def _default_preflop_strength(self, hand_code):
        """Defensive fallback; normally unreachable because the rank table is complete."""
        rank1, rank2 = hand_code[0], hand_code[1]
        suited = len(hand_code) == 3 and hand_code[2] == "s"
        high = RANK_VALUE[rank1]
        low = RANK_VALUE[rank2]
        gap = high - low - 1

        score = 0
        if high >= 14:
            score += 1
        if high >= 12 and low >= 9:
            score += 1
        if suited:
            score += 1
        if gap == 0:
            score += 1
        elif gap <= 2 and low >= 7:
            score += 0.5

        return min(score / 4.0, 0.75)

    def _base_probs_from_strength(self, strength):
        """Map 0.0-1.0 hand strength to fold/call/raise probabilities."""
        if strength >= 0.85:
            return {"fold": 0.02, "call": 0.34, "raise": 0.64}
        if strength >= 0.70:
            return {"fold": 0.06, "call": 0.48, "raise": 0.46}
        if strength >= 0.55:
            return {"fold": 0.14, "call": 0.64, "raise": 0.22}
        if strength >= 0.40:
            return {"fold": 0.30, "call": 0.62, "raise": 0.08}
        if strength >= 0.25:
            return {"fold": 0.48, "call": 0.49, "raise": 0.03}
        return {"fold": 0.66, "call": 0.33, "raise": 0.01}

    def _adjust_for_pressure(self, probs, strength, raise_count, pot_bucket):
        """Tighten weak hands after raises; continue more often in large pots."""
        adjusted = dict(probs)
        pressure = min(raise_count, 4)

        if pressure:
            adjusted["raise"] *= max(0.25, 1.0 - 0.20 * pressure)
            if strength < 0.55:
                adjusted["fold"] *= 1.0 + 0.35 * pressure
                adjusted["call"] *= max(0.55, 1.0 - 0.12 * pressure)
            elif strength < 0.75:
                adjusted["fold"] *= 1.0 + 0.15 * pressure
                adjusted["call"] *= 1.0 + 0.05 * pressure

        if pot_bucket >= 3 and strength >= 0.40:
            adjusted["call"] *= 1.10
            adjusted["fold"] *= 0.90

        return self._normalize(adjusted) or adjusted

    def _adjust_for_position(self, probs, strength, position):
        """Slightly widen in position and tighten weak out-of-position hands."""
        adjusted = dict(probs)

        if position == 1 and strength >= 0.55:
            adjusted["raise"] *= 1.08
        elif position == 0 and strength < 0.40:
            adjusted["fold"] *= 1.08
            adjusted["raise"] *= 0.85

        return self._normalize(adjusted) or adjusted

    def _adjust_for_free_check(self, probs, round_state):
        """Prefer checking over folding when call costs zero chips."""
        street = round_state.get("street")
        histories = round_state.get("action_histories", {}).get(street, [])
        has_raise = any(action.get("action") == "raise" for action in histories)

        if street != "preflop" and not has_raise:
            adjusted = dict(probs)
            fold_mass = adjusted.get("fold", 0.0)
            adjusted["call"] = adjusted.get("call", 0.0) + fold_mass * 0.90
            adjusted["fold"] = fold_mass * 0.10
            return self._normalize(adjusted) or adjusted

        return probs

    # ------------------------------------------------------------------
    # Opponent model
    # ------------------------------------------------------------------

    def _apply_opponent_model(self, probs, street, info_set):
        """Conservatively adjust for observed street aggression."""
        stats = self.opponent_stats.get(street, {"raises": 0, "total": 0})
        total = stats["total"]
        if total < 4:
            return probs

        _, hand_bucket, _, _, _ = info_set
        strength = hand_bucket / 7.0
        aggression = stats["raises"] / total
        adjusted = dict(probs)

        if aggression >= 0.45 and strength < 0.70:
            adjusted["fold"] = adjusted.get("fold", 0.0) * 1.18
            adjusted["raise"] = adjusted.get("raise", 0.0) * 0.78
        elif aggression <= 0.15 and strength >= 0.55:
            adjusted["raise"] = adjusted.get("raise", 0.0) * 1.15
            adjusted["fold"] = adjusted.get("fold", 0.0) * 0.90

        return self._normalize(adjusted) or probs

    # ------------------------------------------------------------------
    # Probability helpers
    # ------------------------------------------------------------------

    def _filter_and_normalize(self, probs, valid_names):
        filtered = {
            action: max(0.0, probs.get(action, 0.0))
            for action in valid_names
            if action in ACTIONS
        }

        normalized = self._normalize(filtered)
        if normalized is not None:
            return normalized

        weight = 1.0 / len(valid_names)
        return {action: weight for action in valid_names}

    def _normalize(self, probs):
        total = sum(max(0.0, prob) for prob in probs.values())
        if total <= 0.0:
            return None
        return {
            action: max(0.0, prob) / total
            for action, prob in probs.items()
        }

    def _sample_action(self, probs):
        return random.choices(
            list(probs.keys()),
            weights=list(probs.values()),
        )[0]


def setup_ai():
    return RaisedPlayer()
