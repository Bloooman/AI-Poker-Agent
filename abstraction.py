"""
abstraction.py — Information set abstraction for CFR+ poker agent.

Maps a raw (hole_card, round_state) from PyPokerEngine into a compact
tuple key used by both train.py (offline CFR+) and raise_player.py
(online lookup). Both scripts must import from here so the mapping
never diverges.

Info set key schema (all values are ints):
    (street, hand_bucket, pot_bucket, raise_count, position)

    street      : 0=preflop, 1=flop, 2=turn, 3=river
    hand_bucket : 0–7  (8 buckets by hand strength percentile)
    pot_bucket  : 0–4  (5 buckets by pot size in chips)
    raise_count : 0–4  (number of raises this street, capped at 4)
    position    : 0=small blind, 1=big blind

Total keyspace: 4 × 8 × 5 × 5 × 2 = 1,600 entries.
"""

from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.engine.card import Card


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def abstract(hole_card, round_state, my_uuid):
    """Convert a raw game state into a compact info set key.

    This is the single source of truth shared by train.py and raise_player.py.
    Any change here invalidates previously saved strategy.pkl files.

    Args:
        hole_card   : list of card strings, e.g. ["CA", "D3"]
        round_state : dict from PyPokerEngine (street, pot, action_histories,
                      seats, small_blind_pos, ...)
        my_uuid     : str — the uuid of our player (to determine position)

    Returns:
        tuple: (street, hand_bucket, pot_bucket, raise_count, position)
    """
    street      = _street_to_int(round_state["street"])
    hand_bucket = _hand_bucket(hole_card, round_state["community_card"])
    pot_bucket  = _pot_bucket(round_state["pot"]["main"]["amount"])
    raise_count = _raise_count(round_state["action_histories"],
                               round_state["street"])
    position    = _position(round_state, my_uuid)

    return (street, hand_bucket, pot_bucket, raise_count, position)


# ---------------------------------------------------------------------------
# Street encoding
# ---------------------------------------------------------------------------

_STREET_MAP = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

def _street_to_int(street_str):
    """Map street name to integer index.

    Args:
        street_str : "preflop" | "flop" | "turn" | "river"

    Returns:
        int: 0–3
    """
    return _STREET_MAP[street_str]


# ---------------------------------------------------------------------------
# Hand strength bucketing
# ---------------------------------------------------------------------------

# Map hand strength strings (from HandEvaluator.gen_hand_rank_info) → 8 buckets.
# Using the string representation avoids fragile bit-mask arithmetic, since
# gen_hand_rank_info already extracts the hand type cleanly.
# Highcard and one pair share bucket 0 (weakest grouping).
_STRENGTH_STR_TO_BUCKET = {
    "HIGHCARD":      0,
    "ONEPAIR":       0,
    "TWOPAIR":       1,
    "THREECARD":     2,
    "STRAIGHT":      3,
    "FLASH":         4,
    "FULLHOUSE":     5,
    "FOURCARD":      6,
    "STRAIGHTFLASH": 7,
}

def _hand_bucket(hole_card_strs, community_card_strs):
    """Bucket hand strength into one of 8 groups.

    During preflop (no community cards), falls back to the Chen formula
    heuristic since no community cards exist to evaluate a full hand.

    Args:
        hole_card_strs      : list of str, e.g. ["CA", "D3"]
        community_card_strs : list of str, e.g. ["H7", "S2", "DK"] or []

    Returns:
        int: 0 (weakest) – 7 (strongest)
    """
    hole = [Card.from_str(c) for c in hole_card_strs]

    if not community_card_strs:
        return _preflop_bucket(hole)

    community    = [Card.from_str(c) for c in community_card_strs]
    info         = HandEvaluator.gen_hand_rank_info(hole, community)
    strength_str = info["hand"]["strength"]
    return _STRENGTH_STR_TO_BUCKET.get(strength_str, 0)


def _preflop_bucket(hole_cards):
    """Estimate preflop hand strength bucket using the Chen formula.

    The Chen formula assigns a numeric score to any two hole cards based
    on rank, pairing, suitedness, and connectedness.  Scores are then
    mapped to 8 buckets matching the post-flop hand-strength scale so
    that the info set keyspace is consistent across all streets.

    Chen formula steps:
        1. Base score = rank score of the highest card (Ace=10 … 2=1)
        2. Pocket pair → multiply base by 2 (minimum 5)
        3. Suited        → +2
        4. Gap penalty   → 0 gap: +1 | gap 1: 0 | gap 2: -1 | gap 3: -2 | gap 4+: -4
        5. Low connector → +1 if both cards < Queen and gap ≤ 1

    Typical score range: –1 (worst) to 20 (AA).

    Bucket thresholds (tuned to match ~equal-frequency bins over all
    169 distinct preflop hand types):
        bucket 0 : score <  3   (rags)
        bucket 1 : score <  5
        bucket 2 : score <  7
        bucket 3 : score <  9
        bucket 4 : score < 11
        bucket 5 : score < 14
        bucket 6 : score < 17
        bucket 7 : score >= 17  (AA, KK, QQ, AKs)

    Args:
        hole_cards : list of two Card objects (already converted from strings)

    Returns:
        int: 0 (weakest) – 7 (strongest)
    """
    score = _chen_score(hole_cards)
    for bucket, threshold in enumerate(_CHEN_THRESHOLDS):
        if score < threshold:
            return bucket
    return 7


# Chen formula rank → base score table.
_CHEN_RANK_SCORE = {
    14: 10,   # Ace
    13:  8,   # King
    12:  7,   # Queen
    11:  6,   # Jack
    10:  5,
     9:  4.5,
     8:  4,
     7:  3.5,
     6:  3,
     5:  2.5,
     4:  2,
     3:  1.5,
     2:  1,
}

# Upper bounds (exclusive) for each of the 8 buckets.
# bucket 7 catches everything >= 16 (AA=20, KK=16).
_CHEN_THRESHOLDS = [3, 5, 7, 9, 11, 14, 16]


def _chen_score(hole_cards):
    """Compute the Chen formula score for two hole cards.

    Args:
        hole_cards : list of two Card objects

    Returns:
        float: Chen score (–1 to 20)
    """
    high, low = sorted(hole_cards, key=lambda c: c.rank, reverse=True)
    score = _CHEN_RANK_SCORE[high.rank]

    if high.rank == low.rank:
        # Pocket pair: double the score, minimum 5.
        score = max(score * 2, 5)
    else:
        if high.suit == low.suit:
            score += 2                      # suited bonus

        gap = high.rank - low.rank - 1     # 0 = connected, 1 = one-gapper …
        if gap == 0:
            score += 1
        elif gap == 2:
            score -= 1
        elif gap == 3:
            score -= 2
        elif gap >= 4:
            score -= 4
        # gap == 1: no adjustment

        if high.rank < 12 and gap <= 1:    # both below Queen, close together
            score += 1

    return score


# ---------------------------------------------------------------------------
# Pot size bucketing
# ---------------------------------------------------------------------------

# Bucket boundaries in chips.  Adjust thresholds after observing typical
# pot sizes in PyPokerEngine games (initial stack = 10,000, blinds 10/20).
_POT_THRESHOLDS = [100, 300, 700, 1500]  # 5 buckets: <100, 100–300, ...

def _pot_bucket(pot_amount):
    """Map pot size (chips) into one of 5 buckets.

    Args:
        pot_amount : int — chips in the main pot

    Returns:
        int: 0 (tiny) – 4 (huge)
    """
    for bucket, threshold in enumerate(_POT_THRESHOLDS):
        if pot_amount < threshold:
            return bucket
    return len(_POT_THRESHOLDS)  # bucket 4


# ---------------------------------------------------------------------------
# Raise count
# ---------------------------------------------------------------------------

def _raise_count(action_histories, street):
    """Count raises made on the current street, capped at 4.

    Args:
        action_histories : dict keyed by street name, values are lists of
                           action dicts {"action": ..., "amount": ...}
        street           : str — current street name

    Returns:
        int: 0–4
    """
    history = action_histories.get(street, [])
    count   = sum(1 for a in history if a.get("action") == "raise")
    return min(count, 4)


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

def _position(round_state, my_uuid):
    """Determine whether we are small blind (0) or big blind (1).

    Args:
        round_state : dict from PyPokerEngine
        my_uuid     : str

    Returns:
        int: 0 = small blind, 1 = big blind
    """
    sb_pos   = round_state["small_blind_pos"]
    seats    = round_state["seats"]
    my_index = next(i for i, s in enumerate(seats) if s["uuid"] == my_uuid)
    return 0 if my_index == sb_pos else 1
