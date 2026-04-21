"""
check_abstraction.py — Smoke tests for abstraction.py.

Verifies that data flows correctly between PyPokerEngine's data structures
and our custom abstraction functions.  Run this before handing off to team
members or after any change to abstraction.py.

Usage:
    python check_abstraction.py

All checks print PASS or FAIL.  No external test framework required.
"""

import sys
from pypokerengine.engine.card import Card
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer

from abstraction import abstract, _chen_score, _preflop_bucket, _pot_bucket, _raise_count, _position


# ---------------------------------------------------------------------------
# Minimal helpers
# ---------------------------------------------------------------------------

def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        global _failures
        _failures += 1

_failures = 0


# ---------------------------------------------------------------------------
# 1. Card string format
# ---------------------------------------------------------------------------

def test_card_format():
    """Verify Card.from_str accepts the exact strings PyPokerEngine emits.

    PyPokerEngine encodes cards as "{suit}{rank}" where:
        suit : C | D | H | S
        rank : 2-9 | T | J | Q | K | A
    """
    print("\n--- Card string format ---")

    c = Card.from_str("CA")
    check("Card.from_str('CA') → suit=Club(2), rank=Ace(14)",
          c.suit == Card.CLUB and c.rank == 14)

    c = Card.from_str("HT")
    check("Card.from_str('HT') → suit=Heart(8), rank=Ten(10)",
          c.suit == Card.HEART and c.rank == 10)

    c = Card.from_str("D3")
    check("Card.from_str('D3') → suit=Diamond(4), rank=3",
          c.suit == Card.DIAMOND and c.rank == 3)

    # Round-trip: str(card) must equal the original string.
    for s in ["CA", "HT", "D3", "SJ", "CK", "H2"]:
        c = Card.from_str(s)
        check(f"Round-trip str(Card.from_str('{s}')) == '{s}'", str(c) == s)


# ---------------------------------------------------------------------------
# 2. Chen formula scores
# ---------------------------------------------------------------------------

def test_chen_score():
    """Verify Chen formula produces expected scores for known hands."""
    print("\n--- Chen formula scores ---")

    def cards(s1, s2):
        return [Card.from_str(s1), Card.from_str(s2)]

    # Pocket aces: base 10 × 2 = 20
    check("AA  → score 20", _chen_score(cards("CA", "DA")) == 20)

    # Pocket kings: base 8 × 2 = 16
    check("KK  → score 16", _chen_score(cards("CK", "DK")) == 16)

    # AK suited: base 10 + 2 (suited) + 1 (connected) = 13, but gap=2 here
    # A=14, K=13 → gap = 14-13-1 = 0 → +1; suited → +2; base=10 → total 13
    check("AKs → score 13", _chen_score(cards("CA", "CK")) == 13)

    # AK offsuit: 10 + 1 (connected) = 11
    check("AKo → score 11", _chen_score(cards("CA", "DK")) == 11)

    # 72 offsuit (the worst hand): base 3.5, gap=5 → -4 = -0.5
    check("72o → score -0.5", _chen_score(cards("C7", "D2")) == -0.5)


# ---------------------------------------------------------------------------
# 3. Preflop bucket
# ---------------------------------------------------------------------------

def test_preflop_bucket():
    """Verify preflop bucketing maps known hands to expected tiers."""
    print("\n--- Preflop buckets ---")

    def bucket(s1, s2):
        return _preflop_bucket([Card.from_str(s1), Card.from_str(s2)])

    check("AA  → bucket 7 (premium)", bucket("CA", "DA") == 7)
    check("KK  → bucket 7 (premium)", bucket("CK", "DK") == 7)  # Chen=16 >= threshold 16
    check("AKs → bucket 5 (strong)",  bucket("CA", "CK") == 5)  # Chen=13, in [11,14)
    check("QQ  → bucket 6",           bucket("CQ", "DQ") == 6)
    check("72o → bucket 0 (rags)",    bucket("C7", "D2") == 0)

    # All buckets should be in valid range
    for s1, s2 in [("CA","DA"), ("CK","DK"), ("CQ","DQ"), ("CJ","DJ"),
                   ("CT","DT"), ("C9","D9"), ("CA","CK"), ("C7","D2")]:
        b = bucket(s1, s2)
        check(f"bucket({s1},{s2})={b} in [0,7]", 0 <= b <= 7)


# ---------------------------------------------------------------------------
# 4. Post-flop hand bucket
# ---------------------------------------------------------------------------

def test_hand_bucket_postflop():
    """Verify HandEvaluator output maps to our bucket scheme correctly."""
    print("\n--- Post-flop hand buckets ---")
    from abstraction import _hand_bucket

    # Straight flush: should be bucket 7
    check("Straight flush → bucket 7",
          _hand_bucket(["CA", "CK"], ["CQ", "CJ", "CT"]) == 7)

    # Four of a kind: bucket 6
    check("Four of a kind → bucket 6",
          _hand_bucket(["CA", "DA"], ["HA", "SA", "C2"]) == 6)

    # Full house: bucket 5
    check("Full house → bucket 5",
          _hand_bucket(["CK", "DK"], ["HK", "CA", "DA"]) == 5)

    # Flush: bucket 4
    check("Flush → bucket 4",
          _hand_bucket(["CA", "CK"], ["CQ", "CJ", "C2"]) == 4)

    # Straight: bucket 3
    check("Straight → bucket 3",
          _hand_bucket(["CA", "DK"], ["HQ", "SJ", "CT"]) == 3)

    # High card / one pair: bucket 0
    check("High card / one pair → bucket 0",
          _hand_bucket(["C2", "D7"], ["HQ", "SJ", "CA"]) == 0)


# ---------------------------------------------------------------------------
# 5. Pot bucket
# ---------------------------------------------------------------------------

def test_pot_bucket():
    print("\n--- Pot buckets ---")
    check("pot=0    → bucket 0", _pot_bucket(0)    == 0)
    check("pot=50   → bucket 0", _pot_bucket(50)   == 0)
    check("pot=100  → bucket 1", _pot_bucket(100)  == 1)
    check("pot=300  → bucket 2", _pot_bucket(300)  == 2)
    check("pot=700  → bucket 3", _pot_bucket(700)  == 3)
    check("pot=1500 → bucket 4", _pot_bucket(1500) == 4)
    check("pot=9999 → bucket 4", _pot_bucket(9999) == 4)


# ---------------------------------------------------------------------------
# 6. Raise count
# ---------------------------------------------------------------------------

def test_raise_count():
    print("\n--- Raise count ---")

    histories = {
        "preflop": [
            {"action": "raise", "amount": 30},
            {"action": "call",  "amount": 30},
            {"action": "raise", "amount": 40},
        ],
        "flop": []
    }
    check("2 raises on preflop → 2",   _raise_count(histories, "preflop") == 2)
    check("0 raises on flop   → 0",    _raise_count(histories, "flop")    == 0)
    check("missing street     → 0",    _raise_count(histories, "turn")    == 0)

    # Cap at 4
    histories["river"] = [{"action": "raise"}] * 6
    check("6 raises capped at 4",      _raise_count(histories, "river")   == 4)


# ---------------------------------------------------------------------------
# 7. Full abstract() via a real captured game state
# ---------------------------------------------------------------------------

class StateCapture(BasePokerPlayer):
    """Minimal player that records the first round_state it sees."""

    def __init__(self):
        super().__init__()
        self.captured_hole      = None
        self.captured_state     = None

    def declare_action(self, valid_actions, hole_card, round_state):
        if self.captured_hole is None:
            self.captured_hole  = hole_card
            self.captured_state = round_state
        return valid_actions[1]["action"]  # always call

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass


def test_abstract_with_real_state():
    """Run a 1-round game and verify abstract() produces a valid key."""
    print("\n--- Full abstract() on live round_state ---")

    p1 = StateCapture()
    p2 = StateCapture()

    config = setup_config(max_round=1, initial_stack=10_000, small_blind_amount=10)
    config.register_player(name="P1", algorithm=p1)
    config.register_player(name="P2", algorithm=p2)
    start_poker(config, verbose=0)

    for name, player in [("P1", p1), ("P2", p2)]:
        hole  = player.captured_hole
        state = player.captured_state

        check(f"{name}: hole_card is list of 2 strings",
              isinstance(hole, list) and len(hole) == 2)

        assert state is not None, f"{name} never received a declare_action call"
        check(f"{name}: round_state has expected keys",
              all(k in state for k in ["street", "pot", "community_card",
                                       "action_histories", "seats",
                                       "small_blind_pos"]))

        key = abstract(hole, state, player.uuid)
        check(f"{name}: abstract() returns 5-tuple", len(key) == 5)

        street, hand_b, pot_b, raise_c, pos = key
        check(f"{name}: street in [0,3]",    0 <= street    <= 3)
        check(f"{name}: hand_bucket in [0,7]", 0 <= hand_b  <= 7)
        check(f"{name}: pot_bucket in [0,4]",  0 <= pot_b   <= 4)
        check(f"{name}: raise_count in [0,4]", 0 <= raise_c <= 4)
        check(f"{name}: position in {{0,1}}",  pos in (0, 1))

        print(f"         captured key = {key}")
        print(f"         hole_card    = {hole}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_card_format()
    test_chen_score()
    test_preflop_bucket()
    test_hand_bucket_postflop()
    test_pot_bucket()
    test_raise_count()
    test_abstract_with_real_state()

    print(f"\n{'='*40}")
    if _failures == 0:
        print("All checks passed.")
    else:
        print(f"{_failures} check(s) FAILED.")
        sys.exit(1)
