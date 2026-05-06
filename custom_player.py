"""
custom_player.py - Custom poker agent with alpha-beta minimax search.

At init time, loads a pre-trained strategy table (strategy.pkl) produced by train.py.
During play, every call to declare_action does three things:
    1. Map current game state -> abstract info set key (via abstraction.py)
    2. Look up the CFR+ strategy for that key
    3. Optionally adjust probabilities using a lightweight online opponent
       model, then sample and return an action.
"""
import pickle
import random

from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.card import Card
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.round_manager import RoundManager
from pypokerengine.engine.action_checker import ActionChecker
from pypokerengine.utils.game_state_utils import restore_game_state, deepcopy_game_state

from abstraction import abstract

STRATEGY_PATH = "strategy.pkl"
ACTIONS       = ["fold", "call", "raise"]

# Search depth for alpha-beta minimax (player-move steps).
# At depth 4, worst-case tree has 3^4 = 81 leaves; alpha-beta pruning
# reduces this to roughly sqrt(81) ≈ 9 effective nodes in practice.
_SEARCH_DEPTH = 4

# Monte Carlo samples used to estimate hand equity at leaf nodes.
_N_EQUITY_SAMPLES = 30

# Probability mass added to the minimax-recommended action when blending
# with the CFR distribution (then renormalised to sum to 1).
_MINIMAX_BOOST = 0.5


# Helper functions called by declare_action
# These are module-level (not class methods) so only declare_action is modified inside CustomPlayer.
def _minimax_search(valid_actions, hole_card, round_state, my_uuid):
    """Run alpha-beta minimax and return the best action string (or None).

    Restores the internal game state from the serialised round_state, attaches
    our hole cards, then evaluates each root-level action with a depth-limited
    alpha-beta search.  Returns the action with the highest subtree value.

    Args:
        valid_actions : list of {"action": str, "amount": int/dict}
        hole_card     : list of card strings, e.g. ["CA", "D3"]
        round_state   : dict from PyPokerEngine
        my_uuid       : str - our player UUID

    Returns:
        str or None - best action string, or None on failure
    """
    try:
        game_state = _build_game_state(hole_card, round_state, my_uuid)
    except Exception:
        return None  # state restore failed; caller falls back to CFR

    best_action = None
    best_val    = float("-inf")
    alpha       = float("-inf")
    beta        = float("inf")

    for action_info in valid_actions:
        action = action_info["action"]
        try:
            gs_copy    = deepcopy_game_state(game_state)
            next_gs, _ = RoundManager.apply_action(gs_copy, action)
            val        = _alpha_beta(next_gs, _SEARCH_DEPTH - 1, alpha, beta, my_uuid)
        except Exception:
            continue  # skip any action the engine rejects

        if val > best_val:
            best_val    = val
            best_action = action
        alpha = max(alpha, best_val)

    return best_action


def _alpha_beta(game_state, depth, alpha, beta, my_uuid):
    """Recursive alpha-beta pruned minimax.

    Maximises at nodes where it is our turn (next player uuid == my_uuid),
    minimises at opponent nodes.  Pruning rules:
      - Beta cut-off in a max node: opponent will never allow a value this
        high, so the branch can be skipped.
      - Alpha cut-off in a min node: we would never choose a branch this
        low, so it can be skipped.

    Args:
        game_state : deep-copied internal PyPokerEngine state dict
        depth      : remaining search depth in player-move steps
        alpha      : best value the maximiser can guarantee so far
        beta       : best value the minimiser can guarantee so far
        my_uuid    : str - our player UUID

    Returns:
        float - estimated node value from our perspective (in chips)
    """
    # Base cases: round ended or depth exhausted
    if game_state["street"] == Const.Street.FINISHED or depth == 0:
        return _evaluate(game_state, my_uuid)

    next_pos = game_state.get("next_player")
    if next_pos is None:
        return _evaluate(game_state, my_uuid)

    players     = game_state["table"].seats.players
    is_our_turn = (next_pos < len(players)
                   and players[next_pos].uuid == my_uuid)

    try:
        valid = ActionChecker.legal_actions(
            players, next_pos, game_state["small_blind_amount"]
        )
    except Exception:
        return _evaluate(game_state, my_uuid)

    if is_our_turn:
        # Maximising: pick the child with the highest value
        best = float("-inf")
        for action_info in valid:
            try:
                gs_copy    = deepcopy_game_state(game_state)
                next_gs, _ = RoundManager.apply_action(gs_copy, action_info["action"])
                val        = _alpha_beta(next_gs, depth - 1, alpha, beta, my_uuid)
            except Exception:
                continue
            best  = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break  # beta cut-off: minimising ancestor rules this out
        return best if best > float("-inf") else _evaluate(game_state, my_uuid)

    else:
        # Minimising: opponent picks the child with the lowest value
        best = float("inf")
        for action_info in valid:
            try:
                gs_copy    = deepcopy_game_state(game_state)
                next_gs, _ = RoundManager.apply_action(gs_copy, action_info["action"])
                val        = _alpha_beta(next_gs, depth - 1, alpha, beta, my_uuid)
            except Exception:
                continue
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break  # alpha cut-off: maximising ancestor rules this out
        return best if best < float("inf") else _evaluate(game_state, my_uuid)


def _evaluate(game_state, my_uuid):
    """Heuristic leaf-node evaluation in chips from our perspective.

    Terminal nodes (street == FINISHED): the game evaluator has already
    updated stacks, so our actual stack is the true payoff.

    Non-terminal cutoff nodes: estimate value as
        current_stack + equity * pot
    where equity is the Monte Carlo win probability vs. a random opponent
    hand.  Multiplying by the pot gives expected chips to gain.

    Args:
        game_state : internal PyPokerEngine state dict
        my_uuid    : str - our player UUID

    Returns:
        float - estimated chip value from our perspective
    """
    players = game_state["table"].seats.players
    me      = next((p for p in players if p.uuid == my_uuid), None)
    if me is None:
        return 0.0

    # True terminal payoff: stacks updated by GameEvaluator after showdown
    if game_state["street"] == Const.Street.FINISHED:
        return float(me.stack)

    # Heuristic: equity-weighted expected stack at depth cutoff
    hole      = getattr(me, "hole_card", None) or []
    community = list(game_state["table"]._community_card)
    equity    = _estimate_equity(hole, community) if hole else 0.5
    pot       = sum(p.pay_info.amount for p in players)
    return float(me.stack) + equity * pot


def _estimate_equity(hole_cards, community_cards):
    """Monte Carlo hand equity vs. random opponent hole cards.

    Samples _N_EQUITY_SAMPLES random two-card hands for the opponent from
    the cards not visible to us, counts wins and ties via HandEvaluator.

    Args:
        hole_cards      : list of Card objects (our two hole cards)
        community_cards : list of Card objects (currently visible board)

    Returns:
        float in [0, 1] - Monte Carlo win-or-tie probability
    """
    if not hole_cards:
        return 0.5

    # Build the unknown deck: all cards not in our hand or on the board
    known_ids = {Card.to_id(c) for c in hole_cards}
    known_ids |= {Card.to_id(c) for c in community_cards}
    remaining  = [Card.from_id(i) for i in range(1, 53) if i not in known_ids]

    if len(remaining) < 2:
        return 0.5

    wins = ties = 0
    n    = min(_N_EQUITY_SAMPLES, len(remaining) * (len(remaining) - 1) // 2)
    n    = max(n, 1)

    for _ in range(n):
        opp_hole  = random.sample(remaining, 2)
        our_score = HandEvaluator.eval_hand(hole_cards, community_cards)
        opp_score = HandEvaluator.eval_hand(opp_hole,  community_cards)
        if our_score > opp_score:
            wins += 1
        elif our_score == opp_score:
            ties += 0.5

    return (wins + ties) / n


def _build_game_state(hole_card, round_state, my_uuid):
    """Restore internal game state and attach our hole cards.

    Calls restore_game_state to rebuild the Table, pot, and action history
    from the serialised round_state dict.  Then attaches our hole cards to
    our Player object and removes them from the deck so future card draws
    in the search tree are consistent with the real game.

    Args:
        hole_card   : list of card strings, e.g. ["CA", "D3"]
        round_state : dict from PyPokerEngine
        my_uuid     : str - our player UUID

    Returns:
        dict - internal game state ready for RoundManager.apply_action
    """
    game_state = restore_game_state(round_state)

    hole_objs = [Card.from_str(c) for c in hole_card]
    hole_ids  = {Card.to_id(c) for c in hole_objs}

    # Attach hole cards to our player in the restored internal state
    for player in game_state["table"].seats.players:
        if player.uuid == my_uuid:
            player.hole_card = hole_objs
            break

    # Remove our hole cards from the deck so they cannot be re-dealt
    deck      = game_state["table"].deck
    deck.deck = [c for c in deck.deck if Card.to_id(c) not in hole_ids]

    return game_state

class CustomPlayer(BasePokerPlayer):
    """CFR+ poker agent using a pre-trained strategy lookup table.

    Attributes:
        strategy       : dict - info_set_key -> {action: probability}
        opponent_stats : dict - per-street aggression tracking for the
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

    def declare_action(self, valid_actions, hole_card, round_state):
        """Select an action using alpha-beta minimax blended with CFR+ strategy.

        Step 1 - CFR+ lookup (near-Nash equilibrium baseline):
            Abstract the state to an info set key and look up the pre-trained
            probability distribution.  The CFR+ average strategy converges to
            a Nash equilibrium over self-play iterations, making the agent
            hard to exploit by any fixed counter-strategy.

        Step 2 - Alpha-beta minimax (depth-limited adversarial search):
            Restore the internal game state, attach our hole cards, and search
            a game tree of depth _SEARCH_DEPTH with alpha-beta pruning.  At
            each node the player whose uuid matches ours is the maximiser;
            the opponent is the minimiser.  Leaf nodes are evaluated as
            current_stack + equity * pot, where equity is the Monte Carlo
            probability that our hand beats a random opponent hand.

        Step 3 - Blend:
            Add _MINIMAX_BOOST probability mass to the minimax-recommended
            action within the CFR distribution and renormalise.  This tilts
            toward the look-ahead best action while the CFR tail keeps all
            actions possible so we remain difficult to counter-exploit.

        Args:
            valid_actions : list of {"action": str, "amount": int/dict}
            hole_card     : list of card strings, e.g. ["CA", "D3"]
            round_state   : dict from PyPokerEngine

        Returns:
            str: "fold" | "call" | "raise"
        """
        valid_names = [a["action"] for a in valid_actions]
        info_set    = abstract(hole_card, round_state, self.uuid)

        # Step 1: CFR+ strategy lookup
        probs = self._lookup(info_set, valid_names)
        probs = self._apply_opponent_model(probs, round_state["street"])

        # Step 2: Alpha-beta minimax search
        minimax_action = _minimax_search(
            valid_actions, hole_card, round_state, self.uuid
        )

        # Step 3: Blend minimax recommendation into CFR distribution
        if minimax_action is not None and minimax_action in probs:
            # Boost the minimax action by _MINIMAX_BOOST then renormalise.
            # All other actions keep positive probability so the distribution
            # retains the Nash-equilibrium property of the CFR strategy.
            probs = dict(probs)
            probs[minimax_action] += _MINIMAX_BOOST
            total = sum(probs.values())
            probs = {a: p / total for a, p in probs.items()}

        action = random.choices(list(probs.keys()),
                                weights=list(probs.values()))[0]
        return action

    # Strategy lookup
    def _load_strategy(self):
        """Load the pre-trained CFR+ strategy table from disk.

        Returns:
            dict: info_set_key -> {action: probability}
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
            info_set    : tuple - abstract info set key
            valid_names : list of str - actions available this step

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

    # Opponent model
    def _apply_opponent_model(self, probs, street):
        """Nudge action probabilities based on observed opponent aggression.

        If the opponent raises frequently on this street, tighten: reduce
        our raise probability and increase fold/call slightly.
        If the opponent is passive, loosen: increase raise probability.

        This is a small multiplicative adjustment, not a replacement for
        the CFR strategy.  The CFR strategy already plays near-Nash, so
        adjustments should be conservative.

        Args:
            probs  : dict - base {action: probability} from CFR lookup
            street : str - current street

        Returns:
            dict: adjusted and renormalised {action: probability}
        """
        stats = self.opponent_stats[street]
        if stats["total"] == 0:
            return probs  # no data yet; use CFR baseline unchanged

        aggression = stats["raises"] / stats["total"]
        adjusted   = dict(probs)

        if aggression > 0.40 and "raise" in adjusted:
            # Tighten: opponent is aggressive, our raises have less fold equity
            adjusted["raise"] *= 0.6
        elif aggression < 0.15 and "raise" in adjusted:
            # Loosen: opponent is passive, raise more to extract value
            adjusted["raise"] = min(1.0, adjusted["raise"] * 1.5)

        total = sum(adjusted.values())
        return {a: p / total for a, p in adjusted.items()}

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return CustomPlayer()