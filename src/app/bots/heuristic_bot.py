from __future__ import annotations
from app.competition.interface import *
from app.bots.helpers import _current_item, _affordable
from typing import Dict
import math


class HeuristicBot(PocketRocketsBot):
    """
    Heuristic bot that values gems based on:
    1. Expected gem value from value chart (using revealed info + uniform estimate for unrevealed)
    2. Trinket completion potential (values suits higher if they help complete trinkets)
    
    Bids a percentage of the calculated value (configurable via bid_percentage parameter).
    """
    
    def __init__(self, bid_percentage: float = 0.7, name: str = "HeuristicBot"):
        """
        Args:
            bid_percentage: Percentage of calculated value to bid (0.0 to 1.0). Default 0.7 (70%).
            name: Bot name for display.
        """
        self.bid_percentage = max(0.0, min(1.0, bid_percentage))  # Clamp to [0, 1]
        self._name = name
    
    @property
    def bot_name(self) -> str:
        return self._name
    
    def _estimate_info_card_counts(self, obs: GameObservation) -> Dict[Suit, float]:
        """
        Estimate the final count of info cards per suit using:
        - Known revealed info cards (public)
        - Our private unrevealed info cards
        - Uniform distribution for other players' unrevealed cards
        """
        # Count known revealed info cards (public)
        known_revealed: Dict[Suit, int] = {s: 0 for s in Suit}
        for p in obs.public.players:
            for c in p.revealed_info:
                known_revealed[c.suit] += 1
        
        # Count our private unrevealed info cards
        my_unrevealed: Dict[Suit, int] = {s: 0 for s in Suit}
        for c in obs.private.info_cards_unrevealed:
            my_unrevealed[c.suit] += 1
        
        # Calculate total info cards and remaining unknown
        total_info = 0
        for p in obs.public.players:
            total_info += p.unrevealed_info_count + len(p.revealed_info)
        
        known_total = sum(known_revealed.values()) + sum(my_unrevealed.values())
        remaining_unknown = max(0, total_info - known_total)
        
        # Estimate: distribute remaining unknown uniformly among suits
        # For each suit: known_revealed + my_unrevealed + (remaining_unknown / 5)
        estimates: Dict[Suit, float] = {}
        for suit in Suit:
            estimates[suit] = (
                known_revealed[suit] + 
                my_unrevealed[suit] + 
                (remaining_unknown / len(Suit))
            )
        
        return estimates
    
    def _get_base_gem_value(self, obs: GameObservation, suit: Suit) -> float:
        """
        Get the base gem value from the value chart based on estimated info card count.
        """
        chart = obs.public.value_chart.mapping
        estimates = self._estimate_info_card_counts(obs)
        
        estimated_count = estimates[suit]
        idx = int(round(estimated_count))
        idx = max(0, min(idx, len(chart) - 1))
        
        return float(chart[idx])
    
    def _calculate_trinket_bonus_value(self, obs: GameObservation, suit: Suit) -> float:
        """
        Calculate additional value for a suit based on trinket completion potential.
        If we have gems of this suit and there are trinkets requiring it, add bonus value.
        """
        # Count our gems by suit
        my_gems: Dict[Suit, int] = {s: 0 for s in Suit}
        for gem in obs.me.gems_owned:
            my_gems[gem.suit] += 1
        
        # Count how many trinkets require this suit (and how many we could complete)
        trinket_bonus = 0.0
        
        for trinket_state in obs.public.trinkets:
            if trinket_state.claimed_by is not None:
                continue  # Already claimed
            
            # Count required gems for this trinket
            required: Dict[Suit, int] = {s: 0 for s in Suit}
            for card in trinket_state.objective.required_cards:
                required[card.suit] += 1
            
            # Check if this suit is required and if we're close to completing
            if required[suit] > 0:
                # Calculate how many we still need
                needed = max(0, required[suit] - my_gems[suit])
                
                # If we already have some, this suit is valuable
                # Bonus scales with: (points / required_count) * (how many we have / how many needed)
                if my_gems[suit] > 0:
                    # We have some, so this suit helps us complete the trinket
                    completion_ratio = my_gems[suit] / required[suit]
                    # Bonus is proportional to trinket points and completion progress
                    trinket_bonus += trinket_state.objective.points * completion_ratio * 0.5
                elif needed == 1:
                    # We need just one more, so it's valuable
                    trinket_bonus += trinket_state.objective.points * 0.3
        
        return trinket_bonus
    
    def _calculate_suit_value(self, obs: GameObservation, suit: Suit) -> float:
        """
        Calculate total value for a suit: base gem value + trinket bonus.
        """
        base_value = self._get_base_gem_value(obs, suit)
        trinket_bonus = self._calculate_trinket_bonus_value(obs, suit)
        return base_value # + trinket_bonus
    
    def get_bid(self, obs: GameObservation) -> Bid:
        """
        Calculate bid based on gem values and trinket potential.
        """
        kind = obs.context.action.kind
        max_bid = legal_max_bid(obs)
        
        # Estimate game stage for small behavior tweaks
        # (upcoming not counted in pile count, so total remaining gems is upcoming + pile)
        remaining_gems = len(obs.context.upcoming_gems) + obs.context.biddable_pile_count
        late = remaining_gems <= 6
        
        if kind in (ActionType.INVESTMENT_5, ActionType.INVESTMENT_10):
            payout = 5 if kind == ActionType.INVESTMENT_5 else 10

            # Investment net gain at end is payout (locked returns), so bid should be <= payout,
            # but we discount early game because locking cash can lose gem opportunities.
            discount = 0.9 if late else 0.65
            bid = int(math.floor(discount * payout))
            return Bid(_affordable(bid, obs))

        if kind in (ActionType.LOAN_10, ActionType.LOAN_20):
            principal = 10 if kind == ActionType.LOAN_10 else 20

            # Loan gives principal now, but you repay principal later; bid is a pure cost.
            # Only worth it if: (a) bid is tiny, and (b) you are cash constrained now.
            cash = obs.me.cash
            constrained = cash <= 3
            if not constrained and not late:
                return Bid(0)

            # If constrained, you might pay 1–2 to gain liquidity.
            # If late, sometimes worth 1 to buy a last gem.
            bid = 2 if constrained else 1
            # Never bid more than a small fraction of principal
            bid = min(bid, max(0, principal // 10))
            return Bid(_affordable(bid, obs))
        
        # Get the gems being auctioned
        cards, _ = _current_item(obs)
        if not cards:
            return Bid(0)
        
        # Calculate total value of the gems being auctioned
        total_value = 0.0
        for card in cards:
            suit_value = self._calculate_suit_value(obs, card.suit)
            total_value += suit_value
        
        # Also check if winning would complete any trinkets (immediate bonus)
        # This is in addition to the suit value calculation
        from app.bots.helpers import _best_trinket_bonus_if_win
        trinket_immediate_bonus = _best_trinket_bonus_if_win(obs, cards)
        total_value += trinket_immediate_bonus
        
        # Small premium late game (values are more known + fewer turns to recover)
        mult = self.bid_percentage * (1.05 if late else 1.0)
        
        # Bid the specified percentage of the calculated value (with late game premium)
        bid = int(max(0, round(total_value * mult)))
        return Bid(_affordable(bid, obs))
    
    def choose_info_to_reveal(self, obs: GameObservation, result: AuctionResult) -> str:
        """
        Slightly more nuanced: reveal a suit that (given public reveals) would
        move market beliefs *toward neutrality* rather than swinging prices.
        We approximate by revealing the suit with the smallest absolute deviation
        between:
          our private count for that suit
        and
          the public expected count for that suit among remaining hidden info.
        """
        unrevealed = list(obs.private.info_cards_unrevealed)
        if not unrevealed:
            return obs.private.info_cards_revealed[0].id

        # Compute public known counts and remaining hidden
        total_info = 0
        known: Dict[Suit, int] = {s: 0 for s in Suit}
        for p in obs.public.players:
            total_info += p.unrevealed_info_count + len(p.revealed_info)
            for c in p.revealed_info:
                known[c.suit] += 1
        remaining = max(0, total_info - sum(known.values()))
        exp_each = remaining / len(Suit) if remaining else 0.0

        # Our private counts
        my_counts: Dict[Suit, int] = {s: 0 for s in Suit}
        for c in unrevealed:
            my_counts[c.suit] += 1

        # Choose card whose suit is closest to "neutral" (exp_each) to minimize info impact
        best = min(
            unrevealed,
            key=lambda c: (abs(my_counts[c.suit] - exp_each), my_counts[c.suit], c.id),
        )
        return best.id

