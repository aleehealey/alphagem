from __future__ import annotations
from app.competition.interface import *
from app.bots.helpers import _current_item, legal_max_bid, _affordable, _best_trinket_bonus_if_win
from typing import Dict, List, Optional, Tuple
import math
from collections import Counter

# -----------------------------
# ValueHeuristicBot: Best of both worlds
# - Uses HeuristicBot's information advantage (private unrevealed info cards)
# - Uses ValueTraderBot's value system and bidding logic (risk-based, late game premium)
# - Strategic trinket evaluation and targeting based on starting hand
# -----------------------------

class ValueHeuristicBot(PocketRocketsBot):
    """
    Hybrid bot combining:
    1. HeuristicBot's information utilization (private unrevealed info cards)
    2. ValueTraderBot's value system and bidding strategy
    3. Strategic trinket evaluation and targeting
    """
    
    def __init__(self, risk: float = 0.9):
        """
        Args:
            risk: Bidding aggressiveness (0.0 to 1.2). Higher = more aggressive.
                  Default 0.9 matches ValueTraderBot's optimal setting.
        """
        self.risk = float(risk)
        self._trinket_strategy: Optional[Dict] = None  # Set at game start
        self._target_suits: Optional[List[Suit]] = None  # Suits to prioritize
    
    @property
    def bot_name(self) -> str:
        return f"ValueHeuristic(r={self.risk:.2f})"
    
    def on_game_start(self, obs: "GameObservation") -> None:
        """Initialize - strategy will be evaluated each turn"""
        self._trinket_strategy = None
        self._target_suits = None
    
    def _evaluate_trinket_strategy(self, obs: "GameObservation") -> None:
        """
        Analyze available trinkets and current game state to form/update strategy.
        Called every turn to adapt to changing game conditions.
        Sets self._trinket_strategy and self._target_suits.
        """
        # Get our unrevealed info cards by suit
        my_info: Dict[Suit, int] = {s: 0 for s in Suit}
        for c in obs.private.info_cards_unrevealed:
            my_info[c.suit] += 1
        
        # Get our current gems by suit
        my_gems: Dict[Suit, int] = {s: 0 for s in Suit}
        for gem in obs.me.gems_owned:
            my_gems[gem.suit] += 1
        
        # Estimate remaining gems available (for feasibility calculation)
        remaining_gems = len(obs.context.upcoming_gems) + obs.context.biddable_pile_count
        late_game = remaining_gems <= 6
        
        # Analyze available trinkets
        trinket_analysis = []
        for trinket_state in obs.public.trinkets:
            if trinket_state.claimed_by is not None:
                continue  # Skip already claimed trinkets
            
            # Count required suits for this trinket
            required: Dict[Suit, int] = {s: 0 for s in Suit}
            for card in trinket_state.objective.required_cards:
                required[card.suit] += 1
            
            # Calculate current progress toward this trinket
            progress_by_suit: Dict[Suit, float] = {}
            total_progress = 0.0
            for suit, needed in required.items():
                current = my_gems[suit]
                progress = min(1.0, current / needed) if needed > 0 else 0.0
                progress_by_suit[suit] = progress
                total_progress += progress
            avg_progress = total_progress / len(required) if required else 0.0
            
            # Calculate alignment: how many info cards we have for required suits
            alignment_score = 0.0
            total_required = sum(required.values())
            for suit, needed in required.items():
                # We get points for having info cards in required suits
                # More info cards = better alignment
                alignment_score += min(my_info[suit], needed) / total_required if total_required > 0 else 0
            
            # Calculate feasibility: can we complete this?
            # Based on current gems, info cards, and remaining game state
            feasibility = 0.0
            for suit, needed in required.items():
                current = my_gems[suit]
                still_needed = max(0, needed - current)
                
                if still_needed == 0:
                    # We already have enough of this suit
                    feasibility += 1.0
                else:
                    # Estimate if we can get enough gems
                    # Factor in: current gems, info cards, remaining game
                    info_advantage = my_info[suit]  # More info = better chance
                    
                    # Estimate gem availability based on remaining gems
                    # Rough estimate: remaining_gems / 5 suits = gems per suit remaining
                    estimated_remaining_per_suit = remaining_gems / len(Suit) if remaining_gems > 0 else 0
                    
                    # Probability we can get the needed gems
                    # Higher if we have info cards, more gems remaining, or late game (values known)
                    if late_game:
                        # Late game: more certain about values, adjust strategy
                        prob = min(1.0, (current + info_advantage * 0.3 + estimated_remaining_per_suit * 0.5) / needed)
                    else:
                        # Early game: more uncertain, be more conservative
                        prob = min(1.0, (current + info_advantage * 0.5 + estimated_remaining_per_suit * 0.3) / needed)
                    
                    feasibility += prob
            
            feasibility /= len(required) if required else 1.0
            
            # Value score: points per gem required
            value_per_gem = trinket_state.objective.points / total_required if total_required > 0 else 0
            
            # Priority calculation: weighted combination
            # Progress gets higher weight if we're already close
            if avg_progress > 0.5:
                # We're making good progress - prioritize this
                priority = (avg_progress * 0.5 + alignment_score * 0.2 + feasibility * 0.2 + value_per_gem * 0.1)
            else:
                # Early stage - focus on alignment and feasibility
                priority = (alignment_score * 0.4 + feasibility * 0.3 + value_per_gem * 0.2 + avg_progress * 0.1)
            
            trinket_analysis.append({
                'trinket': trinket_state,
                'required': required,
                'points': trinket_state.objective.points,
                'alignment': alignment_score,
                'feasibility': feasibility,
                'value_per_gem': value_per_gem,
                'progress': avg_progress,
                'progress_by_suit': progress_by_suit,
                'priority': priority
            })
        
        # Sort by priority
        trinket_analysis.sort(key=lambda x: x['priority'], reverse=True)
        
        # Set strategy: target top 2-3 trinkets (or fewer if not many available)
        available_count = len(trinket_analysis)
        target_count = min(3, max(1, available_count))
        
        self._trinket_strategy = {
            'target_trinkets': trinket_analysis[:target_count],
            'all_trinkets': trinket_analysis,
            'late_game': late_game
        }
        
        # Determine target suits (suits needed for high-priority trinkets)
        target_suits_set = set()
        for analysis in trinket_analysis[:target_count]:
            for suit, needed in analysis['required'].items():
                if needed > 0:
                    target_suits_set.add(suit)
        
        self._target_suits = list(target_suits_set)
    
    def _estimate_info_card_counts(self, obs: "GameObservation") -> Dict[Suit, float]:
        """
        Estimate final info card counts per suit using:
        - Known revealed info cards (public)
        - Our private unrevealed info cards (information advantage!)
        - Uniform distribution for other players' unrevealed cards
        """
        # Count known revealed info cards (public)
        known_revealed: Dict[Suit, int] = {s: 0 for s in Suit}
        for p in obs.public.players:
            for c in p.revealed_info:
                known_revealed[c.suit] += 1
        
        # Count our private unrevealed info cards (KEY ADVANTAGE)
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
        estimates: Dict[Suit, float] = {}
        for suit in Suit:
            estimates[suit] = (
                known_revealed[suit] + 
                my_unrevealed[suit] + 
                (remaining_unknown / len(Suit))
            )
        
        return estimates
    
    def _get_base_gem_value(self, obs: "GameObservation", suit: Suit) -> float:
        """
        Get base gem value from value chart using improved info card estimates
        (includes our private information).
        """
        chart = obs.public.value_chart.mapping
        estimates = self._estimate_info_card_counts(obs)
        
        estimated_count = estimates[suit]
        idx = int(round(estimated_count))
        idx = max(0, min(idx, len(chart) - 1))
        
        return float(chart[idx])
    
    def _calculate_strategic_trinket_bonus(self, obs: "GameObservation", suit: Suit) -> float:
        """
        Calculate strategic trinket bonus based on our current trinket strategy.
        Uses progress tracking to give higher bonuses for trinkets we're close to completing.
        """
        if self._trinket_strategy is None:
            return 0.0
        
        bonus = 0.0
        my_gems: Dict[Suit, int] = {s: 0 for s in Suit}
        for gem in obs.me.gems_owned:
            my_gems[gem.suit] += 1
        
        # Check each target trinket
        for analysis in self._trinket_strategy['target_trinkets']:
            trinket_state = analysis['trinket']
            if trinket_state.claimed_by is not None:
                continue  # Already claimed, skip
            
            required = analysis['required']
            if suit not in required or required[suit] == 0:
                continue
            
            # Use progress information from analysis
            progress = analysis.get('progress', 0.0)
            progress_by_suit = analysis.get('progress_by_suit', {})
            suit_progress = progress_by_suit.get(suit, 0.0)
            
            # Calculate how many more we need
            needed = max(0, required[suit] - my_gems[suit])
            
            if needed == 0:
                # We have enough of this suit for this trinket
                # Bonus scales with overall trinket progress
                if progress > 0.8:
                    # Very close to completing - high bonus
                    bonus += analysis['points'] * 0.4
                elif progress > 0.5:
                    # Good progress - moderate bonus
                    bonus += analysis['points'] * 0.2
                else:
                    # Some progress - small bonus
                    bonus += analysis['points'] * 0.1
            elif needed == 1:
                # We need just one more of this suit
                if progress > 0.7:
                    # Close to completing overall - very high bonus
                    bonus += analysis['points'] * 0.5
                elif progress > 0.4:
                    # Good progress - high bonus
                    bonus += analysis['points'] * 0.3
                else:
                    # Early stage but need this suit - moderate bonus
                    bonus += analysis['points'] * 0.15
            elif suit_progress > 0.5:
                # We have some of this suit already
                # Bonus based on how much we have
                bonus += analysis['value_per_gem'] * suit_progress * 0.4
            elif suit in self._target_suits:
                # This suit is in our target list (strategic alignment)
                # Base bonus for strategic importance
                bonus += analysis['value_per_gem'] * 0.3
        
        return bonus
    
    def _calculate_suit_value(self, obs: "GameObservation", suit: Suit) -> float:
        """
        Calculate total value for a suit:
        - Base gem value (using improved info estimates)
        - Strategic trinket bonus (based on our strategy)
        """
        base_value = self._get_base_gem_value(obs, suit)
        strategic_bonus = self._calculate_strategic_trinket_bonus(obs, suit)
        return base_value + strategic_bonus
    
    def get_bid(self, obs: "GameObservation") -> "Bid":
        """
        Main bidding logic - combines ValueTraderBot's approach with
        improved information utilization.
        """
        kind = obs.context.action.kind
        max_bid = legal_max_bid(obs)
        
        # Estimate game stage
        remaining_gems = len(obs.context.upcoming_gems) + obs.context.biddable_pile_count
        late = remaining_gems <= 6
        
        # Evaluate/update trinket strategy every turn to adapt to game state
        self._evaluate_trinket_strategy(obs)
        
        # Handle investments (same as ValueTraderBot)
        if kind in (ActionType.INVESTMENT_5, ActionType.INVESTMENT_10):
            payout = 5 if kind == ActionType.INVESTMENT_5 else 10
            discount = 0.9 if late else 0.65
            bid = int(math.floor(discount * payout))
            return Bid(_affordable(bid, obs))
        
        # Handle loans (same as ValueTraderBot)
        if kind in (ActionType.LOAN_10, ActionType.LOAN_20):
            principal = 10 if kind == ActionType.LOAN_10 else 20
            cash = obs.me.cash
            constrained = cash <= 3
            if not constrained and not late:
                return Bid(0)
            
            bid = 2 if constrained else 1
            bid = min(bid, max(0, principal // 10))
            return Bid(_affordable(bid, obs))
        
        # Handle gem auctions
        if kind in (ActionType.AUCTION_1, ActionType.AUCTION_2):
            cards, _ = _current_item(obs)
            if not cards:
                return Bid(0)
            
            # Calculate value using improved information
            total_value = 0.0
            for card in cards:
                suit_value = self._calculate_suit_value(obs, card.suit)
                total_value += suit_value
            
            # Add immediate trinket completion bonus
            immediate_bonus = _best_trinket_bonus_if_win(obs, cards)
            total_value += immediate_bonus
            
            # Apply risk multiplier with late game premium (ValueTraderBot style)
            mult = self.risk * (1.05 if late else 1.0)
            
            bid = int(round(mult * total_value))
            return Bid(_affordable(bid, obs))
        
        return Bid(0)
    
    def choose_info_to_reveal(self, obs: "GameObservation", result: "AuctionResult") -> str:
        """
        Strategic info revealing:
        - If we have a strong trinket strategy, reveal suits NOT in our target list
        - Otherwise, use ValueTraderBot's neutral approach
        """
        unrevealed = list(obs.private.info_cards_unrevealed)
        if not unrevealed:
            if obs.private.info_cards_revealed:
                return obs.private.info_cards_revealed[0].id
            return ""
        
        # If we have target suits, prefer revealing non-target suits
        if self._target_suits and len(self._target_suits) > 0:
            # Count unrevealed by suit
            counts: Dict[Suit, int] = {s: 0 for s in Suit}
            for c in unrevealed:
                counts[c.suit] += 1
            
            # Prefer revealing suits that are NOT in our target list
            # (to keep our strategy hidden)
            non_target_cards = [c for c in unrevealed if c.suit not in self._target_suits]
            if non_target_cards:
                # Among non-target suits, use neutral approach
                # Compute public expected counts
                total_info = 0
                known: Dict[Suit, int] = {s: 0 for s in Suit}
                for p in obs.public.players:
                    total_info += p.unrevealed_info_count + len(p.revealed_info)
                    for c in p.revealed_info:
                        known[c.suit] += 1
                remaining = max(0, total_info - sum(known.values()))
                exp_each = remaining / len(Suit) if remaining else 0.0
                
                my_counts: Dict[Suit, int] = {s: 0 for s in Suit}
                for c in non_target_cards:
                    my_counts[c.suit] += 1
                
                best = min(
                    non_target_cards,
                    key=lambda c: (abs(my_counts[c.suit] - exp_each), my_counts[c.suit], c.id),
                )
                return best.id
        
        # Fallback to ValueTraderBot's neutral approach
        total_info = 0
        known: Dict[Suit, int] = {s: 0 for s in Suit}
        for p in obs.public.players:
            total_info += p.unrevealed_info_count + len(p.revealed_info)
            for c in p.revealed_info:
                known[c.suit] += 1
        remaining = max(0, total_info - sum(known.values()))
        exp_each = remaining / len(Suit) if remaining else 0.0
        
        my_counts: Dict[Suit, int] = {s: 0 for s in Suit}
        for c in unrevealed:
            my_counts[c.suit] += 1
        
        best = min(
            unrevealed,
            key=lambda c: (abs(my_counts[c.suit] - exp_each), my_counts[c.suit], c.id),
        )
        return best.id

