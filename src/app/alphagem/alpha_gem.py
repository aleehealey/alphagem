import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from app.competition import interface

class AlphaGem(nn.Module):
    def __init__(self, num_players: int = 3):
        super().__init__()

        if num_players < 3:
            num_players = 3
        if num_players > 5:
            num_players = 5
        self.num_players = num_players

        self.num_inputs = 30 + 19 + (self.num_players - 1) * 15

        self.model = nn.Sequential(
            nn.Linear(self.num_inputs, 160),
            nn.ReLU(),
            nn.Linear(160, 320),
            nn.ReLU(),
            nn.Linear(320, 320),
            nn.ReLU(),
            nn.Linear(320, 160)
        )
        # 2 heads
        self.value_head = nn.Linear(160, 20) # one on the value of the item being auctioned
        self.revealed_card_head = nn.Linear(160, 5) # one on the card of ours being revealed
        
        # State tracking
        self.prev_observation = None
        self.player_order_map = None  # Maps actual player_id to relative position (self is 0)
    
    def forward(self, game_observation: interface.GameObservation):
        encoded_input = self.encode_input(game_observation)
        tensor_input = torch.FloatTensor(encoded_input).unsqueeze(0)
        x = self.model(tensor_input)
        value_output = self.value_head(x)
        revealed_card_output = self.revealed_card_head(x)
        return value_output, revealed_card_output

    def _encode_suit(self, suit):
        """Encode gem suit to number: 0=None, 1=Ruby, 2=Sapphire, 3=Emerald, 4=Amethyst, 5=Diamond"""
        if suit is None:
            return 0
        suit_map = {
            interface.Suit.RUBY: 1,
            interface.Suit.SAPPHIRE: 2,
            interface.Suit.EMERALD: 3,
            interface.Suit.AMETHYST: 4,
            interface.Suit.DIAMOND: 5,
        }
        return suit_map.get(suit, 0)

    def _count_gems_by_suit(self, cards):
        """Count gems by suit in a list of cards. Returns list of 5 values: [Ruby, Sapphire, Emerald, Amethyst, Diamond]"""
        counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # Ruby, Sapphire, Emerald, Amethyst, Diamond
        for card in cards:
            suit_num = self._encode_suit(card.suit)
            if suit_num > 0:
                counts[suit_num] += 1
        return [counts[1], counts[2], counts[3], counts[4], counts[5]]

    def _compute_loans_value(self, loans):
        """Compute total value of loans (negative, as principal must be repaid)."""
        return -sum(loan.principal for loan in loans)

    def _compute_investments_value(self, investments):
        """Compute total value of investments (sum of payouts)."""
        return sum(investment.payout + investment.locked for investment in investments)

    def _encode_game_state(self, public_state: interface.GamePublicState, context: interface.TurnContext):
        """Encode the public game state information."""
        inputs = []
        
        # 1. Encode all 4 trinkets (each trinket: 5 values for gem counts)
        trinkets = public_state.trinkets
        for i in range(4):
            if i < len(trinkets):
                trinket = trinkets[i]
                # If trinket is claimed, encode as all zeros
                if trinket.claimed_by is not None:
                    inputs.extend([0, 0, 0, 0, 0])
                else:
                    # Count gems required for this trinket
                    gem_counts = self._count_gems_by_suit(trinket.objective.required_cards)
                    inputs.extend(gem_counts)
            else:
                # No trinket at this position, encode as all zeros
                inputs.extend([0, 0, 0, 0, 0])

        # 2. Number of gems left in the draw pile
        gems_left = context.biddable_pile_count
        inputs.append(gems_left)

        # 3. Current gem being auctioned (first upcoming gem)
        upcoming_gems = context.upcoming_gems
        if len(upcoming_gems) > 0:
            current_gem_suit = self._encode_suit(upcoming_gems[0].suit)
            inputs.append(current_gem_suit)
        else:
            inputs.append(0)  # No gem available

        # 4. Second gem being shown (for double gem auctions)
        if len(upcoming_gems) > 1:
            second_gem_suit = self._encode_suit(upcoming_gems[1].suit)
            inputs.append(second_gem_suit)
        else:
            inputs.append(0)  # No second gem available

        # 5. Tiebreaker leader index (relative to self)
        tiebreak_leader_id = context.tiebreak_leader_id
        tiebreak_leader_relative = self.player_order_map.get(tiebreak_leader_id, 0)
        inputs.append(tiebreak_leader_relative)

        # 6. Action cards remaining (computed from available action cards)
        action_counts = public_state.action_counts_remaining
        if action_counts is not None:
            # Encode counts for each action type
            action_types = [
                interface.ActionType.AUCTION_1,
                interface.ActionType.AUCTION_2,
                interface.ActionType.LOAN_10,
                interface.ActionType.LOAN_20,
                interface.ActionType.INVESTMENT_5,
                interface.ActionType.INVESTMENT_10,
            ]
            for action_type in action_types:
                count = action_counts.get(action_type, 0)
                inputs.append(count)
        else:
            # If action_counts_remaining is None, encode as all zeros
            inputs.extend([0, 0, 0, 0, 0, 0])

        return inputs

    def _encode_model_player(self, public_state: interface.GamePublicState, private_state: interface.PlayerPrivateState):
        """Encode the model's player state information."""
        inputs = []

        # Fallback: lookup by player_id
        model_player = None
        for p in public_state.players:
            if p.player_id == private_state.player_id:
                model_player = p
                break
        if model_player is None:
            raise ValueError(f"Player with id {private_state.player_id} not found in public players.")
        
        # 1. Revealed info cards - 5 ints (Ruby, Sapphire, Emerald, Amethyst, Diamond)
        revealed_info_counts = self._count_gems_by_suit(private_state.info_cards_revealed)
        inputs.extend(revealed_info_counts)
        
        # 2. Unrevealed info cards - 5 ints (Ruby, Sapphire, Emerald, Amethyst, Diamond)
        unrevealed_info_counts = self._count_gems_by_suit(private_state.info_cards_unrevealed)
        inputs.extend(unrevealed_info_counts)
        
        # 3. Gems owned - 5 ints (Ruby, Sapphire, Emerald, Amethyst, Diamond)
        gems_owned_counts = self._count_gems_by_suit(model_player.gems_owned)
        inputs.extend(gems_owned_counts)
        
        # 4. Loans value - 1 int (negative, total principal to repay)
        loans_value = self._compute_loans_value(model_player.loans)
        inputs.append(loans_value)
        
        # 5. Investments value - 1 int (total payout)
        investments_value = self._compute_investments_value(model_player.investments)
        inputs.append(investments_value)
        
        # 6. Trinkets value - 1 int (total points from claimed trinkets)
        trinkets_value = model_player.trinket_points
        inputs.append(trinkets_value)
        
        # 7. Cash remaining - 1 int
        cash = model_player.cash
        inputs.append(cash)
        
        return inputs

    def _encode_opponent_player(self, player: interface.PlayerPublicState):
        """Encode an opponent player's state information."""
        inputs = []
        
        # 1. Revealed info cards - 5 ints (Ruby, Sapphire, Emerald, Amethyst, Diamond)
        revealed_info_counts = self._count_gems_by_suit(player.revealed_info)
        inputs.extend(revealed_info_counts)
        
        # 2. Unrevealed info cards count - 1 int (just the count, not breakdown)
        unrevealed_count = player.unrevealed_info_count
        inputs.append(unrevealed_count)
        
        # 3. Gems owned - 5 ints (Ruby, Sapphire, Emerald, Amethyst, Diamond)
        gems_owned_counts = self._count_gems_by_suit(player.gems_owned)
        inputs.extend(gems_owned_counts)
        
        # 4. Loans value - 1 int (negative, total principal to repay)
        loans_value = self._compute_loans_value(player.loans)
        inputs.append(loans_value)
        
        # 5. Investments value - 1 int (total payout + locked)
        investments_value = self._compute_investments_value(player.investments)
        inputs.append(investments_value)
        
        # 6. Trinkets value - 1 int (total points from claimed trinkets)
        trinkets_value = player.trinket_points
        inputs.append(trinkets_value)
        
        # 7. Cash remaining - 1 int
        cash = player.cash
        inputs.append(cash)
        
        return inputs

    def encode_input(self, game_observation: interface.GameObservation):
        public_state = game_observation.public
        private_state = game_observation.private
        context = game_observation.context

        self_id = private_state.player_id

        # Establish player order mapping on first observation
        if self.player_order_map is None:
            # Find self in seating order and create relative mapping
            seating_order = context.seating_order
            self_idx = seating_order.index(self_id)
            # Create mapping: self is 0, next player is 1, etc.
            self.player_order_map = {}
            for i, pid in enumerate(seating_order):
                relative_pos = (i - self_idx) % len(seating_order)
                self.player_order_map[pid] = relative_pos

        # Save current observation as previous (for next call)
        self.prev_observation = game_observation

        inputs = []

        # Encode game state
        game_state_inputs = self._encode_game_state(public_state, context)
        inputs.extend(game_state_inputs)

        # Encode model player state
        model_player_inputs = self._encode_model_player(public_state, private_state)
        inputs.extend(model_player_inputs)

        # Encode opponent player states (in order of relative index: 1, 2, 3, etc.)
        # Get all players and sort by relative index, excluding self (index 0)
        opponent_players = []
        for player in public_state.players:
            relative_idx = self.player_order_map.get(player.player_id, -1)
            if relative_idx > 0:  # Exclude self (index 0)
                opponent_players.append((relative_idx, player))
        
        # Sort by relative index to ensure order: 1, 2, 3, etc.
        opponent_players.sort(key=lambda x: x[0])
        
        # Encode each opponent in order
        for _, opponent in opponent_players:
            opponent_inputs = self._encode_opponent_player(opponent)
            inputs.extend(opponent_inputs)

        return np.array(inputs, dtype=np.float32)

    def decode_value_output(self, value_tensor):
        value_probs = value_tensor.squeeze(0).softmax(dim=0)
        value = value_probs.argmax()
        value_confidence = value_probs.max()
        return value, value_confidence

    def decode_revealed_card_output(self, revealed_card_tensor):
        revealed_card_probs = revealed_card_tensor.squeeze(0).softmax(dim=0)

        card_prefereces = [(idx + 1, prob) for idx, prob in enumerate(revealed_card_probs)]
        card_prefereces.sort(key=lambda x: x[1], reverse=True)
        return card_prefereces


