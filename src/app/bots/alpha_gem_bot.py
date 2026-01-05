from __future__ import annotations
from app.competition.interface import *
from app.alphagem.alpha_gem import AlphaGem
import torch

# Bot wrapper that uses AlphaGem model to play
class AlphaGemBot(PocketRocketsBot):
    """Wrapper that makes AlphaGem play as a PocketRocketsBot."""
    
    def __init__(self, model: AlphaGem, name: str = "AlphaGem"):
        self.model = model
        self._name = name
        self.model.eval()  # Set to evaluation mode
        # Reset state tracking for new game
        self.model.prev_observation = None
        self.model.player_order_map = None
    
    @property
    def bot_name(self) -> str:
        return self._name
    
    def on_game_start(self, obs: GameObservation) -> None:
        """Reset state when game starts."""
        self.model.prev_observation = None
        self.model.player_order_map = None
    
    def get_bid(self, obs: GameObservation) -> Bid:
        """Use the model to determine bid amount."""
        # Use the forward function which takes GameObservation directly
        with torch.no_grad():
            value_output, revealed_card_output = self.model.forward(obs)
            
            # Decode the value output using the new decode function
            value, value_confidence = self.model.decode_value_output(value_output)
        
        # Convert decoded value to bid amount (clamp to legal range)
        max_bid = legal_max_bid(obs)
        # The value is an index (0-19), scale it to a reasonable bid range
        # You can adjust this scaling factor based on your value chart
        bid_amount = int(min(value.item() * 2, max_bid))
        bid_amount = max(0, bid_amount)  # Ensure non-negative
        
        return Bid(bid_amount=bid_amount)
    
    def choose_info_to_reveal(self, obs: GameObservation, result: AuctionResult) -> str:
        """Choose which info card to reveal when winning an auction."""
        # Get unrevealed cards
        unrevealed = obs.private.info_cards_unrevealed
        if not unrevealed:
            # Fallback: return first available revealed card (shouldn't happen)
            return obs.private.info_cards_revealed[0].id if obs.private.info_cards_revealed else ""
        
        # Use the forward function to get model outputs
        with torch.no_grad():
            value_output, revealed_card_output = self.model.forward(obs)
            
            # Get the card preferences using the new decode function
            # Returns list of (suit_idx, prob) tuples sorted by probability (highest first)
            # suit_idx is 1-5 (1=Ruby, 2=Sapphire, 3=Emerald, 4=Amethyst, 5=Diamond)
            card_preferences = self.model.decode_revealed_card_output(revealed_card_output)
            
            # Map suit indices to actual suits
            suit_map = {
                1: Suit.RUBY,
                2: Suit.SAPPHIRE,
                3: Suit.EMERALD,
                4: Suit.AMETHYST,
                5: Suit.DIAMOND,
            }
            
            # Go through preferences in order (already sorted by probability)
            for suit_idx, prob in card_preferences:
                preferred_suit = suit_map.get(suit_idx)
                
                # Find the first unrevealed card matching this preferred suit
                if preferred_suit is not None:
                    for card in unrevealed:
                        if card.suit == preferred_suit:
                            return card.id
        
        # If no match found (shouldn't happen), return the first unrevealed card
        return unrevealed[0].id