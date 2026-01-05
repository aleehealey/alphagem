from __future__ import annotations
import sys
from app.competition.interface import *


class HumanBot(PocketRocketsBot):
    """Bot that prompts the human player for input."""
    
    @property
    def bot_name(self) -> str:
        return "You"
    
    def get_bid(self, obs: GameObservation) -> Bid:
        """Prompt human for bid amount."""
        max_bid = legal_max_bid(obs)
        
        while True:
            try:
                print(f"\n{'='*60}")
                print(f"Your turn! You have ${obs.me.cash} cash.")
                print(f"Maximum bid: ${max_bid}")
                print(f"{'='*60}")
                
                bid_input = input("Enter your bid (0 to pass): ").strip()
                bid_amount = int(bid_input)
                
                if bid_amount < 0:
                    print("❌ Bid cannot be negative. Try again.")
                    continue
                if bid_amount > max_bid:
                    print(f"❌ Bid ${bid_amount} exceeds your cash (${max_bid}). Try again.")
                    continue
                
                return Bid(bid_amount=bid_amount)
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n\nGame interrupted by user.")
                sys.exit(0)
    
    def choose_info_to_reveal(self, obs: GameObservation, result: AuctionResult) -> str:
        """Prompt human to choose which info card to reveal."""
        unrevealed = obs.private.info_cards_unrevealed
        
        if not unrevealed:
            # Shouldn't happen, but handle gracefully
            return ""
        
        print(f"\n{'='*60}")
        print("You won the auction! Choose an info card to reveal:")
        print(f"{'='*60}")
        
        for i, card in enumerate(unrevealed):
            print(f"  {i+1}. {card.suit.name} (ID: {card.id})")
        
        while True:
            try:
                choice_input = input(f"\nEnter card number (1-{len(unrevealed)}): ").strip()
                choice_idx = int(choice_input) - 1
                
                if 0 <= choice_idx < len(unrevealed):
                    chosen_card = unrevealed[choice_idx]
                    print(f"✓ Revealing {chosen_card.suit.name} (ID: {chosen_card.id})")
                    return chosen_card.id
                else:
                    print(f"❌ Invalid choice. Please enter a number between 1 and {len(unrevealed)}.")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n\nGame interrupted by user.")
                sys.exit(0)

