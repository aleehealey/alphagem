"""
Unit tests for AlphaGem encoding functions.
"""

import unittest
import numpy as np
from unittest.mock import Mock
from app.alphagem.alpha_gem import AlphaGem
from app.competition import interface


class TestAlphaGemEncoding(unittest.TestCase):
    """Test suite for AlphaGem encoding functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = AlphaGem()

    def test_encode_suit(self):
        """Test suit encoding function."""
        # Test all suits
        self.assertEqual(self.model._encode_suit(interface.Suit.RUBY), 1)
        self.assertEqual(self.model._encode_suit(interface.Suit.SAPPHIRE), 2)
        self.assertEqual(self.model._encode_suit(interface.Suit.EMERALD), 3)
        self.assertEqual(self.model._encode_suit(interface.Suit.AMETHYST), 4)
        self.assertEqual(self.model._encode_suit(interface.Suit.DIAMOND), 5)
        
        # Test None
        self.assertEqual(self.model._encode_suit(None), 0)
        
        # Test invalid suit (should return 0)
        self.assertEqual(self.model._encode_suit("invalid"), 0)

    def test_count_gems_by_suit(self):
        """Test counting gems by suit."""
        # Test empty list
        cards = []
        result = self.model._count_gems_by_suit(cards)
        self.assertEqual(result, [0, 0, 0, 0, 0])
        
        # Test single card of each type
        cards = [
            interface.Card(id="R1", suit=interface.Suit.RUBY),
            interface.Card(id="S1", suit=interface.Suit.SAPPHIRE),
            interface.Card(id="E1", suit=interface.Suit.EMERALD),
            interface.Card(id="A1", suit=interface.Suit.AMETHYST),
            interface.Card(id="D1", suit=interface.Suit.DIAMOND),
        ]
        result = self.model._count_gems_by_suit(cards)
        self.assertEqual(result, [1, 1, 1, 1, 1])
        
        # Test multiple cards of same suit
        cards = [
            interface.Card(id="R1", suit=interface.Suit.RUBY),
            interface.Card(id="R2", suit=interface.Suit.RUBY),
            interface.Card(id="R3", suit=interface.Suit.RUBY),
            interface.Card(id="S1", suit=interface.Suit.SAPPHIRE),
        ]
        result = self.model._count_gems_by_suit(cards)
        self.assertEqual(result, [3, 1, 0, 0, 0])
        
        # Test mixed cards
        cards = [
            interface.Card(id="R1", suit=interface.Suit.RUBY),
            interface.Card(id="R2", suit=interface.Suit.RUBY),
            interface.Card(id="E1", suit=interface.Suit.EMERALD),
            interface.Card(id="D1", suit=interface.Suit.DIAMOND),
            interface.Card(id="D2", suit=interface.Suit.DIAMOND),
        ]
        result = self.model._count_gems_by_suit(cards)
        self.assertEqual(result, [2, 0, 1, 0, 2])

    def test_compute_loans_value(self):
        """Test computing loans value."""
        # Test empty loans
        loans = ()
        result = self.model._compute_loans_value(loans)
        self.assertEqual(result, 0)
        
        # Test single loan
        loans = (
            interface.LoanPosition(id="L1", principal=10, winning_bid=5),
        )
        result = self.model._compute_loans_value(loans)
        self.assertEqual(result, -10)
        
        # Test multiple loans
        loans = (
            interface.LoanPosition(id="L1", principal=10, winning_bid=5),
            interface.LoanPosition(id="L2", principal=20, winning_bid=15),
            interface.LoanPosition(id="L3", principal=10, winning_bid=8),
        )
        result = self.model._compute_loans_value(loans)
        self.assertEqual(result, -40)

    def test_compute_investments_value(self):
        """Test computing investments value."""
        # Test empty investments
        investments = ()
        result = self.model._compute_investments_value(investments)
        self.assertEqual(result, 0)
        
        # Test single investment
        investments = (
            interface.InvestmentPosition(id="I1", payout=5, locked=10),
        )
        result = self.model._compute_investments_value(investments)
        self.assertEqual(result, 15)  # payout + locked
        
        # Test multiple investments
        investments = (
            interface.InvestmentPosition(id="I1", payout=5, locked=10),
            interface.InvestmentPosition(id="I2", payout=10, locked=20),
            interface.InvestmentPosition(id="I3", payout=5, locked=5),
        )
        result = self.model._compute_investments_value(investments)
        self.assertEqual(result, 55)  # (5+10) + (10+20) + (5+5) = 55

    def test_encode_game_state(self):
        """Test encoding game state."""
        # Create test data
        trinket_obj = interface.TrinketObjective(
            id="T1",
            points=10,
            required_cards=(
                interface.Card(id="R1", suit=interface.Suit.RUBY),
                interface.Card(id="R2", suit=interface.Suit.RUBY),
                interface.Card(id="S1", suit=interface.Suit.SAPPHIRE),
            ),
            display_text="Test trinket"
        )
        
        trinkets = (
            interface.TrinketState(objective=trinket_obj, claimed_by=None),
            interface.TrinketState(objective=trinket_obj, claimed_by=0),  # Claimed
        )
        
        value_chart = interface.ValueChart(mapping=[0, 1, 2, 3, 4, 5])
        
        public_state = interface.GamePublicState(
            num_players=3,
            players=(),
            trinkets=trinkets,
            value_chart=value_chart,
            action_discard=(),
            past_auctions=(),
            action_counts_remaining={
                interface.ActionType.AUCTION_1: 5,
                interface.ActionType.AUCTION_2: 2,
                interface.ActionType.LOAN_10: 1,
                interface.ActionType.LOAN_20: 1,
                interface.ActionType.INVESTMENT_5: 1,
                interface.ActionType.INVESTMENT_10: 1,
            }
        )
        
        context = interface.TurnContext(
            turn_index=0,
            action=interface.Action(id="A1", kind=interface.ActionType.AUCTION_1),
            upcoming_gems=(
                interface.Card(id="G1", suit=interface.Suit.RUBY),
                interface.Card(id="G2", suit=interface.Suit.EMERALD),
            ),
            biddable_pile_count=20,
            tiebreak_leader_id=0,
            seating_order=(0, 1, 2),
        )
        
        # Set player order map for tiebreak leader
        self.model.player_order_map = {0: 0, 1: 1, 2: 2}
        
        result = self.model._encode_game_state(public_state, context)
        
        # Check trinkets (2 trinkets, each 5 values)
        # First trinket: [2, 1, 0, 0, 0] (2 Ruby, 1 Sapphire)
        # Second trinket: [0, 0, 0, 0, 0] (claimed)
        # Plus 2 more empty trinkets: [0, 0, 0, 0, 0] each
        self.assertEqual(result[0:5], [2, 1, 0, 0, 0])  # First trinket
        self.assertEqual(result[5:10], [0, 0, 0, 0, 0])  # Second trinket (claimed)
        self.assertEqual(result[10:15], [0, 0, 0, 0, 0])  # Third trinket (empty)
        self.assertEqual(result[15:20], [0, 0, 0, 0, 0])  # Fourth trinket (empty)
        
        # Check gems left (index 20)
        self.assertEqual(result[20], 20)
        
        # Check current gem (index 21) - Ruby = 1
        self.assertEqual(result[21], 1)
        
        # Check second gem (index 22) - Emerald = 3
        self.assertEqual(result[22], 3)
        
        # Check tiebreak leader (index 23) - relative index 0
        self.assertEqual(result[23], 0)
        
        # Check action counts (indices 24-29)
        self.assertEqual(result[24], 5)  # AUCTION_1
        self.assertEqual(result[25], 2)  # AUCTION_2
        self.assertEqual(result[26], 1)  # LOAN_10
        self.assertEqual(result[27], 1)  # LOAN_20
        self.assertEqual(result[28], 1)  # INVESTMENT_5
        self.assertEqual(result[29], 1)  # INVESTMENT_10

    def test_encode_model_player(self):
        """Test encoding model player state."""
        # Create test data
        gems_owned = (
            interface.Card(id="G1", suit=interface.Suit.RUBY),
            interface.Card(id="G2", suit=interface.Suit.RUBY),
            interface.Card(id="G3", suit=interface.Suit.DIAMOND),
        )
        
        loans = (
            interface.LoanPosition(id="L1", principal=10, winning_bid=5),
            interface.LoanPosition(id="L2", principal=20, winning_bid=15),
        )
        
        investments = (
            interface.InvestmentPosition(id="I1", payout=5, locked=10),
            interface.InvestmentPosition(id="I2", payout=10, locked=20),
        )
        
        model_player = interface.PlayerPublicState(
            player_id=0,
            name="TestPlayer",
            cash=50,
            gems_owned=gems_owned,
            loans=loans,
            investments=investments,
            revealed_info=(
                interface.Card(id="I1", suit=interface.Suit.EMERALD),
                interface.Card(id="I2", suit=interface.Suit.EMERALD),
            ),
            unrevealed_info_count=3,
            trinket_points=15,
        )
        
        private_state = interface.PlayerPrivateState(
            player_id=0,
            info_cards_unrevealed=(
                interface.Card(id="U1", suit=interface.Suit.RUBY),
                interface.Card(id="U2", suit=interface.Suit.SAPPHIRE),
                interface.Card(id="U3", suit=interface.Suit.SAPPHIRE),
            ),
            info_cards_revealed=(
                interface.Card(id="I1", suit=interface.Suit.EMERALD),
                interface.Card(id="I2", suit=interface.Suit.EMERALD),
            ),
        )
        
        # Mock public_state.me
        public_state = Mock()
        public_state.me = model_player
        
        result = self.model._encode_model_player(public_state, private_state)
        
        # Check revealed info (5 values) - 2 Emerald
        self.assertEqual(result[0:5], [0, 0, 2, 0, 0])
        
        # Check unrevealed info (5 values) - 1 Ruby, 2 Sapphire
        self.assertEqual(result[5:10], [1, 2, 0, 0, 0])
        
        # Check gems owned (5 values) - 2 Ruby, 1 Diamond
        self.assertEqual(result[10:15], [2, 0, 0, 0, 1])
        
        # Check loans value (index 15) - -30
        self.assertEqual(result[15], -30)
        
        # Check investments value (index 16) - 45 (5+10 + 10+20)
        self.assertEqual(result[16], 45)
        
        # Check trinkets value (index 17) - 15
        self.assertEqual(result[17], 15)
        
        # Check cash (index 18) - 50
        self.assertEqual(result[18], 50)

    def test_encode_opponent_player(self):
        """Test encoding opponent player state."""
        # Create test data
        opponent = interface.PlayerPublicState(
            player_id=1,
            name="Opponent",
            cash=30,
            gems_owned=(
                interface.Card(id="G1", suit=interface.Suit.SAPPHIRE),
                interface.Card(id="G2", suit=interface.Suit.AMETHYST),
            ),
            loans=(
                interface.LoanPosition(id="L1", principal=10, winning_bid=5),
            ),
            investments=(
                interface.InvestmentPosition(id="I1", payout=5, locked=10),
            ),
            revealed_info=(
                interface.Card(id="I1", suit=interface.Suit.RUBY),
            ),
            unrevealed_info_count=4,
            trinket_points=5,
        )
        
        result = self.model._encode_opponent_player(opponent)
        
        # Check revealed info (5 values) - 1 Ruby
        self.assertEqual(result[0:5], [1, 0, 0, 0, 0])
        
        # Check unrevealed count (index 5) - 4
        self.assertEqual(result[5], 4)
        
        # Check gems owned (5 values) - 1 Sapphire, 1 Amethyst
        self.assertEqual(result[6:11], [0, 1, 0, 1, 0])
        
        # Check loans value (index 11) - -10
        self.assertEqual(result[11], -10)
        
        # Check investments value (index 12) - 15 (5+10)
        self.assertEqual(result[12], 15)
        
        # Check trinkets value (index 13) - 5
        self.assertEqual(result[13], 5)
        
        # Check cash (index 14) - 30
        self.assertEqual(result[14], 30)

    def test_encode_input_full(self):
        """Test full encode_input function."""
        # Create comprehensive test data
        trinket_obj = interface.TrinketObjective(
            id="T1",
            points=10,
            required_cards=(
                interface.Card(id="R1", suit=interface.Suit.RUBY),
            ),
            display_text="Test"
        )
        
        value_chart = interface.ValueChart(mapping=[0, 1, 2, 3, 4, 5])
        
        # Model player
        model_player = interface.PlayerPublicState(
            player_id=0,
            name="Model",
            cash=50,
            gems_owned=(interface.Card(id="G1", suit=interface.Suit.RUBY),),
            loans=(),
            investments=(),
            revealed_info=(),
            unrevealed_info_count=5,
            trinket_points=0,
        )
        
        # Opponent player
        opponent = interface.PlayerPublicState(
            player_id=1,
            name="Opponent",
            cash=30,
            gems_owned=(),
            loans=(),
            investments=(),
            revealed_info=(),
            unrevealed_info_count=5,
            trinket_points=0,
        )
        
        public_state = interface.GamePublicState(
            num_players=2,
            players=(model_player, opponent),
            trinkets=(interface.TrinketState(objective=trinket_obj, claimed_by=None),),
            value_chart=value_chart,
            action_discard=(),
            past_auctions=(),
            action_counts_remaining={
                interface.ActionType.AUCTION_1: 5,
                interface.ActionType.AUCTION_2: 2,
                interface.ActionType.LOAN_10: 1,
                interface.ActionType.LOAN_20: 1,
                interface.ActionType.INVESTMENT_5: 1,
                interface.ActionType.INVESTMENT_10: 1,
            }
        )
        
        private_state = interface.PlayerPrivateState(
            player_id=0,
            info_cards_unrevealed=(
                interface.Card(id="U1", suit=interface.Suit.RUBY),
            ),
            info_cards_revealed=(),
        )
        
        context = interface.TurnContext(
            turn_index=0,
            action=interface.Action(id="A1", kind=interface.ActionType.AUCTION_1),
            upcoming_gems=(interface.Card(id="G1", suit=interface.Suit.RUBY),),
            biddable_pile_count=20,
            tiebreak_leader_id=0,
            seating_order=(0, 1),
        )
        
        observation = interface.GameObservation(
            public=public_state,
            private=private_state,
            context=context,
            me=model_player,
        )
        
        result = self.model.encode_input(observation)
        
        # Result should be a numpy array
        self.assertIsInstance(result, np.ndarray)
        self.assertGreater(len(result), 0)
        
        # Verify player order map was created
        self.assertIsNotNone(self.model.player_order_map)
        self.assertEqual(self.model.player_order_map[0], 0)
        self.assertEqual(self.model.player_order_map[1], 1)
        
        # Verify previous observation was saved
        self.assertIsNotNone(self.model.prev_observation)
        self.assertEqual(self.model.prev_observation, observation)


if __name__ == '__main__':
    unittest.main()

