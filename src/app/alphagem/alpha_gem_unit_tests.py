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
        np.testing.assert_array_equal(result, [0, 0, 0, 0, 0])
        
        # Test single card of each type
        cards = [
            interface.Card(id="R1", suit=interface.Suit.RUBY),
            interface.Card(id="S1", suit=interface.Suit.SAPPHIRE),
            interface.Card(id="E1", suit=interface.Suit.EMERALD),
            interface.Card(id="A1", suit=interface.Suit.AMETHYST),
            interface.Card(id="D1", suit=interface.Suit.DIAMOND),
        ]
        result = self.model._count_gems_by_suit(cards)
        np.testing.assert_array_equal(result, [1, 1, 1, 1, 1])
        
        # Test multiple cards of same suit
        cards = [
            interface.Card(id="R1", suit=interface.Suit.RUBY),
            interface.Card(id="R2", suit=interface.Suit.RUBY),
            interface.Card(id="R3", suit=interface.Suit.RUBY),
            interface.Card(id="S1", suit=interface.Suit.SAPPHIRE),
        ]
        result = self.model._count_gems_by_suit(cards)
        np.testing.assert_array_equal(result, [3, 1, 0, 0, 0])
        
        # Test mixed cards
        cards = [
            interface.Card(id="R1", suit=interface.Suit.RUBY),
            interface.Card(id="R2", suit=interface.Suit.RUBY),
            interface.Card(id="E1", suit=interface.Suit.EMERALD),
            interface.Card(id="D1", suit=interface.Suit.DIAMOND),
            interface.Card(id="D2", suit=interface.Suit.DIAMOND),
        ]
        result = self.model._count_gems_by_suit(cards)
        np.testing.assert_array_equal(result, [2, 0, 1, 0, 2])

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
        np.testing.assert_array_equal(result[0:5], [2, 1, 0, 0, 0])  # First trinket
        np.testing.assert_array_equal(result[5:10], [0, 0, 0, 0, 0])  # Second trinket (claimed)
        np.testing.assert_array_equal(result[10:15], [0, 0, 0, 0, 0])  # Third trinket (empty)
        np.testing.assert_array_equal(result[15:20], [0, 0, 0, 0, 0])  # Fourth trinket (empty)
        
        # Check gems left (index 20)
        self.assertEqual(float(result[20]), 20)
        
        # Check current gem (index 21) - Ruby = 1
        self.assertEqual(float(result[21]), 1)
        
        # Check second gem (index 22) - Emerald = 3
        self.assertEqual(float(result[22]), 3)
        
        # Check tiebreak leader (index 23) - relative index 0
        self.assertEqual(float(result[23]), 0)
        
        # Check action counts (indices 24-29)
        self.assertEqual(float(result[24]), 5)  # AUCTION_1
        self.assertEqual(float(result[25]), 2)  # AUCTION_2
        self.assertEqual(float(result[26]), 1)  # LOAN_10
        self.assertEqual(float(result[27]), 1)  # LOAN_20
        self.assertEqual(float(result[28]), 1)  # INVESTMENT_5
        self.assertEqual(float(result[29]), 1)  # INVESTMENT_10

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
        # Construct a mock PublicGameState with players and assign the test player as in interface
        public_state = Mock()
        public_state.players = (model_player,)
        
        result = self.model._encode_model_player(public_state, private_state)
        
        # Check revealed info (5 values) - 2 Emerald
        np.testing.assert_array_equal(result[0:5], [0, 0, 2, 0, 0])
        
        # Check unrevealed info (5 values) - 1 Ruby, 2 Sapphire
        np.testing.assert_array_equal(result[5:10], [1, 2, 0, 0, 0])
        
        # Check gems owned (5 values) - 2 Ruby, 1 Diamond
        np.testing.assert_array_equal(result[10:15], [2, 0, 0, 0, 1])
        
        # Check loans value (index 15) - -30
        self.assertEqual(float(result[15]), -30)
        
        # Check investments value (index 16) - 45 (5+10 + 10+20)
        self.assertEqual(float(result[16]), 45)
        
        # Check trinkets value (index 17) - 15
        self.assertEqual(float(result[17]), 15)
        
        # Check cash (index 18) - 50
        self.assertEqual(float(result[18]), 50)

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
        np.testing.assert_array_equal(result[0:5], [1, 0, 0, 0, 0])
        
        # Check unrevealed count (index 5) - 4
        self.assertEqual(float(result[5]), 4)
        
        # Check gems owned (5 values) - 1 Sapphire, 1 Amethyst
        np.testing.assert_array_equal(result[6:11], [0, 1, 0, 1, 0])
        
        # Check loans value (index 11) - -10
        self.assertEqual(float(result[11]), -10)
        
        # Check investments value (index 12) - 15 (5+10)
        self.assertEqual(float(result[12]), 15)
        
        # Check trinkets value (index 13) - 5
        self.assertEqual(float(result[13]), 5)
        
        # Check cash (index 14) - 30
        self.assertEqual(float(result[14]), 30)

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

    def test_tracking_over_time_multiple_observations(self):
        """Test that state tracking works correctly over multiple observations."""
        # Create initial observation
        trinket_obj = interface.TrinketObjective(
            id="T1",
            points=10,
            required_cards=(interface.Card(id="R1", suit=interface.Suit.RUBY),),
            display_text="Test"
        )
        
        value_chart = interface.ValueChart(mapping=[0, 1, 2, 3, 4, 5])
        
        # Initial state - turn 0
        model_player_t0 = interface.PlayerPublicState(
            player_id=0,
            name="Model",
            cash=50,
            gems_owned=(),
            loans=(),
            investments=(),
            revealed_info=(),
            unrevealed_info_count=5,
            trinket_points=0,
        )
        
        opponent_t0 = interface.PlayerPublicState(
            player_id=1,
            name="Opponent",
            cash=50,
            gems_owned=(),
            loans=(),
            investments=(),
            revealed_info=(),
            unrevealed_info_count=5,
            trinket_points=0,
        )
        
        public_state_t0 = interface.GamePublicState(
            num_players=2,
            players=(model_player_t0, opponent_t0),
            trinkets=(interface.TrinketState(objective=trinket_obj, claimed_by=None),),
            value_chart=value_chart,
            action_discard=(),
            past_auctions=(),
            action_counts_remaining={
                interface.ActionType.AUCTION_1: 10,
                interface.ActionType.AUCTION_2: 5,
                interface.ActionType.LOAN_10: 2,
                interface.ActionType.LOAN_20: 2,
                interface.ActionType.INVESTMENT_5: 2,
                interface.ActionType.INVESTMENT_10: 2,
            }
        )
        
        private_state_t0 = interface.PlayerPrivateState(
            player_id=0,
            info_cards_unrevealed=(
                interface.Card(id="U1", suit=interface.Suit.RUBY),
                interface.Card(id="U2", suit=interface.Suit.SAPPHIRE),
            ),
            info_cards_revealed=(),
        )
        
        context_t0 = interface.TurnContext(
            turn_index=0,
            action=interface.Action(id="A1", kind=interface.ActionType.AUCTION_1),
            upcoming_gems=(interface.Card(id="G1", suit=interface.Suit.RUBY),),
            biddable_pile_count=25,
            tiebreak_leader_id=0,
            seating_order=(0, 1),
        )
        
        observation_t0 = interface.GameObservation(
            public=public_state_t0,
            private=private_state_t0,
            context=context_t0,
            me=model_player_t0,
        )
        
        # First observation - should establish player_order_map
        result_t0 = self.model.encode_input(observation_t0)
        
        # Verify player_order_map was created
        self.assertIsNotNone(self.model.player_order_map)
        self.assertEqual(self.model.player_order_map[0], 0)
        self.assertEqual(self.model.player_order_map[1], 1)
        
        # Verify prev_observation was saved
        self.assertIsNotNone(self.model.prev_observation)
        self.assertEqual(self.model.prev_observation, observation_t0)
        
        # Verify initial state encoding
        self.assertIsInstance(result_t0, np.ndarray)
        self.assertGreater(len(result_t0), 0)
        
        # Second observation - turn 1 (after some changes)
        model_player_t1 = interface.PlayerPublicState(
            player_id=0,
            name="Model",
            cash=45,  # Lost 5 cash (bid on something)
            gems_owned=(
                interface.Card(id="G1", suit=interface.Suit.RUBY),  # Won a gem
            ),
            loans=(),
            investments=(),
            revealed_info=(
                interface.Card(id="I1", suit=interface.Suit.EMERALD),  # Revealed a card
            ),
            unrevealed_info_count=4,  # One less unrevealed
            trinket_points=0,
        )
        
        opponent_t1 = interface.PlayerPublicState(
            player_id=1,
            name="Opponent",
            cash=40,  # Lost 10 cash
            gems_owned=(),
            loans=(
                interface.LoanPosition(id="L1", principal=10, winning_bid=10),  # Took a loan
            ),
            investments=(),
            revealed_info=(),
            unrevealed_info_count=5,
            trinket_points=0,
        )
        
        public_state_t1 = interface.GamePublicState(
            num_players=2,
            players=(model_player_t1, opponent_t1),
            trinkets=(interface.TrinketState(objective=trinket_obj, claimed_by=None),),
            value_chart=value_chart,
            action_discard=(
                interface.Action(id="A1", kind=interface.ActionType.AUCTION_1),
            ),
            past_auctions=(),
            action_counts_remaining={
                interface.ActionType.AUCTION_1: 9,  # One less
                interface.ActionType.AUCTION_2: 5,
                interface.ActionType.LOAN_10: 1,  # One less
                interface.ActionType.LOAN_20: 2,
                interface.ActionType.INVESTMENT_5: 2,
                interface.ActionType.INVESTMENT_10: 2,
            }
        )
        
        private_state_t1 = interface.PlayerPrivateState(
            player_id=0,
            info_cards_unrevealed=(
                interface.Card(id="U1", suit=interface.Suit.RUBY),
                interface.Card(id="U2", suit=interface.Suit.SAPPHIRE),
                interface.Card(id="U3", suit=interface.Suit.EMERALD),
            ),
            info_cards_revealed=(
                interface.Card(id="I1", suit=interface.Suit.EMERALD),
            ),
        )
        
        context_t1 = interface.TurnContext(
            turn_index=1,
            action=interface.Action(id="A2", kind=interface.ActionType.AUCTION_2),
            upcoming_gems=(
                interface.Card(id="G2", suit=interface.Suit.SAPPHIRE),
                interface.Card(id="G3", suit=interface.Suit.EMERALD),
            ),
            biddable_pile_count=23,  # Two less gems
            tiebreak_leader_id=0,  # Still the leader
            seating_order=(0, 1),  # Same seating order
        )
        
        observation_t1 = interface.GameObservation(
            public=public_state_t1,
            private=private_state_t1,
            context=context_t1,
            me=model_player_t1,
        )
        
        # Second observation - should preserve player_order_map and update prev_observation
        result_t1 = self.model.encode_input(observation_t1)
        
        # Verify player_order_map persists (not recreated)
        self.assertIsNotNone(self.model.player_order_map)
        self.assertEqual(self.model.player_order_map[0], 0)
        self.assertEqual(self.model.player_order_map[1], 1)
        
        # Verify prev_observation was updated
        self.assertIsNotNone(self.model.prev_observation)
        self.assertEqual(self.model.prev_observation, observation_t1)
        self.assertNotEqual(self.model.prev_observation, observation_t0)
        
        # Verify encoding reflects changes
        self.assertIsInstance(result_t1, np.ndarray)
        self.assertEqual(len(result_t1), len(result_t0))  # Same structure
        
        # Third observation - turn 2 (more changes, including tiebreak leader change)
        model_player_t2 = interface.PlayerPublicState(
            player_id=0,
            name="Model",
            cash=40,  # Lost more cash
            gems_owned=(
                interface.Card(id="G1", suit=interface.Suit.RUBY),
                interface.Card(id="G2", suit=interface.Suit.SAPPHIRE),  # Won another gem
            ),
            loans=(),
            investments=(
                interface.InvestmentPosition(id="I1", payout=5, locked=5),  # Got an investment
            ),
            revealed_info=(
                interface.Card(id="I1", suit=interface.Suit.EMERALD),
            ),
            unrevealed_info_count=4,
            trinket_points=0,
        )
        
        opponent_t2 = interface.PlayerPublicState(
            player_id=1,
            name="Opponent",
            cash=35,
            gems_owned=(),
            loans=(
                interface.LoanPosition(id="L1", principal=10, winning_bid=10),
            ),
            investments=(),
            revealed_info=(),
            unrevealed_info_count=5,
            trinket_points=0,
        )
        
        public_state_t2 = interface.GamePublicState(
            num_players=2,
            players=(model_player_t2, opponent_t2),
            trinkets=(interface.TrinketState(objective=trinket_obj, claimed_by=0),),  # Claimed!
            value_chart=value_chart,
            action_discard=(
                interface.Action(id="A1", kind=interface.ActionType.AUCTION_1),
                interface.Action(id="A2", kind=interface.ActionType.AUCTION_2),
            ),
            past_auctions=(),
            action_counts_remaining={
                interface.ActionType.AUCTION_1: 9,
                interface.ActionType.AUCTION_2: 4,  # One less
                interface.ActionType.LOAN_10: 1,
                interface.ActionType.LOAN_20: 2,
                interface.ActionType.INVESTMENT_5: 1,  # One less
                interface.ActionType.INVESTMENT_10: 2,
            }
        )
        
        private_state_t2 = interface.PlayerPrivateState(
            player_id=0,
            info_cards_unrevealed=(
                interface.Card(id="U1", suit=interface.Suit.RUBY),
                interface.Card(id="U2", suit=interface.Suit.SAPPHIRE),
                interface.Card(id="U3", suit=interface.Suit.EMERALD),
            ),
            info_cards_revealed=(
                interface.Card(id="I1", suit=interface.Suit.EMERALD),
            ),
        )
        
        context_t2 = interface.TurnContext(
            turn_index=2,
            action=interface.Action(id="A3", kind=interface.ActionType.LOAN_10),
            upcoming_gems=(interface.Card(id="G4", suit=interface.Suit.DIAMOND),),
            biddable_pile_count=22,  # One less gem
            tiebreak_leader_id=1,  # Leader changed!
            seating_order=(0, 1),  # Same seating order
        )
        
        observation_t2 = interface.GameObservation(
            public=public_state_t2,
            private=private_state_t2,
            context=context_t2,
            me=model_player_t2,
        )
        
        # Third observation
        result_t2 = self.model.encode_input(observation_t2)
        
        # Verify player_order_map still persists
        self.assertIsNotNone(self.model.player_order_map)
        self.assertEqual(self.model.player_order_map[0], 0)
        self.assertEqual(self.model.player_order_map[1], 1)
        
        # Verify prev_observation updated again
        self.assertEqual(self.model.prev_observation, observation_t2)
        
        # Verify encoding structure is consistent
        self.assertEqual(len(result_t2), len(result_t0))
        self.assertEqual(len(result_t2), len(result_t1))
        
        # Verify specific changes are reflected in encoding
        # Check that trinket is now claimed (should be all zeros)
        # Trinkets are at indices 0-19 (4 trinkets * 5 values each)
        # First trinket should be [0, 0, 0, 0, 0] because it's claimed
        np.testing.assert_array_equal(result_t2[0:5], [0, 0, 0, 0, 0])  # Claimed trinket
        
        # Check gems left decreased (index 20)
        self.assertEqual(float(result_t2[20]), 22)  # Was 25, then 23, now 22
        self.assertLess(float(result_t2[20]), float(result_t0[20]))
        
        # Check tiebreak leader changed (index 23)
        # Relative to player 0, player 1 has relative index 1
        self.assertEqual(float(result_t2[23]), 1)  # New leader is player 1 (relative index 1)
        self.assertEqual(float(result_t0[23]), 0)  # Original leader was player 0 (relative index 0)
        
        # Check action counts decreased (indices 24-29)
        self.assertEqual(float(result_t2[25]), 4)  # AUCTION_2 decreased from 5 to 4
        self.assertEqual(float(result_t2[28]), 1)  # INVESTMENT_5 decreased from 2 to 1

    def test_player_order_map_persistence(self):
        """Test that player_order_map is set once and persists across observations."""
        # Create a model
        model = AlphaGem()
        
        # First observation
        context1 = interface.TurnContext(
            turn_index=0,
            action=interface.Action(id="A1", kind=interface.ActionType.AUCTION_1),
            upcoming_gems=(),
            biddable_pile_count=20,
            tiebreak_leader_id=0,
            seating_order=(2, 0, 1),  # Player 2 is first in seating
        )
        
        private_state1 = interface.PlayerPrivateState(
            player_id=2,
            info_cards_unrevealed=(),
            info_cards_revealed=(),
        )
        
        player2 = interface.PlayerPublicState(
            player_id=2,
            name="Player2",
            cash=50,
            gems_owned=(),
            loans=(),
            investments=(),
            revealed_info=(),
            unrevealed_info_count=5,
            trinket_points=0,
        )
        
        player0 = interface.PlayerPublicState(
            player_id=0,
            name="Player0",
            cash=50,
            gems_owned=(),
            loans=(),
            investments=(),
            revealed_info=(),
            unrevealed_info_count=5,
            trinket_points=0,
        )
        
        player1 = interface.PlayerPublicState(
            player_id=1,
            name="Player1",
            cash=50,
            gems_owned=(),
            loans=(),
            investments=(),
            revealed_info=(),
            unrevealed_info_count=5,
            trinket_points=0,
        )
        
        public_state1 = interface.GamePublicState(
            num_players=3,
            players=(player2, player0, player1),
            trinkets=(),
            value_chart=interface.ValueChart(mapping=[0, 1, 2]),
            action_discard=(),
            past_auctions=(),
            action_counts_remaining=None,
        )
        
        observation1 = interface.GameObservation(
            public=public_state1,
            private=private_state1,
            context=context1,
            me=player2,
        )
        
        # First call - should create player_order_map
        model.encode_input(observation1)
        
        # Verify map was created with correct relative positions
        # Player 2 is at index 0 in seating, so relative to self (2), it's 0
        # Player 0 is at index 1 in seating, so relative to self (2), it's 1
        # Player 1 is at index 2 in seating, so relative to self (2), it's 2
        self.assertIsNotNone(model.player_order_map)
        self.assertEqual(model.player_order_map[2], 0)  # Self
        self.assertEqual(model.player_order_map[0], 1)  # Next clockwise
        self.assertEqual(model.player_order_map[1], 2)  # After that
        
        # Store the original map
        original_map = model.player_order_map.copy()
        
        # Second observation with different seating order (shouldn't matter)
        context2 = interface.TurnContext(
            turn_index=1,
            action=interface.Action(id="A2", kind=interface.ActionType.AUCTION_1),
            upcoming_gems=(),
            biddable_pile_count=19,
            tiebreak_leader_id=1,
            seating_order=(2, 0, 1),  # Same seating order
        )
        
        observation2 = interface.GameObservation(
            public=public_state1,
            private=private_state1,
            context=context2,
            me=player2,
        )
        
        # Second call - should NOT recreate player_order_map
        model.encode_input(observation2)
        
        # Verify map is the same (not recreated)
        self.assertEqual(model.player_order_map, original_map)
        self.assertEqual(model.player_order_map[2], 0)
        self.assertEqual(model.player_order_map[0], 1)
        self.assertEqual(model.player_order_map[1], 2)

    def test_tiebreaker_leader_id_persistence(self):
        """Test that tiebreaker leader ID relative position is correctly calculated and persists."""
        # Create a model where the model player is NOT player 0
        model = AlphaGem()
        
        # Create players
        player2 = interface.PlayerPublicState(
            player_id=2,
            name="Player2",
            cash=50,
            gems_owned=(),
            loans=(),
            investments=(),
            revealed_info=(),
            unrevealed_info_count=5,
            trinket_points=0,
        )
        
        player0 = interface.PlayerPublicState(
            player_id=0,
            name="Player0",
            cash=50,
            gems_owned=(),
            loans=(),
            investments=(),
            revealed_info=(),
            unrevealed_info_count=5,
            trinket_points=0,
        )
        
        player1 = interface.PlayerPublicState(
            player_id=1,
            name="Player1",
            cash=50,
            gems_owned=(),
            loans=(),
            investments=(),
            revealed_info=(),
            unrevealed_info_count=5,
            trinket_points=0,
        )
        
        value_chart = interface.ValueChart(mapping=[0, 1, 2])
        
        # First observation - model is player 2, leader is player 0
        # Seating order: (2, 0, 1) means: player 2 is first, then 0, then 1 clockwise
        public_state1 = interface.GamePublicState(
            num_players=3,
            players=(player2, player0, player1),
            trinkets=(),
            value_chart=value_chart,
            action_discard=(),
            past_auctions=(),
            action_counts_remaining=None,
        )
        
        private_state1 = interface.PlayerPrivateState(
            player_id=2,
            info_cards_unrevealed=(),
            info_cards_revealed=(),
        )
        
        context1 = interface.TurnContext(
            turn_index=0,
            action=interface.Action(id="A1", kind=interface.ActionType.AUCTION_1),
            upcoming_gems=(),
            biddable_pile_count=20,
            tiebreak_leader_id=0,  # Player 0 is leader
            seating_order=(2, 0, 1),
        )
        
        observation1 = interface.GameObservation(
            public=public_state1,
            private=private_state1,
            context=context1,
            me=player2,
        )
        
        result1 = model.encode_input(observation1)
        
        # Verify player_order_map: player 2 (self) = 0, player 0 = 1, player 1 = 2
        self.assertEqual(model.player_order_map[2], 0)
        self.assertEqual(model.player_order_map[0], 1)
        self.assertEqual(model.player_order_map[1], 2)
        
        # Verify tiebreaker leader encoding: player 0 has relative index 1
        self.assertEqual(float(result1[23]), 1)  # Leader is player 0, relative to self (2) is 1
        
        # Second observation - leader changes to player 1
        context2 = interface.TurnContext(
            turn_index=1,
            action=interface.Action(id="A2", kind=interface.ActionType.AUCTION_1),
            upcoming_gems=(),
            biddable_pile_count=19,
            tiebreak_leader_id=1,  # Player 1 is now leader
            seating_order=(2, 0, 1),  # Same seating order
        )
        
        observation2 = interface.GameObservation(
            public=public_state1,
            private=private_state1,
            context=context2,
            me=player2,
        )
        
        result2 = model.encode_input(observation2)
        
        # Verify player_order_map still persists (not recreated)
        self.assertEqual(model.player_order_map[2], 0)
        self.assertEqual(model.player_order_map[0], 1)
        self.assertEqual(model.player_order_map[1], 2)
        
        # Verify tiebreaker leader encoding: player 1 has relative index 2
        self.assertEqual(float(result2[23]), 2)  # Leader is player 1, relative to self (2) is 2
        
        # Third observation - leader changes to player 2 (self)
        context3 = interface.TurnContext(
            turn_index=2,
            action=interface.Action(id="A3", kind=interface.ActionType.AUCTION_1),
            upcoming_gems=(),
            biddable_pile_count=18,
            tiebreak_leader_id=2,  # Player 2 (self) is now leader
            seating_order=(2, 0, 1),  # Same seating order
        )
        
        observation3 = interface.GameObservation(
            public=public_state1,
            private=private_state1,
            context=context3,
            me=player2,
        )
        
        result3 = model.encode_input(observation3)
        
        # Verify player_order_map still persists
        self.assertEqual(model.player_order_map[2], 0)
        self.assertEqual(model.player_order_map[0], 1)
        self.assertEqual(model.player_order_map[1], 2)
        
        # Verify tiebreaker leader encoding: player 2 (self) has relative index 0
        self.assertEqual(float(result3[23]), 0)  # Leader is player 2 (self), relative index is 0
        
        # Verify all three observations have different leader encodings
        self.assertNotEqual(float(result1[23]), float(result2[23]))
        self.assertNotEqual(float(result2[23]), float(result3[23]))
        self.assertNotEqual(float(result1[23]), float(result3[23]))
        
        # Verify the relative positions are correct throughout
        # result1: leader 0 -> relative 1
        # result2: leader 1 -> relative 2
        # result3: leader 2 -> relative 0
        self.assertEqual(float(result1[23]), 1)
        self.assertEqual(float(result2[23]), 2)
        self.assertEqual(float(result3[23]), 0)


if __name__ == '__main__':
    unittest.main()

