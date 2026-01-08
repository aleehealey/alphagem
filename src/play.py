"""
Interactive game play script - Play against a list of bots (heuristic or model bots).

Usage:
    python self_play.py --bots <bot1> <bot2> ... [--seed SEED]

Bot types:
    - Model bots: model:<path> (e.g., model:src/models/model_final_1767150966.pth)
    - HeuristicBot: heuristic:<percentage> (e.g., heuristic:30, heuristic:40)
    - ValueTraderBot: valuetrader:<risk> (e.g., valuetrader:30, valuetrader:90)
    - ValueHeuristicBot: valueheuristic:<risk> (e.g., valueheuristic:30, valueheuristic:90)
    - AlwaysPass: alwayspass
    - RandomBid: randombid
    - GreedyTrinket: greedytrinket

Examples:
    python self_play.py --bots model:src/models/model_final_1767150966.pth heuristic:30
    python self_play.py --bots heuristic:30 heuristic:40 valuetrader:90
    python self_play.py --bots model:src/models/model_final_1767150966.pth heuristic:30 valueheuristic:90 --seed 42
"""

import sys
import argparse
import torch
from typing import List, Dict, Optional
from app.competition.interface import *
from app.competition.game import PocketRocketsEngine, EngineConfig
from app.alphagem.alpha_gem import AlphaGem
from app.bots.alpha_gem_bot import AlphaGemBot
from app.bots.human_bot import HumanBot
from app.bots.heuristic_bot import HeuristicBot
from app.bots.value_trader_bot import ValueTraderBot
from app.bots.value_heuristic_bot import ValueHeuristicBot
from app.bots.always_pass_bot import AlwaysPassBot
from app.bots.random_bid_bot import RandomBidBot
from app.bots.greedy_trinket_bot import GreedyTrinketBot
from app.competition.simulator import default_value_chart, default_trinkets


def format_suit_name(suit: Suit) -> str:
    """Format suit name for display."""
    return suit.name.capitalize()


def display_game_state(obs: GameObservation, human_player_id: int):
    """Display the current game state in a readable format."""
    public = obs.public
    context = obs.context
    
    print("\n" + "="*80)
    print("GAME STATE")
    print("="*80)
    
    # Current action
    action = context.action
    print(f"\n📋 Current Action: {action.kind.value}")
    
    # Upcoming gems
    upcoming = context.upcoming_gems
    if upcoming:
        gem_str = ", ".join([format_suit_name(g.suit) for g in upcoming])
        print(f"💎 Upcoming Gems: {gem_str}")
    else:
        print("💎 Upcoming Gems: None")
    
    # Value chart info
    print(f"\n💰 Value Chart (per gem based on info cards in that suit):")
    value_chart = public.value_chart
    for i, val in enumerate(value_chart.mapping):
        if i < len(value_chart.mapping):
            print(f"   {i} info cards → ${val} per gem")
    
    # Trinkets
    print(f"\n🏆 Trinkets:")
    for trinket_state in public.trinkets:
        if trinket_state.claimed_by is not None:
            claimed_by_name = next(
                (p.name for p in public.players if p.player_id == trinket_state.claimed_by),
                f"Player {trinket_state.claimed_by}"
            )
            print(f"   ✓ {trinket_state.objective.display_text} - CLAIMED by {claimed_by_name}")
        else:
            print(f"   ○ {trinket_state.objective.display_text} - Available")
    
    # Player states
    print(f"\n👥 Players:")
    for player in public.players:
        is_human = player.player_id == human_player_id
        marker = "👤" if is_human else "🤖"
        
        # Count gems by suit
        gem_counts = count_gems(player.gems_owned)
        gem_str = ", ".join([f"{suit.name}: {count}" for suit, count in gem_counts.items() if count > 0])
        if not gem_str:
            gem_str = "None"
        
        print(f"\n{marker} {player.name} (Player {player.player_id}):")
        print(f"   Cash: ${player.cash}")
        print(f"   Gems: {gem_str} ({len(player.gems_owned)} total)")
        print(f"   Loans: {len(player.loans)}")
        print(f"   Investments: {len(player.investments)}")
        print(f"   Trinket Points: {player.trinket_points}")
        print(f"   Revealed Info: {len(player.revealed_info)} cards")
        print(f"   Unrevealed Info: {player.unrevealed_info_count} cards")
        
        # Show human's private info
        if is_human:
            private = obs.private
            unrevealed_suits = [format_suit_name(c.suit) for c in private.info_cards_unrevealed]
            revealed_suits = [format_suit_name(c.suit) for c in private.info_cards_revealed]
            if unrevealed_suits:
                print(f"   📋 Your Unrevealed Info: {', '.join(unrevealed_suits)}")
            if revealed_suits:
                print(f"   📋 Your Revealed Info: {', '.join(revealed_suits)}")
    
    # Past auctions (last 3)
    if public.past_auctions:
        print(f"\n📜 Recent Auctions (last 3):")
        for result in public.past_auctions[-3:]:
            winner_name = next(
                (p.name for p in public.players if p.player_id == result.winner_id),
                f"Player {result.winner_id}"
            ) if result.winner_id >= 0 else "No winner"
            print(f"   {result.action.kind.value}: {winner_name} won with ${result.winning_bid}")
            if result.auctioned_gems:
                gem_str = ", ".join([format_suit_name(g.suit) for g in result.auctioned_gems])
                print(f"      Gems: {gem_str}")
    
    print("="*80)


def display_auction_result(result: AuctionResult, public: GamePublicState):
    """Display the result of an auction."""
    print("\n" + "="*80)
    print("AUCTION RESULT")
    print("="*80)
    
    winner_name = next(
        (p.name for p in public.players if p.player_id == result.winner_id),
        f"Player {result.winner_id}"
    ) if result.winner_id >= 0 else "No winner (all passed)"
    
    print(f"\n🏆 Winner: {winner_name}")
    print(f"💰 Winning Bid: ${result.winning_bid}")
    
    if result.bids:
        print(f"\n📊 All Bids:")
        for i, bid in enumerate(result.bids):
            player_name = next(
                (p.name for p in public.players if p.player_id == i),
                f"Player {i}"
            )
            print(f"   {player_name}: ${bid}")
    
    if result.auctioned_gems:
        gem_str = ", ".join([format_suit_name(g.suit) for g in result.auctioned_gems])
        print(f"\n💎 Auctioned Gems: {gem_str}")
    
    if result.trinkets_claimed:
        print(f"\n🏆 Trinkets Claimed: {result.trinkets_claimed}")
    
    print("="*80)


def play_interactive_game(engine: PocketRocketsEngine, human_player_id: int) -> Dict[str, object]:
    """Play the game interactively by replicating the engine's play loop with user interaction."""
    turn_index = 0
    engine._refill_upcoming()
    
    started = False
    
    while True:
        # End condition
        if len(engine._state.upcoming) == 0 and len(engine._state.gem_draw_pile) == 0:
            break
        
        # Reshuffle if needed
        if len(engine._state.action_draw_pile) == 0:
            if len(engine._state.action_discard) == 0:
                break
            engine._state.action_draw_pile = engine._state.action_discard[:]
            engine._state.action_discard.clear()
            engine._rng.shuffle(engine._state.action_draw_pile)
        
        action = engine._state.action_draw_pile.pop()
        engine._state.action_discard.append(action)
        
        if engine._state.action_counts_remaining is not None:
            engine._state.action_counts_remaining[action.kind] = max(
                0, engine._state.action_counts_remaining.get(action.kind, 0) - 1
            )
        
        # Skip if not enough gems
        if action.kind == ActionType.AUCTION_1 and len(engine._state.upcoming) < 1:
            continue
        if action.kind == ActionType.AUCTION_2 and engine._cfg.skip_auction2_if_insufficient_gems and len(engine._state.upcoming) < 2:
            continue
        
        # Call on_game_start once
        if not started:
            for pid, bot in enumerate(engine._bots):
                try:
                    bot.on_game_start(engine._build_observation(pid, turn_index, action))
                except Exception:
                    pass
            started = True
        
        # Display game state and collect bids interactively
        human_obs = engine._build_observation(human_player_id, turn_index, action)
        display_game_state(human_obs, human_player_id)
        
        bids: List[int] = [0] * engine._num_players
        for pid, bot in enumerate(engine._bots):
            obs = engine._build_observation(pid, turn_index, action)
            
            if pid == human_player_id:
                # Human player
                out = bot.get_bid(obs)
                amt = int(out.bid_amount)
            else:
                # Bot player
                print(f"\n🤖 {bot.bot_name} is thinking...")
                try:
                    out = bot.get_bid(obs)
                    amt = int(out.bid_amount)
                    print(f"   → {bot.bot_name} bids: ${amt}")
                except Exception as e:
                    print(f"   ❌ {bot.bot_name} error: {e}")
                    amt = 0
            
            # Enforce legality
            if amt < 0 or amt > engine._players[pid].cash:
                amt = 0
            
            bids[pid] = amt
        
        winner_id, winning_bid = engine._resolve_winner(bids)
        
        # Apply effects
        auctioned_gems: List[Card] = []
        trinkets_claimed: Optional[str] = None
        
        new_leader = engine._state.tiebreak_leader_id
        if winner_id >= 0:
            new_leader = winner_id
        
        if winner_id >= 0:
            engine._players[winner_id].cash -= winning_bid
        
        if action.kind == ActionType.AUCTION_1:
            if engine._state.upcoming:
                gem = engine._state.upcoming.pop(0)
                auctioned_gems.append(gem)
                if winner_id >= 0:
                    engine._players[winner_id].gems_owned.append(gem)
            engine._refill_upcoming()
        
        elif action.kind == ActionType.AUCTION_2:
            gems = []
            if len(engine._state.upcoming) >= 1:
                gems.append(engine._state.upcoming.pop(0))
            if len(engine._state.upcoming) >= 1:
                gems.append(engine._state.upcoming.pop(0))
            auctioned_gems.extend(gems)
            if winner_id >= 0:
                engine._players[winner_id].gems_owned.extend(gems)
            engine._refill_upcoming()
        
        elif action.kind in (ActionType.LOAN_10, ActionType.LOAN_20):
            if winner_id >= 0:
                principal = 10 if action.kind == ActionType.LOAN_10 else 20
                engine._players[winner_id].cash += principal
                engine._players[winner_id].loans.append(
                    LoanPosition(id=f"L{turn_index}", principal=principal, winning_bid=winning_bid)
                )
        
        elif action.kind in (ActionType.INVESTMENT_5, ActionType.INVESTMENT_10):
            if winner_id >= 0:
                payout = 5 if action.kind == ActionType.INVESTMENT_5 else 10
                engine._players[winner_id].investments.append(
                    InvestmentPosition(id=f"I{turn_index}", payout=payout, locked=winning_bid)
                )
        
        # Trinkets
        claimed = engine._award_trinkets_if_any(winner_id)
        if claimed:
            trinkets_claimed = ",".join(claimed)
        
        # Build result
        result = AuctionResult(
            turn_index=turn_index,
            action=action,
            winner_id=winner_id,
            winning_bid=winning_bid,
            auctioned_gems=tuple(auctioned_gems),
            new_tiebreak_leader_id=new_leader,
            trinkets_claimed=trinkets_claimed,
            bids=tuple(bids),
        )
        
        engine._state.tiebreak_leader_id = new_leader
        engine._state.past_auctions.append(result)
        
        # Show auction result
        public_state = engine._build_public_state()
        display_auction_result(result, public_state)
        
        # Call on_auction_resolved
        for pid, bot in enumerate(engine._bots):
            try:
                bot.on_auction_resolved(engine._build_observation(pid, turn_index, action), result)
            except Exception:
                pass
        
        # Reveal-on-win
        if winner_id >= 0:
            p = engine._players[winner_id]
            if p.unrevealed_info:
                bot = engine._bots[winner_id]
                obs = engine._build_observation(winner_id, turn_index, action)
                
                if winner_id == human_player_id:
                    chosen_id = bot.choose_info_to_reveal(obs, result)
                else:
                    print(f"\n🤖 {bot.bot_name} won! Choosing info card to reveal...")
                    try:
                        chosen_id = str(bot.choose_info_to_reveal(obs, result))
                        chosen_card = next((c for c in p.unrevealed_info if c.id == chosen_id), None)
                        if chosen_card:
                            print(f"   → {bot.bot_name} reveals: {chosen_card.suit.name} (ID: {chosen_id})")
                    except Exception as e:
                        print(f"   ❌ {bot.bot_name} error: {e}")
                        chosen_id = ""
                
                chosen_idx = next((i for i, c in enumerate(p.unrevealed_info) if c.id == chosen_id), None)
                if chosen_idx is None:
                    chosen_idx = 0
                
                card = p.unrevealed_info.pop(chosen_idx)
                p.revealed_info.append(card)
        
        turn_index += 1
    
    # End game hooks
    final_public = engine._build_public_state()
    for pid, bot in enumerate(engine._bots):
        try:
            bot.on_game_end(GameObservation(
                public=final_public,
                private=engine._build_private_state(pid),
                context=TurnContext(
                    turn_index=turn_index,
                    action=Action(id="END", kind=ActionType.AUCTION_1),
                    upcoming_gems=tuple(engine._state.upcoming),
                    biddable_pile_count=len(engine._state.gem_draw_pile),
                    tiebreak_leader_id=engine._state.tiebreak_leader_id,
                    seating_order=engine._state.seating_order,
                ),
                me=next(p for p in final_public.players if p.player_id == pid),
            ))
        except Exception:
            pass
    
    return engine._finalize()


def load_model(model_path: str, num_players: int = 3) -> AlphaGem:
    """Load an AlphaGem model from a file."""
    model = AlphaGem(num_players=num_players)
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✓ Loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)


def create_bot_from_config(config: str, num_players: int = 3) -> PocketRocketsBot:
    """
    Create a bot from a configuration string.
    
    Supported formats:
    - model:<path> - Load an AlphaGem model
    - heuristic:<percentage> - HeuristicBot with bid_percentage
    - valuetrader:<risk> - ValueTraderBot with risk (0-100, converted to 0.0-1.0)
    - valueheuristic:<risk> - ValueHeuristicBot with risk (0-100, converted to 0.0-1.0)
    - alwayspass - AlwaysPassBot
    - randombid - RandomBidBot
    - greedytrinket - GreedyTrinketBot
    """
    config = config.strip().lower()
    
    if config.startswith("model:"):
        model_path = config[6:].strip()
        if not model_path:
            raise ValueError("Model path cannot be empty")
        model = load_model(model_path, num_players=num_players)
        return AlphaGemBot(model, name=f"AlphaGem({model_path.split('/')[-1]})")
    
    elif config.startswith("heuristic:"):
        try:
            percentage = float(config[10:].strip())
            if not (0 <= percentage <= 100):
                raise ValueError("Percentage must be between 0 and 100")
            return HeuristicBot(bid_percentage=percentage / 100.0, name=f"HeuristicBot{int(percentage)}")
        except ValueError as e:
            raise ValueError(f"Invalid heuristic percentage: {e}")
    
    elif config.startswith("valuetrader:"):
        try:
            risk = float(config[12:].strip())
            if not (0 <= risk <= 100):
                raise ValueError("Risk must be between 0 and 100")
            return ValueTraderBot(risk=risk / 100.0)
        except ValueError as e:
            raise ValueError(f"Invalid valuetrader risk: {e}")
    
    elif config.startswith("valueheuristic:"):
        try:
            risk = float(config[15:].strip())
            if not (0 <= risk <= 100):
                raise ValueError("Risk must be between 0 and 100")
            return ValueHeuristicBot(risk=risk / 100.0)
        except ValueError as e:
            raise ValueError(f"Invalid valueheuristic risk: {e}")
    
    elif config == "alwayspass":
        return AlwaysPassBot()
    
    elif config == "randombid":
        return RandomBidBot()
    
    elif config == "greedytrinket":
        return GreedyTrinketBot()
    
    else:
        raise ValueError(f"Unknown bot configuration: {config}. Use format: model:<path>, heuristic:<%>, valuetrader:<risk>, valueheuristic:<risk>, alwayspass, randombid, or greedytrinket")


def main():
    parser = argparse.ArgumentParser(
        description="Play PocketRockets against a list of bots (heuristic or model bots)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python self_play.py --bots model:src/models/model_final_1767150966.pth heuristic:30
    python self_play.py --bots heuristic:30 heuristic:40 valuetrader:90
    python self_play.py --bots model:src/models/model_final_1767150966.pth heuristic:30 valueheuristic:90 --seed 42
            """
    )
    parser.add_argument("--bots", nargs="+", required=True,
                        help="List of bot configurations. See help for formats.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the game")
    
    args = parser.parse_args()
    
    # Validate bot configurations and create bots
    bots = [HumanBot()]  # Human is always player 0
    
    num_players = len(args.bots) + 1  # +1 for human
    
    if num_players < 3:
        print("⚠️  Warning: Minimum 3 players required. Adding AlwaysPass bots...")
        while num_players < 3:
            bots.append(AlwaysPassBot())
            num_players += 1
    
    if num_players > 5:
        print(f"⚠️  Warning: Maximum 5 players. Limiting to first 4 bots.")
        args.bots = args.bots[:4]
        num_players = 5      
    
    # Create opponent bots
    print("\n🤖 Creating opponent bots...")
    for i, bot_config in enumerate(args.bots):
        try:
            bot = create_bot_from_config(bot_config, num_players=num_players)
            bots.append(bot)
            print(f"   ✓ Bot {i+1}: {bot.bot_name} ({bot_config})")
        except Exception as e:
            print(f"   ❌ Error creating bot from '{bot_config}': {e}")
            sys.exit(1)
    
    # Ensure we have at least 3 players total
    while len(bots) < 3:
        bots.append(AlwaysPassBot())
    
    # Limit to 5 players max
    bots = bots[:5]
    num_players = len(bots)
    
    # Set up game
    seed = args.seed if args.seed is not None else None
    if seed is None:
        import random
        seed = random.randint(0, 1000000)
    
    config = EngineConfig(seed=seed)
    value_chart = default_value_chart()
    trinkets = default_trinkets(seed=seed)
    
    print(f"\n{'='*80}")
    print("POCKETROCKETS - HUMAN vs BOTS")
    print(f"{'='*80}")
    print(f"Players: {num_players}")
    print(f"   👤 You (Human)")
    for i, bot in enumerate(bots[1:], 1):
        print(f"   🤖 {bot.bot_name}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")
    
    # Create engine
    engine = PocketRocketsEngine(
        bots=bots,
        config=config,
        value_chart=value_chart,
        trinkets=trinkets,
        bot_names=[b.bot_name for b in bots],
    )
    
    # Human is always player 0
    # Play game
    try:
        result = play_interactive_game(engine, human_player_id=0)
        
        # Show final scores
        print("\n" + "="*80)
        print("FINAL SCORES")
        print("="*80)
        for player_id, name, score in result["final_scores"]:
            marker = "👤" if player_id == 0 else "🤖"
            print(f"{marker} {name}: ${score}")
        print("="*80)
        
        winner_id = result["winner_id"]
        winner_name = next(
            (name for pid, name, _ in result["final_scores"] if pid == winner_id),
            "Unknown"
        )
        print(f"\n🏆 Winner: {winner_name}!")
        
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during gameplay: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

