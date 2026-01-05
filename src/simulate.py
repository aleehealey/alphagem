from app.competition.simulator import run_pocketrocks_simulation, print_pocketrocks_report, BotEntry
from time import time
import json
import sys
from io import StringIO

# -----------------------------
# Convenience: factories for the simulator
# -----------------------------

from app.bots.always_pass_bot import AlwaysPassBot
from app.bots.random_bid_bot import RandomBidBot
from app.bots.greedy_trinket_bot import GreedyTrinketBot
from app.bots.value_trader_bot import ValueTraderBot
from app.bots.heuristic_bot import HeuristicBot
from app.bots.value_heuristic_bot import ValueHeuristicBot
from app.bots.alpha_gem_bot import AlphaGemBot
from app.alphagem.alpha_gem import AlphaGem
import torch

model = AlphaGem(num_players=3)
model.load_state_dict(torch.load("./src/models/model_final_1767150966.pth"))
model.eval()

def make_bot_entries_for_sim() -> list["BotEntry"]:
    return [
        BotEntry("AlwaysPass", lambda: AlwaysPassBot()),
        BotEntry("RandomBid", lambda: RandomBidBot()),
        BotEntry("GreedyTrinket", lambda: GreedyTrinketBot()),
        BotEntry("ValueTraderRisk10", lambda: ValueTraderBot(risk=0.10)),
        BotEntry("ValueTraderRisk20", lambda: ValueTraderBot(risk=0.20)),
        BotEntry("ValueTraderRisk30", lambda: ValueTraderBot(risk=0.30)),
        BotEntry("ValueTraderRisk40", lambda: ValueTraderBot(risk=0.40)),
        BotEntry("ValueTraderRisk50", lambda: ValueTraderBot(risk=0.50)),
        BotEntry("ValueTraderRisk70", lambda: ValueTraderBot(risk=0.70)),
        BotEntry("HeuristicBot30", lambda: HeuristicBot(bid_percentage=0.30)),
        BotEntry("HeuristicBot20", lambda: HeuristicBot(bid_percentage=0.20)),
        BotEntry("HeuristicBot10", lambda: HeuristicBot(bid_percentage=0.10)),
        BotEntry("HeuristicBot40", lambda: HeuristicBot(bid_percentage=0.40)),
        BotEntry("HeuristicBot50", lambda: HeuristicBot(bid_percentage=0.50)),
        BotEntry("HeuristicBot60", lambda: HeuristicBot(bid_percentage=0.60)),
        BotEntry("HeuristicBot70", lambda: HeuristicBot(bid_percentage=0.70)),
        BotEntry("ValueHeuristicRisk30", lambda: ValueHeuristicBot(risk=0.3)),
        BotEntry("ValueHeuristicRisk40", lambda: ValueHeuristicBot(risk=0.4)),
        BotEntry("ValueHeuristicRisk50", lambda: ValueHeuristicBot(risk=0.5)),
        BotEntry("ValueHeuristicRisk60", lambda: ValueHeuristicBot(risk=0.6)),
        BotEntry("ValueHeuristicRisk70", lambda: ValueHeuristicBot(risk=0.7)),
        BotEntry("ValueHeuristicRisk80", lambda: ValueHeuristicBot(risk=0.8)),
        BotEntry("ValueHeuristicRisk90", lambda: ValueHeuristicBot(risk=0.9)),
        # BotEntry("AlphaGem", lambda: AlphaGemBot(model=model)),
    ]

def run_simulations_with_logging(entries, n_games=1000, log_file=None):
    """
    Run simulations for 3, 4, and 5 player games and optionally log to file.
    
    Args:
        entries: List of BotEntry objects
        n_games: Number of games to run per configuration
        log_file: Optional path to file where logs will be written. If None, no logging.
    """
    log_output = []
    
    # Helper to capture print output
    def capture_print(func, *args, **kwargs):
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        try:
            func(*args, **kwargs)
            output = captured.getvalue()
            return output
        finally:
            sys.stdout = old_stdout
    
    # Run 3 player games
    log_output.append("=" * 80)
    log_output.append("3 PLAYER REPORT")
    log_output.append("=" * 80)
    print("3 player report")
    res_3 = run_pocketrocks_simulation(entries, n_games=n_games, players_per_game=3)
    report_3 = capture_print(print_pocketrocks_report, res_3)
    print(report_3)
    log_output.append(report_3)
    log_output.append("\n")
    
    # Add detailed game logs for 3 player
    log_output.append("-" * 80)
    log_output.append("3 PLAYER GAME LOGS")
    log_output.append("-" * 80)
    for game_log in res_3.game_logs:
        # Convert tuples to lists for JSON serialization
        serializable_log = {
            "game_index": game_log["game_index"],
            "seed": game_log["seed"],
            "seating": game_log["seating"],
            "final_scores": [[pid, name, score] for pid, name, score in game_log["final_scores"]],
            "winner": game_log["winner"]
        }
        log_output.append(json.dumps(serializable_log, indent=2))
    log_output.append("\n")
    
    # Run 4 player games
    log_output.append("=" * 80)
    log_output.append("4 PLAYER REPORT")
    log_output.append("=" * 80)
    print("4 player report")
    res_4 = run_pocketrocks_simulation(entries, n_games=n_games, players_per_game=4)
    report_4 = capture_print(print_pocketrocks_report, res_4)
    print(report_4)
    log_output.append(report_4)
    log_output.append("\n")
    
    # Add detailed game logs for 4 player
    log_output.append("-" * 80)
    log_output.append("4 PLAYER GAME LOGS")
    log_output.append("-" * 80)
    for game_log in res_4.game_logs:
        # Convert tuples to lists for JSON serialization
        serializable_log = {
            "game_index": game_log["game_index"],
            "seed": game_log["seed"],
            "seating": game_log["seating"],
            "final_scores": [[pid, name, score] for pid, name, score in game_log["final_scores"]],
            "winner": game_log["winner"]
        }
        log_output.append(json.dumps(serializable_log, indent=2))
    log_output.append("\n")
    
    # Run 5 player games
    log_output.append("=" * 80)
    log_output.append("5 PLAYER REPORT")
    log_output.append("=" * 80)
    print("5 player report")
    res_5 = run_pocketrocks_simulation(entries, n_games=n_games, players_per_game=5)
    report_5 = capture_print(print_pocketrocks_report, res_5)
    print(report_5)
    log_output.append(report_5)
    log_output.append("\n")
    
    # Add detailed game logs for 5 player
    log_output.append("-" * 80)
    log_output.append("5 PLAYER GAME LOGS")
    log_output.append("-" * 80)
    for game_log in res_5.game_logs:
        # Convert tuples to lists for JSON serialization
        serializable_log = {
            "game_index": game_log["game_index"],
            "seed": game_log["seed"],
            "seating": game_log["seating"],
            "final_scores": [[pid, name, score] for pid, name, score in game_log["final_scores"]],
            "winner": game_log["winner"]
        }
        log_output.append(json.dumps(serializable_log, indent=2))
    log_output.append("\n")
    
    # Write to file if specified
    if log_file:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_output))
        print(f"\nLogs written to: {log_file}")
    
    return res_3, res_4, res_5

entries = make_bot_entries_for_sim()
n_games = 3000

# Optional: specify log file path as command line argument or set to None
log_file = sys.argv[1] if len(sys.argv) > 1 else None

run_simulations_with_logging(entries, n_games=n_games, log_file=log_file)