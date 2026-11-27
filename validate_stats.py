"""Check the stats engine against simulation and make figures for the report.

    python validate_stats.py

Saves PNGs to stats_figures/. No camera or trained models needed.
"""
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from blackjack_stats import (RANKS, SUITS, Shoe, dealer_distribution, hand_value,
                             hit_analysis, stand_outcomes)

OUT = Path(__file__).resolve().parent / "stats_figures"
OUT.mkdir(exist_ok=True)
rng = random.Random(0)


def random_scenario(decks=1, seen_range=(4, 30)):
    """A shoe with some cards already seen. Returns (shoe, upcard, unseen cards)."""
    cards = [(r, s) for r in RANKS for s in SUITS] * decks
    rng.shuffle(cards)
    k = rng.randint(*seen_range)
    shoe = Shoe(decks)
    for rank, suit in cards[:k]:
        shoe.observe(rank, suit)
    return shoe, cards[0][0], [c[0] for c in cards[k:]]


def simulate_dealer(upcard, draws, hit_soft_17=False):
    hand = [upcard]
    for rank in draws:
        hand.append(rank)
        total, soft = hand_value(hand)
        if len(hand) == 2 and total == 21:
            return "blackjack"
        if total > 21:
            return "bust"
        if total >= 18 or (total == 17 and not (soft and hit_soft_17)):
            return total
    return None


def monte_carlo(upcard, unseen, trials, peeked=True):
    counts, valid = {}, 0
    for _ in range(trials):
        outcome = simulate_dealer(upcard, rng.sample(unseen, min(14, len(unseen))))
        if outcome is None:
            continue
        if peeked and upcard in ("A", "10", "J", "Q", "K") and outcome == "blackjack":
            continue  # the dealer peeked, so this hand would not be in play
        counts[outcome] = counts.get(outcome, 0) + 1
        valid += 1
    return {k: v / valid for k, v in counts.items()}


def simulate_known_hand(hand, draws, hit_soft_17=False):
    """Dealer already holds `hand` (e.g. hole card flipped); play it out."""
    hand, draws = list(hand), iter(draws)
    if len(hand) == 2 and hand_value(hand)[0] == 21:
        return "blackjack"
    while True:
        total, soft = hand_value(hand)
        if total > 21:
            return "bust"
        if total >= 18 or (total == 17 and not (soft and hit_soft_17)):
            return total
        nxt = next(draws, None)
        if nxt is None:
            return None
        hand.append(nxt)


def check_known_dealer_hand(n_scenarios=25, trials=20000):
    worst = 0.0
    for _ in range(n_scenarios):
        shoe, up, unseen = random_scenario(1, (6, 30))
        # the dealer's second card is one of the first few cards already seen
        cards = [(r, s) for r in RANKS for s in SUITS]
        second = shoe.log[1][0]
        hand = [shoe.log[0][0], second]
        exact = dealer_distribution(shoe, hand)
        counts, valid = {}, 0
        for _ in range(trials):
            out = simulate_known_hand(hand, rng.sample(unseen, min(14, len(unseen))))
            if out is None:
                continue
            counts[out] = counts.get(out, 0) + 1
            valid += 1
        sim = {k: v / valid for k, v in counts.items()}
        for key in list(range(17, 22)) + ["bust", "blackjack"]:
            worst = max(worst, abs(exact[key] - sim.get(key, 0.0)))
    print(f"known dealer hand vs simulation: largest difference = {worst:.4f}")
    assert worst < 0.02


def check_undo():
    shoe = Shoe(1)
    shoe.observe("K", "hearts"); shoe.observe("5", None)
    assert shoe.undo() == ("5", None) and shoe.undo() == ("K", "hearts") and shoe.undo() is None
    assert shoe.total_remaining == 52 and shoe.running_count() == 0 and shoe.remaining("K") == 4
    print("undo check passed")


def check_basics():
    shoe = Shoe(1)
    assert abs(shoe.p_rank("7") - 1 / 13) < 1e-12 and abs(shoe.p_ten() - 16 / 52) < 1e-12
    assert shoe.running_count() == 0
    assert all(shoe.observe("A", s) for s in SUITS)
    assert shoe.observe("A", "hearts") is False and shoe.observe("Back") is False
    assert shoe.running_count() == -4
    full = Shoe(1)
    for r in RANKS:
        for s in SUITS:
            full.observe(r, s)
    assert full.total_remaining == 0 and full.running_count() == 0  # Hi-Lo is balanced
    big = Shoe(decks=200)  # ~infinite shoe: dealer showing 6 busts about 42% of the time
    bust6 = dealer_distribution(big, "6")["bust"]
    assert 0.41 < bust6 < 0.43, bust6
    print(f"basic checks passed (dealer 6, near-infinite shoe: bust = {bust6:.3f})")


def check_against_simulation(n_scenarios=40, trials=20000):
    worst, predicted, simulated = 0.0, [], []
    for _ in range(n_scenarios):
        shoe, up, unseen = random_scenario()
        exact = dealer_distribution(shoe, up)
        sim = monte_carlo(up, unseen, trials)
        for key in list(range(17, 22)) + ["bust", "blackjack"]:
            worst = max(worst, abs(exact[key] - sim.get(key, 0.0)))
        predicted.append(exact["bust"])
        simulated.append(sim.get("bust", 0.0))
    print(f"engine vs simulation: {n_scenarios} random shoe states x {trials} trials, "
          f"largest difference in any outcome = {worst:.4f} (noise floor ~0.01)")
    assert worst < 0.02

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(predicted, simulated, s=18)
    ax.plot([0, 0.6], [0, 0.6], "k--", lw=1)
    ax.set_xlabel("Engine: P(dealer busts)")
    ax.set_ylabel("Monte Carlo: P(dealer busts)")
    ax.set_title("Engine vs simulation")
    fig.tight_layout()
    fig.savefig(OUT / "engine_vs_simulation.png", dpi=150)


def plot_count_trajectory(decks=6):
    cards = [(r, s) for r in RANKS for s in SUITS] * decks
    rng.shuffle(cards)
    shoe = Shoe(decks)
    true_counts = []
    for rank, suit in cards[: int(len(cards) * 0.75)]:
        shoe.observe(rank, suit)
        true_counts.append(shoe.true_count())
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(shoe.count_trajectory(), label="Hi-Lo running count")
    ax.plot(true_counts, label="Hi-Lo true count")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Cards seen")
    ax.set_ylabel("Count")
    ax.set_title(f"Count over one {decks}-deck shoe (simulated)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "count_trajectory.png", dpi=150)


def plot_true_count_vs_tens(samples=1500):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for decks, color in ((1, "tab:blue"), (6, "tab:orange")):
        xs, ys = [], []
        for _ in range(samples):
            shoe, _, _ = random_scenario(decks, (10, int(52 * decks * 0.75)))
            xs.append(shoe.true_count())
            ys.append(shoe.p_ten())
        ax.scatter(xs, ys, s=5, alpha=0.3, color=color, label=f"{decks} deck(s)")
        slope, intercept = np.polyfit(xs, ys, 1)
        grid = np.linspace(min(xs), max(xs), 50)
        ax.plot(grid, slope * grid + intercept, color=color)
    ax.axhline(16 / 52, color="gray", ls=":", lw=1)
    ax.set_xlabel("Hi-Lo true count")
    ax.set_ylabel("P(next card is ten-value)")
    ax.set_title("Why counting works: high count -> more tens left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "true_count_vs_tens.png", dpi=150)


def time_the_engine():
    shoe, up, _ = random_scenario(1, (6, 10))
    hand = ["5", "7"]
    start = time.perf_counter()
    stand_outcomes(shoe, hand, up)
    hit_analysis(shoe, hand, up)
    print(f"stand + hit analysis, cold cache: {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    check_basics()
    check_undo()
    check_against_simulation()
    check_known_dealer_hand()
    plot_count_trajectory()
    plot_true_count_vs_tens()
    time_the_engine()
    print(f"figures saved to {OUT.resolve()}")
