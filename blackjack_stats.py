"""Blackjack statistics engine (no camera / CV dependencies).

Feed it the cards the vision system has CONFIRMED:

    shoe = Shoe(decks=1)
    shoe.observe("K", "hearts")      # suit may be None if the suit model wasn't sure

then ask for next-card probabilities, card counts, and hand-outcome estimates.

All probabilities are computed from the UNSEEN cards (full shoe minus every card
revealed so far). The dealer's face-down card counts as unseen until it is shown.

Known simplification: when the dealer peeks for blackjack (upcard A or ten), the
hole card is known not to complete it, which very slightly shifts the odds of the
player's next card. That tiny effect is ignored for the "hit" statistics.
"""
from collections import Counter

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["clubs", "diamonds", "hearts", "spades"]  # same names as the suit classifier
TEN_RANKS = ("10", "J", "Q", "K")

# Card-counting systems: rank -> tag. Hi-Lo and Omega II are balanced (true count
# is meaningful); KO is unbalanced (use the running count only).
COUNT_SYSTEMS = {
    "hi_lo": {"2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 0, "8": 0, "9": 0,
              "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1},
    "ko": {"2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 0, "9": 0,
           "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1},
    "omega_ii": {"2": 1, "3": 1, "4": 2, "5": 2, "6": 2, "7": 1, "8": 0, "9": -1,
                 "10": -2, "J": -2, "Q": -2, "K": -2, "A": 0},
}

MIN_DECKS_FOR_TRUE_COUNT = 0.25  # avoid huge true counts at the very end of a shoe


def value_index(rank):
    """Bucket used by the outcome maths: 0=A, 1..8 = 2..9, 9 = ten-value (10/J/Q/K)."""
    if rank == "A":
        return 0
    if rank in TEN_RANKS:
        return 9
    return int(rank) - 1


def _hard_and_ace(ranks):
    return sum(value_index(r) + 1 for r in ranks), "A" in ranks


def hand_value(ranks):
    """Best total and whether it is soft (an ace counted as 11)."""
    hard, ace = _hard_and_ace(ranks)
    if ace and hard + 10 <= 21:
        return hard + 10, True
    return hard, False


def is_blackjack(ranks):
    return len(ranks) == 2 and hand_value(ranks)[0] == 21


class Shoe:
    def __init__(self, decks=1):
        self.decks = decks
        self.reset()

    def reset(self):
        """Call when the shoe is reshuffled."""
        self.seen_rank = Counter()
        self.seen_card = Counter()         # (rank, suit) where the suit was known
        self.unknown_by_rank = Counter()   # seen, but suit not recognised
        self.log = []                      # [(rank, suit)] in the order seen

    # ---- bookkeeping -------------------------------------------------------
    def observe(self, rank, suit=None):
        """Record one confirmed card.

        Returns False (and records nothing) if the card is impossible, e.g. a
        fifth ace in a one-deck shoe, or something that isn't a rank (like
        "Back"). Impossible cards are almost always detector errors, so the
        caller can count them as failure cases.
        """
        if rank not in RANKS:
            return False
        if self.seen_rank[rank] >= 4 * self.decks:
            return False
        if suit not in SUITS:
            suit = None
        if suit is not None and self.seen_card[(rank, suit)] >= self.decks:
            return False

        self.seen_rank[rank] += 1
        if suit is None:
            self.unknown_by_rank[rank] += 1
        else:
            self.seen_card[(rank, suit)] += 1
        self.log.append((rank, suit))
        return True

    def undo(self):
        """Remove the most recently observed card (fixes a misread during a live
        demo). Returns (rank, suit), or None if nothing has been observed."""
        if not self.log:
            return None
        rank, suit = self.log.pop()
        self.seen_rank[rank] -= 1
        if suit is None:
            self.unknown_by_rank[rank] -= 1
        else:
            self.seen_card[(rank, suit)] -= 1
        return rank, suit

    @property
    def cards_seen(self):
        return len(self.log)

    @property
    def total_remaining(self):
        return 52 * self.decks - len(self.log)

    @property
    def decks_remaining(self):
        return self.total_remaining / 52

    @property
    def penetration(self):
        return len(self.log) / (52 * self.decks)

    def remaining(self, rank):
        return 4 * self.decks - self.seen_rank[rank]

    def value_counts(self):
        """Unseen cards per value bucket (A,2..9,ten) - input for the outcome maths."""
        counts = [0] * 10
        for rank in RANKS:
            counts[value_index(rank)] += self.remaining(rank)
        return counts

    # ---- next-card probabilities ------------------------------------------
    def p_rank(self, rank):
        n = self.total_remaining
        return self.remaining(rank) / n if n else 0.0

    def p_rank_distribution(self):
        return {rank: self.p_rank(rank) for rank in RANKS}

    def p_ten(self):
        n = self.total_remaining
        return sum(self.remaining(r) for r in TEN_RANKS) / n if n else 0.0

    def p_ace(self):
        return self.p_rank("A")

    # Suit statistics: cards whose suit wasn't recognised are assumed to have been
    # spread evenly over the suits (an expected-value approximation).
    def _unknown_total(self):
        return sum(self.unknown_by_rank.values())

    def remaining_suit(self, suit):
        seen = sum(c for (r, s), c in self.seen_card.items() if s == suit)
        return max(0.0, 13 * self.decks - seen - self._unknown_total() / 4)

    def p_suit(self, suit):
        n = self.total_remaining
        return self.remaining_suit(suit) / n if n else 0.0

    def p_card(self, rank, suit):
        n = self.total_remaining
        left = self.decks - self.seen_card[(rank, suit)] - self.unknown_by_rank[rank] / 4
        return max(0.0, left) / n if n else 0.0

    # ---- counting ----------------------------------------------------------
    def running_count(self, system="hi_lo"):
        tags = COUNT_SYSTEMS[system]
        return sum(tags[r] * c for r, c in self.seen_rank.items())

    def true_count(self, system="hi_lo"):
        return self.running_count(system) / max(self.decks_remaining, MIN_DECKS_FOR_TRUE_COUNT)

    def count_trajectory(self, system="hi_lo"):
        """Running count after each card seen, in order (for plotting)."""
        tags, total, out = COUNT_SYSTEMS[system], 0, []
        for rank, _ in self.log:
            total += tags[rank]
            out.append(total)
        return out


# ---- dealer outcome distribution (exact, without replacement) --------------
_DEALER_MEMO = {}


def _dealer_final(hard, ace, counts, hit_soft_17):
    """Probabilities [17, 18, 19, 20, 21, bust] for a dealer who holds `hard`
    (aces counted as 1) and draws from `counts` (unseen cards per value bucket).
    Memoised: the state is fully described by (hard, ace, counts)."""
    key = (hard, ace, counts, hit_soft_17)
    cached = _DEALER_MEMO.get(key)
    if cached is not None:
        return cached

    soft = ace and hard + 10 <= 21
    total = hard + 10 if soft else hard

    if total > 21:
        result = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    elif total >= 18 or (total == 17 and not (soft and hit_soft_17)):
        result = tuple(1.0 if i == total - 17 else 0.0 for i in range(5)) + (0.0,)
    else:
        n = sum(counts)
        if n == 0:  # degenerate: shoe exhausted, treat as standing on 17
            result = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            acc = [0.0] * 6
            for idx, c in enumerate(counts):
                if c == 0:
                    continue
                rest = counts[:idx] + (c - 1,) + counts[idx + 1:]
                sub = _dealer_final(hard + idx + 1, ace or idx == 0, rest, hit_soft_17)
                p = c / n
                for i in range(6):
                    acc[i] += p * sub[i]
            result = tuple(acc)

    if len(_DEALER_MEMO) > 400_000:
        _DEALER_MEMO.clear()
    _DEALER_MEMO[key] = result
    return result


def _dealer_distribution_counts(counts, upcard, hit_soft_17, peeked):
    counts = tuple(counts)
    n = sum(counts)
    if n == 0:
        raise ValueError("No unseen cards left to model.")

    up = value_index(upcard)
    acc, natural = [0.0] * 6, 0.0

    for idx, c in enumerate(counts):  # idx = the dealer's hole card
        if c == 0:
            continue
        p = c / n
        hard = (up + 1) + (idx + 1)
        ace = up == 0 or idx == 0
        if ace and hard == 11:  # ace + ten-value = natural blackjack
            natural += p
            continue
        rest = counts[:idx] + (c - 1,) + counts[idx + 1:]
        sub = _dealer_final(hard, ace, rest, hit_soft_17)
        for i in range(6):
            acc[i] += p * sub[i]

    # US rules: with an ace or ten showing the dealer has already peeked, so the
    # hand we are playing is one where the dealer does NOT have blackjack.
    if peeked and up in (0, 9) and natural < 1.0:
        scale = 1.0 / (1.0 - natural)
        acc = [v * scale for v in acc]
        natural = 0.0

    dist = {17 + i: acc[i] for i in range(5)}
    dist["bust"] = acc[5]
    dist["blackjack"] = natural
    return dist


def _as_list(dealer):
    return [dealer] if isinstance(dealer, str) else list(dealer)


def _dealer_dist(dealer, counts, hit_soft_17, peeked):
    """`dealer` is one face-up rank (hole card unknown) or the dealer's whole
    known hand (e.g. after the hole card is flipped and the dealer has drawn)."""
    ranks = _as_list(dealer)
    if len(ranks) == 1:
        return _dealer_distribution_counts(counts, ranks[0], hit_soft_17, peeked)
    if is_blackjack(ranks):
        dist = {t: 0.0 for t in range(17, 22)}
        dist.update({"bust": 0.0, "blackjack": 1.0})
        return dist
    hard, ace = _hard_and_ace(ranks)
    sub_dist = _dealer_final(hard, ace, tuple(counts), hit_soft_17)
    dist = {17 + i: sub_dist[i] for i in range(5)}
    dist["bust"], dist["blackjack"] = sub_dist[5], 0.0
    return dist


def dealer_distribution(shoe, dealer, hit_soft_17=False, peeked=True):
    """Chance the dealer ends on 17..21, busts, or has a natural, given the
    dealer's face-up rank (or full known hand) and the cards still unseen."""
    return _dealer_dist(dealer, shoe.value_counts(), hit_soft_17, peeked)


# ---- player outcomes --------------------------------------------------------
def _settle(total, dist, natural=False, blackjack_payout=1.5):
    """Win/push/lose probabilities and EV (per 1 unit bet) if the player stands."""
    if total > 21:
        return {"win": 0.0, "push": 0.0, "lose": 1.0, "ev": -1.0}
    dealer_bj = dist["blackjack"]
    if natural:
        return {"win": 1 - dealer_bj, "push": dealer_bj, "lose": 0.0,
                "ev": (1 - dealer_bj) * blackjack_payout}
    win = dist["bust"] + sum(dist[t] for t in range(17, 22) if t < total)
    push = dist[total] if 17 <= total <= 21 else 0.0
    lose = max(0.0, 1.0 - win - push)  # includes a dealer natural
    return {"win": win, "push": push, "lose": lose, "ev": win - lose}


def stand_outcomes(shoe, player_ranks, dealer_up, hit_soft_17=False, peeked=True):
    total, _ = hand_value(player_ranks)
    dist = dealer_distribution(shoe, dealer_up, hit_soft_17, peeked)
    return _settle(total, dist, natural=is_blackjack(player_ranks))


def hit_analysis(shoe, player_ranks, dealer_up, hit_soft_17=False, peeked=True):
    """Chance of busting on one more card, and the EV of 'hit exactly once, then
    stand'. The EV is a ONE-CARD LOOKAHEAD, not optimal play: it undervalues hitting
    low totals, where you would normally keep drawing."""
    counts = tuple(shoe.value_counts())
    n = sum(counts)
    hard, ace = _hard_and_ace(player_ranks)

    p_bust, ev = 0.0, 0.0
    for idx, c in enumerate(counts):
        if c == 0:
            continue
        p = c / n
        new_hard, new_ace = hard + idx + 1, ace or idx == 0
        total = new_hard + 10 if new_ace and new_hard + 10 <= 21 else new_hard
        if total > 21:
            p_bust += p
            ev -= p
            continue
        rest = counts[:idx] + (c - 1,) + counts[idx + 1:]
        dist = _dealer_dist(dealer_up, rest, hit_soft_17, peeked)
        ev += p * _settle(total, dist)["ev"]

    return {"p_bust": p_bust, "ev_hit_once": ev}


# ---- everything the display needs, in one dict ------------------------------
def build_snapshot(shoe, dealer_up=None, player_ranks=None, system="hi_lo",
                   hit_soft_17=False, peeked=True):
    """dealer_up: the dealer's face-up rank, or a list of every dealer rank showing
    (None/empty if none). player_ranks: the player's ranks."""
    snap = {
        "decks": shoe.decks,
        "cards_seen": shoe.cards_seen,
        "cards_total": 52 * shoe.decks,
        "decks_remaining": shoe.decks_remaining,
        "penetration": shoe.penetration,
        "system": system,
        "running_count": shoe.running_count(system),
        "true_count": shoe.true_count(system),
        "p_rank": shoe.p_rank_distribution(),
        "p_ten": shoe.p_ten(),
        "p_ace": shoe.p_ace(),
        "p_suit": {s: shoe.p_suit(s) for s in SUITS},
        "recent": shoe.log[-6:],
    }

    dealer_ranks = _as_list(dealer_up) if dealer_up else []
    if (dealer_ranks and all(r in RANKS for r in dealer_ranks) and player_ranks
            and shoe.total_remaining > 0):
        total, soft = hand_value(player_ranks)
        snap["hand"] = {
            "dealer_up": dealer_ranks[0],
            "dealer_ranks": dealer_ranks,
            "dealer_total": hand_value(dealer_ranks)[0] if len(dealer_ranks) > 1 else None,
            "player": list(player_ranks),
            "player_total": total,
            "player_soft": soft,
            "dealer_dist": dealer_distribution(shoe, dealer_ranks, hit_soft_17, peeked),
            "stand": stand_outcomes(shoe, player_ranks, dealer_ranks, hit_soft_17, peeked),
            "hit": hit_analysis(shoe, player_ranks, dealer_ranks, hit_soft_17, peeked)
            if total < 21 else None,
        }
    return snap