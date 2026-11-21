"""Draws the stats as an image you can put next to the camera view.

    snapshot = build_snapshot(shoe, dealer_up="6", player_ranks=["5", "7"])
    panel = render_stats_panel(snapshot, height=table_frame.shape[0])
    cv2.imshow("Blackjack CV", np.hstack([table_frame, panel]))

Only recompute the snapshot when a new card is confirmed, not every frame.
"""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (32, 32, 32)
WHITE = (240, 240, 240)
GREY = (160, 160, 160)
GREEN = (110, 200, 90)
RED = (90, 90, 230)
BLUE = (230, 170, 70)
AMBER = (60, 190, 240)

RANK_ORDER = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]


def _signed(value):
    """+0.21 / -0.38, and never '-0.00'."""
    return f"{0.0 if abs(value) < 0.005 else value:+.2f}"


def _text(img, text, x, y, scale=0.5, color=WHITE, thickness=1):
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _rank_bars(img, p_rank, x0, y0, width, height):
    """Bar per rank; green = more likely than in a fresh shoe, red = less likely."""
    baseline = 1 / 13
    top = max(0.15, max(p_rank.values()) * 1.15)
    slot = width / 13
    base_y = y0 + height
    for i, rank in enumerate(RANK_ORDER):
        p = p_rank[rank]
        bar_h = int(height * p / top)
        x = int(x0 + i * slot)
        color = GREEN if p > baseline + 1e-9 else RED if p < baseline - 1e-9 else GREY
        cv2.rectangle(img, (x + 3, base_y - bar_h), (int(x + slot) - 3, base_y), color, -1)
        _text(img, f"{p * 100:.0f}", x + 4, base_y - bar_h - 4, 0.32, GREY)
        _text(img, rank, x + 5, base_y + 15, 0.4)
    line_y = base_y - int(height * baseline / top)
    cv2.line(img, (x0, line_y), (x0 + width, line_y), WHITE, 1, cv2.LINE_AA)


def _stacked_bar(img, parts, x0, y0, width, height):
    x = x0
    for value, color, label in parts:
        w = int(round(width * value))
        if w <= 0:
            continue
        cv2.rectangle(img, (x, y0), (x + w, y0 + height), color, -1)
        if w > 44:
            _text(img, f"{label} {value * 100:.0f}%", x + 4, y0 + height - 7, 0.4, (20, 20, 20))
        x += w


def render_stats_panel(snapshot, height=700, width=380):
    img = np.full((height, width, 3), BG, np.uint8)
    m = 14  # margin
    inner = width - 2 * m

    _text(img, "SHOE STATS", m, 26, 0.65, WHITE, 2)
    _text(img, f"Seen {snapshot['cards_seen']}/{snapshot['cards_total']}"
               f"   Decks left {snapshot['decks_remaining']:.2f}", m, 50, 0.48, GREY)

    name = {"hi_lo": "Hi-Lo", "ko": "KO", "omega_ii": "Omega II"}[snapshot["system"]]
    rc, tc = snapshot["running_count"], snapshot["true_count"]
    count_color = GREEN if rc > 0 else RED if rc < 0 else WHITE
    _text(img, f"{name}  running {rc:+d}   true {tc:+.1f}", m, 78, 0.55, count_color, 1)

    _text(img, "Next card: % per rank (line = fresh shoe)", m, 112, 0.45, GREY)
    _rank_bars(img, snapshot["p_rank"], m, 130, inner, 120)

    fresh_ten = 16 / 52 * 100
    _text(img, f"Ten-value: {snapshot['p_ten'] * 100:.1f}%  (fresh {fresh_ten:.1f}%)", m, 290, 0.5)
    _text(img, f"Ace: {snapshot['p_ace'] * 100:.1f}%  (fresh 7.7%)", m, 312, 0.5)
    suits = snapshot["p_suit"]
    _text(img, "Suit: " + "  ".join(f"{s[0].upper()} {suits[s] * 100:.0f}%" for s in suits),
          m, 334, 0.45, GREY)

    recent = "  ".join(f"{r}{(s or '?')[0].upper()}" for r, s in snapshot.get("recent", []))
    _text(img, "Last counted: " + (recent or "-"), m, height - 36, 0.42, GREY)
    _text(img, "R new shoe   U undo   P hide panel", m, height - 14, 0.38, GREY)

    hand = snapshot.get("hand")
    if hand is None:
        _text(img, "Waiting for dealer upcard + player hand...", m, 382, 0.45, GREY)
        return img

    cv2.line(img, (m, 356), (width - m, 356), (70, 70, 70), 1)
    kind = "soft" if hand["player_soft"] else "hard"
    if hand["dealer_total"] is None:
        dealer_text = f"Dealer shows {hand['dealer_up']}"
    else:
        dealer_text = f"Dealer has {hand['dealer_total']}"
    _text(img, f"{dealer_text}   You: {hand['player_total']} ({kind})", m, 384, 0.55, WHITE, 1)

    stand = hand["stand"]
    _text(img, "If you stand:", m, 414, 0.45, GREY)
    _stacked_bar(img, [(stand["win"], GREEN, "win"), (stand["push"], AMBER, "push"),
                       (stand["lose"], RED, "lose")], m, 422, inner, 24)
    _text(img, f"EV {_signed(stand['ev'])} per unit bet", m, 466, 0.45, WHITE)

    if hand["hit"] is not None:
        _text(img, f"If you hit: bust {hand['hit']['p_bust'] * 100:.0f}%   "
                   f"EV (1 card) {_signed(hand['hit']['ev_hit_once'])}", m, 492, 0.45, WHITE)

    _text(img, "Dealer's final hand:", m, 528, 0.45, GREY)
    dist = hand["dealer_dist"]
    labels = [("17", dist[17]), ("18", dist[18]), ("19", dist[19]), ("20", dist[20]),
              ("21", dist[21]), ("bust", dist["bust"])]
    slot = inner / len(labels)
    for i, (label, p) in enumerate(labels):
        x = int(m + i * slot)
        bar_h = int(70 * min(1.0, p / 0.5))
        color = RED if label == "bust" else BLUE
        cv2.rectangle(img, (x + 4, 620 - bar_h), (int(x + slot) - 4, 620), color, -1)
        _text(img, f"{p * 100:.0f}%", x + 6, 616 - bar_h, 0.38, WHITE)
        _text(img, label, x + 8, 638, 0.4, GREY)
    return img