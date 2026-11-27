import cv2
import numpy as np
import time
from collections import Counter, deque

MAX_CARDS_PER_HAND = 6
MIN_SLOTS_PER_HAND = 2
CLASS_HISTORY_LENGTH = 7
CARD_TIMEOUT_SECONDS = 1.5




def update_hand_memory(
    detected_cards,
    hand_memory,
    current_time
):
    """
    Maintains card positions and prevents class flickering.
    A card must receive a majority vote before changing labels.
    """

    detected_cards = detected_cards[:MAX_CARDS_PER_HAND]

    for index, card_name in enumerate(detected_cards):

        if index >= len(hand_memory):
            hand_memory.append({
                "card": card_name,
                "last_seen": current_time,
                "history": deque(
                    maxlen=CLASS_HISTORY_LENGTH
                )
            })

        slot = hand_memory[index]

        slot["history"].append(card_name)
        slot["last_seen"] = current_time

        # Count recent predictions for this position
        votes = Counter(slot["history"])
        most_common_card, vote_count = (
            votes.most_common(1)[0]
        )

        history_count = len(slot["history"])
        majority_required = (
            history_count // 2
        ) + 1

        # Change displayed class only after a majority vote
        if vote_count >= majority_required:
            slot["card"] = most_common_card

    stable_cards = []

    for slot in hand_memory:
        time_since_seen = (
            current_time - slot["last_seen"]
        )

        if time_since_seen <= CARD_TIMEOUT_SECONDS:
            stable_cards.append(slot["card"])

    # Remove cards that have disappeared long enough
    hand_memory[:] = [
        slot
        for slot in hand_memory
        if (
            current_time - slot["last_seen"]
            <= CARD_TIMEOUT_SECONDS
        )
    ]

    return stable_cards

def draw_card_slot(
    canvas,
    x,
    y,
    width,
    height,
    card_name=None,
    loading_phase=0
):
    border_color = (130, 150, 150)
    text_color = (220, 220, 220)

    cv2.rectangle(
        canvas,
        (x, y),
        (x + width, y + height),
        border_color,
        2
    )

    if card_name is None:
        dots = "." * (loading_phase % 4)
        text = f"loading{dots}"
    else:
        text = str(card_name)

    cv2.putText(
        canvas,
        text,
        (x + 18, y + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        text_color,
        2
    )


def draw_hand_slots(
    canvas,
    cards,
    start_x,
    start_y,
    slot_width,
    slot_height,
    gap,
    loading_phase
):
    if len(cards) == 0:
        # Initial state: two loading slots
        slot_count = MIN_SLOTS_PER_HAND
    else:
        # Always keep exactly one loading slot available
        slot_count = min(
            MAX_CARDS_PER_HAND + 1,
            max(
                MIN_SLOTS_PER_HAND,
                len(cards) + 1
            )
        )

    for index in range(slot_count):

        if index < len(cards):
            card_name = cards[index]
        else:
            card_name = None

        x = start_x + index * (
            slot_width + gap
        )

        draw_card_slot(
            canvas,
            x,
            start_y,
            slot_width,
            slot_height,
            card_name,
            loading_phase
        )

def render_dashboard(
    dealer_cards,
    player_cards,
    width=1400,
    height=700
):
    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    canvas[:] = (25, 25, 25)

    loading_phase = int(time.time() * 2) % 4

    cv2.putText(
        canvas,
        "DEALER",
        (40, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (220, 220, 220),
        2
    )

    cv2.putText(
        canvas,
        "PLAYER",
        (40, 395),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (220, 220, 220),
        2
    )

    slot_width = 190
    slot_height = 260
    gap = 20
    start_x = 40

    draw_hand_slots(
        canvas,
        dealer_cards,
        start_x,
        80,
        slot_width,
        slot_height,
        gap,
        loading_phase
    )

    draw_hand_slots(
        canvas,
        player_cards,
        start_x,
        420,
        slot_width,
        slot_height,
        gap,
        loading_phase
    )

    return canvas