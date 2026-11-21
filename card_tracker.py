"""Turns noisy per-frame detections into confirmed cards for the Shoe.

The detector fires every frame and flickers, mislabels now and then, and sees
the same card hundreds of times. A card should be counted ONCE, and only once
we are sure what it is. The tracker does that:

  1. Match each detection to a card already being tracked (by the position of
     the box's top-left corner, which stays put when cards overlap).
  2. Keep the last few labels each card received and let them vote.
  3. When a card has been seen enough times and the vote is clear, call
     shoe.observe() exactly once.

Detections are dicts: {"rank": str, "conf": float, "box": (x1, y1, x2, y2),
                       "suit": str | None, "suit_conf": float}
"Back" (a face-down card) is tracked but never counted.
"""
import csv
import math
from collections import Counter, deque

import cv2

GREEN, YELLOW, RED, GREY = (0, 220, 0), (0, 220, 255), (0, 0, 230), (170, 170, 170)


class _Track:
    def __init__(self, track_id, window):
        self.id = track_id
        self.box = (0, 0, 0, 0)
        self.rank_votes = deque(maxlen=window)   # (rank, confidence)
        self.suit_votes = deque(maxlen=window)
        self.hits = 0
        self.missed = 0
        self.counted = False
        self.rejected = False   # shoe said the card is impossible (detector error)
        self.ignored = False    # was already on the table when a new shoe started
        self.disputed = False
        self.rank = None
        self.suit = None
        self.order = None
        self.lost_frame = None

    def add(self, det):
        self.box = det["box"]
        self.hits += 1
        self.missed = 0
        self.rank_votes.append((det["rank"], det["conf"]))
        if det.get("suit"):
            self.suit_votes.append((det["suit"], det.get("suit_conf", 1.0)))

    @property
    def center_y(self):
        return (self.box[1] + self.box[3]) / 2

    @property
    def height(self):
        return self.box[3] - self.box[1]


def _consensus(votes):
    """(label, share of the total confidence) for a collection of (label, weight)."""
    if not votes:
        return None, 0.0
    weights = Counter()
    for label, weight in votes:
        weights[label] += weight
    label, top = weights.most_common(1)[0]
    return label, top / sum(weights.values())


class CardTracker:
    def __init__(self, shoe, confirm_hits=6, window=10, agree=0.7, max_missed=12,
                 ghost_frames=45, match_frac=0.2, min_match_px=20,
                 expect_suits=False, suit_wait=6):
        """
        confirm_hits  frames a card must be seen before it can be counted
        window        how many recent labels vote
        agree         share of the vote the winning label needs
        max_missed    frames a card may vanish before the track is dropped
        ghost_frames  after a counted card vanishes, a same-rank card reappearing
                      at the same spot within this many frames is the same card
        expect_suits  True when a suit model is running: wait up to suit_wait extra
                      frames for a clear suit before counting without one
        """
        self.shoe = shoe
        self.confirm_hits, self.window, self.agree = confirm_hits, window, agree
        self.max_missed, self.ghost_frames = max_missed, ghost_frames
        self.match_frac, self.min_match_px = match_frac, min_match_px
        self.expect_suits, self.suit_wait = expect_suits, suit_wait
        self.events = []          # every counted card, for the session log
        self.stats = Counter()
        self.shoe_number = 1
        self._clear()

    def _clear(self):
        self.tracks, self.ghosts = [], []
        self.frame, self.divider_y = 0, 0
        self._next_id, self._next_order = 1, 1

    # ---- public API ----------------------------------------------------------
    def update(self, detections, divider_y):
        """Call once per frame. Returns the (rank, suit, zone) of cards newly counted."""
        self.frame += 1
        self.divider_y = divider_y
        self.ghosts = [g for g in self.ghosts if self.frame - g.lost_frame <= self.ghost_frames]

        matches, unmatched = self._match(detections)
        for track, det in matches:
            track.add(det)

        matched_ids = {id(t) for t, _ in matches}
        for track in list(self.tracks):
            if id(track) in matched_ids:
                continue
            track.missed += 1
            if track.missed > self.max_missed:
                self.tracks.remove(track)
                if track.counted and not track.ignored:
                    track.lost_frame = self.frame
                    self.ghosts.append(track)

        for det in unmatched:
            self.tracks.append(self._new_track(det))

        counted_now = []
        for track in self.tracks:
            if track.ignored:
                continue
            if track.counted:
                self._check_dispute(track)
            else:
                decision = self._try_confirm(track)
                if decision:
                    counted_now.append(self._count(track, *decision))
        return [c for c in counted_now if c]

    def hands(self):
        """(dealer_ranks, player_ranks): counted cards on the table right now,
        in the order they were counted. Face-down cards are not included."""
        shown = sorted((t for t in self.tracks
                        if t.counted and not t.rejected and not t.ignored),
                       key=lambda t: t.order)
        dealer = [t.rank for t in shown if t.center_y < self.divider_y]
        player = [t.rank for t in shown if t.center_y >= self.divider_y]
        return dealer, player

    def new_shoe(self):
        """Reshuffle: clear the shoe's counts. Cards lying on the table right now
        are ignored, so they are not counted into the fresh shoe."""
        self.shoe.reset()
        for track in self.tracks:
            track.counted = track.ignored = True
        self.ghosts = []
        self._next_order = 1
        self.shoe_number += 1

    def draw(self, frame):
        """Small status line under each tracked card."""
        height = frame.shape[0]
        for t in self.tracks:
            if t.ignored:
                continue
            label, _ = _consensus(t.rank_votes)
            if t.rejected:
                text, color = "IMPOSSIBLE - not counted", RED
            elif t.counted:
                text, color = f"counted #{t.order}", GREEN
            elif label == "Back":
                text, color = "face-down", GREY
            else:
                text, color = f"confirming {min(t.hits, self.confirm_hits)}/{self.confirm_hits}", YELLOW
            x1, _, _, y2 = t.box
            cv2.putText(frame, text, (x1, min(y2 + 20, height - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def summary(self):
        s = self.stats
        return (f"Tracker: {s['counted']} cards counted, {s['rejected']} rejected as impossible, "
                f"{s['late_disagreements']} late label disagreements, "
                f"{s['tracks_created']} tracks created.")

    def save_log(self, path="session_log.csv"):
        fields = ["shoe", "frame", "rank", "suit", "zone", "cards_seen", "running_count"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.events)

    # ---- internals -----------------------------------------------------------
    def _threshold(self, box):
        return max(self.min_match_px, self.match_frac * (box[3] - box[1]))

    def _match(self, detections):
        """Greedy nearest-corner matching. Returns ([(track, det)], [unmatched dets])."""
        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                dist = math.hypot(track.box[0] - det["box"][0], track.box[1] - det["box"][1])
                if dist <= self._threshold(track.box):
                    pairs.append((dist, ti, di))
        pairs.sort()

        used_t, used_d, matches = set(), set(), []
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches.append((self.tracks[ti], detections[di]))
        return matches, [d for i, d in enumerate(detections) if i not in used_d]

    def _new_track(self, det):
        track = _Track(self._next_id, self.window)
        self._next_id += 1
        self.stats["tracks_created"] += 1
        track.add(det)

        # A counted card that flickered away and came back is the same card.
        for ghost in self.ghosts:
            close = math.hypot(ghost.box[0] - det["box"][0],
                               ghost.box[1] - det["box"][1]) <= self._threshold(ghost.box)
            if close and ghost.rank == det["rank"]:
                track.counted, track.rejected = True, ghost.rejected
                track.rank, track.suit, track.order = ghost.rank, ghost.suit, ghost.order
                self.ghosts.remove(ghost)
                break
        return track

    def _try_confirm(self, track):
        if track.hits < self.confirm_hits or len(track.rank_votes) < self.confirm_hits:
            return None
        rank, share = _consensus(track.rank_votes)
        if rank == "Back" or share < self.agree:
            return None

        suit, suit_share = _consensus(track.suit_votes)
        suit_ok = suit is not None and suit_share >= self.agree and len(track.suit_votes) >= 3
        if self.expect_suits and not suit_ok and track.hits < self.confirm_hits + self.suit_wait:
            return None  # give the suit classifier a few more frames
        return rank, (suit if suit_ok else None)

    def _count(self, track, rank, suit):
        track.counted = True
        track.rank, track.suit = rank, suit
        zone = "dealer" if track.center_y < self.divider_y else "player"

        if not self.shoe.observe(rank, suit):
            track.rejected = True
            self.stats["rejected"] += 1
            return None

        track.order = self._next_order
        self._next_order += 1
        self.stats["counted"] += 1
        self.events.append({
            "shoe": self.shoe_number, "frame": self.frame, "rank": rank,
            "suit": suit or "", "zone": zone, "cards_seen": self.shoe.cards_seen,
            "running_count": self.shoe.running_count(),
        })
        return rank, suit, zone

    def _check_dispute(self, track):
        """A counted card whose label later changes: cannot be un-counted, but is
        logged as a failure case (useful for the report)."""
        if track.disputed or track.rejected or len(track.rank_votes) < self.window:
            return
        rank, share = _consensus(track.rank_votes)
        if rank not in (None, "Back", track.rank) and share >= self.agree:
            track.disputed = True
            self.stats["late_disagreements"] += 1