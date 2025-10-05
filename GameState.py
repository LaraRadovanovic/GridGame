# CODE ADAPTED FROM https://www.youtube.com/watch?v=5sZV0Yuh4Io
# Original Author: Maxime Vandegar
# Github: https://github.com/TheCodingAcademy/Minimax-algorithm/blob/main/MinMax.py
# Last accessed: 2025-10-05
# Lara Radovanovic
# CAI5005
# Intro to AI, USF
# 10/05/2025

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import List, Tuple

# ---------- Board / Game constants ----------
ROWS, COLS = 5, 5
BASE_A = (0, 0)    # Player A (Maximizer)
BASE_B = (4, 4)    # Player B (Minimizer)
CAPACITY = 2

DIRS = [(1,0), (-1,0), (0,1), (0,-1)]


@dataclass(frozen=True)
class GameState:
    # Positions
    a_pos: Tuple[int, int]
    b_pos: Tuple[int, int]

    # Previous positions (for anti-backtrack / smoother play)
    a_prev: Tuple[int, int] | None
    b_prev: Tuple[int, int] | None

    # Backpacks (counts only)
    a_bag: int
    b_bag: int

    # Delivered totals (utility is A - B)
    a_delivered: int
    b_delivered: int

    # Remaining resource tiles on the board (positions not yet picked up)
    remaining: frozenset

    # Whose turn: True = A (max), False = B (min)
    a_turn: bool

    # Cached total resources (constant across states)
    total_resources: int

    # Heuristic weights
    # Idea from chatgpt: balance delivered vs distance to needed target
    w_delivered: float = 3.0
    w_need: float = 1.5  # weight for "distance to nearest needed resource/base"
    w_anti_backtrack: float = 0.15  # small nudge to avoid oscillation

    # ------------- Core rules / helpers -------------
    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < ROWS and 0 <= c < COLS

    def is_terminal(self) -> bool:
        """
        Game ends only when:
        - There are no resource tiles left on the board, AND
        - No player is still carrying a resource (both bags are empty).

        This prevents ending while an item is "in transit", so
        a_delivered + b_delivered will always equal total_resources.
        """
        return len(self.remaining) == 0 and self.a_bag == 0 and self.b_bag == 0

    def utility(self) -> int:
        # Zero-sum payoff at terminal: delivered difference
        return self.a_delivered - self.b_delivered

    def score(self) -> float:
        """Evaluation for non-terminal nodes (depth-limited).

        Heuristic focus:
        - Remaining distance to nearest *needed* target."""

        if self.is_terminal():
            return float(self.utility())

        delivered_diff = self.a_delivered - self.b_delivered

        a_need_dist = self._need_distance(is_a=True)
        b_need_dist = self._need_distance(is_a=False)

        # smaller distance is better, so subtract distances
        need_term = -(a_need_dist - b_need_dist)  # A closer than B => positive

        # anti-backtrack slight nudge
        anti_bt = 0.0
        if self.a_prev is not None and self.a_pos == self.a_prev:
            anti_bt -= self.w_anti_backtrack
        if self.b_prev is not None and self.b_pos == self.b_prev:
            anti_bt += self.w_anti_backtrack  # hurts B / helps A in diff

        return self.w_delivered * delivered_diff + self.w_need * need_term + anti_bt

    def _need_distance(self, is_a: bool) -> int:
        """Distance to nearest *needed* goal for the given player.
        If bag < capacity and resources remain: nearest resource.
        Else (bag > 0 or no resources): distance to own base.
        """
        pos = self.a_pos if is_a else self.b_pos
        bag = self.a_bag if is_a else self.b_bag
        base = BASE_A if is_a else BASE_B

        if bag < CAPACITY and self.remaining:
            # nearest resource
            pr, pc = pos
            return min(abs(pr - rr) + abs(pc - cc) for (rr, cc) in self.remaining)
        # need to deliver (or nothing left to pick): go to base
        return manhattan(pos, base)

# adapted from chatgpt openai 
    def get_possible_moves(self):
        """Legal 4-way steps for the active player; cannot move into opponent’s current tile.
        Also avoid immediate backtrack when alternatives exist.
        Additionally: **no cross-base access** (cannot step onto opponent base).
        """
        me = self.a_pos if self.a_turn else self.b_pos
        opp = self.b_pos if self.a_turn else self.a_pos
        prev = self.a_prev if self.a_turn else self.b_prev
        forbidden = BASE_B if self.a_turn else BASE_A  # opponent base

        moves = []
        for dr, dc in DIRS:
            nr, nc = me[0] + dr, me[1] + dc
            if not self.in_bounds(nr, nc):
                continue
            if (nr, nc) == opp:
                continue
            if (nr, nc) == forbidden:
                continue  # disallow stepping on the opponent's base tile
            moves.append((nr, nc))

        # Anti-backtrack filter: drop immediate backtrack if there are other options
        if prev is not None and prev in moves and len(moves) > 1:
            moves = [m for m in moves if m != prev]

        if not moves:
            moves.append(me)  # wait if boxed in
        return moves

    def get_new_state(self, move: Tuple[int,int]) -> "GameState":
        """Apply one move for current player; resolve pickup & delivery; flip turn."""
        if self.a_turn:
            new_a = move
            new_b = self.b_pos
            a_bag, b_bag = self.a_bag, self.b_bag
            a_del, b_del = self.a_delivered, self.b_delivered
            remaining = set(self.remaining)

            # Pickup (no resources on bases by construction/cleaning)
            if new_a in remaining and a_bag < CAPACITY:
                remaining.remove(new_a)
                a_bag += 1

            # Deliver at own base
            if new_a == BASE_A and a_bag > 0:
                a_del += a_bag
                a_bag = 0

            return replace(
                self,
                a_prev=self.a_pos,
                b_prev=self.b_prev,
                a_pos=new_a,
                b_pos=new_b,
                a_bag=a_bag,
                b_bag=b_bag,
                a_delivered=a_del,
                b_delivered=b_del,
                remaining=frozenset(remaining),
                a_turn=False,
            )
        else:
            new_b = move
            new_a = self.a_pos
            a_bag, b_bag = self.a_bag, self.b_bag
            a_del, b_del = self.a_delivered, self.b_delivered
            remaining = set(self.remaining)

            if new_b in remaining and b_bag < CAPACITY:
                remaining.remove(new_b)
                b_bag += 1

            if new_b == BASE_B and b_bag > 0:
                b_del += b_bag
                b_bag = 0

            return replace(
                self,
                a_prev=self.a_prev,
                b_prev=self.b_pos,
                a_pos=new_a,
                b_pos=new_b,
                a_bag=a_bag,
                b_bag=b_bag,
                a_delivered=a_del,
                b_delivered=b_del,
                remaining=frozenset(remaining),
                a_turn=True,
            )


def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# adapted from chatgpt openai 
def _sanitize_resources(resources: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    # Remove any resources placed on bases (no resources on bases)
    clean = [(r, c) for (r, c) in resources if (r, c) not in (BASE_A, BASE_B)]
    # Remove out-of-bounds just in case
    clean = [(r, c) for (r, c) in clean if 0 <= r < ROWS and 0 <= c < COLS]
    return clean


def initial_state(resource_positions: List[Tuple[int,int]]) -> GameState:
    res = _sanitize_resources(resource_positions)
    return GameState(
        a_pos=BASE_A,
        b_pos=BASE_B,
        a_prev=None,
        b_prev=None,
        a_bag=0,
        b_bag=0,
        a_delivered=0,
        b_delivered=0,
        remaining=frozenset(res),
        a_turn=True,  # A (Max) starts
        total_resources=len(res),
    )
