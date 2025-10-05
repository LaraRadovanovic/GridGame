# CODE ADAPTED FROM https://www.youtube.com/watch?v=1n-bhPX38g8
# Original Author: Maxime Vandegar
# Github: https://github.com/TheCodingAcademy/Minimax-algorithm/blob/main/MinMax.py
# Last accessed: 2025-10-05
# Lara Radovanovic
# CAI5005
# Intro to AI, USF
# 10/05/2025


from typing import Optional, Tuple
import math

def minimax(state, depth: int, maximizing: bool, alpha: float = -math.inf, beta: float = math.inf) -> Tuple[float, Optional[tuple]]:
    """Depth-limited minimax with alpha-beta pruning.
    Returns (value, best_move).
    """
    if depth == 0 or state.is_terminal():
        return state.score(), None

    moves = state.get_possible_moves()
    if not moves:
        return state.score(), None

    # Move ordering by shallow score
    ordered_children = []
    for m in moves:
        s2 = state.get_new_state(m)
        ordered_children.append((s2.score(), m))
    ordered_children.sort(reverse=maximizing, key=lambda x: x[0])
    ordered_moves = [m for _, m in ordered_children]

    best_move = None
    if maximizing:
        value = -math.inf
        for m in ordered_moves:
            v, _ = minimax(state.get_new_state(m), depth-1, False, alpha, beta)
            if v > value:
                value = v; best_move = m
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value, best_move
    else:
        value = math.inf
        for m in ordered_moves:
            v, _ = minimax(state.get_new_state(m), depth-1, True, alpha, beta)
            if v < value:
                value = v; best_move = m
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value, best_move
