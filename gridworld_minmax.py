# CODE ADAPTED FROM https://www.youtube.com/watch?v=1n-bhPX38g8 
# MinMaxOriginal Author: Maxime Vandegar
# Github: https://github.com/TheCodingAcademy/Minimax-algorithm/blob/main/MinMax.py
# Last accessed: 2025-10-05
# Tkinter adapted code from https://www.youtube.com/watch?v=UgsVkRwh6mQ
# Tk Original Author: Brianna Ladson
# Github: https://github.com/BriannaLadson/TkWidgets
# Last accessed: 2025-09-21
# Lara Radovanovic
# CAI5005
# Intro to AI, USF
# 10/05/2025


from tkinter import *
from tkinter import ttk, messagebox
import random
import time
from typing import List, Tuple, Callable, Optional
from GameState import initial_state, GameState, BASE_A, BASE_B
from MinMax import minimax as ab_minimax  

ROWS, COLS = 5, 5
CAPACITY = 2

TERRAIN_COLOR = {
    "grassland": "#dff5d8",
    "hills":     "#f3e5c6",
    "swamp":     "#d7e6ef",
    "mountains": "#e7d6e8",
}
GRID_LINE = "#888888"
TILE_OUTLINE = "#333333"

A_BASE_OUTLINE = "#1f6feb"
B_BASE_OUTLINE = "#9e2a2b"
A_AGENT_COLOR  = "#58a6ff"
B_AGENT_COLOR  = "#e63946"
RESOURCE_DOT   = "#ffd166"


def make_maps():
    m1_terrain = [
        ["grassland","grassland","hills","grassland","mountains"],
        ["grassland","swamp","grassland","grassland","hills"],
        ["grassland","hills","grassland","swamp","grassland"],
        ["swamp","grassland","mountains","grassland","grassland"],
        ["grassland","grassland","hills","grassland","swamp"],
    ]
    m1_resources = [(1,3), (3,0), (4,2), (2,1), (0,4), (3,4)] 

    m2_terrain = [
        ["grassland","hills","grassland","grassland","swamp"],
        ["grassland","grassland","hills","grassland","grassland"],
        ["mountains","grassland","grassland","hills","grassland"],
        ["grassland","swamp","grassland","grassland","mountains"],
        ["grassland","hills","grassland","swamp","grassland"],
    ]
    m2_resources = [(1,2), (3,1), (4,3), (2,3), (3,4), (0,3)]

    m3_terrain = [
        ["grassland","grassland","grassland","hills","grassland"],
        ["hills","swamp","grassland","grassland","grassland"],
        ["grassland","grassland","hills","grassland","mountains"],
        ["grassland","mountains","grassland","swamp","grassland"],
        ["swamp","grassland","grassland","hills","grassland"],
    ]
    m3_resources = [(1,1), (2,2), (4,1), (2,4), (3,2), (0,2)]

    def clean(res):
        return [(r,c) for (r,c) in res if (r,c) not in (BASE_A, BASE_B)]
    return [
        {"name":"Map 1", "terrain":m1_terrain, "resources":clean(m1_resources)},
        {"name":"Map 2", "terrain":m2_terrain, "resources":clean(m2_resources)},
        {"name":"Map 3", "terrain":m3_terrain, "resources":clean(m3_resources)},
    ]


# Helpers for evaluation. This portion was adapted from chatgpt. 

def _eval_value(state: GameState) -> float:
    for name in ("utility", "evaluate", "score", "heuristic", "value"):
        f = getattr(state, name, None)
        if callable(f):
            return float(f())
    # If no function available, try attribute
    for name in ("utility", "score", "value"):
        if hasattr(state, name) and not callable(getattr(state, name)):
            try:
                return float(getattr(state, name))
            except Exception:
                pass
    return 0.0


class Counter:
    # Simple object to keep node counts by reference.
    def __init__(self): self.n = 0
    def bump(self): self.n += 1


def plain_minimax(state: GameState, depth: int, maximizing: bool,
                  counter: Optional[Counter] = None) -> tuple[float, Optional[Tuple[int,int]]]:
    """
    Depth-limited MINIMAX without alpha-beta pruning.
    Returns (best_value, best_move)
    Adapted from https://www.youtube.com/watch?v=1n-bhPX38g8
    Github: https://github.com/TheCodingAcademy/Minimax-algorithm
    """
    if counter: counter.bump()

    if depth == 0 or state.is_terminal():
        return _eval_value(state), None

    best_move = None
    if maximizing:
        best_val = float("-inf")
        for mv in state.get_possible_moves():
            child = state.get_new_state(mv)
            v, _ = plain_minimax(child, depth-1, False, counter)
            if v > best_val:
                best_val, best_move = v, mv
        return best_val, best_move
    else:
        best_val = float("+inf")
        for mv in state.get_possible_moves():
            child = state.get_new_state(mv)
            v, _ = plain_minimax(child, depth-1, True, counter)
            if v < best_val:
                best_val, best_move = v, mv
        return best_val, best_move


def alpha_beta_minimax(state: GameState, depth: int, maximizing: bool,
                       counter: Optional[Counter] = None) -> tuple[float, Optional[Tuple[int,int]]]:
    """
    Wrapper around existing MinMax.minimax (assumed alpha–beta).
    """
    if counter: counter.bump() 
    return ab_minimax(state, depth, maximizing)


def benchmark(search_impl: Callable, state: GameState, depth: int, maximizing: bool):
    """Run a search impl, and return (value, move, nodes, seconds)."""
    counter = Counter()
    t0 = time.perf_counter()
    val, move = search_impl(state, depth, maximizing, counter)
    dt = time.perf_counter() - t0
    return val, move, counter.n, dt


# ---------- Agents ----------

class RandomAgent:
    def choose(self, state: GameState):
        return random.choice(state.get_possible_moves())

class SearchAgent:
    """
    Agent that uses either Plain Minimax or Alpha–Beta based on mode string.
    mode {"Minimax", "Alpha-Beta"}
    """
    def __init__(self, depth: int = 4, mode: str = "Alpha-Beta"):
        self.depth = depth
        self.mode = mode

    def choose(self, state: GameState):
        maximizing = state.a_turn
        if self.mode == "Minimax":
            val, best = plain_minimax(state, self.depth, maximizing)
        else:
            val, best = alpha_beta_minimax(state, self.depth, maximizing)
        if best is None:
            moves = state.get_possible_moves()
            if not moves:
                return None
            best = random.choice(moves)
        return best


# ---------- App (UI) ----------

class App(Tk):
    def __init__(self):
        super().__init__()
        self.title("Grid World — Minimax vs Alpha–Beta (depth-limited)")

        self.maps = make_maps()
        self.cur_map_idx = 0

        # Top Controls
        top = Frame(self); top.pack(side=TOP, fill=X)

        Label(top, text="Map:").pack(side=LEFT, padx=(10,5))
        self.map_var = StringVar(value=self.maps[0]["name"])
        self.map_combo = ttk.Combobox(
            top, textvariable=self.map_var,
            values=[m["name"] for m in self.maps],
            state="readonly", width=28
        )
        self.map_combo.pack(side=LEFT, padx=(0,10))
        self.map_combo.bind("<<ComboboxSelected>>", self.on_map_change)

        Label(top, text="Opponent (B):").pack(side=LEFT, padx=(6,5))
        self.agentb_var = StringVar(value="Random")
        self.agentb_combo = ttk.Combobox(
            top, textvariable=self.agentb_var,
            values=["Random","Minimax"],  # Minimax here means a search agent for B
            state="readonly", width=10
        )
        self.agentb_combo.pack(side=LEFT, padx=(0,10))

        Label(top, text="Search Type:").pack(side=LEFT, padx=(6,5))
        self.search_var = StringVar(value="Alpha-Beta")
        self.search_combo = ttk.Combobox(
            top, textvariable=self.search_var,
            values=["Alpha-Beta", "Minimax"], 
            state="readonly", width=10
        )
        self.search_combo.pack(side=LEFT, padx=(0,10))

        Label(top, text="Depth A:").pack(side=LEFT, padx=(6,5))
        self.depth_a = Spinbox(top, from_=1, to=8, width=5)
        self.depth_a.delete(0,"end"); self.depth_a.insert(0,"4")
        self.depth_a.pack(side=LEFT)

        Label(top, text="Depth B:").pack(side=LEFT, padx=(6,5))
        self.depth_b = Spinbox(top, from_=1, to=8, width=5)
        self.depth_b.delete(0,"end"); self.depth_b.insert(0,"4")
        self.depth_b.pack(side=LEFT)

        Label(top, text="Animation:").pack(side=LEFT, padx=(10,5))
        self.speed_var = StringVar(value="Normal")
        self.speed_combo = ttk.Combobox(
            top, textvariable=self.speed_var,
            values=["Slow","Normal","Fast"],
            state="readonly", width=10
        )
        self.speed_combo.pack(side=LEFT, padx=(0,10))

        # Buttons — keep Start/Pause/Step/Reset + Compare
        self.start_btn = Button(top, text="Start", command=self.on_start); self.start_btn.pack(side=LEFT, padx=2)
        self.pause_btn = Button(top, text="Pause", command=self.on_pause); self.pause_btn.pack(side=LEFT, padx=2)
        self.step_btn  = Button(top, text="Step",  command=self.on_step);  self.step_btn.pack(side=LEFT, padx=2)
        self.reset_btn = Button(top, text="Reset", command=self.on_reset); self.reset_btn.pack(side=LEFT, padx=2)
        self.cmp_btn   = Button(top, text="Compare", command=self.on_compare); self.cmp_btn.pack(side=LEFT, padx=10)

        # ===== Main / Canvas + HUD =====
        main = Frame(self); main.pack(fill=BOTH, expand=1)

        self.can = Canvas(main, bg="#ffffff")
        self.can.pack(side=LEFT, fill=BOTH, expand=1)
        self.can.bind("<Configure>", lambda e: self.redraw())

        hud = Frame(main, width=360); hud.pack(side=RIGHT, fill=Y); hud.pack_propagate(False)
        Label(hud, text="HUD", font=("Consolas", 14, "bold")).pack(pady=(12,8))

        self.turn_var = StringVar(value="Turn: A   (moves: 0)")
        Label(hud, textvariable=self.turn_var, font=("Consolas", 12)).pack(anchor="w", padx=10, pady=(0,8))

        Label(hud, text="Positions:", font=("Consolas", 12)).pack(anchor="w", padx=10)
        self.pos_var = StringVar(value="A: (0,0)\nB: (4,4)")
        Label(hud, textvariable=self.pos_var, font=("Consolas", 12)).pack(anchor="w", padx=24, pady=(0,8))

        Label(hud, text="Player A Inventory (max 2):", font=("Consolas", 12), fg="green").pack(anchor="w", padx=10)
        self.inv_a = StringVar(value="S:0  I:0  C:0")
        Label(hud, textvariable=self.inv_a, font=("Consolas", 12)).pack(anchor="w", padx=24, pady=(0,6))

        Label(hud, text="Player B Inventory (max 2):", font=("Consolas", 12), fg="red").pack(anchor="w", padx=10)
        self.inv_b = StringVar(value="S:0  I:0  C:0")
        Label(hud, textvariable=self.inv_b, font=("Consolas", 12)).pack(anchor="w", padx=24, pady=(0,8))

        Label(hud, text="Delivered:", font=("Consolas", 12, "bold")).pack(anchor="w", padx=10)
        self.deliv_var = StringVar(value="A:0  B:0")
        Label(hud, textvariable=self.deliv_var, font=("Consolas", 12)).pack(anchor="w", padx=24, pady=(0,8))

        Label(hud, text="Remaining resources on map:", font=("Consolas", 12, "bold")).pack(anchor="w", padx=10)
        self.rem_var = StringVar(value="0")
        Label(hud, textvariable=self.rem_var, font=("Consolas", 12)).pack(anchor="w", padx=24, pady=(0,8))

        # Comparison results panel
        Label(hud, text="Search Comparison:", font=("Consolas", 12, "bold")).pack(anchor="w", padx=10, pady=(8,0))
        self.cmp_var = StringVar(value="Minimax — nodes: -, time: -\nAlpha-Beta — nodes: -, time: -")
        Label(hud, textvariable=self.cmp_var, font=("Consolas", 11)).pack(anchor="w", padx=24, pady=(0,12))

        # State
        self.running = False
        self.turns = 0
        self.state: GameState | None = None
        self.agentA = None
        self.agentB = None
        self.tile_wh = (1,1)

        self.new_match(self.maps[0]["resources"])
        self.bind("<space>", lambda e: self.on_step())
        self.bind("<Return>", lambda e: self.on_start())

    # ---------- Match / Agents ----------
    def new_match(self, resources: List[Tuple[int,int]]):
        self.state = initial_state(resources)
        self.turns = 0
        self.running = False
        self.configure_agents()
        self.update_hud()
        self.redraw()
        self.cmp_var.set("Minimax — nodes: -, time: -\nAlpha-Beta — nodes: -, time: -")

    def configure_agents(self):
        try:
            da = int(self.depth_a.get())
        except:
            da = 4; self.depth_a.delete(0,"end"); self.depth_a.insert(0,"4")
        try:
            db = int(self.depth_b.get())
        except:
            db = 4; self.depth_b.delete(0,"end"); self.depth_b.insert(0,"4")

        # Agent A uses the selected search type from dropdown
        modeA = self.search_var.get()
        self.agentA = SearchAgent(depth=da, mode=modeA)

        # Agent B uses either Random or Minimax (alpha-beta is fine too)
        if self.agentb_var.get() == "Minimax":
            self.agentB = SearchAgent(depth=db, mode="Alpha-Beta")
        else:
            self.agentB = RandomAgent()

    # ---------- UI Callbacks ----------
    def on_map_change(self, _=None):
        name = self.map_var.get()
        for i, m in enumerate(self.maps):
            if m["name"] == name:
                self.cur_map_idx = i
                break
        self.new_match(self.maps[self.cur_map_idx]["resources"])

    def on_start(self):
        if self.state.is_terminal():
            messagebox.showinfo("Game Over", "All resources are depleted. Reset to play again.")
            return
        self.configure_agents()
        if not self.running:
            self.running = True
            self.loop_once()

    def on_pause(self):
        self.running = False

    def on_step(self):
        if self.state.is_terminal():
            messagebox.showinfo("Game Over", "All resources are depleted. Reset to play again.")
            return
        self.configure_agents()
        self.running = False
        self.loop_once()

    def on_reset(self):
        self.new_match(self.maps[self.cur_map_idx]["resources"])

    def on_compare(self):
        # Compare Minimax vs Alpha–Beta on the current state with depth A.
        if self.state is None:
            return
        try:
            d = int(self.depth_a.get())
        except:
            d = 4
        # Freeze current state
        s = self.state
        maximizing = s.a_turn

        # Benchmark both
        _, _, nodes_mm, t_mm = benchmark(plain_minimax, s, d, maximizing)
        _, _, nodes_ab, t_ab = benchmark(alpha_beta_minimax, s, d, maximizing)

        self.cmp_var.set(
            f"Minimax — nodes: {nodes_mm:,}, time: {t_mm:.4f}s\n"
            f"Alpha-Beta — nodes: {nodes_ab:,}, time: {t_ab:.4f}s"
        )

    # ---------- Game Loop ----------
    # Used chatgpt to debug, and help write this section.
    def speed_delay(self):
        s = self.speed_var.get()
        if s == "Slow":   return 600
        if s == "Fast":   return 80
        return 200

    def loop_once(self):
        if self.state.is_terminal():
            self.update_hud()
            self.running = False
            self.show_winner()
            return

        agent = self.agentA if self.state.a_turn else self.agentB
        move = agent.choose(self.state)
        if move is None:
            self.running = False
            self.show_winner()
            return

        self.state = self.state.get_new_state(move)
        self.turns += 1
        self.update_hud()
        self.redraw()

        if self.state.is_terminal():
            self.running = False
            self.show_winner()
            return

        if self.running:
            self.after(self.speed_delay(), self.loop_once)

    def show_winner(self):
        a, b = self.state.a_delivered, self.state.b_delivered
        if a > b: msg = f"A wins!  A={a}  B={b}"
        elif b > a: msg = f"B wins!  A={a}  B={b}"
        else: msg = f"Tie.  A={a}  B={b}"
        messagebox.showinfo("Result", f"Game over — {msg}")

    # ---------- Rendering ----------
    def grid_geom(self):
        W = self.can.winfo_width()
        H = self.can.winfo_height()
        tw = W / COLS
        th = H / ROWS
        return W, H, tw, th

    def redraw(self):
        self.can.delete("all")
        W, H, tw, th = self.grid_geom()
        self.tile_wh = (tw, th)

        terrain = self.maps[self.cur_map_idx]["terrain"]
        for r in range(ROWS):
            for c in range(COLS):
                x0, y0 = c*tw, r*th
                x1, y1 = x0+tw, y0+th
                t = terrain[r][c]
                self.can.create_rectangle(x0, y0, x1, y1, fill=TERRAIN_COLOR[t], outline=TILE_OUTLINE)

        for c in range(COLS+1):
            x = c*tw; self.can.create_line(x, 0, x, H, fill=GRID_LINE)
        for r in range(ROWS+1):
            y = r*th; self.can.create_line(0, y, W, y, fill=GRID_LINE)

        # coordinates along edges
        for c in range(COLS):
            self.can.create_text(c*tw + tw/2, 12, text=str(c), font=("Consolas", 12))
        for r in range(ROWS):
            self.can.create_text(12, r*th + th/2, text=str(r), font=("Consolas", 12))

        self.draw_base(BASE_A, A_BASE_OUTLINE, "A")
        self.draw_base(BASE_B, B_BASE_OUTLINE, "B")

        if self.state is not None:
            for (rr, cc) in self.state.remaining:
                self.draw_resource(rr, cc)

        if self.state is not None:
            self.draw_agent(self.state.a_pos, A_AGENT_COLOR, "A")
            self.draw_agent(self.state.b_pos, B_AGENT_COLOR, "B")

    def draw_base(self, pos, outline_color, label):
        tw, th = self.tile_wh
        r, c = pos
        x0, y0 = c*tw, r*th
        self.can.create_rectangle(x0, y0, x0+tw, y0+th, outline=outline_color, width=3)
        self.can.create_text(x0+tw/2, y0+th/2, text=label, fill=outline_color,
                             font=("Consolas", 18, "bold"))

    def draw_resource(self, r, c):
        tw, th = self.tile_wh
        cx, cy = c*tw + tw/2, r*th + th/2
        self.can.create_oval(cx-8, cy-8, cx+8, cy+8, fill=RESOURCE_DOT, outline="#0b0b0b")

    def draw_agent(self, pos, fill_color, label):
        tw, th = self.tile_wh
        r, c = pos
        x, y = c*tw + tw/2, r*th + th/2
        d = min(tw, th) * 0.6
        self.can.create_oval(x-d/2, y-d/2, x+d/2, y+d/2, fill=fill_color, outline="#ffffff", width=2)
        self.can.create_text(x, y, text=label, fill="white", font=("Consolas", 14, "bold"))

    # ---------- HUD ----------
    def update_hud(self):
        if self.state is None:
            return
        who = "A" if self.state.a_turn else "B"
        self.turn_var.set(f"Turn: {who}   (moves: {self.turns})")
        self.pos_var.set(f"A: {self.state.a_pos}\nB: {self.state.b_pos}")

        # Placeholder S/I/C formatting tied to bag count
        self.inv_a.set("S:0  I:0  C:0" if self.state.a_bag == 0 else f"S:0  I:{self.state.a_bag}  C:0")
        self.inv_b.set("S:0  I:0  C:0" if self.state.b_bag == 0 else f"S:0  I:{self.state.b_bag}  C:0")

        self.deliv_var.set(f"A:{self.state.a_delivered}  B:{self.state.b_delivered}")

        rem_list = list(self.state.remaining)
        self.rem_var.set(str(len(rem_list)))


if __name__ == "__main__":
    App().mainloop()
