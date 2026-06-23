"""
╔══════════════════════════════════════════════════════════════╗
║   NETWORK INTRUSION DETECTION SYSTEM  —  GUI v3.0           ║
║   Cyberpunk SIEM Dashboard  |  UNSW-NB15 Dataset             ║
║   Random Forest (primary)  +  Decision Tree (baseline)       ║
╚══════════════════════════════════════════════════════════════╝

OVERVIEW:
    This GUI application trains and evaluates two ML classifiers
    (Random Forest and Decision Tree) on the UNSW-NB15 network
    intrusion dataset. It visualises results in a cyberpunk-styled
    Tkinter dashboard with live gauges, charts, and an alert log.

FLOW:
    1. User selects train/test CSV files via sidebar
    2. Clicking "Initiate Scan" launches a background thread
    3. Pipeline: Load → Preprocess → Train → Evaluate → Display
"""

# ── Standard library ──────────────────────────────────────────
import math
import random
import threading
import time
import warnings
from datetime import datetime, timedelta

# ── Third-party ───────────────────────────────────────────────
import matplotlib
matplotlib.use("TkAgg")           # Must be set before importing pyplot
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# ── Scikit-learn ──────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix,
    f1_score, precision_score, recall_score, roc_curve,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
#  DESIGN TOKENS — Neon-on-dark cyberpunk palette
#  All UI colours are defined here; nothing is hardcoded below.
# ══════════════════════════════════════════════════════════════
BG0    = "#050A0F"   # Deepest background (window fill)
BG1    = "#0A1628"   # Panel background (sidebar, cards)
BG2    = "#0F2040"   # Input background, selected rows
BG3    = "#162850"   # Button hover background

CYAN   = "#00F5FF"   # Primary accent — headings, active gauges
GREEN  = "#00FF87"   # Success / low-severity / normal traffic
AMBER  = "#FFB800"   # Warning / high-severity
RED    = "#FF2D55"   # Danger / critical alerts
PURPLE = "#BF5FFF"   # Secondary model colour (Decision Tree)
BLUE   = "#3D9AFF"   # Medium-severity

TEXT1  = "#E0F4FF"   # Primary text (bright)
TEXT2  = "#7BAFC8"   # Secondary text (muted)
TEXT3  = "#3D6080"   # Tertiary text (very muted, labels)
BORDER = "#1A3A5C"   # Subtle panel borders

# Font tuples for tkinter
FONT_HEAD  = ("Courier New", 11, "bold")
FONT_BODY  = ("Courier New", 10)
FONT_SMALL = ("Courier New", 9)
FONT_TINY  = ("Courier New", 8)

# Severity colour map (used throughout the alert table)
SEV_COL = {"CRITICAL": RED, "HIGH": AMBER, "MEDIUM": BLUE, "LOW": GREEN}
# Severity sort order (lower = more important)
SEV_ORD = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

# ── Dataset configuration ─────────────────────────────────────
# Columns that are categorical and need label-encoding
CATEGORICAL_COLS = ["proto", "service", "state"]
# Columns to drop before training (identifiers, timestamps, raw target text)
DROP_COLS = ["id", "srcip", "sport", "dstip", "dsport", "Stime", "Ltime", "attack_cat"]
TARGET    = "label"   # Binary target column (0 = normal, 1 = attack)

# ── Model hyperparameters ─────────────────────────────────────
RF_PARAMS = dict(
    n_estimators=100,       # Number of trees in the forest
    max_depth=None,         # Grow full trees (pruned implicitly by min_samples_split)
    min_samples_split=5,    # Minimum samples required to split a node
    n_jobs=-1,              # Use all CPU cores
    random_state=42,
    class_weight="balanced" # Compensate for class imbalance
)
DT_PARAMS = dict(
    max_depth=15,           # Limit depth to avoid overfitting
    random_state=42,
    class_weight="balanced"
)

# Confidence thresholds for alert severity classification
SEV_THRESH = {"CRITICAL": 0.90, "HIGH": 0.75, "MEDIUM": 0.60, "LOW": 0.50}

def assign_sev(confidence: float) -> str:
    """Map a model confidence score (0–1) to a severity label."""
    if   confidence >= SEV_THRESH["CRITICAL"]: return "CRITICAL"
    elif confidence >= SEV_THRESH["HIGH"]:     return "HIGH"
    elif confidence >= SEV_THRESH["MEDIUM"]:   return "MEDIUM"
    else:                                       return "LOW"


# ══════════════════════════════════════════════════════════════
#  WIDGET: ANIMATED DONUT GAUGE
#  Draws a circular arc gauge that animates smoothly toward a
#  target value. Used on the dashboard for each metric.
# ══════════════════════════════════════════════════════════════
class DonutGauge(tk.Canvas):
    def __init__(self, parent, label="", color=CYAN, size=108, **kw):
        super().__init__(parent, width=size, height=size,
                         bg=parent["bg"], highlightthickness=0, **kw)
        self._size    = size
        self._color   = color
        self._label   = label
        self._target  = 0.0    # Value we're animating toward (0–1)
        self._current = 0.0    # Currently displayed value (0–1)
        self._draw()
        self._tick()           # Start the animation loop

    def set_value(self, v: float):
        """Set the target value (0.0–1.0). The gauge animates toward it."""
        self._target = max(0.0, min(1.0, v))

    def _draw(self):
        """Redraw the gauge for the current animation frame."""
        self.delete("all")
        s, p = self._size, 10
        cx = cy = s / 2

        # Radius leaves room for padding
        r = s / 2 - p

        # Dashed glow ring (decorative outer halo)
        self.create_arc(cx-r-3, cy-r-3, cx+r+3, cy+r+3,
                        start=0, extent=360, style="arc",
                        outline=self._color, width=1, dash=(2, 6))

        # Grey track (full circle)
        self.create_arc(cx-r, cy-r, cx+r, cy+r,
                        start=0, extent=360, style="arc",
                        outline=BG2, width=10)

        # Coloured value arc (drawn clockwise from top)
        ext = self._current * 359.9
        self.create_arc(cx-r, cy-r, cx+r, cy+r,
                        start=90, extent=-ext,
                        style="arc", outline=self._color, width=10)

        # Centre percentage label
        pct = f"{self._current*100:.1f}%"
        self.create_text(cx, cy-7, text=pct,
                         fill=self._color, font=("Courier New", 12, "bold"))

        # Optional sub-label beneath the percentage
        if self._label:
            self.create_text(cx, cy+9, text=self._label,
                             fill=TEXT3, font=("Courier New", 7))

    def _tick(self):
        """Animation tick: ease the current value toward the target."""
        if abs(self._current - self._target) > 0.002:
            self._current += (self._target - self._current) * 0.09
        else:
            self._current = self._target
        self._draw()
        self.after(30, self._tick)   # ~33 fps


# ══════════════════════════════════════════════════════════════
#  WIDGET: ANIMATED VERTICAL THREAT BAR
#  Fills upward from green → amber → red based on a 0–1 threat
#  level. Sits in the sidebar as an at-a-glance risk indicator.
# ══════════════════════════════════════════════════════════════
class ThreatBar(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, width=30, bg=parent["bg"],
                         highlightthickness=0, **kw)
        self._level  = 0.0   # Currently displayed level
        self._target = 0.0   # Target level
        self.bind("<Configure>", lambda e: self._draw())
        self._draw()
        self._tick()

    def set_level(self, v: float):
        """Set the target threat level (0.0–1.0)."""
        self._target = max(0.0, min(1.0, v))

    def _draw(self):
        """Redraw the bar for the current animation frame."""
        self.delete("all")
        w = 30
        h = self.winfo_height() or 120

        # Dark grey track
        self.create_rectangle(10, 4, 20, h-4, fill=BG2, outline="")

        # Filled portion — colour changes with level
        fh = int((h-8) * self._level)
        if fh > 0:
            col = (RED   if self._level > 0.7  else
                   AMBER if self._level > 0.35 else GREEN)
            self.create_rectangle(10, h-4-fh, 20, h-4, fill=col, outline="")

        # Tick marks at 25%, 50%, 75%
        for pct in [0.25, 0.5, 0.75]:
            y = h - 4 - int((h-8) * pct)
            self.create_line(8, y, 22, y, fill=TEXT3, width=1)

    def _tick(self):
        """Animation tick: ease the current level toward the target."""
        if abs(self._level - self._target) > 0.002:
            self._level += (self._target - self._level) * 0.07
        self._draw()
        self.after(40, self._tick)


# ══════════════════════════════════════════════════════════════
#  WIDGET: PULSING DOT BADGE
#  A small circle that pulses when active — used as a status
#  indicator next to the "IDLE / CRITICAL" label in the sidebar.
# ══════════════════════════════════════════════════════════════
class PulseBadge(tk.Canvas):
    def __init__(self, parent, color=RED, size=12, **kw):
        super().__init__(parent, width=size, height=size,
                         bg=parent["bg"], highlightthickness=0, **kw)
        self._color  = color
        self._size   = size
        self._phase  = 0.0     # Current position in the sine wave
        self._active = False
        self._tick()

    def activate(self, color=None):
        """Start pulsing (optionally change colour)."""
        if color:
            self._color = color
        self._active = True

    def deactivate(self):
        """Stop pulsing and clear the canvas."""
        self._active = False

    def _tick(self):
        """Redraw each frame, varying radius using a sine wave."""
        self.delete("all")
        s = self._size
        if self._active:
            r = int(s/2 * (0.55 + 0.45 * abs(math.sin(self._phase))))
            self.create_oval(s//2-r, s//2-r, s//2+r, s//2+r,
                             fill=self._color, outline="")
            self._phase += 0.14
        self.after(48, self._tick)


# ══════════════════════════════════════════════════════════════
#  WIDGET: SCROLLING LOG
#  Auto-scrolling text widget used in the sidebar to display
#  pipeline progress messages (INFO / OK / WARN / ERR levels).
# ══════════════════════════════════════════════════════════════
class ScrollLog(tk.Frame):
    def __init__(self, parent, height=8, **kw):
        super().__init__(parent, bg=BG0, **kw)

        vsb = tk.Scrollbar(self, orient="vertical", bg=BG1,
                           troughcolor=BG0, width=8)
        vsb.pack(side="right", fill="y")

        self.txt = tk.Text(
            self, font=("Courier New", 8), bg=BG0, fg=TEXT2,
            relief="flat", bd=0, state="disabled",
            height=height, wrap="word",
            yscrollcommand=vsb.set,
            highlightthickness=0, padx=6, pady=4
        )
        self.txt.pack(fill="both", expand=True)
        vsb.config(command=self.txt.yview)

        # Colour tags for each log level
        self.txt.tag_config("ok",   foreground=GREEN)
        self.txt.tag_config("info", foreground=CYAN)
        self.txt.tag_config("warn", foreground=AMBER)
        self.txt.tag_config("err",  foreground=RED)

    def log(self, msg: str, level: str = "info"):
        """Append a timestamped message to the log."""
        ts = datetime.now().strftime("%H:%M:%S")
        self.txt.config(state="normal")
        self.txt.insert("end", f"[{ts}] {msg}\n", level)
        self.txt.see("end")          # Auto-scroll to the latest line
        self.txt.config(state="disabled")


# ══════════════════════════════════════════════════════════════
#  MAIN APPLICATION
#  Subclasses tk.Tk and owns all UI state and the ML pipeline.
# ══════════════════════════════════════════════════════════════
class IDSApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("NETWATCH-IDS  //  UNSW-NB15  //  v3.0")
        self.configure(bg=BG0)
        self.state("zoomed")
        self.minsize(1280, 800)

        # File paths selected by the user
        self.train_path = tk.StringVar()
        self.test_path  = tk.StringVar()

        # Holds all results after a successful scan
        self.results   = None
        self._running  = False   # Prevents concurrent pipeline runs

        # Sidebar counter variables (bound to Labels)
        self._disp_total  = tk.StringVar(value="——")
        self._disp_crit   = tk.StringVar(value="——")
        self._disp_high   = tk.StringVar(value="——")
        self._disp_normal = tk.StringVar(value="——")

        self._build_ui()

    # ──────────────────────────────────────────────────────────
    #  TOP-LEVEL LAYOUT
    #  Title bar (top) → body (fills remaining height)
    #  Body: sidebar (fixed width left) + main content (flexible right)
    # ──────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_titlebar()

        body = tk.Frame(self, bg=BG0)
        body.pack(fill="both", expand=True)

        # Sidebar — fixed 292px width
        self.sidebar = tk.Frame(body, bg=BG1, width=292)
        self.sidebar.pack(side="left", fill="y", padx=(6, 0), pady=6)
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        # Main content area — fills remaining space
        self.main = tk.Frame(body, bg=BG0)
        self.main.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        self._build_main()

    # ──────────────────────────────────────────────────────────
    #  TITLE BAR
    #  Contains the logo, a live-updating status label, and a clock.
    # ──────────────────────────────────────────────────────────
    def _build_titlebar(self):
        bar = tk.Frame(self, bg=BG0, height=54)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        # Logo (left side)
        logo = tk.Frame(bar, bg=BG0)
        logo.place(x=16, rely=0.5, anchor="w")
        tk.Label(logo, text="◈ NETWATCH", font=("Courier New", 16, "bold"),
                 fg=CYAN, bg=BG0).pack(side="left")
        tk.Label(logo, text="-IDS", font=("Courier New", 16, "bold"),
                 fg=GREEN, bg=BG0).pack(side="left")
        tk.Label(logo, text="  //  UNSW-NB15  //  v3.0",
                 font=("Courier New", 9), fg=TEXT3, bg=BG0).pack(
                     side="left", padx=(8, 0))

        # Clock + status (right side)
        right = tk.Frame(bar, bg=BG0)
        right.place(relx=1, rely=0.5, anchor="e", x=-16)

        self._clock_var = tk.StringVar()
        tk.Label(right, textvariable=self._clock_var,
                 font=("Courier New", 9), fg=TEXT3, bg=BG0).pack(side="right")

        self._status_var = tk.StringVar(value="● IDLE")
        self._status_lbl = tk.Label(right, textvariable=self._status_var,
                                    font=("Courier New", 10, "bold"),
                                    fg=TEXT3, bg=BG0)
        self._status_lbl.pack(side="right", padx=(0, 20))

        self._tick_clock()

    def _tick_clock(self):
        """Update the clock label every second."""
        self._clock_var.set(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.after(1000, self._tick_clock)

    def _set_status(self, txt: str, color=TEXT2):
        """Update the status label in the title bar."""
        self._status_var.set(txt)
        self._status_lbl.config(fg=color)
        self.update_idletasks()

    # ──────────────────────────────────────────────────────────
    #  SIDEBAR
    #  From top to bottom:
    #    • Threat bar + level label + pulse badge
    #    • Dataset file selectors
    #    • Summary counters (total, critical, high, normal)
    #    • "Initiate Scan" button
    #    • Scrolling system log
    # ──────────────────────────────────────────────────────────
    def _build_sidebar(self):
        sb = self.sidebar
        p  = dict(padx=12)   # Common horizontal padding shorthand

        # ── Section: System Status ────────────────────────────
        tk.Label(sb, text="SYSTEM STATUS", font=FONT_TINY,
                 fg=TEXT3, bg=BG1, **p).pack(anchor="w", pady=(12, 4))

        # Threat bar (left) + text labels (right)
        trow = tk.Frame(sb, bg=BG1)
        trow.pack(fill="x", **p, pady=(0, 8))

        self._threat_bar = ThreatBar(trow, height=110)
        self._threat_bar.pack(side="left")

        tinfo = tk.Frame(trow, bg=BG1)
        tinfo.pack(side="left", fill="both", expand=True, padx=8)

        tk.Label(tinfo, text="THREAT\nLEVEL", font=FONT_TINY,
                 fg=TEXT3, bg=BG1, justify="left").pack(anchor="w")

        self._threat_lbl = tk.Label(tinfo, text="LOW",
                                     font=("Courier New", 16, "bold"),
                                     fg=GREEN, bg=BG1)
        self._threat_lbl.pack(anchor="w", pady=(4, 0))

        # Pulse badge + status text
        badge_row = tk.Frame(tinfo, bg=BG1)
        badge_row.pack(anchor="w", pady=(6, 0))
        self._pulse = PulseBadge(badge_row, size=12, color=RED)
        self._pulse.pack(side="left")
        self._pulse_lbl = tk.Label(badge_row, text=" IDLE",
                                    font=FONT_TINY, fg=TEXT3, bg=BG1)
        self._pulse_lbl.pack(side="left")

        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x", **p, pady=8)

        # ── Section: Dataset file selectors ───────────────────
        tk.Label(sb, text="01 / DATASET", font=FONT_TINY,
                 fg=TEXT3, bg=BG1, **p).pack(anchor="w", pady=(0, 6))
        self._file_row(sb, "Training CSV", self.train_path)
        self._file_row(sb, "Testing  CSV", self.test_path)

        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x", **p, pady=8)

        # ── Section: Detection summary counters ───────────────
        tk.Label(sb, text="02 / DETECTION SUMMARY", font=FONT_TINY,
                 fg=TEXT3, bg=BG1, **p).pack(anchor="w", pady=(0, 6))

        grid = tk.Frame(sb, bg=BG1)
        grid.pack(fill="x", **p)
        for row, (label, var, col) in enumerate([
            ("TOTAL LOGS",     self._disp_total,  CYAN),
            ("CRITICAL",       self._disp_crit,   RED),
            ("HIGH",           self._disp_high,   AMBER),
            ("NORMAL TRAFFIC", self._disp_normal, GREEN),
        ]):
            tk.Label(grid, text=label, font=FONT_TINY,
                     fg=TEXT3, bg=BG1, anchor="w").grid(
                         row=row, column=0, sticky="w", pady=3)
            tk.Label(grid, textvariable=var,
                     font=("Courier New", 13, "bold"),
                     fg=col, bg=BG1, anchor="e").grid(
                         row=row, column=1, sticky="e", pady=3)
        grid.columnconfigure(1, weight=1)

        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x", **p, pady=8)

        # ── Run button ────────────────────────────────────────
        self.run_btn = tk.Button(
            sb, text="▶  INITIATE SCAN",
            font=("Courier New", 11, "bold"),
            bg=CYAN, fg=BG0, relief="flat", bd=0,
            cursor="hand2",
            activebackground="#00C8D4", activeforeground=BG0,
            command=self._run_thread
        )
        self.run_btn.pack(fill="x", **p, ipady=10, pady=(0, 4))

        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x", **p, pady=8)

        # ── System log ────────────────────────────────────────
        tk.Label(sb, text="03 / SYSTEM LOG", font=FONT_TINY,
                 fg=TEXT3, bg=BG1, **p).pack(anchor="w", pady=(0, 4))
        self.log = ScrollLog(sb, height=10)
        self.log.pack(fill="both", expand=True, **p, pady=(0, 12))

    def _file_row(self, parent, label: str, var: tk.StringVar):
        """Helper: create a labelled file-path entry + browse button."""
        tk.Label(parent, text=label, font=FONT_TINY,
                 fg=TEXT2, bg=BG1, anchor="w").pack(
                     fill="x", padx=12, pady=(4, 1))
        row = tk.Frame(parent, bg=BG1)
        row.pack(fill="x", padx=12)

        # Text entry bound to the StringVar
        e = tk.Entry(row, textvariable=var, font=FONT_TINY,
                     bg=BG2, fg=TEXT1, insertbackground=CYAN,
                     relief="flat", bd=0, highlightthickness=1,
                     highlightbackground=BORDER, highlightcolor=CYAN)
        e.pack(side="left", fill="x", expand=True, ipady=5)

        # "…" browse button opens a file-picker dialog
        tk.Button(row, text="…", font=FONT_TINY, bg=BG3, fg=CYAN,
                  relief="flat", bd=0, cursor="hand2", padx=6,
                  activebackground=BORDER, activeforeground=CYAN,
                  command=lambda v=var: self._browse(v)
                  ).pack(side="right", ipady=5, padx=(2, 0))

    def _browse(self, var: tk.StringVar):
        """Open a file-picker and store the selected path in var."""
        p = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            var.set(p)

    # ──────────────────────────────────────────────────────────
    #  MAIN NOTEBOOK (5 tabs)
    #    1. Dashboard  — gauges + model comparison + severity chart
    #    2. Confusion Matrix
    #    3. ROC Curve
    #    4. Feature Map
    #    5. Log Feed   — filterable alert table
    # ──────────────────────────────────────────────────────────
    def _build_main(self):
        # Style the notebook to match the dark theme
        st = ttk.Style(self)
        st.theme_use("default")
        st.configure("IDS.TNotebook",
                      background=BG0, borderwidth=0, tabmargins=0)
        st.configure("IDS.TNotebook.Tab",
                      background=BG1, foreground=TEXT3,
                      font=("Courier New", 10, "bold"),
                      padding=[16, 7], borderwidth=0)
        st.map("IDS.TNotebook.Tab",
               background=[("selected", BG2)],
               foreground=[("selected", CYAN)])

        nb = ttk.Notebook(self.main, style="IDS.TNotebook")
        nb.pack(fill="both", expand=True)

        # Create tab frames
        self.t_dash      = tk.Frame(nb, bg=BG0)
        self.t_confusion = tk.Frame(nb, bg=BG0)
        self.t_roc       = tk.Frame(nb, bg=BG0)
        self.t_features  = tk.Frame(nb, bg=BG0)
        self.t_logs      = tk.Frame(nb, bg=BG0)

        nb.add(self.t_dash,      text="  DASHBOARD  ")
        nb.add(self.t_confusion, text="  CONFUSION MATRIX  ")
        nb.add(self.t_roc,       text="  ROC CURVE  ")
        nb.add(self.t_features,  text="  FEATURE MAP  ")
        nb.add(self.t_logs,      text="  LOG FEED  ")

        # Build static tabs immediately
        self._build_dashboard_tab()
        self._build_logs_tab()

        # Chart tabs show a placeholder until a scan completes
        self._placeholder(self.t_confusion, "Run scan to view Confusion Matrix")
        self._placeholder(self.t_roc,       "Run scan to view ROC Curve")
        self._placeholder(self.t_features,  "Run scan to view Feature Importance")

    def _placeholder(self, frame, msg: str):
        """Fill a frame with a gridded placeholder message canvas."""
        c = tk.Canvas(frame, bg=BG0, highlightthickness=0)
        c.place(relwidth=1, relheight=1)

        def _draw(e=None):
            c.delete("all")
            w = c.winfo_width()  or 800
            h = c.winfo_height() or 500
            # Subtle grid lines
            for x in range(0, w, 50):
                c.create_line(x, 0, x, h, fill="#0A1828", width=1)
            for y in range(0, h, 50):
                c.create_line(0, y, w, y, fill="#0A1828", width=1)
            c.create_text(w//2, h//2,    text=msg, fill=TEXT3,
                          font=("Courier New", 13))
            c.create_text(w//2, h//2+26, text="◈", fill=BORDER,
                          font=("Courier New", 18))

        c.bind("<Configure>", _draw)

    # ──────────────────────────────────────────────────────────
    #  DASHBOARD TAB
    #  Row 1: Six metric donut gauges (accuracy, precision, etc.)
    #  Row 2: Model comparison table (left) | Severity chart (right)
    # ──────────────────────────────────────────────────────────
    def _build_dashboard_tab(self):
        f = self.t_dash

        # ── Row 1: Metric gauges ──────────────────────────────
        r1 = tk.Frame(f, bg=BG0)
        r1.pack(fill="x", padx=8, pady=(8, 4))

        # Dict of { metric_key: (DonutGauge, StringVar) }
        self._gauges = {}

        for col, (key, label, color) in enumerate([
            ("accuracy",  "ACCURACY",  GREEN),
            ("precision", "PRECISION", CYAN),
            ("recall",    "RECALL",    CYAN),
            ("f1",        "F1 SCORE",  GREEN),
            ("roc_auc",   "ROC AUC",   PURPLE),
            ("fpr",       "FALSE POS", RED),
        ]):
            card = tk.Frame(r1, bg=BG1, highlightthickness=1,
                             highlightbackground=BORDER)
            card.grid(row=0, column=col, padx=4, sticky="nsew")
            r1.columnconfigure(col, weight=1)

            # Coloured left-edge accent strip
            tk.Frame(card, bg=color, width=3).pack(side="left", fill="y")

            inner = tk.Frame(card, bg=BG1)
            inner.pack(fill="both", expand=True, padx=(4, 0), pady=8)

            tk.Label(inner, text=label, font=FONT_TINY,
                     fg=TEXT3, bg=BG1).pack()

            g = DonutGauge(inner, color=color, size=100)
            g.pack(pady=2)

            val_var = tk.StringVar(value="—")
            tk.Label(inner, textvariable=val_var,
                     font=("Courier New", 9), fg=color, bg=BG1).pack()

            self._gauges[key] = (g, val_var)

        # ── Row 2: Two-column card row ─────────────────────────
        r2 = tk.Frame(f, bg=BG0)
        r2.pack(fill="both", expand=True, padx=8, pady=4)
        r2.columnconfigure(0, weight=1)
        r2.columnconfigure(1, weight=1)

        # Left card: RF vs DT comparison table
        lc = tk.Frame(r2, bg=BG1, highlightthickness=1,
                       highlightbackground=BORDER)
        lc.grid(row=0, column=0, padx=(0, 4), sticky="nsew")

        tk.Label(lc, text="MODEL COMPARISON",
                 font=FONT_HEAD, fg=CYAN, bg=BG1).pack(
                     anchor="w", padx=12, pady=(10, 4))
        tk.Frame(lc, bg=BORDER, height=1).pack(fill="x", padx=12)

        # Style the comparison Treeview
        st2 = ttk.Style()
        st2.configure("Cmp.Treeview",
                       background=BG1, foreground=TEXT1,
                       rowheight=26, fieldbackground=BG1,
                       font=("Courier New", 9), borderwidth=0)
        st2.configure("Cmp.Treeview.Heading",
                       background=BG2, foreground=CYAN,
                       font=("Courier New", 9, "bold"), relief="flat")
        st2.map("Cmp.Treeview",
                background=[("selected", BG3)],
                foreground=[("selected", CYAN)])

        self.cmp_tree = ttk.Treeview(
            lc, columns=("metric", "rf", "dt"),
            show="headings", style="Cmp.Treeview", height=10
        )
        for cid, txt, w in [("metric", "METRIC", 120),
                              ("rf",     "RANDOM FOREST", 130),
                              ("dt",     "DECISION TREE", 130)]:
            self.cmp_tree.heading(cid, text=txt)
            self.cmp_tree.column(cid, width=w, anchor="center")

        # Row colour tags: green = RF wins, red = RF loses
        self.cmp_tree.tag_configure("better", foreground=GREEN)
        self.cmp_tree.tag_configure("worse",  foreground=RED)
        self.cmp_tree.tag_configure("even",   foreground=TEXT1)
        self.cmp_tree.pack(fill="both", expand=True, padx=4, pady=8)

        # Right card: Severity breakdown chart (populated after scan)
        rc = tk.Frame(r2, bg=BG1, highlightthickness=1,
                       highlightbackground=BORDER)
        rc.grid(row=0, column=1, padx=(4, 0), sticky="nsew")

        tk.Label(rc, text="SEVERITY BREAKDOWN",
                 font=FONT_HEAD, fg=CYAN, bg=BG1).pack(
                     anchor="w", padx=12, pady=(10, 4))
        tk.Frame(rc, bg=BORDER, height=1).pack(fill="x", padx=12)

        self.sev_chart_frame = tk.Frame(rc, bg=BG1)
        self.sev_chart_frame.pack(fill="both", expand=True)
        self._placeholder(self.sev_chart_frame, "No data yet")

    # ──────────────────────────────────────────────────────────
    #  LOG FEED TAB
    #  Filterable table of all predicted attacks. Rows are colour-
    #  coded by severity. Severity radio buttons filter the table.
    # ──────────────────────────────────────────────────────────
    def _build_logs_tab(self):
        f = self.t_logs

        # Header bar with severity filter radio buttons
        hdr = tk.Frame(f, bg=BG1, pady=8)
        hdr.pack(fill="x")
        tk.Label(hdr, text="LOG FEED",
                 font=FONT_HEAD, fg=CYAN, bg=BG1).pack(
                     side="left", padx=(14, 16))
        tk.Frame(hdr, bg=BORDER, width=1).pack(side="left", fill="y", padx=4)

        self.sev_filter = tk.StringVar(value="ALL")
        for sev in ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            col = SEV_COL.get(sev, TEXT2)
            tk.Radiobutton(
                hdr, text=sev, variable=self.sev_filter, value=sev,
                font=("Courier New", 9, "bold"), fg=col, bg=BG1,
                selectcolor=BG2, activebackground=BG1, activeforeground=col,
                relief="flat", command=self._refresh_logs
            ).pack(side="left", padx=6)

        # Counter label showing how many rows are visible
        self.log_count_lbl = tk.Label(hdr, text="",
                                       font=FONT_TINY, fg=TEXT3, bg=BG1)
        self.log_count_lbl.pack(side="right", padx=14)

        # Style the log Treeview
        st3 = ttk.Style()
        st3.configure("log.Treeview",
                       background=BG0, foreground=TEXT1,
                       rowheight=22, fieldbackground=BG0,
                       font=("Courier New", 9), borderwidth=0)
        st3.configure("log.Treeview.Heading",
                       background=BG2, foreground=CYAN,
                       font=("Courier New", 9, "bold"), relief="flat")
        st3.map("log.Treeview",
                background=[("selected", BG3)],
                foreground=[("selected", CYAN)])

        tree_f = tk.Frame(f, bg=BG0)
        tree_f.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        vsb = ttk.Scrollbar(tree_f, orient="vertical")
        vsb.pack(side="right", fill="y")

        a_cols = ("ts", "id", "sev", "conf", "result", "actual")
        self.log_tree = ttk.Treeview(
            tree_f, columns=a_cols,
            show="headings", style="log.Treeview",
            yscrollcommand=vsb.set
        )
        vsb.config(command=self.log_tree.yview)
        self.log_tree.pack(fill="both", expand=True)

        for cid, txt, w in [
            ("ts",     "TIMESTAMP",    150),
            ("id",     "LOG ID",       100),
            ("sev",    "SEVERITY",      90),
            ("conf",   "CONFIDENCE",   100),
            ("result", "RESULT",       130),
            ("actual", "ACTUAL LABEL", 110),
        ]:
            self.log_tree.heading(cid, text=txt)
            self.log_tree.column(cid, width=w, anchor="center")

        # Row colour tags — one per severity level
        for sev, col in SEV_COL.items():
            self.log_tree.tag_configure(sev, foreground=col)

    # ──────────────────────────────────────────────────────────
    #  PIPELINE ENTRY POINT
    #  Spawns a daemon thread so the UI stays responsive.
    # ──────────────────────────────────────────────────────────
    def _run_thread(self):
        if self._running:
            return   # Ignore duplicate clicks while a scan is in progress
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    # ──────────────────────────────────────────────────────────
    #  ML PIPELINE (runs in background thread)
    #
    #  Steps:
    #    1. Validate file paths
    #    2. Load CSV files
    #    3. Preprocess (drop, fill NaN, encode, scale)
    #    4. Train Random Forest + Decision Tree
    #    5. Evaluate both models (metrics, confusion matrix, ROC)
    #    6. Build alert log dataframe
    #    7. Schedule UI update on the main thread
    # ──────────────────────────────────────────────────────────
    def _run_pipeline(self):
        self._running = True
        self.run_btn.config(state="disabled")
        self._set_status("● SCANNING", AMBER)
        self._pulse.activate(AMBER)

        try:
            # ── Step 1: Validate paths ────────────────────────
            tp = self.train_path.get().strip()
            sp = self.test_path.get().strip()
            if not tp or not sp:
                messagebox.showerror(
                    "Missing Files",
                    "Please select both Training and Testing CSV files.")
                return

            # ── Step 2: Load datasets ─────────────────────────
            self.log.log("Loading datasets...", "info")
            train = pd.read_csv(tp, low_memory=False)
            test  = pd.read_csv(sp, low_memory=False)
            self.log.log(
                f"Train {train.shape[0]:,} rows | Test {test.shape[0]:,} rows",
                "ok")

            # ── Step 3: Preprocess ────────────────────────────
            # Remove constant columns (zero variance — useless for training)
            const = [c for c in train.columns if train[c].nunique() == 1]
            drops = list(set(DROP_COLS + const))

            for df in [train, test]:
                df.drop_duplicates(inplace=True)
                df.drop(columns=[c for c in drops if c in df.columns],
                        inplace=True)
                # Fill missing numeric values with the column median
                df.fillna(df.median(numeric_only=True), inplace=True)

            # Label-encode categorical columns
            # (test set maps unseen categories to -1)
            for col in CATEGORICAL_COLS:
                if col not in train.columns:
                    continue
                le = LabelEncoder()
                train[col] = le.fit_transform(train[col].astype(str))
                test[col]  = test[col].astype(str).apply(
                    lambda x: le.transform([x])[0]
                    if x in le.classes_ else -1
                )

            X_tr = train.drop(columns=[TARGET])
            y_tr = train[TARGET].values
            X_te = test.drop(columns=[TARGET])
            y_te = test[TARGET].values
            feat = list(X_tr.columns)    # Feature names (for importance chart)

            # Standardise features to zero-mean, unit-variance
            sc   = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)
            self.log.log("Preprocessing complete.", "ok")

            # ── Step 4: Train models ──────────────────────────
            self._set_status("● TRAINING RF", CYAN)
            self.log.log("Training Random Forest...", "info")
            rf = RandomForestClassifier(**RF_PARAMS)
            rf.fit(X_tr, y_tr)
            self.log.log("Random Forest trained.", "ok")

            self._set_status("● TRAINING DT", CYAN)
            self.log.log("Training Decision Tree...", "info")
            dt = DecisionTreeClassifier(**DT_PARAMS)
            dt.fit(X_tr, y_tr)
            self.log.log("Decision Tree trained.", "ok")

            # ── Step 5: Evaluate models ───────────────────────
            def get_metrics(model, name: str) -> dict:
                """Compute all evaluation metrics for a fitted model."""
                y_pred  = model.predict(X_te)
                y_proba = model.predict_proba(X_te)[:, 1]
                cm      = confusion_matrix(y_te, y_pred)
                tn, fp, fn, tp_ = cm.ravel()
                fpr_arr, tpr_arr, _ = roc_curve(y_te, y_proba)
                return {
                    "name"     : name,
                    "y_pred"   : y_pred,
                    "y_proba"  : y_proba,
                    "accuracy" : accuracy_score(y_te, y_pred),
                    "precision": precision_score(y_te, y_pred, zero_division=0),
                    "recall"   : recall_score(y_te, y_pred, zero_division=0),
                    "f1"       : f1_score(y_te, y_pred, zero_division=0),
                    "fpr_val"  : fp / (fp + tn) if (fp + tn) > 0 else 0,
                    "roc_auc"  : auc(fpr_arr, tpr_arr),
                    "fpr_arr"  : fpr_arr,
                    "tpr_arr"  : tpr_arr,
                    "tp": tp_, "tn": tn, "fp": fp, "fn": fn,
                    "cm": cm,
                }

            self._set_status("● EVALUATING", CYAN)
            rf_m = get_metrics(rf, "Random Forest")
            dt_m = get_metrics(dt, "Decision Tree")
            self.log.log("Metrics computed.", "ok")

            # ── Step 6: Build alert log dataframe ─────────────
            # Only rows predicted as attacks (label = 1) become alerts.
            self.log.log("Generating alert logs...", "info")
            y_pred  = rf_m["y_pred"]
            y_proba = rf_m["y_proba"]
            attack_indices = np.where(y_pred == 1)[0]

            rows = []
            base_time = datetime.now()
            for i, idx in enumerate(attack_indices):
                conf = y_proba[idx]
                ts   = base_time - timedelta(seconds=random.randint(0, 86400))
                rows.append({
                    "timestamp"   : ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "log_id"      : f"IDS-{i+1:06d}",
                    "severity"    : assign_sev(conf),
                    "confidence"  : round(float(conf), 4),
                    "result"      : ("True Positive"
                                     if int(y_te[idx]) == 1
                                     else "False Positive"),
                    "actual_label": ("ATTACK" if int(y_te[idx]) == 1
                                     else "NORMAL"),
                })

            adf = pd.DataFrame(rows)
            if not adf.empty:
                # Sort by severity (most critical first), then confidence (desc)
                adf["_r"] = adf["severity"].map(SEV_ORD)
                adf = adf.sort_values(["_r", "confidence"],
                                       ascending=[True, False])
                adf.drop(columns=["_r"], inplace=True)
                adf.reset_index(drop=True, inplace=True)

            self.log.log(f"Alert logs: {len(adf):,} generated.", "ok")

            # Store results and update UI on the main thread
            self.results = {
                "rf_m": rf_m, "dt_m": dt_m,
                "log_df": adf, "feat": feat, "rf": rf
            }
            self.after(0, self._update_ui)

        except Exception as ex:
            self.log.log(f"ERROR: {ex}", "err")
            messagebox.showerror("Pipeline Error", str(ex))
            self._set_status("● ERROR", RED)
        finally:
            self._running = False
            self.run_btn.config(state="normal")

    # ──────────────────────────────────────────────────────────
    #  UI UPDATE (called on the main thread after pipeline completes)
    #  Populates gauges, comparison table, all charts, and the log tab.
    # ──────────────────────────────────────────────────────────
    def _update_ui(self):
        rf_m = self.results["rf_m"]
        dt_m = self.results["dt_m"]
        adf  = self.results["log_df"]
        feat = self.results["feat"]
        rf   = self.results["rf"]

        # ── Update donut gauges ───────────────────────────────
        gauge_map = {
            "accuracy" : (rf_m["accuracy"],  lambda v: f"{v*100:.1f}%"),
            "precision": (rf_m["precision"], lambda v: f"{v*100:.1f}%"),
            "recall"   : (rf_m["recall"],    lambda v: f"{v*100:.1f}%"),
            "f1"       : (rf_m["f1"],        lambda v: f"{v*100:.1f}%"),
            "roc_auc"  : (rf_m["roc_auc"],   lambda v: f"{v:.4f}"),
            "fpr"      : (rf_m["fpr_val"],   lambda v: f"{v*100:.1f}%"),
        }
        for key, (val, fmt) in gauge_map.items():
            g, var = self._gauges[key]
            g.set_value(val)
            var.set(fmt(val))

        # ── Update sidebar counters ───────────────────────────
        crit = len(adf[adf["severity"] == "CRITICAL"])
        high = len(adf[adf["severity"] == "HIGH"])
        self._disp_total.set(f"{len(adf):,}")
        self._disp_crit.set(f"{crit:,}")
        self._disp_high.set(f"{high:,}")
        self._disp_normal.set(f"{rf_m['tn']:,}")

        # ── Update threat bar ─────────────────────────────────
        # Weighted score: criticals count more than highs
        threat = min(1.0, (crit * 0.9 + high * 0.4) / max(1, len(adf)))
        self._threat_bar.set_level(threat)
        lvl, col = (
            ("CRITICAL", RED)   if threat > 0.6 else
            ("ELEVATED", AMBER) if threat > 0.3 else
            ("LOW",      GREEN)
        )
        self._threat_lbl.config(text=lvl, fg=col)

        if crit > 0:
            self._pulse.activate(RED)
            self._pulse_lbl.config(text=f" {crit} CRITICAL", fg=RED)
        else:
            self._pulse.activate(GREEN)
            self._pulse_lbl.config(text=" MONITORING", fg=GREEN)

        # ── Populate model comparison table ───────────────────
        for row in self.cmp_tree.get_children():
            self.cmp_tree.delete(row)

        for metric, rv, dv, higher_is_better in [
            ("Accuracy",  rf_m["accuracy"],  dt_m["accuracy"],  True),
            ("Precision", rf_m["precision"], dt_m["precision"], True),
            ("Recall",    rf_m["recall"],    dt_m["recall"],    True),
            ("F1 Score",  rf_m["f1"],        dt_m["f1"],        True),
            ("ROC AUC",   rf_m["roc_auc"],   dt_m["roc_auc"],   True),
            ("FPR",       rf_m["fpr_val"],   dt_m["fpr_val"],   False),  # Lower FPR is better
        ]:
            # Tag rows green if RF is better, red if worse
            tag = ("better"
                   if (rv >= dv and higher_is_better)
                   or (rv <= dv and not higher_is_better)
                   else "worse")
            self.cmp_tree.insert("", "end",
                values=(metric, f"{rv*100:.2f}%", f"{dv*100:.2f}%"),
                tags=(tag,))

        # ── Draw all charts ───────────────────────────────────
        self._draw_severity_chart(adf)
        self._draw_confusion(rf_m, dt_m)
        self._draw_roc(rf_m, dt_m)
        self._draw_features(rf, feat)
        self._refresh_logs()

        self._set_status("● SCAN COMPLETE", GREEN)
        self.log.log("All steps complete. Results ready.", "ok")

    # ──────────────────────────────────────────────────────────
    #  CHART HELPERS
    # ──────────────────────────────────────────────────────────
    def _mpl_fig(self, w=9, h=5) -> Figure:
        """Create a matplotlib Figure with the dark background."""
        fig = Figure(figsize=(w, h))
        fig.patch.set_facecolor(BG0)
        return fig

    def _embed(self, fig: Figure, frame):
        """Destroy any previous chart in frame and embed a new Figure."""
        for w in frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _ax_style(self, ax, title="", xlabel="", ylabel=""):
        """Apply the dark cyberpunk style to a matplotlib Axes."""
        ax.set_facecolor(BG1)
        ax.tick_params(colors=TEXT2, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        if title:
            ax.set_title(title, color=TEXT1, fontsize=10,
                          fontweight="bold", fontfamily="Courier New", pad=8)
        if xlabel:
            ax.set_xlabel(xlabel, color=TEXT2, fontsize=8)
        if ylabel:
            ax.set_ylabel(ylabel, color=TEXT2, fontsize=8)
        ax.grid(True, alpha=0.12, color=BORDER)

    # ── Chart: Severity breakdown ──────────────────────────────
    def _draw_severity_chart(self, adf):
        """Draw a donut pie + horizontal bar showing alerts by severity."""
        fig = self._mpl_fig(5, 3.6)
        gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32,
                                left=0.06, right=0.97,
                                top=0.86, bottom=0.12)

        labels_s = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        sizes    = [len(adf[adf["severity"] == s]) for s in labels_s]
        colors   = [RED, AMBER, BLUE, GREEN]

        # Donut chart (left panel)
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor(BG1)
        valid = [(s, c, l) for s, c, l in zip(sizes, colors, labels_s) if s > 0]
        if valid:
            sv, cv, lv = zip(*valid)
            wedges, _ = ax1.pie(
                sv, colors=cv, startangle=90,
                wedgeprops=dict(width=0.44, edgecolor=BG0, linewidth=2),
            )
            ax1.legend(wedges, lv, loc="lower center", ncol=2,
                        fontsize=7, facecolor=BG2, edgecolor=BORDER,
                        labelcolor=TEXT1,
                        prop={"family": "Courier New", "size": 7})
        ax1.set_title("BY SEVERITY", color=TEXT2,
                       fontsize=8, fontfamily="Courier New")

        # Horizontal bar chart (right panel)
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor(BG1)
        for sp in ax2.spines.values():
            sp.set_edgecolor(BORDER)
        ax2.tick_params(colors=TEXT2, labelsize=7)
        ax2.grid(True, axis="x", alpha=0.12, color=BORDER)
        ax2.set_title("COUNT", color=TEXT2,
                       fontsize=8, fontfamily="Courier New")

        ypos = range(len(labels_s))
        bars = ax2.barh(ypos, sizes, color=colors,
                         edgecolor=BG0, linewidth=0.5, height=0.55)
        ax2.set_yticks(list(ypos))
        ax2.set_yticklabels(labels_s, fontfamily="Courier New",
                             fontsize=7, color=TEXT1)

        # Annotate each bar with the count value
        mx = max(sizes) if max(sizes) > 0 else 1
        for bar, val in zip(bars, sizes):
            if val > 0:
                ax2.text(bar.get_width() + mx * 0.02,
                          bar.get_y() + bar.get_height() / 2,
                          f"{val:,}", va="center", fontsize=7,
                          color=TEXT2, fontfamily="Courier New")

        self._embed(fig, self.sev_chart_frame)
        plt.close(fig)

    # ── Chart: Confusion Matrix ────────────────────────────────
    def _draw_confusion(self, rf_m, dt_m):
        """Draw side-by-side confusion matrices for RF (cyan) and DT (purple)."""
        fig = self._mpl_fig(11, 5)
        gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                                left=0.06, right=0.97,
                                top=0.88, bottom=0.10)

        for i, (m, tcol) in enumerate([(rf_m, CYAN), (dt_m, PURPLE)]):
            ax  = fig.add_subplot(gs[i])
            cm  = m["cm"]
            ax.set_facecolor(BG1)

            # Custom colour map from dark background to the model's accent colour
            cmap = LinearSegmentedColormap.from_list("ids", [BG1, tcol], N=256)
            ax.imshow(cm, cmap=cmap, aspect="auto", vmin=0)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Normal", "Attack"], color=TEXT1,
                                fontsize=9, fontfamily="Courier New")
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Normal", "Attack"], color=TEXT1,
                                fontsize=9, rotation=90, va="center",
                                fontfamily="Courier New")
            ax.tick_params(colors=TEXT2)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.set_xlabel("Predicted", color=TEXT2, fontsize=9)
            ax.set_ylabel("Actual",    color=TEXT2, fontsize=9)
            ax.set_title(m["name"], color=tcol, fontweight="bold",
                          fontsize=11, fontfamily="Courier New", pad=10)

            # Annotate each cell with its label (TN/FP/FN/TP) and count
            cell_labels = [["TN", "FP"], ["FN", "TP"]]
            thresh = cm.max() / 2
            for r in range(2):
                for c in range(2):
                    # Use dark text on bright cells, light text on dark cells
                    fc = BG0 if cm[r, c] > thresh else TEXT1
                    ax.text(c, r,
                             f"{cell_labels[r][c]}\n{cm[r,c]:,}",
                             ha="center", va="center",
                             fontsize=11, fontweight="bold",
                             color=fc, fontfamily="Courier New")

        self._embed(fig, self.t_confusion)
        plt.close(fig)

    # ── Chart: ROC Curve ──────────────────────────────────────
    def _draw_roc(self, rf_m, dt_m):
        """Plot ROC curves for both models with AUC annotations."""
        fig = self._mpl_fig(9, 5.5)
        ax  = fig.add_subplot(111)
        fig.patch.set_facecolor(BG0)
        ax.set_facecolor(BG1)

        # Shaded area under each curve
        ax.fill_between(rf_m["fpr_arr"], rf_m["tpr_arr"], alpha=0.08, color=CYAN)
        ax.fill_between(dt_m["fpr_arr"], dt_m["tpr_arr"], alpha=0.05, color=PURPLE)

        # ROC lines
        for m, col, ls, lw in [
            (rf_m, CYAN,   "-",  2.5),
            (dt_m, PURPLE, "--", 2.0),
        ]:
            ax.plot(m["fpr_arr"], m["tpr_arr"],
                     color=col, linestyle=ls, linewidth=lw,
                     label=f"{m['name']}   AUC = {m['roc_auc']:.4f}")

        # Random-guess baseline
        ax.plot([0, 1], [0, 1], color=BORDER, linestyle=":",
                 linewidth=1, label="Random Guess  AUC = 0.50")

        # AUC annotation box
        ax.annotate(f"AUC={rf_m['roc_auc']:.4f}",
                     xy=(0.28, 0.74), color=CYAN,
                     fontsize=9, fontfamily="Courier New",
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor=BG2, edgecolor=CYAN, alpha=0.8))

        self._ax_style(ax,
                       "ROC CURVE  —  Random Forest vs Decision Tree",
                       "False Positive Rate",
                       "True Positive Rate")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.legend(loc="lower right",
                   facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT1,
                   prop={"family": "Courier New", "size": 9})
        fig.tight_layout(pad=2)
        self._embed(fig, self.t_roc)
        plt.close(fig)

    # ── Chart: Feature Importance ──────────────────────────────
    def _draw_features(self, rf, feat, top: int = 20):
        """Horizontal bar chart of the top-N Random Forest feature importances."""
        imp  = rf.feature_importances_
        idxs = np.argsort(imp)[-top:]   # Indices of the top-N features

        # Colour bars by impact tier
        cols = [
            RED   if imp[i] > 0.05 else
            AMBER if imp[i] > 0.02 else
            CYAN
            for i in idxs
        ]

        fig = self._mpl_fig(10, 7)
        ax  = fig.add_subplot(111)
        fig.patch.set_facecolor(BG0)
        ax.set_facecolor(BG1)

        bars = ax.barh(range(top), imp[idxs],
                        color=cols, edgecolor=BG0, linewidth=0.5)
        ax.set_yticks(range(top))
        ax.set_yticklabels([feat[i] for i in idxs],
                            color=TEXT1, fontfamily="Courier New", fontsize=8)

        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.tick_params(colors=TEXT2, labelsize=8)
        ax.grid(True, axis="x", alpha=0.12, color=BORDER)
        ax.set_title(f"TOP {top} FEATURE IMPORTANCES  —  RANDOM FOREST",
                      color=TEXT1, fontweight="bold",
                      fontfamily="Courier New", fontsize=11, pad=10)
        ax.set_xlabel("Gini Importance Score", color=TEXT2, fontsize=9)

        # Value labels at the end of each bar
        for bar, idx in zip(bars, idxs):
            ax.text(bar.get_width() + 0.001,
                     bar.get_y() + bar.get_height() / 2,
                     f"{imp[idx]:.4f}",
                     va="center", fontsize=8,
                     color=TEXT2, fontfamily="Courier New")

        # Legend for impact tiers
        patches = [
            mpatches.Patch(color=RED,   label="> 5%  HIGH IMPACT"),
            mpatches.Patch(color=AMBER, label="2-5%  MEDIUM IMPACT"),
            mpatches.Patch(color=CYAN,  label="< 2%  LOW IMPACT"),
        ]
        ax.legend(handles=patches, loc="lower right",
                   facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT1,
                   prop={"family": "Courier New", "size": 8})
        fig.tight_layout(pad=2)
        self._embed(fig, self.t_features)
        plt.close(fig)

    # ──────────────────────────────────────────────────────────
    #  LOG TABLE REFRESH
    #  Re-populates the Log Feed Treeview based on the current
    #  severity filter selection. Capped at 1,000 visible rows.
    # ──────────────────────────────────────────────────────────
    def _refresh_logs(self):
        if self.results is None:
            return   # Nothing to show before a scan

        adf  = self.results["log_df"]
        filt = self.sev_filter.get()

        # Apply the severity filter (or show all rows)
        df = adf if filt == "ALL" else adf[adf["severity"] == filt]

        # Clear existing rows
        for row in self.log_tree.get_children():
            self.log_tree.delete(row)

        # Insert up to 1,000 rows
        for _, row in df.head(1000).iterrows():
            self.log_tree.insert("", "end", values=(
                row["timestamp"],
                row["log_id"],
                row["severity"],
                f"{row['confidence']*100:.1f}%",
                row["result"],
                row["actual_label"],
            ), tags=(row["severity"],))

        self.log_count_lbl.config(
            text=f"Showing {min(len(df), 1000):,} / {len(df):,} logs")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = IDSApp()
    app.mainloop()
