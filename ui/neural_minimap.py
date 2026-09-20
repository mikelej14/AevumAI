from __future__ import annotations

import bisect
import math
import queue
import threading
import tkinter as tk
from functools import lru_cache

from compact_engram import header, iter_routes, iter_stored_frames
from .theme import ACCENT, BG, BORDER_SOFT, SURFACE_2, TEXT_DIM, TEXT_MUTED

# The HUD must never become a second decoder workload. For small engrams we can
# afford to fold recurrent-route density into the glow. For larger memories the
# minimap uses the actual stored firing-frame stream only; that is still real
# neural activity, while keeping render preparation bounded regardless of how
# many recurrent transmissions the document contains.
ROUTE_DETAIL_LIMIT = 12_000
MAX_RENDER_FRAMES = 60
COLOR_LEVELS = 24


def _mix(c0: str, c1: str, t: float) -> str:
    t = max(0.0, min(1.0, float(t)))
    a = tuple(int(c0[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(c1[i:i + 2], 16) for i in (1, 3, 5))
    rgb = tuple(round(x + (y - x) * t) for x, y in zip(a, b))
    return '#%02x%02x%02x' % rgb


def _bucketed_firing_frames(blob: bytes, max_frames: int):
    sparse = [(int(t), tuple(int(n) for n in active)) for t, active in iter_stored_frames(blob) if active]
    if not sparse:
        return [], []

    target = max(12, min(MAX_RENDER_FRAMES, int(max_frames)))
    bucket_size = max(1, math.ceil(len(sparse) / target))
    buckets = []
    end_times = []
    for start in range(0, len(sparse), bucket_size):
        group = sparse[start:start + bucket_size]
        counts = [0] * 128
        for _t, active in group:
            for n in active:
                if 0 <= n < 128:
                    counts[n] += 1
        peak = max(counts) if counts else 0
        vals = [0.0] * 128
        if peak:
            denom = math.log1p(peak)
            for n, count in enumerate(counts):
                if count:
                    # Real firing frequency inside this sampled time window.
                    vals[n] = 0.34 + 0.56 * (math.log1p(count) / denom)
        t0 = group[0][0]
        t1 = group[-1][0]
        buckets.append([int(t1), vals, sum(1 for c in counts if c), 0])
        end_times.append(int(t1))
    return buckets, end_times


@lru_cache(maxsize=16)
def _sample_cached(blob: bytes, max_frames: int):
    h = header(blob)
    buckets, end_times = _bucketed_firing_frames(blob, max_frames)
    if not buckets:
        return tuple()

    # Recurrent-route traffic gives an additional real strength cue for short
    # memories. Large memories intentionally skip this pass so the minimap cost
    # never scales with hundreds of thousands/millions of route events.
    if int(h.get('route_count', 0)) <= ROUTE_DETAIL_LIMIT:
        route_counts = [[0] * 128 for _ in buckets]
        route_totals = [0] * len(buckets)
        for s, d, t in iter_routes(blob):
            idx = bisect.bisect_left(end_times, int(t))
            if idx >= len(buckets):
                idx = len(buckets) - 1
            if 0 <= int(s) < 128:
                route_counts[idx][int(s)] += 1
            if 0 <= int(d) < 128:
                route_counts[idx][int(d)] += 1
            route_totals[idx] += 1

        for idx, counts in enumerate(route_counts):
            peak = max(counts) if counts else 0
            if peak:
                denom = math.log1p(peak)
                vals = buckets[idx][1]
                for n, count in enumerate(counts):
                    if count:
                        route_v = 0.94 * (math.log1p(count) / denom)
                        # Blend actual firing with route participation so small
                        # engrams preserve graded route strength instead of
                        # collapsing active neurons to one brightness level.
                        vals[n] = min(1.0, 0.45 * vals[n] + 0.65 * route_v)
            buckets[idx][3] = route_totals[idx]

    return tuple((t, tuple(vals), active, routes) for t, vals, active, routes in buckets)


def sample_activity_strength(blob: bytes, max_frames: int = 48):
    """Return bounded animation frames from actual stored neural evidence.

    The primary intensity signal is real firing frequency in a sampled temporal
    window. Small engrams additionally use recurrent-route density. Large
    engrams deliberately skip the route-density pass so UI cost remains bounded.
    No synthetic firing nodes are introduced.
    """
    if not blob:
        return []
    return list(_sample_cached(bytes(blob), max(12, min(MAX_RENDER_FRAMES, int(max_frames)))))


class NeuralMiniMap(tk.Frame):
    """Small StarCraft-style live neural minimap for the Aevum header.

    Rendering is intentionally decoupled from neural work: preparation happens
    off the Tk thread, incoming animations are coalesced, draw rate is capped,
    and canvas items are reused rather than recreated every frame.
    """

    QUALITY_PROFILES = {
        'smooth': (66, 36),
        'balanced': (50, 48),
        'detailed': (45, 60),
    }

    def __init__(self, parent, *, speed_ms: int = 50, max_frames: int = 48, quality: str | None = None, **kwargs):
        super().__init__(parent, bg=SURFACE_2, highlightthickness=1, highlightbackground=BORDER_SOFT, **kwargs)
        # Protect old configs from accidentally restoring the heavy 29 FPS/90-frame path.
        self.speed_ms = max(45, int(speed_ms))
        self.max_frames = max(24, min(MAX_RENDER_FRAMES, int(max_frames)))
        if quality:
            self.set_quality(quality)
        self._node_pos = []
        self._rings = []
        self._cores = []
        self._heat = [0.0] * 128
        self._last_levels = [-1] * 128
        self._frames = []
        self._i = 0
        self._after = None
        self._prep_q: queue.Queue = queue.Queue(maxsize=2)
        self._ready_q: queue.Queue = queue.Queue(maxsize=2)
        self._sequence = 0
        self._outer_lut = [_mix('#010302', '#1c7d3c', (i / (COLOR_LEVELS - 1)) ** 0.70) for i in range(COLOR_LEVELS)]
        self._core_lut = [_mix('#102619', ACCENT, (i / (COLOR_LEVELS - 1)) ** 0.46) for i in range(COLOR_LEVELS)]

        top = tk.Frame(self, bg=SURFACE_2)
        top.pack(fill='x', padx=7, pady=(4, 1))
        tk.Label(top, text='NEURAL', bg=SURFACE_2, fg=TEXT_DIM,
                 font=('Cascadia Mono', 6)).pack(side='left')
        self.mode = tk.Label(top, text='IDLE', bg=SURFACE_2, fg=TEXT_MUTED,
                             font=('Cascadia Mono', 6))
        self.mode.pack(side='right')
        self.canvas = tk.Canvas(self, width=182, height=55, bg=BG, bd=0, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=5, pady=(0, 4))
        self.canvas.bind('<Configure>', self._redraw)

        threading.Thread(target=self._prep_worker, name='neural-minimap-prep', daemon=True).start()
        self.after(60, self._poll_ready)
        self.after(180, self._idle_decay)


    def set_quality(self, quality: str):
        key = str(quality or 'smooth').strip().lower()
        if key not in self.QUALITY_PROFILES:
            key = 'smooth'
        speed, frames = self.QUALITY_PROFILES[key]
        self.speed_ms = int(speed)
        self.max_frames = int(frames)
        # A quality change should apply to the next neural event without
        # rebuilding the current animation on the Tk thread.
        return key

    def _positions(self, w: int, h: int):
        rows, cols = 8, 16
        mx, my = 5.0, 4.0
        dx = (max(20, w) - 2 * mx) / (cols - 0.5)
        dy = (max(20, h) - 2 * my) / max(1, rows - 1)
        pts = []
        for r in range(rows):
            shift = dx * 0.5 if r % 2 else 0.0
            for c in range(cols):
                pts.append((mx + c * dx + shift, my + r * dy))
        return pts

    def _redraw(self, _event=None):
        w = max(120, self.canvas.winfo_width())
        h = max(44, self.canvas.winfo_height())
        self.canvas.delete('all')
        self._node_pos = self._positions(w, h)
        self._rings = []
        self._cores = []
        self._last_levels = [-1] * 128
        for x, y in self._node_pos:
            ring = self.canvas.create_oval(x - 2.7, y - 2.7, x + 2.7, y + 2.7, fill='#020603', outline='')
            core = self.canvas.create_oval(x - 0.95, y - 0.95, x + 0.95, y + 0.95, fill='#12311d', outline='')
            self._rings.append(ring)
            self._cores.append(core)
        self._paint(self._heat, force=True)

    def _paint(self, strengths, *, force=False):
        if len(self._cores) != 128:
            return
        for i, val in enumerate(strengths):
            v = max(0.0, min(1.0, float(val)))
            level = min(COLOR_LEVELS - 1, int(round(v * (COLOR_LEVELS - 1))))
            if not force and level == self._last_levels[i]:
                continue
            self._last_levels[i] = level
            self.canvas.itemconfigure(self._rings[i], fill=self._outer_lut[level])
            self.canvas.itemconfigure(self._cores[i], fill=self._core_lut[level])

    def play_blob(self, blob: bytes | None, *, mode: str = 'RECALL', label: str = ''):
        if not blob:
            return
        self._sequence += 1
        item = (self._sequence, bytes(blob), str(mode or 'RECALL').upper(), str(label or ''))
        # Keep only the newest pending real neural event. A minimap should show
        # current activity, not replay an ever-growing backlog after a busy turn.
        try:
            self._prep_q.put_nowait(item)
        except queue.Full:
            try:
                self._prep_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._prep_q.put_nowait(item)
            except queue.Full:
                pass

    def _prep_worker(self):
        while True:
            item = self._prep_q.get()
            # Coalesce bursts before doing any CPU work.
            try:
                while True:
                    item = self._prep_q.get_nowait()
            except queue.Empty:
                pass
            seq, blob, mode, label = item
            try:
                frames = sample_activity_strength(blob, self.max_frames)
                ready = (seq, frames, mode, label)
            except Exception as exc:
                ready = (seq, [], 'ERROR', str(exc))
            try:
                self._ready_q.put_nowait(ready)
            except queue.Full:
                try:
                    self._ready_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._ready_q.put_nowait(ready)
                except queue.Full:
                    pass

    def _poll_ready(self):
        newest = None
        try:
            while True:
                newest = self._ready_q.get_nowait()
        except queue.Empty:
            pass
        if newest is not None:
            seq, frames, mode, label = newest
            if seq >= self._sequence:
                self._start(frames, mode, label)
        self.after(60, self._poll_ready)

    def _start(self, frames, mode, label):
        if self._after:
            try:
                self.after_cancel(self._after)
            except Exception:
                pass
            self._after = None
        self._frames = list(frames or [])
        self._i = 0
        self.mode.configure(text=(mode[:9] if mode else 'ACTIVE'), fg=ACCENT if self._frames else TEXT_MUTED)
        if self._frames:
            self._advance()

    def _advance(self):
        if self._i >= len(self._frames):
            self.mode.configure(text='IDLE', fg=TEXT_MUTED)
            self._frames = []
            self._after = None
            return
        _t, vals, _active, _routes = self._frames[self._i]
        # Visual persistence only; it does not invent activity.
        self._heat = [max(float(vals[i]), self._heat[i] * 0.43) for i in range(128)]
        self._paint(self._heat)
        self._i += 1
        self._after = self.after(self.speed_ms, self._advance)

    def _idle_decay(self):
        if not self._frames:
            changed = False
            nxt = []
            for v in self._heat:
                nv = v * 0.78
                if nv > 0.018:
                    changed = True
                else:
                    nv = 0.0
                nxt.append(nv)
            self._heat = nxt
            if changed:
                self._paint(self._heat)
        self.after(180, self._idle_decay)
