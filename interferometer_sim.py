"""
interferometer_sim.py
---------------------
Toy model of the camera pattern from the three-path PBS interferometer
(paths A, B, C; microscope-slide phase shifters in A and B).

The three beams arrive at the camera side by side along x and slightly tilted
toward each other, so where they overlap they form straight fringes. Each beam
can be steered by the optic that launches it:

    path A  <-  PBS1   (reflected arm)
    path B  <-  PBS3   (reflected arm)
    path C  <-  M2     (right-hand mirror)

For each of those optics you can set
    * a translation (stage) along the incoming beam: moves the beam sideways at
      the camera by the same amount and lengthens that path by the same amount,
    * a horizontal / vertical tilt delta: the beam angle changes by 2*delta and,
      over the lever arm L to the camera, the beam moves by 2*delta*L.

Units: wavelengths in nm, lengths in mm, angles in mrad (optics) or deg (slides).
Fields are scalar; the analyzer is assumed to project all three paths onto the
same polarization with the amplitudes given in `Setup.amp`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

PATHS = ("A", "B", "C")
LAUNCH_OPTIC = {"A": "PBS1", "B": "PBS3", "C": "M2"}


# ----------------------------------------------------------------- slides
def n_glass(lam_nm: float, A: float = 1.5046, B: float = 0.00420) -> float:
    """Cauchy index (BK7-like; soda-lime slides are within ~0.01)."""
    l_um = lam_nm / 1000.0
    return A + B / l_um**2


def slide_phase(theta_deg: float, lam_nm: float, t_mm: float) -> float:
    """Phase (rad) added by a tilted plate of thickness t relative to air."""
    th = np.deg2rad(theta_deg)
    n = n_glass(lam_nm)
    s = np.sin(th)
    return 2 * np.pi * (t_mm * 1e6 / lam_nm) * (np.sqrt(n**2 - s**2) - np.cos(th))


def slide_walk(theta_deg: float, lam_nm: float, t_mm: float) -> float:
    """Sideways beam displacement (mm) from a tilted plate."""
    th = np.deg2rad(theta_deg)
    n = n_glass(lam_nm)
    s = np.sin(th)
    return t_mm * s * (1 - np.cos(th) / np.sqrt(n**2 - s**2))


def slide_slope(theta_deg: float, lam_nm: float, t_mm: float) -> float:
    """d(phase)/d(theta) in rad per degree."""
    h = 1e-4
    return (slide_phase(theta_deg + h, lam_nm, t_mm)
            - slide_phase(theta_deg - h, lam_nm, t_mm)) / (2 * h)


# ----------------------------------------------------------------- beams
@dataclass
class Launch:
    """Adjustments on the optic that launches one path."""
    stage_mm: float = 0.0     # translation of PBS1 / PBS3 / M2
    tilt_x_mrad: float = 0.0  # horizontal tilt of that optic
    tilt_y_mrad: float = 0.0  # vertical tilt of that optic
    blocked: bool = False
    drift_rad: float = 0.0    # extra phase (simulated drift)


def _default_launch() -> dict:
    return {"A": Launch(tilt_x_mrad=0.6), "B": Launch(), "C": Launch(tilt_x_mrad=-0.6)}


@dataclass
class Setup:
    lam_nm: float = 810.0
    slide_t_mm: float = 1.0
    theta_A_deg: float = 10.0
    theta_B_deg: float = 12.0
    waist_mm: float = 1.0
    # where each beam lands with stage = 0 and tilt = 0 (mm on the camera)
    base_x_mm: dict = field(default_factory=lambda: {"A": -1.75, "B": -0.07, "C": 1.36})
    # distance from launching optic to camera (mm) - sets how far a tilt moves the beam
    lever_mm: dict = field(default_factory=lambda: {"A": 600.0, "B": 450.0, "C": 300.0})
    amp: dict = field(default_factory=lambda: {"A": 1.0, "B": 1.0, "C": 1.0})
    stage_changes_phase: bool = True
    launch: dict = field(default_factory=_default_launch)

    # -------------------------------------------------------------- per-beam
    def k_per_mm(self) -> float:
        return 2 * np.pi / (self.lam_nm * 1e-6)

    def beam(self, p: str) -> dict:
        """Centre, wave-vector, phase and amplitude of path p at the camera."""
        L = self.launch[p]
        ang_x = 2 * L.tilt_x_mrad * 1e-3          # beam angle (rad)
        ang_y = 2 * L.tilt_y_mrad * 1e-3
        x0 = self.base_x_mm[p] + L.stage_mm + ang_x * self.lever_mm[p]
        y0 = ang_y * self.lever_mm[p]
        phase = L.drift_rad
        if p == "A":
            x0 += slide_walk(self.theta_A_deg, self.lam_nm, self.slide_t_mm)
            phase += slide_phase(self.theta_A_deg, self.lam_nm, self.slide_t_mm)
        elif p == "B":
            x0 += slide_walk(self.theta_B_deg, self.lam_nm, self.slide_t_mm)
            phase += slide_phase(self.theta_B_deg, self.lam_nm, self.slide_t_mm)
        if self.stage_changes_phase:
            phase += 2 * np.pi * (L.stage_mm * 1e6) / self.lam_nm
        k = self.k_per_mm()
        return dict(x0=x0, y0=y0, kx=k * ang_x, ky=k * ang_y,
                    angle_x_mrad=ang_x * 1e3, angle_y_mrad=ang_y * 1e3,
                    phase=phase, amp=self.amp[p], blocked=L.blocked)

    def phases(self) -> dict:
        """Phase of each path, mod 2*pi."""
        return {p: float(np.mod(self.beam(p)["phase"], 2 * np.pi)) for p in PATHS}

    # -------------------------------------------------------------- fields
    def _envelope(self, X, Y, b):
        return b["amp"] * np.exp(-((X - b["x0"])**2 + (Y - b["y0"])**2) / self.waist_mm**2)

    def field(self, X, Y):
        E = np.zeros(np.broadcast(X, Y).shape, dtype=complex)
        for p in PATHS:
            b = self.beam(p)
            if b["blocked"]:
                continue
            E += self._envelope(X, Y, b) * np.exp(1j * (b["kx"] * X + b["ky"] * Y + b["phase"]))
        return E

    def intensity(self, X, Y):
        return np.abs(self.field(X, Y))**2

    def max_intensity(self, X, Y) -> float:
        """Brightest value possible on this grid (all paths open and in phase)."""
        env = sum(self._envelope(X, Y, self.beam(p)) for p in PATHS)
        return float(np.max(env**2))

    def camera(self, fov_mm: float = 3.0, n: int = 241):
        """Return x, y (mm) and intensity on an n x n grid of half-width fov_mm."""
        x = np.linspace(-fov_mm, fov_mm, n)
        X, Y = np.meshgrid(x, x)
        return x, x, self.intensity(X, Y)

    def fringe_periods(self) -> dict:
        """Fringe period (mm) for each open pair, from the angle between beams."""
        out = {}
        open_paths = [p for p in PATHS if not self.launch[p].blocked]
        for i, p in enumerate(open_paths):
            for q in open_paths[i + 1:]:
                bp, bq = self.beam(p), self.beam(q)
                dk = np.hypot(bp["kx"] - bq["kx"], bp["ky"] - bq["ky"])
                out[p + q] = np.inf if dk == 0 else 2 * np.pi / dk
        return out



# ----------------------------------------------------------------- drawing
def draw_setup(ax, s: "Setup") -> None:
    """Draw the optical layout on `ax`, reflecting the current state of `s`
    (slide tilts, blocks, stage and tilt values on PBS1 / PBS3 / M2)."""
    from matplotlib.patches import Rectangle, FancyArrowPatch

    ax.clear()
    ax.set_xlim(10, 1210)
    ax.set_ylim(680, 30)          # SVG-style coordinates: y grows downward
    ax.set_aspect("equal")
    ax.axis("off")

    BEAM, OFF = "#d62728", 0.12
    blk = {p: s.launch[p].blocked for p in PATHS}
    anyopen = not all(blk.values())

    def beam(xs, ys, on=True, lw=3.0):
        ax.plot(xs, ys, color=BEAM, lw=lw, alpha=1.0 if on else OFF,
                solid_capstyle="butt", zorder=1)

    # ---- beams
    beam([140, 820], [580, 580])
    beam([370, 370], [580, 451]); beam([370, 370], [451, 113], not blk["A"])
    beam([370, 1020, 1110], [113, 113, 118], not blk["A"], lw=2)
    beam([580, 580], [580, 451]); beam([580, 580], [451, 120], not blk["B"])
    beam([580, 1110], [120, 120], not blk["B"], lw=2)
    beam([820, 820], [580, 451]); beam([820, 820], [451, 127], not blk["C"])
    beam([820, 1020, 1110], [127, 127, 122], not blk["C"], lw=2)
    beam([1040, 1040], [120, 66], anyopen, lw=1.5)

    # ---- optics helpers
    def cube(cx, cy, label, where="below"):
        ax.add_patch(Rectangle((cx - 20, cy - 20), 40, 40, fc="#e9eef5", ec="#7d8899", lw=1.2, zorder=3))
        ax.plot([cx - 20, cx + 20], [cy + 20, cy - 20], color="#7d8899", lw=1.2, zorder=4)
        if where == "below":
            ax.text(cx, cy + 38, label, ha="center", fontsize=9, zorder=5)
        elif where == "above":
            ax.text(cx, cy - 30, label, ha="center", fontsize=9, zorder=5)
        else:  # upper left, clear of the vertical beam
            ax.text(cx - 26, cy - 26, label, ha="right", fontsize=9, zorder=5)

    def hwp(cx, cy, vertical_beam=False, label=None):
        if vertical_beam:
            ax.add_patch(Rectangle((cx - 19, cy - 4), 38, 8, fc="#1d6b73", ec="none", zorder=3))
        else:
            ax.add_patch(Rectangle((cx - 4, cy - 21), 8, 42, fc="#1d6b73", ec="none", zorder=3))
        if label:
            ax.text(cx, cy - 30, label, ha="center", fontsize=8, color="#555")

    def mirror(cx, cy):
        ax.plot([cx - 16, cx + 16], [cy + 16, cy - 16], color="#222", lw=5, solid_capstyle="round", zorder=4)

    def slide(cx, cy, theta_deg, name):
        th = np.deg2rad(theta_deg)
        dx, dy = 26 * np.cos(th), 26 * np.sin(th)
        ax.plot([cx - dx, cx + dx], [cy + dy, cy - dy], color="#e8c21c", lw=7, solid_capstyle="butt", zorder=4)
        ax.plot([cx - dx, cx + dx], [cy + dy, cy - dy], color="#a88400", lw=0.8, zorder=5)
        ax.text(cx - 40, cy + 4, f"{name}\n{theta_deg:.3f}°", ha="right", va="center", fontsize=8.5)

    def block(cx, p):
        x0 = cx - 18 if blk[p] else cx + 14
        ax.add_patch(Rectangle((x0, 446), 36, 10, fc="#222" if blk[p] else "#bbb", ec="none", zorder=4))
        ax.plot([cx - 18, cx + 50], [463, 463], color="#aaa", lw=1)
        if blk[p]:
            ax.text(cx + 54, 455, "blocked", fontsize=8, color="#222", va="center")

    def adjust_label(cx, cy, p, dx=0, dy=0, ha="center"):
        L = s.launch[p]
        txt = (f"stage {L.stage_mm*1e3:+.1f} µm\n"
               f"tilt x {L.tilt_x_mrad:+.2f}, y {L.tilt_y_mrad:+.2f} mrad")
        ax.text(cx + dx, cy + dy, txt, ha=ha, va="top", fontsize=7.5, color="#0d8774",
                bbox=dict(fc="white", ec="#0d8774", lw=0.8, boxstyle="round,pad=0.3"), zorder=6)

    def stage_arrow(cx, cy):
        ax.add_patch(FancyArrowPatch((cx - 26, cy), (cx + 26, cy), arrowstyle="<|-|>",
                                     mutation_scale=9, color="#0d8774", lw=1.4, zorder=6))

    # ---- source
    ax.add_patch(Rectangle((30, 562), 110, 36, fc="#222", ec="none", zorder=3))
    ax.add_patch(Rectangle((132, 570), 8, 20, fc=BEAM, ec="none", zorder=4))
    ax.text(30, 552, f"Laser {s.lam_nm:.0f} nm", fontsize=9)
    hwp(250, 580, label="HWP")

    # ---- optics
    cube(370, 580, "PBS1", "left"); stage_arrow(370, 612); adjust_label(370, 625, "A", dx=-30, ha="center")
    cube(580, 580, "PBS3", "left"); stage_arrow(580, 612); adjust_label(580, 625, "B", dx=0, ha="center")
    mirror(820, 580); ax.text(850, 590, "M2", fontsize=9)
    stage_arrow(820, 612); adjust_label(820, 625, "C", dx=40, ha="center")
    hwp(475, 580); hwp(700, 580)

    slide(370, 340, s.theta_A_deg, "slide A")
    slide(580, 340, s.theta_B_deg, "slide B")
    block(370, "A"); block(580, "B"); block(820, "C")

    mirror(370, 113); ax.text(345, 98, "M1", fontsize=9, ha="right")
    hwp(470, 116)
    cube(580, 120, "PBS2", "above")
    hwp(700, 120)
    cube(820, 120, "PBS4", "above")
    hwp(900, 120)
    cube(1040, 120, "analyzer PBS", "below")
    ax.add_patch(Rectangle((1028, 50), 24, 16, fc="#222", ec="none")); ax.text(1060, 62, "dump", fontsize=8, color="#555")
    ax.add_patch(Rectangle((1110, 92), 66, 56, fc="#222", ec="none", zorder=3))
    ax.add_patch(plt_circle((1112, 120), 14))
    ax.text(1143, 80, "PI-MAX 4", ha="center", fontsize=9)

    # ---- path labels
    for x, p in [(330, "A"), (540, "B"), (780, "C")]:
        ax.text(x, 250, p, fontsize=18, fontweight="bold", color="#999", ha="center")
    ax.text(870, 215, "A, B, C side by side,\nslightly converging", fontsize=8, color="#555")


def plt_circle(xy, r):
    from matplotlib.patches import Circle
    return Circle(xy, r, fc="#8c96a5", ec="none", zorder=4)

if __name__ == "__main__":
    s = Setup()
    x, y, I = s.camera()
    print("phases mod 2pi:", {k: round(v, 3) for k, v in s.phases().items()})
    print("fringe periods (mm):", {k: round(v, 3) for k, v in s.fringe_periods().items()})
    print("dphi/dtheta A (rad/deg):", round(slide_slope(s.theta_A_deg, s.lam_nm, s.slide_t_mm), 1))
    print("I range:", I.min(), I.max(), "max possible:", s.max_intensity(*np.meshgrid(x, y)))
