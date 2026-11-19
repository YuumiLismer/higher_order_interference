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

Polarization is tracked with Jones calculus. Every half-wave plate has an angle
(fast axis from H = table plane) and every PBS transmits H and reflects V, with
an optional leakage `pbs_leak` (fraction of the wrong polarization's power that
goes the wrong way). The amplitude each path delivers to the camera therefore
follows from the HWP angles; `equal_intensity_angles()` gives the settings that
make all three equal (1/9 of the laser power each, the maximum possible).

    HWP0  input, before PBS1            HWP3  arm C, between PBS3 and M2
    HWP1  between PBS1 and PBS3          HWP4  between PBS2 and PBS4
    HWP2  arm A, between M1 and PBS2     HWP5  between PBS4 and the analyzer PBS

Units: wavelengths in nm, lengths in mm, angles in mrad (optics) or deg (slides,
waveplates).
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


# ----------------------------------------------------------------- polarization
HWP_NAMES = ("HWP0", "HWP1", "HWP2", "HWP3", "HWP4", "HWP5")
HWP_WHERE = {"HWP0": "input", "HWP1": "PBS1 → PBS3", "HWP2": "arm A (top)",
             "HWP3": "arm C (bottom)", "HWP4": "before PBS4", "HWP5": "before analyzer"}


def hwp_matrix(theta_deg: float) -> np.ndarray:
    """Jones matrix of a half-wave plate with fast axis at theta from H."""
    t = np.deg2rad(2 * theta_deg)
    return np.array([[np.cos(t), np.sin(t)], [np.sin(t), -np.cos(t)]])


def _pbs_t(leak: float) -> np.ndarray:   # transmitted port: H passes, V leaks
    return np.diag([1.0, np.sqrt(leak)])


def _pbs_r(leak: float) -> np.ndarray:   # reflected port: V passes, H leaks
    return np.diag([np.sqrt(leak), 1.0])


def jones_paths(hwp_deg: dict, input_pol_deg: float = 0.0, leak: float = 0.0) -> dict:
    """Jones vector (E_H, E_V) that each path delivers to the camera, for unit
    laser amplitude linearly polarized at input_pol_deg from H."""
    W = lambda n: hwp_matrix(hwp_deg[n])
    T, R = _pbs_t(leak), _pbs_r(leak)
    r = np.deg2rad(input_pol_deg)
    E = W("HWP0") @ np.array([np.cos(r), np.sin(r)])
    A, bottom = R @ E, T @ E                      # PBS1: reflected -> A, transmitted -> B/C
    A = T @ (W("HWP2") @ A)                       # arm A enters PBS2 from the left: transmitted
    bottom = W("HWP1") @ bottom
    B, C = R @ bottom, T @ bottom                 # PBS3: reflected -> B, transmitted -> C
    B = R @ B                                     # arm B enters PBS2 from below: reflected
    A, B = (T @ (W("HWP4") @ v) for v in (A, B))  # A+B enter PBS4 from the left: transmitted
    C = R @ (W("HWP3") @ C)                       # arm C enters PBS4 from below: reflected
    return {p: T @ (W("HWP5") @ v) for p, v in (("A", A), ("B", B), ("C", C))}  # analyzer


def equal_intensity_angles(input_pol_deg: float = 0.0) -> dict:
    """HWP angles (deg) giving equal power from A, B, C at the camera with the
    largest possible throughput (1/9 of the laser each), for ideal PBSs.

    HWP0 puts 1/3 into A, HWP1 splits the rest equally into B and C, HWP2 / HWP3
    rotate A and C by 90 deg so PBS2 / PBS4 pass them, HWP4 mixes A and B equally,
    and HWP5 weights (A+B) : C as 2 : 1 at the analyzer."""
    alpha = np.degrees(np.arcsin(1 / np.sqrt(3)))          # 35.26 deg
    return {"HWP0": float((alpha + input_pol_deg) / 2), "HWP1": 22.5, "HWP2": 45.0,
            "HWP3": 45.0, "HWP4": 22.5, "HWP5": float(np.degrees(np.arccos(np.sqrt(2 / 3))) / 2)}


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
    # A and C tilts are deliberately unequal so the AB and BC fringes have different periods
    return {"A": Launch(tilt_x_mrad=0.6), "B": Launch(), "C": Launch(tilt_x_mrad=-1.0)}


@dataclass
class Setup:
    lam_nm: float = 810.0
    slide_t_mm: float = 1.0
    theta_A_deg: float = 10.0
    theta_B_deg: float = 12.0
    waist_mm: float = 1.0
    # where each beam lands with stage = 0 and tilt = 0 (mm on the camera)
    base_x_mm: dict = field(default_factory=lambda: {"A": -1.75, "B": -0.07, "C": 1.6})
    # distance from launching optic to camera (mm) - sets how far a tilt moves the beam
    lever_mm: dict = field(default_factory=lambda: {"A": 600.0, "B": 450.0, "C": 300.0})
    amp: dict = field(default_factory=lambda: {"A": 1.0, "B": 1.0, "C": 1.0})   # extra loss factors
    hwp_deg: dict = field(default_factory=equal_intensity_angles)
    input_pol_deg: float = 0.0   # laser polarization angle from H
    pbs_leak: float = 0.0        # e.g. 1e-3 for a 1000:1 PBS
    stage_changes_phase: bool = True
    launch: dict = field(default_factory=_default_launch)
    # camera: PI-MAX 4 1024 x 1024 (pixel pitch assumed 12.8 um - check your model)
    npix: int = 1024
    pixel_um: float = 12.8

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
        jones = self.amp[p] * jones_paths(self.hwp_deg, self.input_pol_deg, self.pbs_leak)[p]
        return dict(x0=x0, y0=y0, kx=k * ang_x, ky=k * ang_y,
                    angle_x_mrad=ang_x * 1e3, angle_y_mrad=ang_y * 1e3,
                    phase=phase, jones=jones, amp=float(np.linalg.norm(jones)),
                    blocked=L.blocked)

    def camera_powers(self) -> dict:
        """Fraction of the laser power each path delivers to the camera."""
        J = jones_paths(self.hwp_deg, self.input_pol_deg, self.pbs_leak)
        return {p: float(self.amp[p]**2 * np.sum(np.abs(J[p])**2)) for p in PATHS}

    def phases(self) -> dict:
        """Phase of each path, mod 2*pi."""
        return {p: float(np.mod(self.beam(p)["phase"], 2 * np.pi)) for p in PATHS}

    # -------------------------------------------------------------- fields
    def _gauss(self, X, Y, b):
        return np.exp(-((X - b["x0"])**2 + (Y - b["y0"])**2) / self.waist_mm**2)

    def field(self, X, Y):
        """Jones field at the camera, shape (2, *X.shape): (E_H, E_V)."""
        E = np.zeros((2,) + np.broadcast(X, Y).shape, dtype=complex)
        for p in PATHS:
            b = self.beam(p)
            if b["blocked"]:
                continue
            scalar = self._gauss(X, Y, b) * np.exp(1j * (b["kx"] * X + b["ky"] * Y + b["phase"]))
            E += b["jones"].reshape(2, *([1] * scalar.ndim)) * scalar
        return E

    def intensity(self, X, Y):
        return np.sum(np.abs(self.field(X, Y))**2, axis=0)

    def max_intensity(self, X, Y) -> float:
        """Brightest value on this grid for the ideal equal split (each path
        delivering 1/9 of the laser power) with all paths in phase. Used to
        normalise plots, so a badly set waveplate shows up as a dimmer image."""
        env = sum(self._gauss(X, Y, self.beam(p)) / 3 for p in PATHS)
        return float(np.max(env**2))

    # -------------------------------------------------------------- camera
    def mm_to_px(self, v_mm):
        """Camera position in mm (0 = sensor centre) -> pixel index."""
        return np.asarray(v_mm) / (self.pixel_um * 1e-3) + (self.npix - 1) / 2

    def px_to_mm(self, v_px):
        return (np.asarray(v_px) - (self.npix - 1) / 2) * self.pixel_um * 1e-3

    def pixel_window(self, n_view: int | None = None) -> np.ndarray:
        """Pixel indices of a centred n_view-wide window (whole sensor if None)."""
        n = self.npix if n_view is None else min(int(n_view), self.npix)
        i0 = (self.npix - n) // 2
        return np.arange(i0, i0 + n)

    def camera(self, n_view: int | None = None):
        """Intensity on every pixel of a centred n_view x n_view window.
        Returns (pixel indices, intensity[row, col])."""
        px = self.pixel_window(n_view)
        c = self.px_to_mm(px)
        X, Y = np.meshgrid(c, c)
        return px, self.intensity(X, Y)

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

    def fringe_periods_px(self) -> dict:
        """Fringe period in camera pixels for each open pair."""
        return {k: v / (self.pixel_um * 1e-3) for k, v in self.fringe_periods().items()}



# ----------------------------------------------------------------- drawing
def draw_setup(ax, s: "Setup", lock: bool = False) -> None:
    """Draw the optical layout on `ax`, reflecting the current state of `s`
    (slide tilts, blocks, stage and tilt values on PBS1 / PBS3 / M2).
    lock=True adds the phase-compensation hardware: 630 nm reference laser and
    dichroic DM1, short-pass filter blocks, dichroic DM2, two pinholes + photodiodes,
    and the PC + Newport XPS turning the slides."""
    from matplotlib.patches import Rectangle, FancyArrowPatch

    ax.clear()
    ax.set_xlim(10, 1210)
    ax.set_ylim(680, 30)          # SVG-style coordinates: y grows downward
    ax.set_aspect("equal")
    ax.axis("off")

    BEAM, OFF = "#d62728", 0.12
    blk = {p: s.launch[p].blocked for p in PATHS}

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
            ax.text(cx, cy - 25, label, ha="center", va="bottom", fontsize=7.5, color="#1d6b73",
                    linespacing=1.1, zorder=5)

    def mirror(cx, cy):
        ax.plot([cx - 16, cx + 16], [cy + 16, cy - 16], color="#222", lw=5, solid_capstyle="round", zorder=4)

    def slide(cx, cy, theta_deg, name):
        th = np.deg2rad(theta_deg)
        dx, dy = 26 * np.cos(th), 26 * np.sin(th)
        ax.plot([cx - dx, cx + dx], [cy + dy, cy - dy], color="#e8c21c", lw=7, solid_capstyle="butt", zorder=4)
        ax.plot([cx - dx, cx + dx], [cy + dy, cy - dy], color="#a88400", lw=0.8, zorder=5)
        ax.text(cx - 40, cy + 4, f"{name}\n{theta_deg:.3f}°", ha="right", va="center", fontsize=8.5)

    def block(cx, p):
        if not blk[p]:
            return
        if lock:   # short-pass filter: stops 810, passes 630
            ax.add_patch(Rectangle((cx - 18, 446), 36, 10, fc="#7b4fd6", ec="#4b2a99", alpha=0.75, zorder=4))
            ax.text(cx + 26, 455, "blocked (SP filter)", fontsize=8, color="#4b2a99", va="center")
        else:
            ax.add_patch(Rectangle((cx - 18, 446), 36, 10, fc="#222", ec="none", zorder=4))
            ax.text(cx + 26, 455, "blocked", fontsize=8, color="#222", va="center")

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
    hl = {n: f"{n}\n{s.hwp_deg[n]:.2f}°" for n in HWP_NAMES}
    hwp(250, 580, label=hl["HWP0"])

    # ---- optics
    cube(370, 580, "PBS1", "left"); stage_arrow(370, 612); adjust_label(370, 625, "A", dx=-30, ha="center")
    cube(580, 580, "PBS3", "left"); stage_arrow(580, 612); adjust_label(580, 625, "B", dx=0, ha="center")
    mirror(820, 580); ax.text(850, 590, "M2", fontsize=9)
    stage_arrow(820, 612); adjust_label(820, 625, "C", dx=40, ha="center")
    hwp(475, 580, label=hl["HWP1"]); hwp(700, 580, label=hl["HWP3"])

    slide(370, 340, s.theta_A_deg, "slide A")
    slide(580, 340, s.theta_B_deg, "slide B")
    block(370, "A"); block(580, "B"); block(820, "C")

    mirror(370, 113); ax.text(345, 98, "M1", fontsize=9, ha="right")
    hwp(470, 116, label=hl["HWP2"])
    cube(580, 120, "PBS2", "above")
    hwp(700, 120, label=hl["HWP4"])
    cube(820, 120, "PBS4", "above")
    hwp(900, 120, label=hl["HWP5"])
    cube(1040, 120, "analyzer PBS", "below")
    ax.add_patch(Rectangle((1110, 92), 66, 56, fc="#222", ec="none", zorder=3))
    ax.add_patch(plt_circle((1112, 120), 14))
    ax.text(1143, 80, "PI-MAX 4", ha="center", fontsize=9)

    # ---- path labels
    for x, p in [(330, "A"), (540, "B"), (780, "C")]:
        ax.text(x, 250, p, fontsize=18, fontweight="bold", color="#999", ha="center")
    ax.text(870, 215, "A, B, C side by side,\nslightly converging", fontsize=8, color="#555")

    if not lock:
        return
    # ---- phase-compensation hardware
    REF, CTL, NEW = "#ff8c00", "#0d8774", "#0d8774"

    def ref(xs, ys, lw=1.3):
        ax.plot(xs, ys, color=REF, lw=lw, zorder=2, solid_capstyle="butt")

    # reference laser and DM1 (dichroic: passes 810, reflects 630 into the beam line)
    ax.add_patch(Rectangle((183, 640), 24, 44, fc="#222", ec="none", zorder=3))
    ax.add_patch(Rectangle((188, 634), 14, 6, fc=REF, ec="none", zorder=4))
    ax.text(175, 668, "Ref laser\n630 nm", ha="right", va="center", fontsize=8, color=REF)
    ref([195, 195], [634, 580])
    ax.plot([181, 209], [594, 566], color=NEW, lw=4, solid_capstyle="round", zorder=4)
    ax.text(195, 548, "DM1", ha="center", fontsize=8, color=NEW)
    # reference beam follows every path (filter blocks pass 630)
    ref([195, 820], [580, 580])
    ref([370, 370], [580, 113]); ref([580, 580], [580, 120]); ref([820, 820], [580, 127])
    ref([370, 1020, 1080], [113, 113, 118]); ref([580, 1080], [120, 120]); ref([820, 1020, 1080], [127, 127, 122])
    # DM2 after the analyzer: 810 on to the PI-MAX, 630 down to two pinholes + photodiodes
    ax.plot([1068, 1092], [108, 132], color=NEW, lw=4, solid_capstyle="round", zorder=4)
    ax.text(1080, 92, "DM2", ha="center", fontsize=8, color=NEW)
    ref([1080, 1080], [120, 200], lw=1.6)
    for x0, x1 in ((1054, 1069), (1075, 1085), (1091, 1106)):        # pinhole plate, two holes
        ax.plot([x0, x1], [200, 200], color="#222", lw=3, solid_capstyle="butt", zorder=4)
    ref([1072, 1072], [200, 214]); ref([1088, 1088], [200, 214])
    for xc in (1072, 1088):                                          # photodiodes
        ax.add_patch(Rectangle((xc - 6, 214), 12, 10, fc="#222", ec="none", zorder=4))
    ax.text(1112, 200, "pinholes", va="center", fontsize=7.5, color=NEW)
    ax.text(1080, 240, "PD1  PD2", ha="center", va="top", fontsize=8, color=NEW)
    # PC + Newport XPS driving the slide rotation stages
    ax.add_patch(Rectangle((880, 385), 170, 46, fc="#d9f2ec", ec=NEW, lw=1.4, zorder=5))
    ax.text(965, 400, "PC + Newport XPS", ha="center", fontsize=9, fontweight="bold", color=NEW, zorder=6)
    ax.text(965, 418, "hold PD1, PD2 constant", ha="center", fontsize=7, color=NEW, zorder=6)
    dash = dict(color=CTL, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.plot([1080, 1080, 1050], [256, 408, 408], **dash)
    ax.plot([880, 440], [408, 408], **dash)
    for cx, th, axis in ((370, s.theta_A_deg, 1), (580, s.theta_B_deg, 2)):
        r = np.deg2rad(th)
        tip = (cx + 30 * np.cos(r), 340 - 30 * np.sin(r) + 4)       # just right of the slide's end
        ax.plot([cx + 70, cx + 70], [408, 372], **dash)
        ax.add_patch(FancyArrowPatch((cx + 70, 372), tip, arrowstyle="-|>", mutation_scale=12,
                                     color=CTL, lw=1.2, ls=(0, (4, 3)), zorder=6))
        ax.text(cx + 76, 394, f"XPS axis {axis}", fontsize=7, color=CTL, va="center")
    ax.text(30, 380, "red: 810 nm\norange: 630 nm reference\ndashed: feedback via XPS\n\nHWPs achromatic\n"
            "beam blocks: short-pass\nfilters (stop 810, pass 630)", fontsize=8, color="#444", va="top")


# ----------------------------------------------------------------- phase compensation
def wrap(phi):
    """Wrap to (-pi, pi]."""
    return (np.asarray(phi) + np.pi) % (2 * np.pi) - np.pi


def pinhole_intensity(s: "Setup", x_mm, d_um: float = 50.0) -> np.ndarray:
    """Mean intensity through pinholes of diameter d_um centred at x_mm (y = 0)."""
    x_mm = np.atleast_1d(np.asarray(x_mm, dtype=float))
    o = np.array([-0.35, 0.0, 0.35]) * d_um * 1e-3          # 3 x 3 samples inside the hole
    X, Y = np.broadcast_arrays(x_mm[:, None, None] + o[None, :, None], o[None, None, :])
    return s.intensity(X, Y).mean(axis=(1, 2))


def slide_jacobian(s: "Setup", x_mm, d_um: float = 50.0, h_deg: float = 2e-4) -> np.ndarray:
    """dI/dtheta (per degree) at each pinhole for slides A and B. In the lab this is
    measured at start-up by stepping each slide a little and recording the photodiodes."""
    import copy
    J = np.zeros((len(np.atleast_1d(x_mm)), 2))
    for j, attr in enumerate(("theta_A_deg", "theta_B_deg")):
        up, dn = copy.deepcopy(s), copy.deepcopy(s)
        setattr(up, attr, getattr(s, attr) + h_deg); setattr(dn, attr, getattr(s, attr) - h_deg)
        J[:, j] = (pinhole_intensity(up, x_mm, d_um) - pinhole_intensity(dn, x_mm, d_um)) / (2 * h_deg)
    return J


def choose_pinholes(s: "Setup", n: int = 2, half_range_mm: float = 2.5, n_grid: int = 161,
                    d_um: float = 50.0) -> np.ndarray:
    """Pinhole positions (mm, along x at y = 0) for the photodiodes: bright points (at
    least 30 % of the peak) on steep fringe slopes, never near a dark fringe. n = 1: the
    most sensitive point. n = 2: the pair whose sensitivities to the two slides are most
    independent (largest |det| of the 2 x 2 Jacobian), so both phases can be held."""
    x = np.linspace(-half_range_mm, half_range_mm, n_grid)
    I = pinhole_intensity(s, x, d_um)
    J = slide_jacobian(s, x, d_um) / I.max()
    ok = np.where(I > 0.3 * I.max())[0]
    if n == 1:
        return x[[ok[np.argmax(np.linalg.norm(J[ok], axis=1))]]]
    best, pair = -1.0, None
    for a_ in ok:
        for b_ in ok:
            if b_ <= a_:
                continue
            d = abs(J[a_, 0] * J[b_, 1] - J[a_, 1] * J[b_, 0])
            if d > best:
                best, pair = d, (a_, b_)
    return x[list(pair)]


class XPSAxis:
    """A slide rotation stage on the Newport XPS: a command takes effect after
    `latency_s` (PC -> XPS -> trajectory start), the stage then moves at up to
    `vel_deg_s`, and every target is rounded to `step_deg` (minimum incremental motion)."""

    def __init__(self, pos_deg: float, latency_s: float = 0.05, vel_deg_s: float = 2.0,
                 step_deg: float = 0.001):
        self.pos = self.target = self.cmd = pos_deg
        self.latency, self.vel, self.step = latency_s, vel_deg_s, step_deg
        self.queue = []

    def command(self, t: float, target_deg: float):
        self.cmd = target_deg
        self.queue.append((t + self.latency, round(target_deg / self.step) * self.step))

    def advance(self, t: float, dt: float):
        while self.queue and self.queue[0][0] <= t:
            self.target = self.queue.pop(0)[1]
        d = self.target - self.pos
        self.pos += np.clip(d, -self.vel * dt, self.vel * dt)


class IntensityLockSim:
    """Step-by-step simulation of the intensity lock (see simulate_intensity_lock).

    The 630 nm reference passes all three paths (short-pass filter blocks, achromatic
    HWPs) and lands on n_pd pinholes + photodiodes. At lock time the photodiode readings
    are stored as setpoints and the Jacobian dI/dtheta is measured. Every 1/loop_hz the
    PC reads the photodiodes (relative noise pd_noise) and sends integral corrections
    gain * pinv(J) @ (setpoint - I) to the two XPS axes, which apply them after latency_s
    at up to vel_deg_s. Drift: each arm's path length does a random walk (walk_nm per
    sqrt(s)) plus a steady rate (nm/s), plus optional sudden kicks [(t_s, arm, nm), ...].
    A path-length change dL shifts the phase by 2*pi*dL/lambda at both wavelengths."""

    def __init__(self, main: "Setup", n_pd: int = 2, ref_nm: float = 630.0, pinhole_um: float = 50.0,
                 dt_s: float = 0.01, loop_hz: float = 10.0, latency_s: float = 0.05,
                 vel_deg_s: float = 2.0, step_deg: float = 0.001, gain: float = 0.5,
                 walk_nm: float = 3.0, rate_nm_s: dict | None = None, pd_noise: float = 0.002,
                 kicks_nm=(), seed: int = 1):
        import copy
        self.rng = np.random.default_rng(seed)
        self.rate = {"A": 2.0, "B": -1.0, "C": 0.5} if rate_nm_s is None else rate_nm_s
        self.walk, self.pd_noise, self.gain, self.pinhole_um = walk_nm, pd_noise, gain, pinhole_um
        self.dt, self.kicks = dt_s, sorted(kicks_nm)
        self.locked, self.free = copy.deepcopy(main), copy.deepcopy(main)
        for s_ in (self.locked, self.free):
            for L in s_.launch.values():
                L.drift_rad, L.blocked = 0.0, False
        self.ref = copy.deepcopy(self.locked)
        self.ref.lam_nm = ref_nm
        self.x_ph = choose_pinholes(self.ref, n=n_pd, d_um=pinhole_um)
        self.J = slide_jacobian(self.ref, self.x_ph, pinhole_um)
        self.Jinv = np.linalg.pinv(self.J)
        self.I_set = pinhole_intensity(self.ref, self.x_ph, pinhole_um)
        self.I_meas = self.I_set.copy()
        self.axes = [XPSAxis(self.locked.theta_A_deg, latency_s, vel_deg_s, step_deg),
                     XPSAxis(self.locked.theta_B_deg, latency_s, vel_deg_s, step_deg)]
        self.theta0 = (self.locked.theta_A_deg, self.locked.theta_B_deg)
        self.target = self.rel(self.locked)
        self.dL = {p: 0.0 for p in PATHS}
        self.every_loop = max(1, int(round(1 / (loop_hz * dt_s))))
        self.i, self.t, self.n_updates, self.events = 0, 0.0, 0, []

    @staticmethod
    def rel(s_):
        ph = {p: s_.beam(p)["phase"] for p in PATHS}
        return np.array([ph["A"] - ph["C"], ph["B"] - ph["C"]])

    def step(self):
        """Advance the simulation by one time step dt_s."""
        self.i += 1
        self.t = self.i * self.dt
        for p in PATHS:
            self.dL[p] += self.rate[p] * self.dt + self.walk * np.sqrt(self.dt) * self.rng.standard_normal()
        while self.kicks and self.kicks[0][0] <= self.t:
            tk, arm, nm = self.kicks.pop(0)
            self.dL[arm] += nm
            self.events.append((self.t, f"arm {arm} path jumps {nm:+.0f} nm"))
        for ax_ in self.axes:
            ax_.advance(self.t, self.dt)
        for s_ in (self.locked, self.free, self.ref):
            for p in PATHS:
                s_.launch[p].drift_rad = 2 * np.pi * self.dL[p] / s_.lam_nm
        self.locked.theta_A_deg = self.ref.theta_A_deg = self.axes[0].pos
        self.locked.theta_B_deg = self.ref.theta_B_deg = self.axes[1].pos
        if self.i % self.every_loop == 0:
            self.I_meas = pinhole_intensity(self.ref, self.x_ph, self.pinhole_um) * (
                1 + self.pd_noise * self.rng.standard_normal(len(self.x_ph)))
            d_theta = self.gain * self.Jinv @ (self.I_set - self.I_meas)
            for ax_, d in zip(self.axes, d_theta):
                ax_.command(self.t, ax_.cmd + d)
            self.n_updates += 1

    def snapshot(self) -> dict:
        """Current errors, photodiode signals and stage states."""
        status = []
        for ax_ in self.axes:
            if ax_.queue:
                status.append("command sent, waiting (XPS latency)")
            elif abs(ax_.pos - ax_.target) > 1e-12:
                status.append("moving")
            else:
                status.append("settled")
        return dict(t=self.t, err_free=wrap(self.rel(self.free) - self.target),
                    err_lock=wrap(self.rel(self.locked) - self.target),
                    pd=pinhole_intensity(self.ref, self.x_ph, self.pinhole_um) / self.I_set,
                    theta=[a.pos for a in self.axes], cmd=[a.cmd for a in self.axes],
                    status=status, dL=dict(self.dL))


def simulate_intensity_lock(main: "Setup", n_pd: int = 2, ref_nm: float = 630.0, pinhole_um: float = 50.0,
                            duration_s: float = 300.0, dt_s: float = 0.01, loop_hz: float = 10.0,
                            latency_s: float = 0.05, vel_deg_s: float = 2.0, step_deg: float = 0.001,
                            gain: float = 0.5, walk_nm: float = 3.0, rate_nm_s: dict | None = None,
                            pd_noise: float = 0.002, record_every_s: float = 0.1, seed: int = 1,
                            kicks_nm=()) -> dict:
    """Run IntensityLockSim for duration_s and record every record_every_s. Holding the
    630 nm phases fixed leaves a small 810 nm residual because the slide glass disperses
    slightly differently from air."""
    sim = IntensityLockSim(main, n_pd=n_pd, ref_nm=ref_nm, pinhole_um=pinhole_um, dt_s=dt_s,
                           loop_hz=loop_hz, latency_s=latency_s, vel_deg_s=vel_deg_s, step_deg=step_deg,
                           gain=gain, walk_nm=walk_nm, rate_nm_s=rate_nm_s, pd_noise=pd_noise,
                           kicks_nm=kicks_nm, seed=seed)
    n = int(round(duration_s / dt_s))
    every_rec = max(1, int(round(record_every_s / dt_s)))
    keys = ("t", "err_free", "err_lock", "pd", "theta_A", "theta_B", "cmd_A", "cmd_B", "dL_A", "dL_B", "dL_C")
    out = {k: [] for k in keys}
    for i in range(1, n + 1):
        sim.step()
        if i % every_rec == 0:
            snap = sim.snapshot()
            for k in ("t", "err_free", "err_lock", "pd"):
                out[k].append(snap[k])
            out["theta_A"].append(snap["theta"][0]); out["theta_B"].append(snap["theta"][1])
            out["cmd_A"].append(snap["cmd"][0]); out["cmd_B"].append(snap["cmd"][1])
            for p in PATHS:
                out["dL_" + p].append(snap["dL"][p])
    out = {k: np.array(v) for k, v in out.items()}
    out.update(locked=sim.locked, free=sim.free, ref=sim.ref, x_pinholes=sim.x_ph, J=sim.J, I_set=sim.I_set)
    return out


def plt_circle(xy, r):
    from matplotlib.patches import Circle
    return Circle(xy, r, fc="#8c96a5", ec="none", zorder=4)

if __name__ == "__main__":
    s = Setup()
    px, I = s.camera()
    print("frame shape:", I.shape)
    print("phases mod 2pi:", {k: round(v, 3) for k, v in s.phases().items()})
    print("fringe periods (px):", {k: round(v, 1) for k, v in s.fringe_periods_px().items()})
    print("dphi/dtheta A (rad/deg):", round(slide_slope(s.theta_A_deg, s.lam_nm, s.slide_t_mm), 1))
    print("equal-intensity HWP angles:", {k: round(v, 3) for k, v in equal_intensity_angles().items()})
    print("power to camera per path:", {k: round(v, 5) for k, v in s.camera_powers().items()})
