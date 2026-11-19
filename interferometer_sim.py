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


if __name__ == "__main__":
    s = Setup()
    x, y, I = s.camera()
    print("phases mod 2pi:", {k: round(v, 3) for k, v in s.phases().items()})
    print("fringe periods (mm):", {k: round(v, 3) for k, v in s.fringe_periods().items()})
    print("dphi/dtheta A (rad/deg):", round(slide_slope(s.theta_A_deg, s.lam_nm, s.slide_t_mm), 1))
    print("I range:", I.min(), I.max(), "max possible:", s.max_intensity(*np.meshgrid(x, y)))
