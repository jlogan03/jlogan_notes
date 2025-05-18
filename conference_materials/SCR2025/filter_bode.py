import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def rc_bode(r, c, w=None):
    """LTI system and frequency response for first-order analog RC filter"""

    # Transfer function: H(s) = 1 / (RCs + 1)
    num = [1]
    den = [r * c, 1]
    system = signal.TransferFunction(num, den)

    # Frequency range for Bode plot
    w = w if w is not None else np.logspace(1, 6, 500)  # from 10 to 1,000,000 rad/s

    # Frequency response
    w, mag, phase = signal.bode(system, w)

    return system, w, mag, phase


def butter_bode(cutoff_ratio, order, w_norm=None):
    """Normalized digital butterworth filter"""

    # Transfer function
    system = signal.butter(order, Wn=cutoff_ratio * 2.0 * np.pi, analog=False, fs=1.0)

    # Frequency range for Bode plot
    w_norm = w_norm if w_norm is not None else np.logspace(-3, 0, 500)

    w_norm, mag, phase = signal.dbode(signal.dlti(*system, dt=1.0), w_norm)

    return system, w_norm, mag, phase


# RC analog filter params
r = 10.0  # [ohm]
c = 1e-6  # [F]

# Reference points
internal_samplerate = 40e3  # [Hz]
reporting_rate = 1e3  # [Hz]
rc_cutoff = 1 / (2.0 * np.pi * r * c)  # [Hz]
#   [dimensionless] butterworth filter cutoff as a fraction of samplerate
butter_cutoff_ratio = reporting_rate / internal_samplerate

# RC filter freq response
rc_sys, rc_w, rc_mag, rc_phase = rc_bode(r, c)

# Butterworth filter freq response
reporting_rate_rad_per_s = reporting_rate * 2.0 * np.pi
w = rc_w[np.where(rc_w < reporting_rate_rad_per_s)]
w_norm = w / reporting_rate_rad_per_s
butter_sys, butter_w, butter_mag, butter_phase = butter_bode(
    butter_cutoff_ratio, order=2, w_norm=w_norm
)
butter_w = butter_w * reporting_rate * 2.0 * np.pi  # [rad/s] from normalized form

# Magnitude
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.semilogx(rc_w / (2.0 * np.pi), rc_mag, color="k", linestyle="dotted")
plt.semilogx(butter_w / (2.0 * np.pi), butter_mag, color="k", linestyle="--")
plt.title("Bode Plot of First-order RC Low-pass Filter")
plt.ylabel("Magnitude [dB]")
plt.grid(which="both", linewidth=0.5)

plt.axvline(x=40e3, color="k", linestyle="--", label="Internal Samplerate")
plt.axvline(x=1e3, color="k", linestyle="-", label="Typical Reporting Rate")
plt.axvline(
    x=1 / (2.0 * np.pi * r * c), color="k", linestyle="dotted", label="RC Cutoff"
)

# Phase
plt.subplot(2, 1, 2)
plt.semilogx(rc_w / (2.0 * np.pi), rc_phase, color="k", linestyle="dotted")
plt.semilogx(butter_w / (2.0 * np.pi), butter_phase, color="k", linestyle="--")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase [deg]")
plt.grid(which="both", linewidth=0.5)

plt.axvline(x=40e3, color="k", linestyle="--", label="Internal Samplerate")
plt.axvline(x=1e3, color="k", linestyle="-", label="Typical Reporting Rate")
plt.axvline(
    x=1 / (2.0 * np.pi * r * c), color="k", linestyle="dotted", label="RC Cutoff"
)

plt.legend()

plt.tight_layout()
plt.show()
