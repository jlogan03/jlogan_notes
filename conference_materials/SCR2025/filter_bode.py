import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def rc_bode(r, c, w=None):
    """LTI system and frequency response for first-order analog RC filter"""

    # Transfer function: H(s) = 1 / (RCs + 1)
    num = [1]
    den = [r * c, 1]
    system = signal.TransferFunction(num, den)

    # Frequency response
    w = w if w is not None else np.logspace(1, 6, 500)  # [rad/s]
    w, mag, phase = signal.bode(system, w)

    return system, w, mag, phase


def butter_bode(cutoff_ratio, order, fs, w=None):
    """Digital Butterworth with some cutoff as a fraction of samplerate"""

    # Transfer function
    system = signal.butter(order, Wn=cutoff_ratio * fs, analog=False, fs=fs)

    # Frequency response
    w = w if w is not None else np.logspace(1, 6, 500)  # [rad/s]
    w, mag, phase = signal.dbode(signal.dlti(*system, dt=1.0 / fs), w=w / fs)

    return system, w, mag, phase


def inv_db(x):
    """Inverse decibel scaling"""
    return 10.0 ** (x / 20.0)


def db(x):
    """Decibel scaling"""
    return 20.0 * np.log10(x)


# RC analog filter params
r = 10.0  # [ohm]
c = 1e-6  # [F]

# Reference points
internal_samplerate = 40e3  # [Hz]
reporting_rate = 1e3  # [Hz]
rc_cutoff = 1 / (2.0 * np.pi * r * c)  # [Hz]
#   [dimensionless] butterworth filter cutoff as a fraction of samplerate
#   with cutoff parked at reporting rate, allowing some aliasing in exchange for
#   reduced phase distortion
butter_cutoff_ratio = reporting_rate / internal_samplerate

# RC filter freq response
rc_sys, rc_w, rc_mag, rc_phase = rc_bode(r, c)

# Butterworth filter freq response
# Examined only below the internal samplerate
inds = np.where(rc_w < internal_samplerate * 2.0 * np.pi)
w = rc_w[inds]
butter_sys, butter_w, butter_mag, butter_phase = butter_bode(
    butter_cutoff_ratio, order=2, fs=internal_samplerate, w=w
)

# Total freq response
total_mag = db(inv_db(rc_mag[inds]) * inv_db(butter_mag))  # [dB]
total_phase = rc_phase[inds] + butter_phase  # [deg]

# Magnitude
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.semilogx(
    rc_w / (2.0 * np.pi),
    rc_mag,
    color="k",
    linestyle="dotted",
    label="ADC Input Filter",
)
plt.semilogx(
    butter_w / (2.0 * np.pi),
    butter_mag,
    color="k",
    linestyle="--",
    label="Digital Filter",
)
plt.semilogx(butter_w / (2.0 * np.pi), total_mag, color="k", linestyle="-", label="Aggregate Filter")
plt.title("Bode Plot of Filter Stages 2 & 3")
plt.ylabel("Magnitude [dB]")
plt.grid(which="both", linewidth=0.5)

ymin, ymax = plt.ylim()
dy = ymax - ymin

plt.axvline(x=internal_samplerate, color="k", linestyle="--")
plt.annotate(
    "Internal Samplerate",
    xy=(internal_samplerate, ymax - dy / 4),
    xytext=(20, -10),
    textcoords='offset points',
    arrowprops={"linewidth": 1, "facecolor": 'k', "arrowstyle": '->,head_width=.25'},
    ha="left",
)

plt.axvline(x=internal_samplerate / 2, color="k", linestyle="--")
plt.annotate(
    "Nyquist of\nInternal Samplerate",
    xy=(internal_samplerate / 2, ymax - dy / 8),
    xytext=(-20, -10),
    textcoords='offset points',
    arrowprops={"linewidth": 1, "facecolor": 'k', "arrowstyle": '->,head_width=.25'},
    ha="right",
)

plt.axvline(x=reporting_rate, color="k", linestyle="--")
plt.annotate(
    "Typical Reporting Rate",
    xy=(reporting_rate, ymax - 2 * dy / 4 ),
    xytext=(20, -5),
    textcoords='offset points',
    arrowprops={"linewidth": 1, "facecolor": 'k', "arrowstyle": '->,head_width=.25'},
    ha="left",
)

plt.axvline(x=reporting_rate / 2, color="k", linestyle="-", linewidth=3)
plt.annotate(
    "Nyquist of\nTypical Reporting Rate",
    xy=(reporting_rate / 2, ymax - 3 * dy / 4 ),
    xytext=(-20, -10),
    textcoords='offset points',
    arrowprops={"linewidth": 1, "facecolor": 'k', "arrowstyle": '->,head_width=.25'},
    ha="right",
)

plt.axvline(x=rc_cutoff, color="k", linestyle="--")
plt.annotate(
    "ADC Input\nFilter Cutoff",
    xy=(rc_cutoff, ymax - 3 * dy / 4 ),
    xytext=(-20, -10),
    textcoords='offset points',
    arrowprops={"linewidth": 1, "facecolor": 'k', "arrowstyle": '->,head_width=.25'},
    ha="right",
)
plt.xlim(left=np.min(rc_w))

plt.legend()

# Phase
plt.subplot(2, 1, 2)
plt.semilogx(rc_w / (2.0 * np.pi), rc_phase, color="k", linestyle="dotted", label="ADC Input Filter")
plt.semilogx(butter_w / (2.0 * np.pi), butter_phase, color="k", linestyle="--", label="Digital Filter")
plt.semilogx(butter_w / (2.0 * np.pi), total_phase, color="k", linestyle="-", label="Aggregate Filter")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase [deg]")
plt.grid(which="both", linewidth=0.5)

plt.axvline(x=internal_samplerate, color="k", linestyle="--")
plt.axvline(x=internal_samplerate / 2, color="k", linestyle="--")
plt.axvline(x=reporting_rate, color="k", linestyle="--")
plt.axvline(x=reporting_rate / 2, color="k", linestyle="-", linewidth=3)
plt.axvline(x=rc_cutoff, color="k", linestyle="--")

plt.xlim(left=np.min(rc_w))

plt.legend()

plt.tight_layout()
plt.show()
