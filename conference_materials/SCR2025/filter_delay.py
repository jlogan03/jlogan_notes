import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy import signal


# Timing components
adc_clock_speed = 50e6  # [Hz]
adc_clock_period = 1.0 / adc_clock_speed  # [s]
adc_sample_hold_cycles = 16.5  # [dimensionless]
adc_sample_hold = adc_sample_hold_cycles * adc_clock_period  # [s]
adc_conversion_time = 0.0  # [s] TODO: figure out how long this is

# Calculate fractional delay needed for each channel to align with the first sample group
#   Each group starts as soon as the previous one is done
delay_per_group = adc_sample_hold + adc_conversion_time  # [s]

groups = [
    [8, 9, 0],
    [10, 12, 1],
    [11, 13, 2],
    [14, 15, 3],
    [16, 17, 4],
    [18, 5],
    [19, 6],
    [7]
]

delays = np.zeros(20)  # [s] delay of each channel
for i, group in enumerate(groups):
    delays[group] = i * delay_per_group


# Build fractional-delay filters for each channel

def poly_taps(order: int, delay: float) -> NDArray:

    taps = np.zeros(order)

    for k in range(order):
        coeff = 1.0
        kv = float(k)
        for m in range(order):
            mv = float(m)
            if m != k:
                coeff *= (delay - mv) / (kv - mv)
        taps[k] = coeff
    
    return taps


# NOTE: We can apply the delay at either the full or decimated frequency
samplerate = 40e3  # [Hz]
sample_period = 1.0 / samplerate  # [s]
reporting_rate = 1e3  # [Hz]
fractional_delays = delays / sample_period  # [dimensionless] 

for order in [3, 4, 6]:

    taps = [poly_taps(order, delay) for delay in fractional_delays]

    w = 2.0 * np.pi * samplerate * np.logspace(-4, np.log10(0.5), 1000)
    responses = []
    for b in taps:
        _, h = signal.freqz(b, worN=w, fs=2.0 * np.pi * samplerate)
        mag = 20.0 * np.log10(np.abs(h))  # [dB] frequency response magnitude
        phase = np.unwrap(np.angle(h, deg=True))  # [deg] phase lag frequency response
        responses.append((mag, phase))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    plt.suptitle(f"Per-Channel Fractional Delays\nPoly Order = {order - 1} ({order}-Tap), Samplerate = {samplerate * 1e-3:.2f} kHz")
    for mag, phase in responses:
        plt.sca(ax1)
        plt.semilogx(
            w / (2.0 * np.pi),
            mag,
            color="k",
            alpha=0.4,
            linestyle="-",
        )
        plt.ylabel("Magnitude [dB]")

        plt.sca(ax2)
        plt.semilogx(
            w / (2.0 * np.pi),
            phase,
            color="k",
            alpha=0.4,
            linestyle="-",
        )
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Phase [deg]")

    plt.sca(ax1)
    plt.axvline(samplerate / 2, color='k', linewidth=1, label="Nyquist of Samplerate")
    plt.axvline(reporting_rate, color='k', linewidth=3, label="Typical Reporting Rate")
    plt.legend()
    plt.sca(ax2)
    plt.axvline(samplerate / 2, color='k', linewidth=1,)
    plt.axvline(reporting_rate, color='k', linewidth=3,)

    plt.tight_layout()

    plt.savefig(f"./fractional_delay_order{order}.svg")

plt.show()