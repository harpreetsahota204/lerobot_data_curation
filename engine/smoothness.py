"""Smoothness math on a 1-D speed profile: SPARC, LDLJ and a low-pass helper.

Ported from ``demo_quality_scorer.engine.motion``. Source: Balasubramanian et
al., "A robust and sensitive metric for quantifying movement smoothness"
(IEEE TBME 2012). SPARC is amplitude- and duration-invariant, which is why it is
the primary smoothness metric. LDLJ is duration-sensitive (about duration**4 on
a stationary signal), so it must only be compared across equal-length windows.
"""

import numpy as np
from scipy.signal import butter, filtfilt

FILTER_ORDER = 4


def lowpass_filtfilt(signal, cutoff_hz, fs, order=FILTER_ORDER):
    """Zero-phase Butterworth low-pass, skipped gracefully when ill-defined.

    Returns `signal` unchanged when `cutoff_hz` is not below Nyquist or the
    signal is too short for filtfilt's padding requirement.
    """
    nyquist = fs / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyquist:
        return signal
    b, a = butter(order, cutoff_hz, fs=fs)
    padlen = 3 * max(len(a), len(b))
    if len(signal) <= padlen:
        return signal
    return filtfilt(b, a, signal)


def sparc(speed, fs, fc=10.0, amp_threshold=0.05, pad_level=4):
    """Spectral arc length of a speed profile (closer to 0 is smoother).

    Args:
        speed: a 1-D speed profile
        fs: sampling rate, in Hz
        fc (10.0): band limit in Hz. Values computed at different `fc` are not
            comparable
        amp_threshold (0.05): spectral magnitude below which content is noise
        pad_level (4): zero-padding exponent, for arc-length resolution
    """
    speed = np.asarray(speed, dtype=np.float64)
    if len(speed) < 4 or fs <= 0 or not np.all(np.isfinite(speed)):
        return np.nan

    n = len(speed)
    nfft = int(2 ** (np.ceil(np.log2(n)) + pad_level))
    spectrum = np.abs(np.fft.fft(speed, nfft))[: nfft // 2]
    freqs = np.fft.fftfreq(nfft, d=1.0 / fs)[: nfft // 2]

    peak = spectrum.max()
    if peak == 0:
        return np.nan
    spectrum = spectrum / peak

    # Trim to the band that carries movement. Including the noise floor out to
    # fc would add arc length proportional to how quiet the channel is.
    in_band = freqs <= fc
    above = in_band & (spectrum >= amp_threshold)
    if not above.any():
        return np.nan

    lo = np.argmax(above)
    hi = len(above) - 1 - np.argmax(above[::-1])
    band_freqs, band_spectrum = freqs[lo : hi + 1], spectrum[lo : hi + 1]
    if len(band_freqs) < 2:
        return np.nan

    span = band_freqs[-1] - band_freqs[0]
    if span == 0:
        return np.nan

    d_freq = np.diff(band_freqs) / span
    d_amp = np.diff(band_spectrum)
    return float(-np.sum(np.sqrt(d_freq**2 + d_amp**2)))


def ldlj(speed, fs):
    """Log dimensionless jerk of a speed profile (closer to 0 is smoother)."""
    speed = np.asarray(speed, dtype=np.float64)
    if len(speed) < 5 or fs <= 0 or not np.all(np.isfinite(speed)):
        return np.nan

    peak = np.max(np.abs(speed))
    if peak == 0:
        return np.nan

    dt = 1.0 / fs
    duration = (len(speed) - 1) * dt
    jerk = np.diff(speed, n=2) / dt**2
    dimensionless_jerk = (duration**3 / peak**2) * np.sum(jerk**2) * dt
    if dimensionless_jerk <= 0:
        return np.nan
    return float(-np.log(dimensionless_jerk))


def jerk_rms(speed, fs, cutoff_hz=10.0):
    """RMS jerk of a zero-phase-filtered speed profile (lower is smoother).

    Differentiating twice amplifies noise by omega^2, so the profile is low-pass
    filtered first and the samples most affected by filtfilt's edge padding are
    trimmed before differentiating. Noise-sensitive and correlated with SPARC and
    LDLJ, so it is opt-in and never scored.
    """
    speed = np.asarray(speed, dtype=np.float64)
    if len(speed) < 5 or fs <= 0 or not np.all(np.isfinite(speed)):
        return np.nan

    nyquist = fs / 2.0
    padlen = 3 * (FILTER_ORDER + 1)
    filter_applied = 0 < cutoff_hz < nyquist and len(speed) > padlen
    filtered = lowpass_filtfilt(speed, cutoff_hz, fs)
    trim = min(len(filtered) // 4, 3 * FILTER_ORDER) if filter_applied else 0
    trimmed = filtered[trim : len(filtered) - trim] if trim > 0 else filtered
    if len(trimmed) < 5:
        return np.nan
    jerk = np.diff(trimmed, n=2) * fs**2
    return float(np.sqrt(np.mean(jerk**2)))


def psd_lf_hf(speed, fs, cutoff_hz=None):
    """Log ratio of low-band to high-band power (higher is smoother).

    A Welch band-power ratio on the speed profile. Unreliable at LeRobot frame
    rates, because tremor sits near or above Nyquist, so it is opt-in and never
    scored. Returned as a natural log so it is symmetric around 0.
    """
    from scipy.signal import welch

    speed = np.asarray(speed, dtype=np.float64)
    if len(speed) < 8 or fs <= 0 or not np.all(np.isfinite(speed)):
        return np.nan
    freqs, power = welch(speed, fs=fs, nperseg=min(len(speed), 64))
    cutoff_hz = cutoff_hz if cutoff_hz is not None else fs / 4.0
    low = power[freqs <= cutoff_hz].sum()
    high = power[freqs > cutoff_hz].sum()
    if low <= 0 or high <= 0:
        return np.nan
    return float(np.log(low / high))
