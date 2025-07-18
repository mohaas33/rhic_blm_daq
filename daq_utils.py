import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import logging as log

def timebase_to_interval_ns(timebase: int):
    if timebase <= 2:
        return 2**timebase
    elif timebase < 2**32-1:
        return (timebase - 2) / 125e-3
    else:
        raise ValueError(f"Invalid timebase: {timebase}")


# https://stackoverflow.com/questions/50365310/python-rising-falling-edge-oscilloscope-like-trigger
def rising_edges(data, thresh):
    sign = data >= thresh
    pos = np.where(np.convolve(sign, [1, -1]) == 1)
    return pos[0]


def analyze_buffer(
    x: np.array,
    revsig: np.array,
    revsig_threshold: float,
    peak_height_threshold_above_mean: float,
    peak_width_threshold: float,
    peak_distance_threshold: float,
    integral_range: list[float],
    skip_integrals=False,
    return_extra_properties=False,
):
    results = {}

    revsig_edges = rising_edges(revsig, revsig_threshold)

    integral_samples_pre = integral_range[0]
    integral_samples_post = integral_range[1]

    # Restrict range of x for the peak finding to make sure we have no peaks
    # with integration ranges outside the bounds of the array
    peak_height_threshold = x.mean() + peak_height_threshold_above_mean
    # Peak finding appears to slow down (alot?) if the signal is mostly zero?
    # Perhaps because we can skip regions if there is more structure to the signal?
    peak_indicies, properties = find_peaks(
        x[integral_samples_pre:-integral_samples_post],
        height=peak_height_threshold,
        width=peak_width_threshold,
        distance=peak_distance_threshold,
    )
    log.info(f"Found {len(peak_indicies)} peaks")
    # Fix indicies as those are now shifted due to the restricted range
    peak_indicies = peak_indicies + integral_samples_pre
    peak_heights = properties["peak_heights"]

    peak_bin_idx = np.digitize(peak_indicies, bins=revsig_edges) - 1
    dist_to_revsig = revsig_edges[peak_bin_idx] - peak_indicies

    # Straight forward way, btw this will ignore if the range is out of bounds???
    # integrals = np.array([np.trapezoid(x[p - integral_samples_pre : p + integral_samples_post]) for p in peaks_add])
    # The slighly (like 50ms) faster way
    if not skip_integrals:
        integrals = np.trapezoid(
            x[peak_indicies[:, None] - integral_samples_pre + np.arange(integral_samples_pre + integral_samples_post)],
            axis=1,
        )

    results = {}
    results["peak_indicies"] = peak_indicies
    results["peak_heights"] = peak_heights
    results["widths"] = properties["widths"]
    results["dist_to_revsig"] = dist_to_revsig
    if return_extra_properties:
        for kk, vv in properties.items():
            if kk not in results.keys():
                results[kk] = vv
    if not skip_integrals:
        results["integrals"] = integrals
    # TODO: Perhaps also save some sample waveforms (just the samples near some of the peaks)?

    return results
