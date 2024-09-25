
# Joseph Kessler
# 2024 September 20
# class4_simulator.py
################################################################################
# Spin echo experiment

import matplotlib.pyplot as plt
import numpy as np
import wavebin_parser

import time, humanize
from numba import njit

################################################################################
# Convinience functions

# Finds the minimum and maximum along given axes.
def minimax(arr, axis=None):
    return np.array([np.min(arr, axis=axis), np.max(arr, axis=axis)])

# Timer function usable with a with directive.
# with Timer():
#     ...
class Timer:
    def fmttime(t0_ns, t1_ns):
        return humanize.metric((t1_ns-t0_ns)*1e-9, 's')
    def printtime(t0_ns, t1_ns):
        print(Timer.fmttime(t0_ns, t1_ns))
    
    def __init__(self, prefix=""):
        self.prefix = prefix
    def __enter__(self):
        self.t0 = time.time_ns()
        return self
    def __exit__(self, *args):
        t1 = time.time_ns()
        if self.prefix == "":
            Timer.printtime(self.t0, t1)
        else:
            print(self.prefix, "in", Timer.fmttime(self.t0, t1))


################################################################################
# NMR Experiment code

def plot_waves(wave):
    fig, ax = plt.subplots()
    for sig in wave.T[1:]:
        ax.plot(wave.T[0], sig)
    plt.show()

def plot_fft_mag(wave):
    fig, ax = plt.subplots()
    fft = np.fft.fft(wave.T[1])
    k = np.fft.fftfreq(len(wave), d=wave[1,0] - wave[0,0])
    f = np.abs(1/k)
    ax.plot(f, np.real(np.abs(fft)))#, s=1, alpha=0.1)

    fft = np.fft.fft(wave.T[2])
    ax.plot(f, np.real(np.abs(fft)))#, s=1, alpha=0.1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()

def test_plot():
    path = "09_20_2024/RigolDS0.bin"

    with Timer("bin loaded"):
        wave = wavebin_parser.parse(path)

    # Add noise to largest signal difference to reduce quantization
    noise_range = wave[2,1] - wave[1,1]
    wave.T[2] = wave.T[1] + (np.random.random(size=len(wave.T[1]))-0.5)*noise_range

    plot_waves(wave)
    #plot_fft_mag(wave)

test_plot()
