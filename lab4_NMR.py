
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

def load_T1():
    delays = np.array([88, 138, 238, 338, 438, 538, 638, 738,
        788, 688, 588, 488, 388, 288, 188, 88, 38,
        78, 68, 58, 48, 28, 18, 8])*1e-3

    waves = []
    with Timer("T1 dataset loaded"):
        for n in range(len(delays)):
            path = "09_20_2024/T1/RigolDS%d.bin"%n
            waves.append(wavebin_parser.parse(path))

    waves[0] = waves[0][:, (0,2)]
    return waves, delays

def process_T1(waves, delays):

    # The first plot is formatted strangely. We have a duplicate sample, so we
    #     can remove it.
    waves = waves[1:]
    delays = delays[1:]

    # Downsample the waveform to 1000 samples.
    for n in range(len(waves)):
        dt = waves[n].T[0,1] - waves[n].T[0,0]
        fft = np.fft.fft(waves[n].T[1])
        k = np.fft.fftfreq(len(waves[n].T[0]))
        kernel = np.exp(-(100*np.pi*k)**2)
        fft = fft*kernel
        filtered = np.real(np.fft.ifft(fft))

        waves[n] = np.array([
            np.reshape(waves[n].T[0], (1000, -1))[:,0],
            np.reshape(filtered, (1000, -1))[:,0]
        ])

    # Flip the negative signs on the smaller samples
    means = [np.mean(w[1,-200:]) for w in waves]
    mean = np.mean(means)

    for n in range(len(waves)):
        waves[n] = waves[n][:, 275:] # Arbitrary cutoff
        if delays[n] < 80e-3:
            waves[n][1] = mean - waves[n][1]

    return np.array([[0*t+d, t, w] for d,(t,w) in zip(delays, waves)])

def compute_T1():
    T1_waves, T1_delays = load_T1()
    T1_data = process_T1(T1_waves, T1_delays)

    def plot_T1_2d():
        fig, ax = plt.subplots()
        
        for wave in T1_data:
            delay, t, ch2 = wave
            ax.plot(t, ch2)
            
        plt.show()

    def plot_T1_3d():
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        
        for delay, time, ch2 in T1_data:
            ax.plot(delay, time, ch2)

        plt.show()
        globals().update(locals())

    #plot_waves(T1_waves[0])
    #plot_T1_2d()
    plot_T1_3d()
    globals().update(locals())

if __name__ == '__main__':
    compute_T1()
