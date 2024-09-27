
# Joseph Kessler
# 2024 September 20
# class4_simulator.py
################################################################################
# Spin echo experiment

import matplotlib.pyplot as plt
import numpy as np
import wavebin_parser

from time import time_ns
import humanize
from numba import njit

################################################################################
# Convinience functions

# Finds the minimum and maximum along given axes.
def minimax(arr, axis=None):
    return np.array([np.min(arr, axis=axis), np.max(arr, axis=axis)])

# Linearly interpolates between start to end by pct.
def lerp(start, end, pct):
    return start*(1.0-pct) + end*pct

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
        self.t0 = time_ns()
    def __enter__(self):
        self.t0 = time_ns()
        return self
    def __exit__(self, *args):
        t1 = time_ns()
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
    #means = [np.mean(w[1,-200:]) for w in waves]
    #mean = np.mean(means)
    #print(mean)
    # Since this mean is arbitrary, instead use it as an additional parameter.
    # Movable are the datasets that have this learable parameter.

    movable = np.zeros_like(delays)
    for n in range(len(waves)):
        waves[n] = waves[n][:, 275:] # Arbitrary cutoff
        if delays[n] < 80e-3:
            waves[n][1] = -waves[n][1]
            movable[n] = 1
    
    return np.array([[0*t+m, 0*t+d, t, w] for m,d,(t,w) in zip(movable, delays, waves)])

def T1_model(params, d, t):
    off, A, B, C, d0, τ = params
    return B + (A + C*np.exp(-d/d0))*np.exp(-t/τ)

def T1_model_grad(params, d, t, T1M, yoff):
    off, A, B, C, d0, τ = params

    err = T1M - yoff
    dA = np.exp(-t/τ)
    dB = t*0 + 1
    dC = dA*np.exp(-d/d0)
    dd0= dC*C*d/d0**2
    dτ = (T1M - B)*t/τ**2
    
    return np.einsum('nt,pnt->p', err, [dA*0, dA, dB, dC, dd0, dτ])

def adagrad(iters, grad, grad_m, grad_v, β1=0.9, β2=0.99):
    grad_m += (β1 - 1)*(grad_m - grad)
    grad_v += (β2 - 1)*(grad_v - grad**2)

    mh = grad_m / (1 - β1**(iters+1))
    mv = grad_v / (1 - β2**(iters+1))

    return mh / np.sqrt(mv + 1e-8)

def fit_T1(T1_data):
    movable, delays, times, waves = T1_data.transpose((1,0,2))
    
    params = np.random.random(6)
    grad_m = np.zeros_like(params)
    grad_v = np.ones_like(params)

    # saved parameters that are useful starting points
    #params = np.array([0.015, 0.25, 0.025, -0.4, 0.1, 0.00035])
    params = np.array([0.015, 4.104857e-01, 5.029003e-01,
        1.921413e-01, 4.254114e-01, 7.756779e-01]) # anim 4

    #params = np.array([2.66328203e-02, 3.15024587e-01, 1.66554928e-02,
    #    -6.18351606e-01, 1.31578148e-01, 5.65475746e-04]) #optimized result

    N = 3000
    for i in range(N):
        T1M = T1_model(params, delays, times)
        y_adj = waves + params[0]*movable
        loss = np.var(T1M - y_adj)

        if i%(N//(250)) == 0:
            print(i, '%.6e'%loss, '***', *('%.6e,'%p for p in params))
        
        grad = T1_model_grad(params, delays, times, T1M, y_adj)
        grad[0] = -np.sum(movable*(T1M - y_adj))
        grad /= waves.size

        #grad = adagrad(i, grad, grad_m, grad_v, β1=0.99, β2=0.999)
        #params -= 5e-3 * grad

        # Used with the animation
        grad = adagrad(i, grad, grad_m, grad_v, β1=0.5, β2=0.9)
        params[:-2] -= 1e-4 * grad[:-2]
        params[-2:] *= np.exp(-1e-3*grad[-2:])
        
    print(f'{params = }')
    print(f'{np.linalg.norm(grad) = }')
    return params

def compute_T1():
    T1_waves, T1_delays = load_T1()
    T1_data = process_T1(T1_waves, T1_delays)
    with Timer("T1 fit completed in"):
        params = fit_T1(T1_data)

    def plot_T1_2d():
        fig, ax = plt.subplots()
        
        for wave in T1_data:
            delay, t, ch2 = wave
            ax.plot(t, ch2)
            
        plt.show()

    def plot_T1_3d():
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        ax.view_init(30, 90+45)
        
        bounds = minimax(T1_data[:, 1:], axis=(0,2))
        delay_bounds, time__bounds, wave__bounds = bounds.T

        for movable, delay, time, wave in T1_data:
            ch2 = wave + params[0]*movable
            ax.plot(delay, time, ch2)

        l = np.linspace(0, 1, 1000)
        short = np.linspace(0, 1, 16)

        for s in short:
            s = l*0 + s
            
            d = lerp(*delay_bounds, s)
            t = lerp(*time__bounds, l)
            ax.plot(d, t, T1_model(params, d, t), color='black')

            d = lerp(*delay_bounds, l)
            t = lerp(*time__bounds, s)
            ax.plot(d, t, T1_model(params, d, t), color='black')

        plt.show()
        globals().update(locals())

    #plot_waves(T1_waves[0])
    #plot_T1_2d()
    plot_T1_3d()
    globals().update(locals())

if __name__ == '__main__':
    compute_T1()
