
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

        grad = adagrad(i, grad, grad_m, grad_v, β1=0.99, β2=0.999)
        params -= 5e-3 * grad

        # Used with the animation
        #grad = adagrad(i, grad, grad_m, grad_v, β1=0.5, β2=0.9)
        #params[:-2] -= 1e-4 * grad[:-2]
        #params[-2:] *= np.exp(-1e-3*grad[-2:])
        
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

def load_T2():
    # Magic numbers
    # Minimum thresholds for each channel
    dch0, dch1 = 0.003981947898864746, 0.07901215553283691
    dch = np.array([dch0, dch1])

    waves = []
    with Timer("T2 dataset loaded"):
        for n in range(8):
            path = "09_20_2024/T2/RigolDS%d.bin"%n
            wave = wavebin_parser.parse(path)
            
            wave = wave[84000:]

            #noise = (np.random.random(wave[:, 1:].shape)-1/2) * dch
            #wave[:, 1:] += noise
            waves.append(wave)

    return np.array([w[:, :2] for w in waves[:6]])

@njit
def T2_model(params, tf, tn, t):
    A, C, P, τ, σ = params
    #return C + A*np.power(P, tn)/(1+B*((tf-τ)/σ)**2)
    return C + A*np.exp(-t/P)/(1+((tf-τ)/σ)**2)

@njit
def T2_model_grad(params, tf, tn, t, T2M, y):
    A, C, P, τ, σ = params

    err = T2M - y
    dA = (T2M - C)/A
    #dP = (T2M - C) * tn/P
    dP = (T2M - C) * (t/P**2)
    dC = tf*0 + 1
    
    tt = (tf-τ)/σ
    D = 1+tt**2
    dM_dD = -(T2M - C)/D

    dD_dtt = 2*tt
    dτ = dM_dD * dD_dtt * (-1/σ)
    dσ = dM_dD * dD_dtt * (-tt/σ)

    #grad = np.array([dA, dC, dP, dτ, dσ], np.float64)
    grad = np.empty((len(params), err.shape[0], err.shape[1]))
    grad[0] = dA
    grad[1] = dC
    grad[2] = dP
    grad[3] = dτ
    grad[4] = dσ

    result = np.zeros_like(params)
    for t in range(err.shape[1]):
        for n in range(err.shape[0]):
            for p in range(len(params)):
                result[p] += err[n,t] * grad[p,n,t]

    return result
    
    #return grad @ err
    #return np.einsum('nt,pnt->p', err, grad)

def fit_T2(T2_data):
    params = np.random.random(5)
    grad_m = np.zeros_like(params)
    grad_v = np.ones_like(params)

    params = np.array([
        #A, C, P, τ, σ = params
        #0.266, 0.01, 0.0949, 0.005, 0.0005
        #$1, 0.001, 0.1035, 0.005, 0.001
        2.432559e-01, 9.000208e-03, 1.028314e-01, 5.022274e-03, 5.596152e-04
    ])

    T2_data = T2_data[:, :60000]
    T2_data = T2_data[:, ::100]
    
    t, ch0 = T2_data.transpose(2,0,1)

    tp = 0.01
    tf, tn = np.modf(t/tp)
    tf *= tp

    N = 100000
    
    for i in range(N):
        T2M = T2_model(params, tf, tn, t)
        loss = np.var(T2M - ch0)

        #if i%(N//25) == 0:
        if i%(N//100) == 0:
            print(i, '%.6e'%loss, '***', *('%.6e,'%p for p in params))
        
        grad = T2_model_grad(params, tf, tn, t, T2M, ch0)
        # grad += np.random.normal(size=params.shape) * np.abs(params)*1e-3
        grad = adagrad(i, grad, grad_m, grad_v, β1=0.9, β2=0.99)
        grad = grad * [1,1,1,1,1e-6]
        grad /= T2_data[0].size
        params -= 1e-3 * grad

        #if i%(N//10) == 0:
        #    #print(i, loss, '***', *params)
        #    Params[i//10+1] = params
        
        #params[:-2] -= 1e-2 * grad[:-2]
        #params[-2:] *= np.exp(-1e-2*grad[-2:])
    return params

def plot_T2(T2_data, params):

    fig, ax = plt.subplots()
    for w in T2_data[:1]:
    
        t, ch0 = w.T
        ax.plot(t, ch0)

    t0 = np.linspace(0, 1, 1000001)
    tp = 0.01
    tf, tn = np.modf(t0/tp)
    tf *= tp
    ax.plot(t0, T2_model(params, tf, tn, t0))

    ax.set_xbound(0, 0.3)

    plt.show()

def plot_T2_spectra(T2_data):
    fig, ax = plt.subplots()
    
    for w in T2_data:
        t, ch0 = w.T
        k = np.fft.fftfreq(len(t), t[1]-t[0])
        fft = np.fft.fft(ch0)
        ax.scatter(np.abs(k), np.real(np.abs(fft)), s=1, alpha=0.1)

    #ax.set_xscale('log')
    ax.set_yscale('log')

    plt.show()

def compute_T2():
    T2_data = load_T2()
    #params = fit_T2(T2_data)
    params = np.array([2.432379e-01, 8.994032e-03, 1.033462e-01, 5.022145e-03, 5.595870e-04])
    #plot_T2(T2_data, params)
    plot_T2_spectra(T2_data)

if __name__ == '__main__':
    #compute_T1()
    compute_T2()
