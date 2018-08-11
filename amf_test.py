'''
Created on Dec 26, 2017

@author: richard
'''

import numpy as np
import amf 
import scipy.signal as signal

Fs = 44100.0    #sample rate

M=2048         # samples per FFT frame
redundancy=8

pghi = amf.Rtpghi(a_a=int(M/redundancy), M=M,tol = 1e-6, show_plots = False, show_frames=20)
pghi.logprint ('samples per second={}'.format(Fs))

# sweep test
freq_high =4000
freq_low = 0
pghi.title('sweep test, {:.0f}Hz,{:.0f}Hz'.format(freq_low, freq_high))

dur = int(2*Fs)  #swept sine
method=('linear','quadratic','hyperbolic','logarithmic')[0]
signal_in = signal.chirp(range(dur), freq_low/Fs, dur, freq_high/Fs, method=method)
signal_in2 = signal.chirp(range(dur), freq_high/Fs, dur, freq_low/Fs, method=method)
signal_in = np.concatenate([signal_in,signal_in2])
pghi.logprint ('duration of sound = {0:10.7} seconds'.format(signal_in.shape[0]/Fs)) 

magnitude_frames, phase_original_frames = pghi.signal_to_magphase_frames(signal_in)
phase_estimated_frames = pghi.magnitude_to_phase_estimate(magnitude_frames)
signal_out = pghi.magphase_frames_to_signal(magnitude_frames, phase_estimated_frames)

pghi.plt.plot_waveforms('Signal in, Signal out', [signal_in[:4000], signal_out[:4000]])

def normalize(mono):
    return (mono - np.min(mono))/np.ptp(mono) -.5

def compute_frobenious(original_magnitude, estimated_phase, sig):
    reconstructed_magnitude, reconstructed_phase = pghi.signal_to_magphase_frames(sig)
    s1 = normalize(original_magnitude[1:]) # s1 is delayed by 1 frame with respect to s2
    s2 = normalize(reconstructed_magnitude[:-1])
    # pghi.plt.plot_3d('magnitude_frames, reconstructed_magnitude', [s1[100:110], s2[100:110] ])   
    E = np.linalg.norm(s2- s1)/np.linalg.norm(s1)    # Frobenius norm
    pghi.logprint ("\nerror measure = {:8.4f} dB".format(20*np.log10(E)))    


compute_frobenious(magnitude_frames, phase_estimated_frames, signal_out)

# pulse test
pghi.title( 'pulse test,')
magnitude_frames = np.zeros_like(magnitude_frames)
magnitude_frames[20,:]= 1
phase_estimated_frames = pghi.magnitude_to_phase_estimate(magnitude_frames)

# pure sine test
f = 23
pghi.title( 'pure sine test, {:6f}Hz'.format(f*Fs/M))
signal_in = signal.chirp(range(dur), f/M, dur, f/M, method=method)
magnitude_frames, phase_original_frames = pghi.signal_to_magphase_frames(signal_in)
phase_estimated_frames = pghi.magnitude_to_phase_estimate(magnitude_frames)
signal_out = pghi.magphase_frames_to_signal(magnitude_frames, phase_estimated_frames)
pghi.plt.plot_waveforms('Signal in, Signal out', [signal_in[:4000], signal_out[:4000]])
compute_frobenious(magnitude_frames, phase_estimated_frames, signal_out)
