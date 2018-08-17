'''
Created on Dec 26, 2017

based upon

"A Non-iterative Method for (Re)Construction of Phase from STFT Magnitude"
Zdenek Prusa, Peter Balazs, Peter L. Sondergaard

@author: richard
'''

import numpy as np
import pghi
import scipy.signal as signal

def sine_test():
    f = 10*p.Fs/p.M # fft bin #10
    p.test( 'pure sine test {:4.0f}Hz'.format(f))
    dur = int(2*p.Fs)  #2 seconds 
    signal_in = signal.chirp(range(dur), f/p.Fs, dur, f/p.Fs)
    signal_out = p.signal_to_signal(signal_in)
    p.plt.plot_waveforms('Signal in, Signal out', [signal_in, signal_out])
    
def pulse_test():
    p.test( 'pulse test')
    magnitude_frames = np.zeros((300,int(p.M/2+1)))
    p.original_phase = magnitude_frames    
    magnitude_frames[20,:]= 1
    phase_estimated_frames = p.magnitude_to_phase_estimate(magnitude_frames)    
    
def sweep_test():
    freq_high = 5000 #Hz
    freq_low = 0
    p.test('sweep test {:.0f}Hz,{:.0f}Hz'.format(freq_low, freq_high))
    dur = int(2*p.Fs)  #swept sine 2 seconds
    method=('linear','quadratic','hyperbolic','logarithmic')[0]
    signal_in = signal.chirp(range(dur), freq_low/p.Fs, dur, freq_high/p.Fs, method=method)
    signal_in2 = signal.chirp(range(dur), freq_high/p.Fs, dur, freq_low/p.Fs, method=method)
    signal_in = np.concatenate([signal_in,signal_in2])
    p.logprint ('duration of sound = {0:10.7} seconds'.format(signal_in.shape[0]/p.Fs)) 
    signal_out = p.signal_to_signal(signal_in)
    p.plt.plot_waveforms('Signal in, Signal out', [signal_in, signal_out])        
          
def audio_test():
    p.plt.fileCount =0    
    for nfile in range(100): # arbitrary file limit
        song_title, audio_in = p.plt.get_song()
        if audio_in is None: 
            break          
        stereo = []
        for i in range(audio_in.shape[0]): # channels = 2 for stereo
            p.test( 'audio test{} ch {}'.format(nfile,i))             
            signal_in = audio_in[i]
            signal_out = p.signal_to_signal(signal_in)         
            p.plt.plot_waveforms('Signal in, Signal out', [signal_in, signal_out])
            stereo.append( signal_out)
        p.test( 'audio test{}'.format(nfile))
        p.plt.signal_to_file(np.stack(stereo), song_title, override_verbose = True) 
              
############################  program start ###############################

p = pghi.PGHI(tol = 1e-3, show_plots = False, show_frames=100, verbose=True)

# gl = 2048
# g = signal.windows.hann(gl)    
# gamma =gl**2*.25645
# p = pghi.PGHI(tol = 1e-6, show_plots = False, show_frames=10, g=g,gamma = gamma, gl=gl)

pulse_test()
sine_test()
sweep_test()
p.setverbose(False)    
audio_test()


     

    
  

