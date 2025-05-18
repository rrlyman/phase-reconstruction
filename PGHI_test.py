"""
Created on Jul 7, 2018

based upon

"A Non-iterative Method for (Re)Construction of Phase from STFT Magnitude"
Zdenek Prusa, Peter Balazs, Peter L. Sondergaard

@author: richard

to run:
    python PGHI_test.py
or
    python3 PGHI_test.py

    look in the "results" directory for the program output
"""

# REQUIRES FFMEG TO BE INSTALLED IN THE SYSTEM

import numpy as np
import rtpghi as rtpghi
import pghi as pghi
import scipy.signal as signal
import time as tm


"""
    use scale-up to shift the frequency. 1.0 is no frequency change, 
    2 expands the frequency so a 1 kHz tone becomes 2 kHz.
    .5 shrinks the frequency so 1 kHz tone becomes  500 hz.
"""

scale_up = 1.5


def sine_test():
    """
    Conduct a pure sine wave test to verify signal processing

    """
    f = 440  # frequency in Hz
    dur = 2  # duration in seconds
    algorithm.test_name("pure sine test {:4.0f}Hz".format(f))
    dur = int(2 * algorithm.Fs)  # 2 seconds
    signal_in = signal.chirp(range(dur), f / algorithm.Fs, dur, f / algorithm.Fs)
    signal_out = algorithm.signal_to_signal(signal_in)
    algorithm.plt.signal_to_file(
        signal_out,
        "====== END RESULT of sine test ======",
        override_verbose=True,
    )
    sz = min(signal_in.shape[0], signal_out.shape[0])

    harmony = np.add(signal_in[:sz], signal_out[:sz])
    algorithm.plt.signal_to_file(
        harmony,
        "harmony",
        override_verbose=True,
    )


def pulse_test():
    """
    Perform a pulse test by generating a pulse in magnitude frames and estimating the resultant signal

    :param p: Object holding methods for testing, plotting and signal processing
    :type p: object

    :returns: None, but plots the waveform of the output signal from the pulse test
    :return type: None
    """
    algorithm.test_name("pulse test")
    magnitude_frames = np.zeros((300, int(algorithm.M / 2 + 1)))
    algorithm.corig_frames = None  # kludge to keep from plotting original_phase
    magnitude_frames[20, :] = 1  # pulse at frame 20
    phase_estimated_frames = algorithm.magnitude_to_phase_estimate(magnitude_frames)
    signal_out = algorithm.magphase_frames_to_signal(
        magnitude_frames, phase_estimated_frames
    )
    algorithm.plt.plot_waveforms("Signal out", [signal_out])


def sweep_test():
    freq_high = 5000  # Hz
    """
        Generate a swept sine wave signal for testing purposes
    
        :param p: The object or instance used for logging and accessing sample rate (Fs)
        :type p: object with attributes 'Fs' (sample rate) and methods 'test', 'logprint', 'signal_to_signal'
    
        :returns: Concatenated swept sine wave signal from low to high and back from high to low frequencies
        :return type: numpy.ndarray
    """
    freq_low = 0
    algorithm.test_name("sweep test {:.0f}Hz,{:.0f}Hz".format(freq_low, freq_high))
    dur = int(2 * algorithm.Fs)  # swept sine 2 seconds
    method = ("linear", "quadratic", "hyperbolic", "logarithmic")[0]
    signal_in = signal.chirp(
        range(dur),
        freq_low / algorithm.Fs,
        dur,
        freq_high / algorithm.Fs,
        method=method,
    )
    signal_in2 = signal.chirp(
        range(dur),
        freq_high / algorithm.Fs,
        dur,
        freq_low / algorithm.Fs,
        method=method,
    )
    signal_in = np.concatenate([signal_in, signal_in2])
    algorithm.logprint(
        "duration of sound = {0:10.7} seconds".format(signal_in.shape[0] / algorithm.Fs)
    )
    algorithm.signal_to_signal(signal_in)


def audio_test():
    """
    Conducts a test on audio files and processes them for waveform analysis

    :param song_title: The title of the song being processed
    :type song_title: str
    :param audio_in: The audio data of the song to be tested
    :type audio_in: ndarray

    :returns: None
    :return type: None
    """
    algorithm.rewind_file_list()
    for nfile in range(100):  # 100 arbitrary file limit
        etime = tm.time()
        # algorithm.test_name("audio test " + song_title)
        song_title, audio_in = algorithm.plt.get_song()
        if audio_in is None:
            break
        stereo = []
        harmony_stereo = []
        for i in range(audio_in.shape[0]):  # channels = 2 for stereo
            algorithm.test_name("audio test " + song_title + " ch{}".format(i))
            signal_in = audio_in[i]
            signal_out = algorithm.signal_to_signal(signal_in)
            sz = min(signal_in.shape[0], signal_out.shape[0])
            harmony = np.add(signal_in[:sz], signal_out[:sz])
            algorithm.plt.plot_waveforms(
                "Signal in, Signal out", [signal_in, signal_out]
            )
            stereo.append(signal_out)
            harmony_stereo.append(harmony)
        # saved = algorithm.setverbose(True)
        #         saved = algorithm.setverbose(True)
        audio_out = np.stack(stereo)
        harmony_out = np.stack(harmony_stereo)
        algorithm.plt.signal_to_file(
            audio_out,
            "====== END RESULT of audio test ======",
            override_verbose=True,
        )
        algorithm.plt.signal_to_file(
            harmony_out,
            "====== HARMONY RESULT of audio test ======",
            override_verbose=True,
        )

        algorithm.logprint("elapsed time = {:8.2f} seconds\n".format(tm.time() - etime))
        # algorithm.setverbose(saved)


def warble_test():
    """
    Conduct a warble test by alternating signals between two frequencies

    :param p: A parameter set that contains properties like sampling frequency (Fs) and a method to execute test and log results.
    :type p: module or class with specific attributes and methods

    :returns: Logs the duration of the generated sound signal and processes it through the system
    :return type: None
    """
    f1 = 32 * algorithm.Fs / algorithm.M  # cycles per second
    f2 = 128 * algorithm.Fs / algorithm.M
    # set so there is no discontinuity in phase when changing frequencies
    samples_for_2_pi_radians = int(algorithm.Fs / f1)
    algorithm.test_name("warble test {:.0f}Hz,{:.0f}Hz".format(f1, f2))
    dur = int(0.25 * algorithm.Fs / samples_for_2_pi_radians)
    signal_in = []
    for k in range(dur):
        signal_in.append(
            signal.chirp(
                range(samples_for_2_pi_radians),
                f1 / algorithm.Fs,
                samples_for_2_pi_radians,
                f1 / algorithm.Fs,
            )
        )
        signal_in.append(
            signal.chirp(
                range(samples_for_2_pi_radians),
                f2 / algorithm.Fs,
                samples_for_2_pi_radians,
                f2 / algorithm.Fs,
            )
        )
    signal_in = np.concatenate(signal_in)
    algorithm.logprint(
        "duration of sound = {0:10.7} seconds".format(signal_in.shape[0] / algorithm.Fs)
    )
    algorithm.signal_to_signal(signal_in)


############################  program start ###############################
if __name__ == "__main__":

    algorithm = pghi.PGHI(
        tol=1e-3,
        show_frames=100,
        time_scale=1 / scale_up,
        freq_scale=scale_up,
        show_plots=False,
        verbose=True,
        logfile="log_pghi.txt",
    )

    warble_test()
    pulse_test()
    sine_test()
    sweep_test()
    audio_test()

    algorithm = rtpghi.PGHI(
        tol=1e-3,
        show_frames=100,
        time_scale=1 / scale_up,
        freq_scale=scale_up,
        show_plots=False,
        verbose=True,
        logfile="log_rtpghi.txt",
    )

    audio_test()

    # gl = 2048
    # g = signal.windows.hann(gl)
    # gamma =gl**2*.25645
    # algorithm =pghi.PGHI(tol = 1e-6, show_plots = False, show_frames=10, g=g,gamma = gamma, gl=gl)
