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

"""

import numpy as np
import heapq
import scipy.signal as signal
import pghi_plot
from scipy import ndimage

dtype = np.float64


class PGHI_base(object):
    """
    implements the Phase Gradient Heap Integration - PGHI algorithm
    """

    def __init__(
        self,
        redundancy=8,
        time_scale=1,
        freq_scale=1,
        M=2048,
        gl=None,
        g=None,
        tol=1e-6,
        lambdasqr=None,
        gamma=None,
        h=0.01,
        plt=None,
        show_plots=False,
        show_frames=25,
        verbose=True,
        Fs=44100,
        logfile="",
    ):
        params = locals()
        params.pop("self", None)  # Remove 'self' from the dictionary
        print(params)

        self.__dict__.update(params)
        """
        Parameters
            redundancy
                number of hops per window
            time_scale
                multiplier to lengthen/shorten time, higher number is slower output
            freq_scale
                multiplier to expand/contract frequency scale
            M
                number of samples in for each FFT calculation
                measure: samples
            gl length of the sampling window
                measure: samples
            g
                windowing function of shape (gl,)
            lambdasqr
                constant for windowing function
                measure: samples**2
            gamma
                alternative to lambdasqr
                measure 2*pi*samples**2
            tol
                small signal relative magnitude filtering size
                measure: filtering height/maximum magnitude height
            h
                the relative height of the Gaussian window function at edges
                of the window, h = 1 mid window
            show_plots
                if True, each plot window becomes active and must be closed to continue
                the program. Handy for rotating the plot with the cursor for 3d plots
                if False, plots are saved to the ./pghi_plots sub directory
            show_frames
                The number of frames to plot on each side of the algorithm start point
            verbose
                boolean, if True then save output to ./pghi_plots directory

            Fs
                sampling frequency
                measure - samples per second
        Example
            p = pghi.PGHI(redundancy=8, M=2048,tol = 1e-6, show_plots = False, show_frames=20)
        """
        if self.gl is None:
            self.gl = self.M
        if self.gamma is not None:
            self.lambdasqr = self.gamma / (2 * np.pi)
        if self.g is None:
            # Auger, Motin, Flandrin #19
            self.lambda_ = (-(self.gl**2) / (8 * np.log(self.h))) ** 0.5
            self.lambdasqr = self.lambda_**2
            self.gamma = 2 * np.pi * self.lambdasqr
            self.g = np.array(
                signal.windows.gaussian(2 * self.gl + 1, self.lambda_ * 2, sym=False),
                dtype=dtype,
            )[1 : 2 * self.gl + 1 : 2]

        self.M2 = int(self.M / 2) + 1

        self.a_s = int(self.M / self.redundancy)
        self.a_a = int(self.a_s / self.time_scale)

        # self.corig = None

        self.plt = pghi_plot.Pghi_Plot(
            show_plots=self.show_plots,
            show_frames=self.show_frames,
            verbose=self.verbose,
            time_scale=self.time_scale,
            freq_scale=self.freq_scale,
            logfile=self.logfile,
        )

        if self.lambdasqr is None:
            self.logprint("parameter error: must supply lambdasqr and g")
        self.logprint("a_a(analysis time hop size) = {} samples".format(self.a_a))
        self.logprint("a_s(synthesis time hop size) = {} samples".format(self.a_s))
        self.logprint("M, samples per frame = {}".format(self.M))
        self.logprint("tol, small signal filter tolerance ratio = {}".format(self.tol))
        self.logprint("lambdasqr = {:9.4f} 2*pi*samples**2  ".format(self.lambdasqr))
        self.logprint("gamma = {:9.4f} 2*pi*samples**2  ".format(self.gamma))
        self.logprint(
            "h, window height at edges = {} relative to max height".format(self.h)
        )
        self.logprint("fft bins = {}".format(self.M2))
        self.logprint("redundancy = {}".format(self.redundancy))
        self.logprint("time_scale = {}".format(self.time_scale))
        self.logprint("freq_scale = {}".format(self.freq_scale))
        self.plt.plot_waveforms("Window Analysis pghi", [self.g])

        denom = 0  # calculate the synthesis window
        self.gsynth = np.zeros_like(self.g, dtype=dtype)
        for l in range(int(self.gl)):
            denom = 0
            for n in range(-self.redundancy, self.redundancy + 1):
                dl = l - n * self.a_s
                if dl >= 0 and dl < self.M:
                    denom += self.g[dl] ** 2
            self.gsynth[l] = self.g[l] / denom
        self.plt.plot_waveforms("Window Synthesis pghi", [self.gsynth])

    def setverbose(self, verbose):
        saved_d = self.plt.verbose
        self.plt.verbose = verbose
        return saved_d

    def test_name(self, testName, module_name):
        self.plt.pre_title = str.format(
            module_name + " " + testName + " time_scale {0:3.2} freq_scale {1:3.2} ",
            self.time_scale,
            self.freq_scale,
        )

    def logprint(self, txt):
        self.plt.logprint(txt)

    # def clear(self):
    #     self.corig = None
    def rewind_file_list(self):
        self.plt.fileCount = 0

    def sigstretch(self, samples):
        """
        modify the FFT magnitude coefficients to translate and scale the
            frequency
            parameter:
                magnitude
                    np.array the absolute values of the FFT coefficients
            return
                magnitude
                    np.array
        """
        if self.freq_scale == 1:
            return samples

        newMs = np.linspace(
            0, samples.size, int(self.freq_scale * samples.size), endpoint=False
        )
        newsig = np.empty_like(newMs)

        if self.freq_scale < 1:
            lowpassfir = signal.firwin(32, 0.9 * self.freq_scale)
            samples = np.convolve(lowpassfir, samples, mode="same")

        for m, v in enumerate(newMs):
            oldMhigh = min(samples.size - 1, int(np.ceil(v)))
            oldMlow = max(0, int(np.floor(v)))
            dv = v - oldMlow
            assert oldMhigh >= 0 and oldMhigh < samples.size
            assert oldMlow >= 0 and oldMlow < samples.size
            newsig[m] = (1 - dv) * samples[oldMlow] + dv * samples[oldMhigh]
        return newsig

    def debugInfo(self, n1, m1, n0, m0, phase, original_phase):
        dif = (phase[n1, m1] - phase[n0, m0]) % (2 * np.pi)
        if original_phase is None:
            dif_orig = dif
        else:
            if n1 != n0:
                dif_orig = (original_phase[n1, m1] - original_phase[n0, m0]) % (
                    2 * np.pi
                )
            elif m1 != m0:
                dif_orig = (original_phase[n1, m1] - original_phase[n0, m0]) % (
                    2 * np.pi
                )
        if dif_orig == 0:
            err_new = 0
        else:
            err_new = (dif - dif_orig) / dif_orig
        self.q_errors.append((n0, m0, 0, n1 - n0, m1 - m0, err_new / (2 * np.pi)))

        if self.debug_count < 10:
            if m1 == m0 + 1:
                self.logprint(
                    "###############################   POP   ###############################"
                )
            self.logprint(
                ["", "NORTH", "SOUTH"][m1 - m0] + ["", "EAST", "WEST"][n1 - n0]
            )
            self.logprint("n1,m1=({},{}) n0,m0=({},{})".format(n1, m1, n0, m0))
            self.logprint(
                "\testimated phase[n,m]={:13.4f}, phase[n0,m0]         =:{:13.4f}, dif(2pi)     ={:9.4f}".format(
                    (phase[n1, m1]), (phase[n0, m0]), dif
                )
            )
            if original_phase is not None:
                self.logprint(
                    "\toriginal_phase[n,m] ={:13.4f}, original_phase[n0,m0]=:{:13.4f}, dif_orig(2pi)={:9.4f}".format(
                        (original_phase[n1, m1]), (original_phase[n0, m0]), dif_orig
                    )
                )
                self.logprint("error ={:9.4f}%".format(100 * err_new))
        self.debug_count += 1

    def magphase_to_complex(self, magnitude, phase):
        return magnitude * (np.cos(phase) + np.sin(phase) * 1j)

    def magphase_frames_to_signal(self, magnitude, phase):
        return self.complex_frames_to_signal(self.magphase_to_complex(magnitude, phase))

    def complex_to_magphase(self, corig):
        return np.absolute(corig), np.angle(corig)

    def signal_to_frames(self, s):  # applies window function, g
        self.plt.signal_to_file(s, "signal_to_frames")
        self.plt.spectrogram(s, "spectrogram signal_to_frames")
        L = s.shape[0] - self.M
        self.corig_frames = np.stack(
            [np.fft.rfft(self.g * s[ix : ix + self.M]) for ix in range(0, L, self.a_a)]
        )
        return self.corig_frames

    def complex_frames_to_signal(self, complex_frames):
        M2 = complex_frames.shape[1]
        N = complex_frames.shape[0]
        M = self.M
        a_s = self.a_s

        vr = np.fft.irfft(complex_frames)
        sig = np.zeros((N * a_s + self.M))
        cum_waveforms = []
        n1 = 15
        n2 = 25
        for n in range(N):
            vs = vr[n] * self.gsynth
            if self.verbose and n >= n1 and n < n2:
                vout = np.zeros(((n2 - n1) * a_s + M))
                na = (n - n1) * a_s
                vout[na : na + M] = vs
                cum_waveforms.append(vout)
            sig[n * a_s : n * a_s + M] += vs
        self.plt.plot_waveforms("Gabor Contributions", cum_waveforms)
        self.plt.signal_to_file(sig, "complex_frames_to_signal")
        self.plt.spectrogram(sig, "spectrogram signal out")
        return sig

    def signal_to_magphase_frames(self, s):
        return self.complex_to_magphase(self.signal_to_frames(s))

    def signal_to_signal(self, signal_in):
        """
        convert signal_in to frames
          throw away the phase
          reconstruct the phase from the magnitudes
          re-run fft and compute the frobenius norm for an error value

          parameter:
              signal_in numpy array (length,)

          return:
               reconstructed signal
        """
        self.plt.signal_to_file(signal_in, "original_sound_track")
        self.plt.spectrogram(signal_in, "original_sound_track in")
        s = self.sigstretch(signal_in)
        magnitude_frames, _ = self.signal_to_magphase_frames(s)
        phase_estimated_frames = self.magnitude_to_phase_estimate(magnitude_frames)

        signal_out = self.magphase_frames_to_signal(
            magnitude_frames, phase_estimated_frames
        )
        self.plt.plot_waveforms("Signal in, Signal out", [signal_in, signal_out])

        saved_verbose = self.setverbose(False)
        reconstructed_magnitude, _ = self.signal_to_magphase_frames(signal_out)
        self.setverbose(saved_verbose)
        if magnitude_frames.shape[0] > 1 and reconstructed_magnitude.shape[0] > 1:
            s1 = self.plt.normalize(
                magnitude_frames[1:]
            )  # s1 is delayed by 1 frame with respect to s2
            s2 = self.plt.normalize(reconstructed_magnitude[:-1])
            minlen = min(s1.shape[0], s2.shape[0])
            s1 = s1[:minlen]
            s2 = s2[:minlen]
            mn = min(minlen, 100) - 15
            dif = s2 - s1
            E = np.sqrt(np.sum(dif * dif)) / np.sqrt(np.sum(s1 * s1))  # Frobenius norm
            self.plt.plot_3d(
                "magnitude_frames, reconstructed_magnitude",
                [s1[mn : mn + 10], s2[mn : mn + 10]],
            )
        return signal_out
