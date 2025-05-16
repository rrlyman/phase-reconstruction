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

# import scipy.signal as signal
# import pghi_plot
# from scipy import ndimage
from PGHI_base import PGHI_base


class PGHI(PGHI_base):
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
    ):
        super().__init__(
            redundancy=redundancy,
            time_scale=time_scale,
            freq_scale=freq_scale,
            M=M,
            gl=gl,
            g=g,
            tol=tol,
            lambdasqr=lambdasqr,
            gamma=gamma,
            h=h,
            plt=plt,
            show_plots=show_plots,
            show_frames=show_frames,
            verbose=verbose,
            Fs=Fs,
            logfile="log_rtpghi.txt",
        )

        self.magnitude = np.zeros((3, self.M2))
        self.phase = np.zeros((3, self.M2))
        self.fgrad = np.zeros((3, self.M2))
        self.tgrad = np.zeros((3, self.M2))
        self.logs = np.zeros((3, self.M2))
        self.original_phase = np.zeros((3, self.M2))

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
            p = pghi.RTPGHI(redundancy=8, M=2048,tol = 1e-6, show_plots = False, show_frames=20)
        """

    def test_name(self, testName):
        super().test_name(testName, __name__)

    def dxdw(self, x):
        """return the derivative of x with respect to frequency"""
        xp = np.pad(x, 1, mode="edge")
        #         dw = (np.multiply(3,(xp[1:-1,:-2]) + np.multiply(2,xp[1:-1,1:-1]) + np.multiply(3,xp[1:-1,2:])) - np.multiply(6,(xp[1:-1,:-2] + xp[1:-1,1:-1] + xp[1:-1,2:])))/6
        dw = (xp[2:] - xp[:-2]) / 2
        return dw

    def dxdt(self, x):
        """return the derivative of x with respect to time"""
        xp = np.pad(x, 1, mode="edge")
        #         dt = (np.multiply(3,(xp[:-2,1:-1]) + np.multiply(2,xp[1:-1,1:-1]) + np.multiply(3,xp[2:,1:-1])) - np.multiply(6,(xp[:-2,1:-1] + xp[1:-1,1:-1] + xp[2:,1:-1])))/6
        dt = (xp[1, 1:-1] - xp[1, 1:-1]) / (2)

        return dt

    def magnitude_to_phase_estimate(self, magnitude):
        """
        run the hop by hop magnitude to phase algorithm through the
        entire sound sample to produce graphs
        """
        original_phase = np.zeros_like(magnitude)
        if self.plt.verbose:  # for debugging
            self.debug_count = 0
            try:
                original_phase = np.angle(self.corig_frames)
            except:
                pass
            self.q_errors = []
        phase, fgrad, tgrad = [], [], []

        for n in range(magnitude.shape[0]):
            #             self.mask = np.roll(self.mask,-1,axis=0)
            #             self.mask[2] = magnitude[n] > (self.tol*np.max(magnitude[n]))
            #             print('STEP')
            p, f, t = self.magnitude_to_phase_estimatex(magnitude[n], original_phase[n])
            phase.append(p)
            fgrad.append(f)
            tgrad.append(t)

        mask = magnitude > (self.tol * np.max(magnitude))
        phase = np.stack(phase)
        tgrad = np.stack(tgrad)
        fgrad = np.stack(fgrad)

        if self.plt.verbose:
            nprocessed = np.sum(np.where(mask, 1, 0))
            self.logprint(
                "magnitudes processed above threshold tolerance={}, magnitudes rejected below threshold tolerance={}".format(
                    nprocessed, magnitude.size - nprocessed
                )
            )
            self.plt.plot_3d("magnitude", [magnitude], mask=mask)
            self.plt.plot_3d("fgrad", [fgrad], mask=mask)
            self.plt.plot_3d("tgrad", [tgrad], mask=mask)
            self.plt.plot_3d("Phase estimated", [phase], mask=mask)
            if original_phase is not None:
                self.plt.plot_3d("Phase original", [original_phase], mask=mask)
                self.plt.plot_3d(
                    "Phase original, Phase estimated",
                    [(original_phase) % (2 * np.pi), (phase) % (2 * np.pi)],
                    mask=mask,
                )
                self.plt.colorgram(
                    "Phase original minus Phase estimated",
                    np.abs((original_phase) % (2 * np.pi) - (phase) % (2 * np.pi)),
                    mask=mask,
                )
                self.plt.quiver("phase errors", self.q_errors)
        return phase

    def magnitude_to_phase_estimatex(self, magnitude, original_phase):
        """estimate the phase frames from the magnitude
        parameter:
            magnitude
                numpy array containing the real absolute values of the
                magnitudes of each FFT frame.
                shape (n,m) where n is the frame step and
                m is the frequency step
        return
            estimated phase of each fft coefficient
                shape (n,m) where n is the frame step and
                m is the frequency step
                measure: radians per sample
        """

        N = magnitude.shape[0]
        M2, M, a_a = self.M2, self.M, self.a_a
        wbin = 2 * np.pi / self.M
        self.magnitude = np.roll(self.magnitude, -1, axis=0)
        self.phase = np.roll(self.phase, -1, axis=0)
        self.fgrad = np.roll(self.fgrad, -1, axis=0)
        self.tgrad = np.roll(self.tgrad, -1, axis=0)
        self.logs = np.roll(self.logs, -1, axis=0)
        self.original_phase = np.roll(self.original_phase, -1, axis=0)
        self.magnitude[2] = magnitude
        self.original_phase[2] = original_phase
        self.logs[2] = np.log(magnitude + 1e-50)

        # alternative
        #         fmul = self.lambdasqr*wbin/a

        fmul = self.gamma / (a_a * M)
        self.tgradplus = (2 * np.pi * a_a / M) * np.arange(M2)
        self.tgrad[2] = self.dxdw(self.logs[2]) / fmul + self.tgradplus

        self.fgradplus = np.pi
        self.fgrad[1] = -fmul * self.dxdt(self.logs) + self.fgradplus

        hp = []

        mask = magnitude > (self.tol * np.max(magnitude))
        n0 = 0
        for m0 in range(M2):
            heapq.heappush(hp, (-self.magnitude[n0, m0], n0, m0))

        while len(hp) > 0:
            s = heapq.heappop(hp)
            n, m = s[1], s[2]
            if n == 1 and m < M2 - 1 and mask[m + 1]:  # North
                mask[m + 1] = False
                self.phase[n, m + 1] = (
                    self.phase[n, m] + (self.fgrad[n, m] + self.fgrad[n, m + 1]) / 2
                )
                heapq.heappush(hp, (-self.magnitude[n, m + 1], n, m + 1))
                if self.plt.verbose and self.debug_count <= 2000:
                    self.debugInfo(n, m + 1, n, m, self.phase, self.original_phase)

            if n == 1 and m > 0 and mask[m - 1]:  # South
                mask[m - 1] = False
                self.phase[n, m - 1] = (
                    self.phase[n, m] - (self.fgrad[n, m] + self.fgrad[n, m - 1]) / 2
                )
                heapq.heappush(hp, (-self.magnitude[n, m - 1], n, m - 1))
                if self.plt.verbose and self.debug_count <= 2000:
                    self.debugInfo(n, m - 1, n, m, self.phase, self.original_phase)

            if n == 0 and mask[m]:  # East
                mask[m] = False
                self.phase[(n + 1), m] = (
                    self.phase[n, m]
                    + self.time_scale * (self.tgrad[n, m] + self.tgrad[(n + 1), m]) / 2
                )
                heapq.heappush(hp, (-self.magnitude[n + 1, m], 1, m))
                if self.plt.verbose and self.debug_count <= 2000:
                    self.debugInfo(n + 1, m, n, m, self.phase, self.original_phase)

        return self.phase[0], self.fgrad[0], self.tgrad[0]
