'''
Created on Jul 7, 2018

based upon

"A Non-iterative Method for (Re)Construction of Phase from STFT Magnitude"
Zdenek Prusa, Peter Balazs, Peter L. Sondergaard

@author: richard lyman
'''
import numpy as np
import heapq 
import scipy.signal as signal
import pghi_plot
from theano.scalar.basic import sgn

dtype = np.float64

class PGHI(object):
    '''
    implements the Phase Gradient Heap Integration - PGHI algorithm
    '''

    def __init__(self, a_a=128, M=2048, gl=None, g=None, tol = 1e-6, lambdasqr = None, gamma = None, h = .01, plt=None, alg='p2015', pre_title='init', show_plots = False,  show_frames = 25, verbose=True, Fs=44100):
        '''
        Parameters
            a_a    
                analysis hop size
                measure:  samples
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
            pre_title
                string to prepend to the file names when storing plots and sound files
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
            p = pghi.PGHI(a_a=256, M=2048,tol = 1e-6, show_plots = False, show_frames=20)
        '''  
        if gl is None: gl = M  
        if gamma is not None:
            lambdasqr = gamma/(2*np.pi)
        if g is None:     
            # Auger, Motin, Flandrin #19
            lambda_ = (-gl**2/(8*np.log(h)))**.5
            lambdasqr = lambda_**2   
            gamma = 2*np.pi*lambdasqr         
            g=np.array(signal.windows.gaussian(gl, lambda_, sym=False), dtype = dtype)

        self.a_a,self.M,self.tol,self.lambdasqr,self.g,self.gl,self.h, self.pre_title,self.verbose,self.Fs, self.gamma = a_a,M,tol,lambdasqr,g,gl,h,pre_title,verbose,Fs, gamma

        self.M2 = int(self.M/2) + 1          
        self.redundancy = int (self.M/self.a_a)         
        
        self.plt = pghi_plot.Pghi_Plot( show_plots = show_plots,  show_frames = show_frames, pre_title=pre_title)    
           
        self.setverbose(verbose)
        if lambdasqr is None: self.logprint('parameter error: must supply lambdasqr and g')            
        self.logprint('a_a(analysis time hop size) = {} samples'.format(a_a))    
        self.logprint('M, samples per frame = {}'.format(M))     
        self.logprint('tol, small signal filter tolerance ratio = {}'.format(tol))  
        self.logprint('lambdasqr = {:9.4f} 2*pi*samples**2  '.format(self.lambdasqr))
        self.logprint('h, window height at edges = {} relative to max height'.format(h))            
        self.logprint('fft bins = {}'.format(self.M2))                              
        self.logprint ('redundancy = {}'.format(self.redundancy))  
        self.plt.plot_waveforms("Window Function", [self.g])         

        denom = 0    # calculate the synthesis window 
        self.gsynth = np.zeros_like(self.g, dtype = dtype)
        for l in range (self.gl):
            denom = 0                
            for n in range(-self.redundancy, self.redundancy+1):
                dl = l-n*self.a_a
                if dl >=0 and dl < self.M:
                    denom += self.g[dl]**2
            self.gsynth[l] = (1/self.M)*self.g[l]/denom 
    
    def setverbose(self, verbose):
        saved_d = self.plt.verbose
        self.plt.verbose = verbose
        return saved_d
        
    def test(self, title):
        self.plt.pre_title = title
        self.logprint ('\n'+title)            
        
    def logprint(self, txt):
        self.plt.logprint(txt)        
        
    def dxdw(self,x):
        ''' return the derivative of x with respect to frequency'''
        xp = np.pad(x,1,mode='edge')
#         dw = (np.multiply(3,(xp[1:-1,:-2]) + np.multiply(2,xp[1:-1,1:-1]) + np.multiply(3,xp[1:-1,2:])) - np.multiply(6,(xp[1:-1,:-2] + xp[1:-1,1:-1] + xp[1:-1,2:])))/6           
        dw = (xp[1:-1,2:]-xp[1:-1,:-2])/2    
        return dw
    
    def dxdt(self,x):
        ''' return the derivative of x with respect to time'''   
        xp = np.pad(x,1,mode='edge')      
#         dt =    (np.multiply(3,(xp[:-2,1:-1])    + np.multiply(2,xp[1:-1,1:-1])    + np.multiply(3,xp[2:,1:-1]))   - np.multiply(6,(xp[:-2,1:-1]    + xp[1:-1,1:-1]    + xp[2:,1:-1])))/(*6 )        
        dt = (xp[2:,1:-1]-xp[:-2,1:-1])/(2)         

        return dt
            
    def magnitude_to_phase_estimate(self, magnitude):  
        ''' estimate the phase frames from the magnitude
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
        '''

        self.magnitude = magnitude
        N,a,M,M2 = magnitude.shape[0],self.a_a,self.M,self.M2  
        wbin = 2*np.pi/self.M
#         wt = np.fromfunction(lambda n,m : a*n*m*wbin - m*a, (N,M2)) 
#         wt = np.fromfunction(lambda n,m :  - m*a, (N,M2)) 
        
               
        # debugging
        if self.plt.verbose:
            self.debug_count=0
            try:              
                self.original_phase = np.angle(self.corig_frames)   
#                 print (self.original_phase[0,:10])
                self.original_phase -= self.original_phase[0,:]  
#                 print (self.original_phase[0,:10])                  
#                 self.original_phase -=  wt          
#                 print (self.original_phase[0,:10])                             
            except:
                self.original_phase = None     
            self.q_errors=[]
        
        # small signal filter
        mask = magnitude > self.tol*np.max(magnitude)     
        # padded mask is offset by 1
        self.active_padded = np.pad(mask,1,mode='constant',constant_values=False)     
        
        from scipy import ndimage
#         labeled, nr_objects = ndimage.label(magnitude > self.tol) 
#         self.logprint("number of objects = {}".format(nr_objects) )                     
        
        logs = np.log(magnitude+1e-50)  

        # alternative
#         fmul = self.lambdasqr*wbin/self.a_a
        fmul = self.gamma/(a * M)
        tgradplus = (2*np.pi*a/M)*np.arange(M2)
        tgrad = self.dxdw(logs)/fmul + tgradplus
        
        fgradplus =  a
        fgrad = -fmul*self.dxdt(logs) - fgradplus        
   
#         dphaseNE =   self.dxdNE(logs) /self.lambdasqr  - times/2  - self.lambdasqr*self.dxdSE(logs)   
#         dphaseSE = self.dxdNE(logs) /self.lambdasqr  - times/2  - self.lambdasqr*self.dxdSE(logs)            
#         dphaseNE =  tgrad  - fgrad + wbin*self.a_a
#         dphaseSE =  tgrad + fgrad  - wbin*self.a_a 

        # for debugging
        fgradplusOffsets = np.reshape(np.arange(M2), (1,M2))*fgradplus
        
        self.phase = np.random.random_sample(magnitude.shape)*2*np.pi      
        self.startpoints = []  
        self.h=[]   
        
#         #add known phase borders to the heap
#         for n in range(N): 
#             if self.active_padded[n+1,1]:
#                 heapq.heappush(self.h, (-magnitude[n,0], n, 0))     
#             if self.active_padded[n+1,-1]:
#                 heapq.heappush(self.h, (-magnitude[n,self.M2-1], n, self.M2-1)) 
#         for m in range(self.M2): 
#             if self.active_padded[1,m+1]:
#                 heapq.heappush(self.h, (-magnitude[0,m], 0, m))     
#             if self.active_padded[-1,m+1]:
#                 heapq.heappush(self.h, (-magnitude[N-1,m], N-1, m))                                     
                          
        while np.any(self.active_padded): 
            
            # original PGHI algorithm, start at highest point                
            nm = np.argmax(magnitude*self.active_padded[1:-1,1:-1])
            m = nm % self.M2
            n = nm // self.M2
            self.startpoints.append((n,m))        
            self.phase[n,m] = 0
            heapq.heappush(self.h, (-magnitude[n,m],n,m)) 
            self.active_padded[n+1,m+1]=False
            
 
            if self.plt.verbose: 
                self.logprint('Processing Island: start point=[{},{}]'.format(n,m))
                
                #  For debugging purposes, try to align the original phase with the phase generated by this algorithm.
                if self.original_phase is not None:
                    self.original_phase = self.original_phase  - self.original_phase[n,m] 
                    self.original_phase = self.original_phase - fgradplusOffsets                   

            while len(self.h) > 0:
                s=heapq.heappop(self.h)            
                n,m = s[1],s[2]
                if self.active_padded[n+1,m+2]: self.integrate(n,m+1,n,m, 1, fgrad) # North                             
                if self.active_padded[n+1,m]: self.integrate(n,m-1,n,m, -1, fgrad) # South                         
                if self.active_padded[n+2,m+1]: self.integrate(n+1,m,n,m,1, tgrad)  # East                            
                if self.active_padded[n,m+1]: self.integrate(n-1,m,n,m, -1, tgrad) # West                                               
#                 if self.active_padded[n+2,m+2]: self.integrate(n+1,m+1,n,m, 1, dphaseNE) # NE                          
#                 if self.active_padded[n+2,m]: self.integrate(n+1 ,m-1,n,m, 1, dphaseSE) #SE      
#                 if self.active_padded[n,m]: self.integrate(n-1,m-1,n,m, -1, dphaseNE) # SW    
#                 if self.active_padded[n,m+2]: self.integrate(n-1,m+1,n,m, -1, dphaseSE) # NE 
        if self.plt.verbose:
            nprocessed = np.sum(np.where(mask,1,0))                           
            self.logprint ('magnitudes processed above threshold tolerance={}, magnitudes rejected below threshold tolerance={}'.format(nprocessed, magnitude.size-nprocessed) ) 
            self.plt.plot_3d('magnitude', [magnitude], mask=mask, startpoints=self.startpoints)             
            self.plt.plot_3d('fgrad',[fgrad], mask=mask, startpoints=self.startpoints)
            self.plt.plot_3d('tgrad',[tgrad], mask=mask, startpoints=self.startpoints)      
            if self.original_phase is not None: 
                self.plt.plot_3d('Phase original', [self.original_phase], mask=mask,startpoints=self.startpoints)          
                self.plt.plot_3d('Phase original, Phase estimated', [(self.original_phase) %(2*np.pi), ( self.phase) %(2*np.pi)], mask=mask, startpoints=self.startpoints)       
                self.plt.quiver('phase errors (2pi)%', self.q_errors, startpoints= self.startpoints)                                   
            self.plt.plot_3d('Phase estimated', [self.phase], mask=mask,startpoints=self.startpoints)                      
         
        return self.phase
    
    def integrate(self, n1, m1, n0, m0, c1, dphase):
        '''
        implements y = phase[n0,m0] + c1*dphase*t 
        
        parameters:
            n1,m1 = destination for integrated phase
            n0,m0 = source of integrated phase
            c1
                scalar unit change 
                    should be +/- 2*pi*m/M when going north or south
                    should be +/- a hop when going east or west       
            dphase
                derivative of the phase with respect to w or t
                should be dphasew when going north or south
                should be dphaset when going east or west                       
        '''
#         if self.active_padded[1+n1,1+m1] == False: return
        self.active_padded[1+n1,1+m1]=False                  
        self.phase[n1,m1]=  self.phase[n0,m0] + c1*(dphase[n0,m0] + dphase[n1,m1])/2 
        heapq.heappush(self.h, (-self.magnitude[n1,m1],n1,m1))        
         
        # plot the first 2000 points
        if self.plt.verbose and self.debug_count <= 2000:    
            dif = (self.phase[n1,m1] - self.phase[n0,m0]) %(2*np.pi)
            if self.original_phase is None:        
                dif_orig = dif                     
            else:
                if n1 != n0:
                    dif_orig = (self.original_phase[n1,m1] - self.original_phase[n0,m0] )%(2*np.pi)       
                elif m1 != m0:
                    dif_orig = (self.original_phase[n1,m1] - self.original_phase[n0,m0])%(2*np.pi)   
            if dif==0:
                err_new = 0
            else:         
                err_new = (dif - dif_orig) / (2*np.pi)
            # print a few heap pops
            if self.debug_count < 10:  
                if m1 == m0+1: 
                    self.logprint('###############################   POP   ###############################')  
                self.logprint(['','NORTH','SOUTH'][m1-m0]+ ['','EAST','WEST'][n1-n0])         
                self.logprint ('n1,m1=({},{}) n0,m0=({},{})'.format(n1,m1,n0,m0))
                self.logprint ('\testimated phase[n,m]={:13.4f}, phase[n0,m0]         =:{:13.4f}, dif(2pi)     ={:9.4f}'.format((self.phase[n1,m1]) , (self.phase[n0,m0]), dif )) 
                if self.original_phase is not None:    
                    self.logprint ('\toriginal_phase[n,m] ={:13.4f}, original_phase[n0,m0]=:{:13.4f}, dif_orig(2pi)={:9.4f}'.format((self.original_phase[n1,m1]) , (self.original_phase[n0,m0])  ,dif_orig))                                          
                    self.logprint('error ={:9.4f}%'.format(100*err_new))  
            self.q_errors.append((n0,m0,abs(dif) ,n1-n0,m1-m0,err_new))                                                                      
            self.debug_count += 1          
        return
   
    def magphase_to_complex(self,magnitude, phase):
        return magnitude*(np.cos(phase)+ np.sin(phase)*1j)
    
    def magphase_frames_to_signal(self, magnitude, phase):
        return self.complex_frames_to_signal(self.magphase_to_complex(magnitude, phase))
    
    def complex_to_magphase(self, corig ):
        return  np.absolute(corig),np.angle(corig)     

    def signal_to_frames(self, s):   # applies window function, g
        self.plt.signal_to_file(s , 'signal_in' )
        self.plt.spectrogram(s,'spectrogram signal in')        
        L = s.shape[0] - self.M
        self.corig_frames = np.stack( [np.fft.rfft(self.g*s[ix:ix + self.M]) for ix in range(0, L, self.a_a)])
        return self.corig_frames  
    
    def complex_frames_to_signal(self, complex_frames): 
        M2 = complex_frames.shape[1]
        N = complex_frames.shape[0]      
        vr=np.fft.irfft(complex_frames)         
        sig = np.zeros((N*self.a_a+self.M))
        for k in range(N):
            vs = vr[k]*self.gsynth*self.M
            sig[k*self.a_a: k*self.a_a+self.M] += vs

        self.plt.signal_to_file(sig , 'signal_out')
        self.plt.spectrogram(sig, 'spectrogram signal out')          
        return sig
    
    def signal_to_magphase_frames(self, s):    
        return self.complex_to_magphase(self.signal_to_frames(s))
         
    def signal_to_signal(self,signal_in):
        ''' 
          convert signal_in to frames
            throw away the phase
            reconstruct the phase from the magnitudes
            re-run fft and compute the frobenius norm for an error value  
                  
            parameter:
                signal_in numpy array (length,)
  
            return:
                 reconstructed signal
        '''
        magnitude_frames, _ = self.signal_to_magphase_frames(signal_in)
        phase_estimated_frames = self.magnitude_to_phase_estimate(magnitude_frames)
        signal_out = self.magphase_frames_to_signal(magnitude_frames, phase_estimated_frames)
        self.plt.plot_waveforms('Signal in, Signal out', [signal_in, signal_out])
        saved_verbose = self.setverbose(False)        
        reconstructed_magnitude, _ = self.signal_to_magphase_frames(signal_out)
        self.setverbose(saved_verbose)
        s1 = self.plt.normalize(magnitude_frames[1:]) # s1 is delayed by 1 frame with respect to s2
        s2 = self.plt.normalize(reconstructed_magnitude[:-1])
        self.plt.plot_3d('magnitude_frames, reconstructed_magnitude', [s1[100:110], s2[100:110] ])   
        E = np.linalg.norm(s2- s1)/np.linalg.norm(s1)    # Frobenius norm
        self.logprint ("\nerror measure = {:8.4f} dB".format(20*np.log10(E)))  
        return signal_out              
