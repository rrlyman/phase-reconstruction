'''
Created on Jul 7, 2018

@author: richard lyman
'''
import numpy as np
import heapq 
import scipy.signal as signal
import pghi_plot

dtype = np.float64

class Rtpghi(object):
    '''
    classdocs
    '''

    def __init__(self, a_a=128, M=2048, gl=None, g=None, tol = 1e-6, lambdasqr = None, h = .01, plt=None, alg='p2015', pre_title='init', show_plots = False,  show_frames = 25):
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
        Example
            pghi = amf.Rtpghi(a_a=256, M=2048,tol = 1e-6, show_plots = False, show_frames=20)
        '''  
        if gl is None: gl = M  
        if g is None:  
            # Auger,Motin,Flandrin #19            
            lambda_ = (-gl**2/(8*np.log(h)))**.5
            lambdasqr = lambda_**2
            g=np.array(signal.windows.gaussian(gl, lambda_, sym=False), dtype = dtype)
         
        if lambdasqr is None: self.logprint('parameter error: must supply lambdasqr and g')
        self.a_a,self.M,self.tol,self.lambdasqr,self.g,self.gl,self.h, self.pre_title  = a_a,M,tol,lambdasqr,g,gl,h,pre_title

        self.M2 = int(self.M/2) + 1          
        self.redundancy = int (self.M/self.a_a)         
        
        self.plt = pghi_plot.Pghi_Plot( show_plots = show_plots,  show_frames = show_frames, pre_title=pre_title)       

        self.logprint('a_a(analysis time hop size) = {} samples'.format(a_a))    
        self.logprint('M, samples per frame = {}'.format(M))     
        self.logprint('tol, small signal filter tolerance ratio = {}'.format(tol))  
        self.logprint('lambdasqr = {:9.4f} samples**2 '.format(self.lambdasqr))
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
    
    def title(self, title):
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
#         dt =    (np.multiply(3,(xp[:-2,1:-1])    + np.multiply(2,xp[1:-1,1:-1])    + np.multiply(3,xp[2:,1:-1]))   - np.multiply(6,(xp[:-2,1:-1]    + xp[1:-1,1:-1]    + xp[2:,1:-1])))/6         
        dt = (xp[2:,1:-1]-xp[:-2,1:-1])/2         

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
        N = magnitude.shape[0]      
        wbin = 2*np.pi/self.M  # bin size in radians, one frequency step          
        
        # debugging
        self.debug_count=0
        self.original_phase = np.angle(self.corig_frames)
        self.q_errors=[]
        
        # small signal filter
        mask = magnitude > self.tol*np.max(magnitude)
        nprocessed = np.sum(np.where(mask,1,0))           
        self.logprint ('magnitudes processed above threshold tolerance={}, magnitudes rejected below threshold tolerance={}'.format(nprocessed, magnitude.size-nprocessed) )          
        
        logs = np.log(magnitude+1e-50)  
        
        tadd = self.a_a*wbin*np.arange(self.M2) 
        tgrad = self.a_a*self.dxdw(logs)/(wbin*self.lambdasqr) 
        
        fgrad = wbin* self.lambdasqr*self.dxdt(logs)/self.a_a
        fadd = self.a_a/2       
        
        dphasedt =   tgrad + tadd
        dphasedw = -(fgrad + fadd)       
        
        dphaseNE =  tgrad + tadd -(fgrad + fadd) 
        dphaseSE =  tgrad + tadd +(fgrad + fadd)      

        self.phase = np.random.random_sample(magnitude.shape)*2*np.pi      
        self.active_mask = np.pad(mask,1,mode='constant',constant_values=False)
              
        self.startpoints = []  
        self.h=[]   
        
        #add known phase borders to the heap
#         for n in range(N): 
#             if self.active_mask[n+1,1]:
#                 heapq.heappush(self.h, (-magnitude[n,0], n, 0))     
#             if self.active_mask[n+1,-1]:
#                 heapq.heappush(self.h, (-magnitude[n,self.M2-1], n, self.M2-1)) 
#         for m in range(self.M2): 
#             if self.active_mask[1,m+1]:
#                 heapq.heappush(self.h, (-magnitude[0,m], 0, m))     
#             if self.active_mask[-1,m+1]:
#                 heapq.heappush(self.h, (-magnitude[N-1,m], N-1, m))                                     
                          
        while np.any(self.active_mask): 
            # original PGHI algorithm, start at highest point                
            nm = np.argmax(magnitude*self.active_mask[1:-1,1:-1])
            m = nm % self.M2
            n = nm // self.M2
            self.startpoints.append((n,m))        
            self.logprint('Processing Island: start point=[{},{}]'.format(n,m))
            self.phase[n,m] = 0

            heapq.heappush(self.h, (-magnitude[n,m],n,m)) 
            self.active_mask[n+1,m+1]=False
            self.original_phase-=self.original_phase[n] + (np.arange(self.M2) -m ) *self.a_a/2

            while len(self.h) > 0:
                s=heapq.heappop(self.h)            
                n,m = s[1],s[2]
                self.integrate(n,m+1,n,m, 1, dphasedw) # North                             
                self.integrate(n,m-1,n,m, -1, dphasedw) # South                         
                self.integrate(n+1,m,n,m,1, dphasedt)  # East                            
                self.integrate(n-1,m,n,m, -1, dphasedt) # West                                               
#                     self.integrate(n+1,m+1,n,m, 1, dphaseNE) # NE                          
#                     self.integrate(n+1 ,m-1,n,m, 1, dphaseSE) #SE      
#                     self.integrate(n-1,m-1,n,m, -1, dphaseNE) # SW    
#                     self.integrate(n-1,m+1,n,m, -1, dphaseSE) # NE                            
 
        self.plt.plot_3d('magnitude', [magnitude], mask=mask, startpoints=self.startpoints)             
        self.plt.plot_3d('dphasedw',[dphasedw], mask=mask, startpoints=self.startpoints)
        self.plt.plot_3d('dphasedt',[dphasedt], mask=mask, startpoints=self.startpoints)       
        self.plt.plot_3d('Phase original', [self.original_phase], mask=mask,startpoints=self.startpoints)                 
        self.plt.plot_3d('Phase estimated', [self.phase], mask=mask,startpoints=self.startpoints)                      
        self.plt.plot_3d('Phase original, Phase estimated', [(self.original_phase) %(2*np.pi), ( self.phase) %(2*np.pi)], mask=mask, startpoints=self.startpoints)  
        self.plt.quiver('phase errors (2pi)%', self.q_errors, startpoints= self.startpoints)            
                           
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
                note: the step size, wbin or self.a_a is included in dphasew/dphaset                       
        '''
        if self.active_mask[1+n1,1+m1] == False: return
        self.active_mask[1+n1,1+m1]=False                  
        self.phase[n1,m1]=  self.phase[n0,m0] + c1*(dphase[n0,m0] + dphase[n1,m1])/2 
        heapq.heappush(self.h, (-self.magnitude[n1,m1],n1,m1))        
         
        # plot the first 2000 points
        if self.debug_count <= 2000:    
            dif = (self.phase[n1,m1] - self.phase[n0,m0]) %(2*np.pi) 
            if n1 != n0:
                dif_orig = (self.original_phase[n1,m1] - self.original_phase[n0,m0] )%(2*np.pi)       
            elif m1 != m0:
                dif_orig = (self.original_phase[n1,m1] - self.original_phase[n0,m0])%(2*np.pi)             
            err_new = (dif - dif_orig) / abs(dif)
            # print a few heap pops
            if (self.debug_count>=0) and (self.debug_count < 10):  
                if m1 == m0+1: 
                    self.logprint('###############################   POP   ###############################')  
                self.logprint(['','NORTH','SOUTH'][m1-m0]+ ['','EAST','WEST'][n1-n0])         
                self.logprint ('n1,m1=({},{}) n0,m0=({},{})'.format(n1,m1,n0,m0))
                self.logprint ('\testimated phase[n,m]={:13.4f}, phase[n0,m0]         =:{:13.4f}, dif(2pi)     ={:9.4f}'.format((self.phase[n1,m1]) , (self.phase[n0,m0]), dif ))     
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
        self.plt.signal_to_file(sig , 'signal out')
        self.plt.spectrogram(sig, 'spectrogram signal out')          
        return sig
    
    def signal_to_magphase_frames(self, s):    
        return self.complex_to_magphase(self.signal_to_frames(s))
         