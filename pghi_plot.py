'''
Created on Jul 26, 2018

@author: richard
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as signal
import numpy as np
import os
from pydub import AudioSegment
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


colors =  ['r', 'g', 'b', 'c', 'm', 'y', 'k']
file_sep = ' '
class Pghi_Plot(object):
    '''
    classdocs
    '''

    def __init__(self, show_plots=True, show_frames = 5, pre_title=''):
        '''
        parameters:
            show_plots
                if True, then display each plot on the screen before saving
                to the disk. Useful for rotating 3D plots with the mouse
                if False, just save the plot to the disk in the './pghi_plots' directory
            pre_title
                string: pre_titleription to be prepended to each plot title
        '''
        
        self.show_plots, self.show_frames, self.pre_title = show_plots, show_frames, pre_title
        try:
            os.mkdir('./pghi_plots')
        except:
            pass
        self.openfile = ''
        
    def save_plots(self, title):  
        file =  './pghi_plots/' + title +  '.png' 
        print ('saving plot to file:' + file)        
        plt.savefig(file,  dpi=300)    
        if self.show_plots:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()            
            plt.show()             
        else:               
            plt.clf() # savefig does not clear the figure like show does
            plt.cla()      
            plt.close()
         
    def spectrogram(self, samples, title):
        title = self.pre_title  +file_sep+title      
        plt.title( title )           
        ff, tt, Sxx = signal.spectrogram(samples, fs=44100, nfft=8192)
    
        plt.pcolormesh(tt, ff[:1025], Sxx[:1025], cmap='gray_r')
        plt.xlabel('samples')
        plt.ylabel('Frequency (Hz)')
        plt.grid()
        self.save_plots(title)
        
    prop_cycle = plt.rcParams['axes.prop_cycle']    

    def plot_waveforms(self, title, sigs,fontsize=None):    
        title = self.pre_title  + file_sep + title  

        fig = plt.figure()
        plt.title(title)   
        plt.ylabel('amplitude', color='b',fontsize=fontsize)
        plt.xlabel('Samples',fontsize=fontsize)    
        ax = plt.gca()
    
        for i,s in enumerate(sigs):
            xs = np.arange(s.shape[0])
            ys = s
            ax.scatter(xs, ys, color = colors[i],s=3)      
        plt.grid()
        plt.axis('tight')
        self.save_plots(title)
        
    def minmax(self, startpoints, stime, sfreq):
        if startpoints is None:
            minfreq = mintime = 0
            maxfreq = maxtime = 2*self.show_frames
        else:
            starttimes = [s[0] for s in startpoints]
            startfreqs = [s[1] for s in startpoints]                
            mintime = max(0,min(starttimes)-self.show_frames)
            maxtime = min(stime,max(starttimes)+self.show_frames)
            minfreq = max(0,min(startfreqs)-self.show_frames)
            maxfreq = min(sfreq,max(startfreqs)+self.show_frames)  
        return mintime, maxtime, minfreq, maxfreq
    
    def subplot(self, figax, sigs, r, c, p, elev, azim, mask, startpoints, fontsize=None):

        ax = figax.add_subplot(r,c,p, projection='3d',elev = elev, azim=azim)     
        for i, s in enumerate(sigs):
            mintime, maxtime, minfreq, maxfreq = self.minmax(startpoints, s.shape[0], s.shape[1])            
            values = s[mintime:maxtime, minfreq:maxfreq]           
            if mask is None:  #plot all values                                      
                xs = np.arange(values.size) % values.shape[0]
                ys = np.arange(values.size) // values.shape[1]
                zs = np.reshape(values,(values.size))
            else:                         
                indices = np.where(mask[mintime:maxtime, minfreq:maxfreq]   == True)
                xs = indices[0] + mintime 
                ys = indices[1] + minfreq      
                zs = values[indices]  
            if i==0:
                sn=8
            else:
                sn=3  
            ax.scatter(xs, ys, zs, s=sn, color = colors[i+1]) 
        if startpoints is not None:
            for stpt in startpoints:                
                n = stpt[0] 
                m = stpt[1]
            ax.scatter([n],[m], [s[n,m]], s=30, color = colors[0])
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.zaxis.set_major_formatter(StrMethodFormatter('{x:.2e}'))            
                       
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 
            tick.label.set_rotation('vertical')
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 
            tick.label.set_rotation('vertical') 
        for tick in ax.zaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 
            tick.label.set_rotation('horizontal')    
                                                             
        ax.set_zlabel('mag',fontsize=fontsize)   
        ax.set_ylabel('freq',fontsize=fontsize)
        ax.set_xlabel('frames',fontsize=fontsize)
        
    def signal_to_file(self, mono, title, Fs = 44100 ):       
        title = self.pre_title  + file_sep+ title              
        print('saving signal to file: {}'.format(title))
        w = .5*(np.max(mono)-np.min(mono))
        mono = mono/w
        a = np.max(mono)+np.min(mono)
        mono = mono - a/2    
        mono = np.array(mono) *( 2**15-1)
        stereo = [mono, mono]
        stereo = np.array(stereo, dtype=np.int16)
        stereo = np.rollaxis(stereo, 1)
        stereo = stereo.flatten()
        stereo = stereo[: 4*(stereo.shape[0]//4)]
        output_sound = AudioSegment(data=stereo, sample_width=2,frame_rate=Fs, channels=2)    
        try:
            os.mkdir('./pghi_plots')
        except:
            pass     
        output_sound.export("./pghi_plots/"+title+".mp3", format="mp3")  
                        
    def plot_3d(self, title, sigs, mask=None, startpoints=None):
        title = self.pre_title  + file_sep + title       
        figax = plt.figure()   
        plt.axis('off')    
        plt.title( title ) 

        if self.show_plots:      
            self.subplot(figax, sigs, 1,1, 1, 45, 45, mask,startpoints,fontsize=8)   
        else:         
            self.subplot(figax, sigs, 2,2, 1, 45, 45, mask,startpoints,fontsize=6)
            self.subplot(figax, sigs, 2,2, 2, 0,  0,  mask,startpoints,fontsize=6)
            self.subplot(figax, sigs, 2,2, 3, 0,  45, mask,startpoints,fontsize=6)
            self.subplot(figax, sigs, 2,2, 4, 0,  90, mask,startpoints,fontsize=6)                           
        self.save_plots(title)     
        
    def quiver(self, title, qtuples, mask=None, startpoints=None):   
        title = self.pre_title + file_sep + title

        figax = plt.figure()
        ax = figax.add_subplot(111, projection='3d',elev = 45, azim=45)
        plt.title(title)        
        stime = max([q[0] + q[3] for q in qtuples])
        sfreq = max([q[1] + q[4] for q in qtuples])
        mintime, maxtime, minfreq, maxfreq = self.minmax(startpoints, stime, sfreq)             
        x, y, z, u, v, w = [],[],[],[],[],[]
        for q in qtuples:        
            if q[0] < mintime or q[0] > maxtime or q[1] < minfreq or q[1] > maxfreq:
                continue;
            x.append(q[0])
            y.append(q[1])
            z.append(q[2])
            u.append(q[3])
            v.append(q[4])
            w.append(q[5])                        
                   
        ax.quiver(x,y,z,u,v,w,length=.5, arrow_length_ratio=.3, pivot='tail', color = colors[1], normalize=False)
        if startpoints is not None:
            for stpt in startpoints:                
                n = stpt[0]
                m = stpt[1]
                ax.scatter([n],[m], [z[0]], s=30, color = colors[0])     
        self.save_plots(title)
         
    def logprint(self, txt):   
        if self.openfile != './pghi_plots/' + self.pre_title + '.txt' :
            self.openfile = './pghi_plots/' + self.pre_title+ '.txt' 
            self.file = open(self.openfile, mode='w')
        print(txt, file=self.file, flush=True)
        print(txt)
        
                     