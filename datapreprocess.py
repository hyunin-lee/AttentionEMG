
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import pickle

class emgDataProcess() :
    def __init__(self,read_path,starttime,endtime,middle_time_list, emgfps,resample_fs):
        self.read_path = read_path
        self.data = np.transpose(self.loadTrajectory())
        self.starttime = starttime
        self.middle_time_list = middle_time_list
        self.endtime = endtime
        self.resample_fs = resample_fs
        self.time = self.data[0,int(starttime * emgfps) : int(endtime * emgfps)]
        self.rawEmg = self.data[1:,int(starttime * emgfps) : int(endtime * emgfps)]
        self.resampleEmg, self.resampleTime = self.resampling_total()
        self.emglist, self.timelist = self.resampling_part(middle_time_list)

    def loadTrajectory(self):
        import pickle
        with open(self.read_path, 'rb') as f:
            a = pickle.load(f)
        return a

    def findindex(self,time):
        index1 = next(x for x, val in enumerate(self.time)if val > time-0.5*duration)
        index2 = next(x for x, val in enumerate(self.time) if val > time+0.5*duration)
        return index1 , index2

    def dividevidieo(self,time):
        idx1,idx2 =self.findindex(time)
        t = self.time[idx1: idx2]
        rawEmg = self.rawEmg[:, idx1: idx2]
        return t,rawEmg

    def resampling_total(self):
        size = int((self.endtime-self.starttime)*self.resample_fs)
        resample_data = np.zeros((4,size))
        for i in range(4) :
            resample_data[i,:] = signal.resample(self.rawEmg[i,:],size,axis=0)
        return resample_data, np.linspace(self.time[0], self.time[-1],size)

    def resampling_part(self,middletime_list):
        timelist , emglist = [], []
        for middle_time in middletime_list :
            t,rawEmg = self.dividevidieo(middle_time)
            size = int((t[-1]-t[0])*self.resample_fs)
            resample_data = np.zeros((4, size))
            for i in range(4):
                resample_data[i, :] = signal.resample(rawEmg[i, :], size, axis=0)
            timelist.append(np.linspace(t[0], t[-1],size))
            emglist.append(resample_data)

        assert len(emglist) == dataLength
        assert len(timelist) == dataLength

        return emglist ,timelist


    def plotEmg(self):
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(411)
        ax2 = fig1.add_subplot(412)
        ax3 = fig1.add_subplot(413)
        ax4 = fig1.add_subplot(414)
        ax1.plot(self.time, self.rawEmg[0, :] , label='0')
        #ax1.plot(self.resampleTime, self.resampleEmg[0, :],'x')
        ax2.plot(self.time, self.rawEmg[1, :], label='1')
        #ax2.plot(self.resampleTime, self.resampleEmg[1, :],'x', label='1')
        ax3.plot(self.time, self.rawEmg[2, :], label='2')
        #ax3.plot(self.resampleTime, self.resampleEmg[2, :],'x', label='2')
        ax4.plot(self.time, self.rawEmg[3, :], label='3')
        #ax4.plot(self.resampleTime, self.resampleEmg[3, :],'x', label='3')

        plt.show()

    def plotEMGpart(self):

        for idx,(time,emg) in enumerate(zip(self.timelist,self.emglist)) :
            fig1 = plt.figure(idx)
            ax1 = fig1.add_subplot(411)
            ax2 = fig1.add_subplot(412)
            ax3 = fig1.add_subplot(413)
            ax4 = fig1.add_subplot(414)

            ax1.plot(time, emg[0, :], label='0')
            ax1.set_ylim([0, 1])
            ax2.plot(time, emg[1, :], label='1')
            ax2.set_ylim([0, 1])
            ax3.plot(time, emg[2, :], label='2')
            ax3.set_ylim([0, 1])
            ax4.plot(time, emg[3, :], label='3')
            ax4.set_ylim([0, 1])

            plt.show()

class angleDataprocess() :
    def __init__(self,read_path,starttime,endtime,middletime_list,videofps,resample_fs):
        self.read_path = read_path
        self.starttime = starttime
        self.endtime = endtime
        self.middletime_list = middletime_list
        self.time =  starttime
        self.data = self.loadTrajectory()
        self.resample_fs = resample_fs
        self.time = self.data[14, int(starttime * videofps) : int(endtime * videofps)]
        self.rawAngles = self.data[:14, int(starttime * videofps) : int(endtime * videofps)]
        self.smoothAngles = self.datasmoothing()
        self.normalizeNsmoothAngles =self.makeZeroAndOne()
        self.resampleAngles, self.resampleTime = self.resampling_total()
        self.anglelist , self.timelist = self.resampling_part(middletime_list)


    def loadTrajectory(self):
        import pickle
        with open(self.read_path, 'rb') as f:
            a = pickle.load(f)
        return a

    def dividevidieo(self,middletime):
        idx1,idx2 =self.findindex(middletime)
        t_part = self.time[idx1: idx2]
        normalizeNsmoothAngles_part = self.normalizeNsmoothAngles[:, idx1: idx2]
        return t_part,normalizeNsmoothAngles_part

    def findindex(self,time):
        index1 = next(x for x, val in enumerate(self.time)if val > time-0.5*duration)
        index2 = next(x for x, val in enumerate(self.time) if val > time+0.5*duration)
        return index1 , index2

    def datasmoothing(self):
        smoothAngles = np.zeros_like(self.rawAngles)
        windowsize = int(2 / (self.time[1] - self.time[0])) if not int(2 / (self.time[1] - self.time[0]))%2 == 0 else int(2 / (self.time[1] - self.time[0])) + 1
        poly_deg = 6

        for i in range(14) :
            smoothAngles[i,:] = savgol_filter(self.rawAngles[i,:], windowsize, poly_deg)

        return smoothAngles

    def makeZeroAndOne(self):
        normalizeNsmoothAngles = np.zeros_like(self.rawAngles)
        for i in range(14) :
            angles = self.smoothAngles[i,:]-np.min(self.smoothAngles[i,:])
            normalizeNsmoothAngles[i,:] = angles * 1/np.max(angles)
            #normalizeNsmoothAngles[i,:] = -(normalizeNsmoothAngles[i,:]-1)

        return normalizeNsmoothAngles

    def resampling_total(self):
        size = (self.endtime-self.starttime)*self.resample_fs
        resample_data = np.zeros((14,size))
        for i in range(14) :
            resample_data[i,:] = signal.resample(self.normalizeNsmoothAngles[i,:],size,axis=0)
        return resample_data, np.linspace(self.time[0], self.time[-1],size)

    def resampling_part(self,middletime_list):
        anglelist, timelist = [], []
        for middle_time in middletime_list :
            t,angles = self.dividevidieo(middle_time)
            size = int((t[-1]-t[0])*self.resample_fs)
            resample_data = np.zeros((14, size))
            for i in range(14):
                resample_data[i, :] = signal.resample(angles[i, :], size, axis=0)
            timelist.append(np.linspace(t[0], t[-1],size))
            anglelist.append(resample_data)
            #self.inittime += 2.0

        assert len(anglelist) == dataLength
        assert len(timelist) == dataLength

        return anglelist, timelist

    def plotAngles(self):
        fig2 = plt.figure(2,figsize=(16.0, 10.0))
        ax1 = fig2.add_subplot(211)
        ax2 = fig2.add_subplot(212)
        ax1.title.set_text('Thumb')
        ax1.plot(self.time, self.rawAngles[0, :] , label='0')
        ax1.plot(self.time, self.smoothAngles[0,:], label='0')
        ax1.plot(self.time, self.normalizeNsmoothAngles[0, :], label='0')
        #ax1.plot(self.resampleTime, self.resampleAngles[0, :],'x',label='0')
        ax2.plot(self.time, self.rawAngles[1, :], label='1')
        ax2.plot(self.time, self.smoothAngles[1,:], label='1')
        ax2.plot(self.time, self.normalizeNsmoothAngles[1, :], label='1')
        #ax2.plot(self.resampleTime, self.resampleAngles[1, :],'x', label='1')
        fig2.savefig(savepath+filename+'_thumb')

        fig3 = plt.figure(3,figsize=(16.0, 10.0))
        ax1 = fig3.add_subplot(311)
        ax2 = fig3.add_subplot(312)
        ax3 = fig3.add_subplot(313)
        ax1.title.set_text('index')
        ax1.plot(self.time, self.rawAngles[2, :] , label='2')
        ax1.plot(self.time, self.smoothAngles[2,:], label='2')
        ax1.plot(self.time, self.normalizeNsmoothAngles[2, :], label='2')
        ax2.plot(self.time, self.rawAngles[3, :], label='3')
        ax2.plot(self.time, self.smoothAngles[3,:], label='3')
        ax2.plot(self.time, self.normalizeNsmoothAngles[3, :], label='3')
        ax3.plot(self.time, self.rawAngles[4, :], label='4')
        ax3.plot(self.time, self.smoothAngles[4,:], label='4')
        ax3.plot(self.time, self.normalizeNsmoothAngles[4, :], label='4')
        fig3.savefig(savepath+filename + '_index')

        fig4 = plt.figure(4,figsize=(16.0, 10.0))
        ax1 = fig4.add_subplot(311)
        ax2 = fig4.add_subplot(312)
        ax3 = fig4.add_subplot(313)
        ax1.title.set_text('middle')
        ax1.plot(self.time, self.rawAngles[5, :] , label='5')
        ax1.plot(self.time, self.smoothAngles[5,:], label='5')
        ax1.plot(self.time, self.normalizeNsmoothAngles[5, :], label='5')
        ax2.plot(self.time, self.rawAngles[6, :], label='6')
        ax2.plot(self.time, self.smoothAngles[6,:], label='6')
        ax2.plot(self.time, self.normalizeNsmoothAngles[6, :], label='6')
        ax3.plot(self.time, self.rawAngles[7, :], label='7')
        ax3.plot(self.time, self.smoothAngles[7,:], label='7')
        ax3.plot(self.time, self.normalizeNsmoothAngles[7, :], label='7')
        fig4.savefig(savepath+filename + '_middle')

        fig5 = plt.figure(5,figsize=(16.0, 10.0))
        ax1 = fig5.add_subplot(311)
        ax2 = fig5.add_subplot(312)
        ax3 = fig5.add_subplot(313)
        ax1.title.set_text('ring')
        ax1.plot(self.time, self.rawAngles[8, :] , label='8')
        ax1.plot(self.time, self.smoothAngles[8,:], label='8')
        ax1.plot(self.time, self.normalizeNsmoothAngles[8, :], label='8')
        ax2.plot(self.time, self.rawAngles[9, :], label='9')
        ax2.plot(self.time, self.smoothAngles[9,:], label='9')
        ax2.plot(self.time, self.normalizeNsmoothAngles[9, :], label='9')
        ax3.plot(self.time, self.rawAngles[10, :], label='10')
        ax3.plot(self.time, self.smoothAngles[10,:], label='10')
        ax3.plot(self.time, self.normalizeNsmoothAngles[10, :], label='10')
        fig5.savefig(savepath + filename + '_ring')

        fig6 = plt.figure(6,figsize=(16.0, 10.0))
        ax1 = fig6.add_subplot(311)
        ax2 = fig6.add_subplot(312)
        ax3 = fig6.add_subplot(313)
        ax1.title.set_text('pinky')
        ax1.plot(self.time, self.rawAngles[11, :] , label='11')
        ax1.plot(self.time, self.smoothAngles[11,:], label='11')
        ax1.plot(self.time, self.normalizeNsmoothAngles[11, :], label='11')
        ax2.plot(self.time, self.rawAngles[12, :], label='12')
        ax2.plot(self.time, self.smoothAngles[12,:], label='12')
        ax2.plot(self.time, self.normalizeNsmoothAngles[12, :], label='12')
        ax3.plot(self.time, self.rawAngles[13, :], label='13')
        ax3.plot(self.time, self.smoothAngles[13,:], label='13')
        ax3.plot(self.time, self.normalizeNsmoothAngles[13, :], label='13')
        fig6.savefig(savepath + filename + '_pinky',bbox_inches='tight')

        plt.show()

    def plotAnglespart(self):

        for idx,(time,angles) in enumerate(zip(self.timelist,self.anglelist)) :
            fig2 = plt.figure(1)
            ax1 = fig2.add_subplot(3,5,1)
            ax2 = fig2.add_subplot(3,5,6)
            ax1.title.set_text('Thumb')
            ax1.plot(time, angles[0, :], label='0')
            ax2.plot(time, angles[1, :], label='1')
            ax1.set_ylim([0, 1])
            ax2.set_ylim([0, 1])

            #fig3 = plt.figure(2)
            ax3 = fig2.add_subplot(3,5,2)
            ax4 = fig2.add_subplot(3,5,7)
            ax5 = fig2.add_subplot(3,5,12)
            ax3.title.set_text('index')
            ax3.plot(time, angles[2, :], label='0')
            ax4.plot(time, angles[3, :], label='1')
            ax5.plot(time, angles[4, :], label='1')
            ax3.set_ylim([0, 1])
            ax4.set_ylim([0, 1])
            ax5.set_ylim([0, 1])

            #fig4 = plt.figure(3)
            ax6 = fig2.add_subplot(3,5,3)
            ax7 = fig2.add_subplot(3,5,8)
            ax8 = fig2.add_subplot(3,5,13)
            ax6.title.set_text('middle')
            ax6.plot(time, angles[5, :], label='0')
            ax7.plot(time, angles[6, :], label='1')
            ax8.plot(time, angles[7, :], label='1')
            ax6.set_ylim([0, 1])
            ax7.set_ylim([0, 1])
            ax8.set_ylim([0, 1])

            #fig5 = plt.figure(4)
            ax9 = fig2.add_subplot(3,5,4)
            ax10 = fig2.add_subplot(3,5,9)
            ax11 = fig2.add_subplot(3,5,14)
            ax9.title.set_text('ring')
            ax9.plot(time, angles[8, :], label='0')
            ax10.plot(time, angles[9, :], label='1')
            ax11.plot(time, angles[10, :], label='1')
            ax9.set_ylim([0, 1])
            ax10.set_ylim([0, 1])
            ax11.set_ylim([0, 1])

            #fig6 = plt.figure(5)
            ax12 = fig2.add_subplot(3,5,5)
            ax13 = fig2.add_subplot(3,5,10)
            ax14 = fig2.add_subplot(3,5,15)
            ax12.title.set_text('pinky')
            ax12.plot(time, angles[11, :], label='0')
            ax13.plot(time, angles[12, :], label='1')
            ax14.plot(time, angles[13, :], label='1')
            ax12.set_ylim([0, 1])
            ax13.set_ylim([0, 1])
            ax14.set_ylim([0, 1])

            plt.show()

class Data():
    def __init__(self,x_data,y_data):
        self.x_data = x_data
        self.y_data = y_data

def dataprepare(data_path,emg_file_name,angle_file_name) :
    path_emg = data_path + emg_file_name
    path_angle = data_path + angle_file_name

    import pickle
    with open(path_emg,'rb') as f :
        emgdata = pickle.load(f)
    with open(path_angle,'rb') as f :
        angledata = pickle.load(f)

    return emgdata , angledata

def synctime(inputdata,outputdata,starttime,videofs,emgfs, sync=True) :

    t_emg = inputdata[:starttime * emgfs, 0]
    x_emg = inputdata[:starttime * emgfs, 1]

    t_angle = outputdata[14,:int(starttime * videofs)]
    x_angle = outputdata[0,:int(starttime * videofs)]

    peaks_emg, _ = find_peaks(x_emg, distance=0.7 * emgfs)
    peaks_angle, _ = find_peaks(-x_angle, distance=0.7 * videofs)

    if sync :
        ## 2 sec file
        # train_lhi_1 -> emg : 4, angle : 3
        # train_lhi_2 -> emg : 3, angle : 3
        # train_lhi_3 -> emg : 5, angle : 4
        # train_lhi_4 -> emg : 8, angle : 5
        # train_lhi_5 -> emg : 2, angle : 3
        # train_lhi_6 -> emg : 9, angle : 11
        # train_lhi_7 -> emg : 9, angle : 8
        # train_lhi_8 -> emg : 9, angle : 7
        # train_lhi_9 -> emg : 4, angle : 1
        # test_lhi_1 -> emg : 4, angle : 1

        ## 1 sec file
        # train_lhi_1 -> emg : 3, angle : 3
        # train_lhi_2 -> emg : 2, angle : 3
        # train_lhi_3 -> emg : 0, angle : 5
        emg_idx = peaks_emg[4]
        angle_idx = peaks_angle[5]
        t_emg_sync = t_emg[emg_idx]
        t_anlge_sync = t_angle[angle_idx]

    plt.figure(figsize=(16.0, 10.0))
    plt.plot(t_emg,x_emg )
    plt.plot(t_emg[peaks_emg], x_emg[peaks_emg], "x")

    plt.figure(figsize=(16.0, 10.0))
    plt.plot(t_angle,x_angle )
    plt.plot(t_angle[peaks_angle], x_angle[peaks_angle], "x")

    plt.figure(figsize=(16.0, 10.0))
    plt.plot(t_angle-t_anlge_sync, x_angle)
    plt.plot(t_emg-t_emg_sync,x_emg)
    plt.savefig(savepath+filename+'_sync')
    plt.show()

    return t_emg_sync,t_anlge_sync , emg_idx , angle_idx

def savedata(save_name,x_data,y_data):
    assert len(x_data) == len(y_data)

    with open(save_name, 'wb') as output:
        data = Data(x_data, y_data)
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    data_path = '/home/hyuninlee/PycharmProjects/xcorps/data/data0622_differentsecond/lhi/convertdata/train/'
    savepath = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/trainData/lhi/1sec/'
    filename_angles = 'angles_0622_exp_lhi_one_1sec_4.npy'
    filename_emg = 'emg_smooth_0622_exp_lhi_one_1sec_4.npy'

    filename = 'train_lhi_4'
    savename = filename +'.pkl'


    video_fs = 239.88
    emg_fs = 1260

    #train_lhi_1.pkl
    #start_time_angle = 14 #must be integer
    #end_time_angle = 65 # must be integer
    #middle_time_list_angle = [15.2, 17.3, 19.3, 21.3, 23.3, 25.3, 27.6, 29.6, 31.8, 33.8, 35.7, 37.6, 39.7, 41.5, 43.5, 45.5, 47.6, 49.7, 51.7, 53.7, 55.7, 57.7, 59.7, 61.8, 63.99 ]

    ##train_lhi_2.pkl
    #start_time_angle = 11  # must be integer
    #end_time_angle = 62  # must be integer
    #middle_time_list_angle = [12.7, 14.7, 16.6, 18.6, 20.6, 22.7, 24.7, 26.7, 28.7, 30.7, 32.7, 34.7, 36.7, 38.7, 40.7, 42.7, 44.7, 46.8, 49.0, 51.0, 53.0, 54.9, 56.7, 58.7, 60.7]

    ##train_lhi_3.pkl
    #start_time_angle = 14  # must be integer
    #end_time_angle = 70  # must be integer
    #middle_time_list_angle = [16.2, 18.3, 20.3, 22.5, 25.0, 27.3, 29.3, 31.3, 33.5, 35.5, 37.7, 39.8, 42.2, 44.3, 46.8, 48.8, 51.0, 53.3, 55.6, 57.6, 59.8, 61.9, 63.8, 66.2, 68.4]

    ##train_yya_1.pkl
    #start_time_angle = 14  # must be integer
    #end_time_angle = 65  # must be integer
    #middle_time_list_angle = [15.7, 17.6, 19.5, 21.5, 23.5, 25.5, 27.5, 29.4, 31.2, 33.1, 34.8, 36.6, 38.0, 40.0, 42.0, 44.0, 46.1, 48.1, 50.3, 52.3, 54.7, 56.7, 58.9, 61.2, 63.2]

    ##train_yya_2.pkl
    #start_time_angle = 17  # must be integer
    #end_time_angle = 69  # must be integer
    #middle_time_list_angle = [18.9, 20.9, 23.3, 25.5, 27.5, 29.5, 31.3, 33.3, 35.2, 37.2, 39.2, 41.2, 43.2, 45.2, 47.2, 49.1, 51.0, 53.0, 55.0, 57.0, 59.0, 61.0, 63.6, 65.6, 67.6]

    ##train_yya_3.pkl
    #start_time_angle = 16  # must be integer
    #end_time_angle = 67  # must be integer
    #middle_time_list_angle = [17.7, 19.7, 21.7, 23.7, 25.7, 27.6, 29.6, 31.6, 33.6, 35.7, 37.6, 39.7, 41.7, 43.7, 45.7, 47.7, 49.7, 52.0, 54.0, 56.0, 58.0, 60.0, 61.8, 63.8, 65.8]

    ##train_lhi_4.pkl
    #start_time_angle = 14  # must be integer
    #end_time_angle = 75  # must be integer
    #middle_time_list_angle = [2*x+16.0 for x in range(25)]

    ##train_lhi_5.pkl
    #start_time_angle = 17  # must be integer
    #end_time_angle = 69  # must be integer
    #middle_time_list_angle = [2*x+19.2 for x in range(25)]

    ##train_lhi_6.pkl
    #start_time_angle = 16  # must be integer
    #end_time_angle = 68  # must be integer
    #middle_time_list_angle = [2*x+18.5 for x in range(25)]

    ##train_lhi_7.pkl
    #start_time_angle = 15  # must be integer
    #end_time_angle = 67  # must be integer
    #middle_time_list_angle = [17.6, 19.6, 21.6, 23.6, 25.6, 27.6, 29.6, 31.6, 33.6, 35.6, 37.6, 39.6, 41.6, 43.6, 45.6, 47.5, 49.3, 51.6, 53.6, 55.6, 57.3, 59.6, 61.6, 63.6,65.6]

    ##train_lhi_8.pkl
    #start_time_angle = 12  # must be integer
    #end_time_angle = 64  # must be integer
    #middle_time_list_angle = [2*x+14.2 for x in range(25)]

    ##train_lhi_9.pkl
    #start_time_angle = 12  # must be integer
    #end_time_angle = 63  # must be integer
    #middle_time_list_angle = [2*x+13.8 for x in range(25)]

    ##test_lhi_1.pkl
    #start_time_angle = 9  # must be integer
    #end_time_angle = 58  # must be integer
    #middle_time_list_angle = [10.1, 12.1, 14.0, 15.8, 17.8, 19.7, 21.7, 23.7, 25.7, 27.7, 29.7, 31.7, 33.7, 35.7, 37.7, 39.7, 41.7, 43.5, 45.5, 47.2, 49.2, 50.6, 52.9, 54.9, 56.9]

    ## 1sec_train_lhi_1.pkl
    #start_time_angle = 14
    #end_time_angle = 80
    #middle_time_list_angle = [15.2, 16.2, 17.1, 18.1, 19.2, 20.2, 21.2, 22.2, 23.2, 24.1, 25.1, 26.1, 27.1, 28.1, 29.1, 30.0, 31.2, 32.1, 33.1, 34.1, 35.1, 36.1, 37.1, 38.1, 39.1,
    #                          40.1, 41.1, 42.1, 43.1, 44.1, 45.1, 46.1, 47.1, 48.1, 49.1, 50.1, 51.2, 52.2, 53.2, 54.2, 55.2, 56.1, 57.2, 58.2, 59.2, 60.2, 61.2, 62.2, 63.3, 64.3,
    #                          65.3, 66.3, 67.3, 68.3, 69.3, 70.3, 71.3, 72.3, 73.3, 74.3, 75.3, 76.3, 77.3, 78.3, 79.4]
    #dataLength = 65
    #duration = 1

    ## 1sec_train_lhi_2.pkl
    #start_time_angle = 11
    #end_time_angle = 72
    #middle_time_list_angle = [12.3, 13.3, 14.3, 15.3, 16.3, 17.3, 18.3, 19.3, 20.3, 21.3, 22.3, 23.3, 24.3, 25.3, 26.3, 27.3, 28.3, 29.2, 30.2, 31.2, 32.2, 33.2, 34.2, 35.2, 36.2,
    #                          37.2, 38.3, 39.3, 40.3, 41.3, 42.3, 43.4, 44.4, 45.4, 46.5, 47.5, 48.5, 49.4, 50.4, 51.3, 52.4, 53.5, 54.4, 55.4, 56.5, 57.5, 58.5, 59.5, 60.4, 61.4,
    #                          62.4, 63.4, 64.5, 65.5, 66.5, 67.5, 68.5, 69.4, 70.4, 71.4]
    #dataLength = 60
    #duration = 1

    ## 1sec_train_lhi_3.pkl
    #start_time_angle = 13
    #end_time_angle = 79
    #middle_time_list_angle = [13.6, 14.7, 15.7, 16.7, 17.7, 18.7, 19.7, 20.7, 21.7, 22.8, 23.7, 24.7, 25.8, 26.8, 27.8, 28.8, 29.9, 30.9, 31.9, 32.8, 33.8, 34.8, 35.8, 36.8, 37.8,
    #                          38.8, 39.8, 40.8, 41.8, 42.8, 43.8, 44.9, 50.9, 51.9, 52.9, 53.9, 54.9, 55.9, 56.9, 57.9, 58.9, 59.9, 60.9, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0,
    #                          69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0]
    #dataLength = 60
    #duration = 1

    ## 1sec_train_lhi_4.pkl
    start_time_angle = 8
    end_time_angle = 75
    middle_time_list_angle = [9.3, 10.3, 11.3, 12.3, 13.3, 14.3, 15.3, 16.3, 17.3, 18.3, 19.3, 20.4, 21.4, 22.4, 23.3, 24.3, 25.3, 26.3, 27.3, 28.3, 29.4, 30.5, 31.5, 32.5, 33.4,
                              34.3, 35.4, 36.4, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.6, 47.6, 48.6, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.6, 56.6, 57.6, 58.6,
                              59.6, 60.5, 61.5, 62.5, 63.5, 64.5, 65.5, 66.5, 67.5, 68.5, 69.6, 70.6, 71.6, 72.6, 73.6]
    dataLength = 65
    duration = 1

    assert len(middle_time_list_angle) == dataLength

    resample_fs = 100

    #prepare data
    input_data, output_data = dataprepare(data_path, filename_emg, filename_angles)
    emg_sync, angle_sync , _ , _ = synctime(input_data,output_data,start_time_angle,video_fs,emg_fs)

    AngleSmoothing = angleDataprocess(data_path+filename_angles,start_time_angle,end_time_angle,middle_time_list_angle,video_fs,resample_fs)
    #AngleSmoothing.plotAngles()
    #AngleSmoothing.plotAnglespart()


    emgsmothing = emgDataProcess(data_path+filename_emg,start_time_angle+emg_sync-angle_sync,end_time_angle+emg_sync-angle_sync,middle_time_list_angle+emg_sync-angle_sync,emg_fs,resample_fs)
    emgsmothing.plotEmg()
    emgsmothing.plotEMGpart()


    emgTime = emgsmothing.resampleTime
    emgData = emgsmothing.resampleEmg #(4,52000)
    emgTimeList = emgsmothing.timelist
    emgDataList = emgsmothing.emglist


    anglesTime = AngleSmoothing.resampleTime
    anglesData = AngleSmoothing.resampleAngles #(14,52000)
    anglesTimelist = AngleSmoothing.timelist
    angelsDataList = AngleSmoothing.anglelist

    plt.figure()
    plt.plot(emgTime-emg_sync,emgData[0,:])
    plt.plot(anglesTime-angle_sync,anglesData[2,:])
    plt.show()

    savedata(savepath + savename, emgDataList, angelsDataList)











