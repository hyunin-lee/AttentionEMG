import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd
import os , glob

cur_path = os.path.dirname(__file__)
################## change emg_path and save_path #########################

emg_path = os.path.relpath('./data/data0702/khj', cur_path)
save_path = os.path.relpath('./data/data0702/khj/convertdata/', cur_path)
plotname = '0702_khj_random_2sec'
filepath = emg_path+'/'+plotname+'.csv'
fs = 1260
# set start time when DataPreprocess  = False
starttime = 20

Datapreprocess = True
#########################################################################
#exp1_khj_1 : starttime = 15
#exp1_khj_2 : starttime = 15


def normalize(emg_data,fs,start_time) :
    emg_max = np.max(emg_data[start_time*fs:])
    return emg_data/emg_max

def butter_lowpass_filter(data, cutoff, order,nyq):
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff*1.25, btype='low', analog=False)
    y=filtfilt(b, a, data)
    return y

def datasmooth(emg_data, cutoff, order,nyq) :
    emg_data_1 = abs(emg_data - np.mean(emg_data))
    emg_data_2 = butter_lowpass_filter(emg_data_1,cutoff, order,nyq)
    return emg_data_2

def savedata(save_name,file):
  import pickle
  with open(save_name,'wb') as f :
    pickle.dump(file,f)

import numpy as np

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


cutoff = 5 #cutoff frequency
nyq= 0.5*fs
order=2 #filter order
csv_list = {}
csv_list_smooth={}



#for filepath in glob.glob(os.path.join(emg_path, '*.csv')):
print(filepath)
with open(os.path.join(os.getcwd(), filepath)) as f:
    data = pd.read_csv(f)
    time = data['X[s]'].to_numpy()
    emg1 =data['Avanti sensor 1: EMG 1'].to_numpy()
    emg1_smooth = datasmooth(emg1,cutoff, order,nyq)
    emg1_smooth = normalize(emg1_smooth,fs,starttime)

    emg2 =data['Avanti sensor 2: EMG 2'].to_numpy()
    emg2_smooth = datasmooth(emg2,cutoff, order,nyq)
    emg2_smooth = normalize(emg2_smooth, fs, starttime)

    emg3 =data['Avanti sensor 3: EMG 3'].to_numpy()
    emg3_smooth = datasmooth(emg3,cutoff, order,nyq)
    emg3_smooth = normalize(emg3_smooth, fs, starttime)

    emg4 =data['Avanti sensor 4: EMG 4'].to_numpy()
    emg4_smooth = datasmooth(emg4,cutoff, order,nyq)
    emg4_smooth = normalize(emg4_smooth, fs, starttime)

    nans, x = nan_helper(emg1_smooth)
    emg1_smooth[nans] = np.interp(x(nans), x(~nans), emg1_smooth[~nans])

    emg1_nosmooth = normalize(emg1, fs, starttime)
    emg2_nosmooth = normalize(emg2, fs, starttime)
    emg3_nosmooth = normalize(emg3, fs, starttime)
    emg4_nosmooth = normalize(emg4, fs, starttime)

    filename = os.path.basename(filepath).split(".")[0]
    csv_list[filename]=np.concatenate((time.reshape(-1,1),emg1_nosmooth.reshape(-1,1),emg2_nosmooth.reshape(-1,1),emg3_nosmooth.reshape(-1,1),emg4_nosmooth.reshape(-1,1)),axis=1)
    csv_list_smooth[filename]=np.concatenate((time.reshape(-1,1),emg1_smooth.reshape(-1,1),emg2_smooth.reshape(-1,1),emg3_smooth.reshape(-1,1),emg4_smooth.reshape(-1,1)),axis=1)

if Datapreprocess :
    save_name = save_path + '/emg_smooth_'+filename +'.npy'
    savedata(save_name,csv_list_smooth[filename])




fig1 = plt.figure(1)
#plt.plot(csv_list[plotname][:,0],csv_list[plotname][:,1])
if Datapreprocess :
    plt.plot(csv_list[plotname][:, 0], csv_list[plotname][:, 1])
    plt.plot(csv_list_smooth[plotname][:,0],csv_list_smooth[plotname][:,1])
plt.title("emg1")

fig2 = plt.figure(2)
#plt.plot(csv_list[plotname][:,0],csv_list[plotname][:,2])
if Datapreprocess :
    plt.plot(csv_list[plotname][:, 0], csv_list[plotname][:, 2])
    plt.plot(csv_list_smooth[plotname][:,0],csv_list_smooth[plotname][:,2])
plt.title("emg2")

fig3 = plt.figure(3)
#plt.plot(csv_list[plotname][:,0],csv_list[plotname][:,3])
if Datapreprocess :
    plt.plot(csv_list[plotname][:, 0], csv_list[plotname][:, 3])
    plt.plot(csv_list_smooth[plotname][:,0],csv_list_smooth[plotname][:,3])
plt.title("emg3")

fig4 = plt.figure(4)
#plt.plot(csv_list[plotname][:,0],csv_list[plotname][:,4])
if Datapreprocess :
    plt.plot(csv_list[plotname][:, 0], csv_list[plotname][:, 4])
    plt.plot(csv_list_smooth[plotname][:,0],csv_list_smooth[plotname][:,4])
plt.title("emg4")
plt.show()