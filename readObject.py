import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


read_path = "./data/data0504/lhi/convertdata/angles_exp_lhi_1.npy'"

def loadTrajectory():
    import pickle
    with open(read_path,'rb') as f :
        a = pickle.load(f)
    return a


trajectory = loadTrajectory()
angles = trajectory[:14,:]
time = trajectory[14,:]
"""
plot1 = plt.figure(1)
plt. plot(time, angles[0,:]*180/np.pi,label='0')
plt. plot(time, angles[1,:]*180/np.pi,label='1')

plot2 = plt.figure(2)
plt. plot(time, angles[2,:]*180/np.pi,label='2')
plt. plot(time, angles[3,:]*180/np.pi,label='3')
plt. plot(time, angles[4,:]*180/np.pi,label='4')

plot3 = plt.figure(3)
plt. plot(time, angles[5,:]*180/np.pi,label='5')
plt. plot(time, angles[6,:]*180/np.pi,label='6')
plt. plot(time, angles[7,:]*180/np.pi,label='7')

plot4 = plt.figure(4)
plt. plot(time, angles[8,:]*180/np.pi,label='8')
plt. plot(time, angles[9,:]*180/np.pi,label='9')
plt. plot(time, angles[10,:]*180/np.pi,label='10')

plot5 = plt.figure(5)
plt. plot(time, angles[11,:]*180/np.pi,label='11')
plt. plot(time, angles[12,:]*180/np.pi,label='12')
plt. plot(time, angles[13,:]*180/np.pi,label='13')
"""
# fig = plt.figure(1)
# ax1 = fig.add_subplot(511)
# ax2 = fig.add_subplot(512)
# ax3 = fig.add_subplot(513)
# ax4 = fig.add_subplot(514)
# ax5 = fig.add_subplot(515)
#
# ax1.title.set_text('Thumb')
# ax1. plot(time, angles[0,:]*180/np.pi,label='0')
# ax1. plot(time, angles[1,:]*180/np.pi,label='1')
#
# ax2.title.set_text('index finger')
# ax2. plot(time, angles[2,:]*180/np.pi,label='2')
# ax2. plot(time, angles[3,:]*180/np.pi,label='3')
# ax2. plot(time, angles[4,:]*180/np.pi,label='4')
#
# ax3.title.set_text('middle finger')
# ax3. plot(time, angles[5,:]*180/np.pi,label='5')
# ax3. plot(time, angles[6,:]*180/np.pi,label='6')
# ax3. plot(time, angles[7,:]*180/np.pi,label='7')
#
# ax4.title.set_text('ring finger')
# ax4. plot(time, angles[8,:]*180/np.pi,label='8')
# ax4. plot(time, angles[9,:]*180/np.pi,label='9')
# ax4. plot(time, angles[10,:]*180/np.pi,label='10')
#
# ax5.title.set_text('ring finger')
# ax5. plot(time, angles[11,:]*180/np.pi,label='11')
# ax5. plot(time, angles[12,:]*180/np.pi,label='12')
# ax5. plot(time, angles[13,:]*180/np.pi,label='13')


windowsize = int(2/(time[1]-time[0]))  if not int(2/(time[1]-time[0])) == 0 else int(2/(time[1]-time[0]))+1
poly_deg = 5

fig2 = plt.figure(2)
ax1 = fig2.add_subplot(211)
ax2 = fig2.add_subplot(212)
ax1.title.set_text('Thumb')
ax1. plot(time, angles[0,:]*180/np.pi,label='0')
ax1. plot(time, savgol_filter(angles[0,:], windowsize, poly_deg) *180/np.pi,label='0')
ax2. plot(time, angles[1,:]*180/np.pi,label='1')
ax2. plot(time, savgol_filter(angles[1,:], windowsize, poly_deg) *180/np.pi,label='0')

fig3 = plt.figure(3)
ax1 = fig3.add_subplot(311)
ax2 = fig3.add_subplot(312)
ax3 = fig3.add_subplot(313)
ax1.title.set_text('index')
ax1. plot(time, angles[2,:]*180/np.pi,label='2')
ax1. plot(time, savgol_filter(angles[2,:], windowsize, poly_deg) *180/np.pi,label='0')
ax2. plot(time, angles[3,:]*180/np.pi,label='3')
ax2. plot(time, savgol_filter(angles[3,:], windowsize, poly_deg) *180/np.pi,label='0')
ax3. plot(time, angles[4,:]*180/np.pi,label='4')
ax3. plot(time, savgol_filter(angles[4,:], windowsize, poly_deg) *180/np.pi,label='0')

fig4 = plt.figure(4)
ax1 = fig4.add_subplot(311)
ax2 = fig4.add_subplot(312)
ax3 = fig4.add_subplot(313)
ax1.title.set_text('middle')
ax1. plot(time, angles[5,:]*180/np.pi,label='5')
ax1. plot(time, savgol_filter(angles[5,:], windowsize, poly_deg) *180/np.pi,label='0')
ax2. plot(time, angles[6,:]*180/np.pi,label='6')
ax2. plot(time, savgol_filter(angles[6,:], windowsize, poly_deg) *180/np.pi,label='0')
ax3. plot(time, angles[7,:]*180/np.pi,label='7')
ax3. plot(time, savgol_filter(angles[7,:], windowsize, poly_deg) *180/np.pi,label='0')

fig5 = plt.figure(5)
ax1 = fig5.add_subplot(311)
ax2 = fig5.add_subplot(312)
ax3 = fig5.add_subplot(313)
ax1.title.set_text('ring')
ax1. plot(time, angles[8,:]*180/np.pi,label='5')
ax1. plot(time, savgol_filter(angles[8,:], windowsize, poly_deg) *180/np.pi,label='0')
ax2. plot(time, angles[9,:]*180/np.pi,label='6')
ax2. plot(time, savgol_filter(angles[9,:], windowsize, poly_deg) *180/np.pi,label='0')
ax3. plot(time, angles[10,:]*180/np.pi,label='7')
ax3. plot(time, savgol_filter(angles[10,:], windowsize, poly_deg) *180/np.pi,label='0')

# fig6 = plt.figure(6)
# ax1 = fig6.add_subplot(311)
# ax2 = fig6.add_subplot(312)
# ax3 = fig6.add_subplot(313)
# ax1.title.set_text('pinky')
# ax1. plot(time, angles[11,:]*180/np.pi,label='5')
# ax1. plot(time, savgol_filter(angles[11,:], windowsize, poly_deg) *180/np.pi,label='0')
# ax2. plot(time, angles[12,:]*180/np.pi,label='6')
# ax2. plot(time, savgol_filter(angles[12,:], windowsize, poly_deg) *180/np.pi,label='0')
# ax3. plot(time, angles[13,:]*180/np.pi,label='7')
# ax3. plot(time, savgol_filter(angles[13,:], windowsize, poly_deg) *180/np.pi,label='0')

plt.show()
