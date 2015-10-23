# import numpy as np
# import matplotlib.pyplot as plt
# 
# 
# x = np.linspace(0, 10)
# line = plt.plot(x, np.sin(x), '--', linewidth=2)
# 
# #dashes = [10, 5, 100, 5] # 10 points on, 5 off, 100 on, 5 off
# #line.set_dashes(dashes)
# 
# plt.show()


# save_training_info = 1 
# if save_training_info:
#             # save extra information on evolution of training
#     print 'inside'


import matplotlib.pyplot as plt
import numpy as np

# Simple data to display in various forms
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

plt.close('all')

# # Just a figure and one subplot
# f, ax = plt.subplots()
# ax.plot(x, y)
# ax.set_title('Simple plot')

 # Two subplots, the axes array is 1-d
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(x, y,label='layer 0')
axarr[0].set_title('Sharing X axis')
axarr[1].scatter(x, y,label='layer 1')
#f.subplots_adjust(hspace=0)
#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
#plt.setp([a.get_xticklabels() for a in axarr], visible=False)
#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
plt.xlabel('Pt epochs')
axarr[0].set_ylabel('reconstruction cost',rotation=90)
axarr[1].set_ylabel('reconstruction cost',rotation=90)
axarr[0].legend()
axarr[1].legend()
#axarr[0].legend("layer 0")
#axarr[1].legend()
# line_up, = plt.plot([1,2,3], label='Line 2')
# line_down, = plt.plot([3,2,1], label='Line 1')
#plt.legend([axarr], ['Line Up', 'Line Down'])

#axarr[1].set_legend('label1')
#plt.ylabel('reconstruction cost',rotation=90)
#plt.ylabel('reconstruction cost',rotation=90)

# # Three subplots sharing both x/y axes
# f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
# ax1.plot(x, y)
# ax1.set_title('Sharing both axes')
# ax2.scatter(x, y)
# ax3.scatter(x, 2 * y ** 2 - 1, color='r')
# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
#f.subplots_adjust(hspace=0)
#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

plt.show()