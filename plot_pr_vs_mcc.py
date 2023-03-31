import numpy as np
import matplotlib.pyplot as plt

def get_metrics(CM):
    tp = CM[0,0]
    tn = CM[1,1]
    fp = CM[1,0]
    fn = CM[0,1]

    tpr = tp/(tp+fn)
    fpr = fp/(fp + tn)
    mcc = (tn*tp - fn*fp)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    
    return tpr, fpr, mcc


pAs = []
grid = 100
tprs = np.zeros((grid,grid))
fprs = np.zeros((grid,grid))
mccs = np.zeros((grid,grid))

for i in range(grid):
    for j in range(grid):   
        CM = np.array([[grid-i, i],[grid-j,j]])
        tpr, fpr, mcc = get_metrics(CM)
        tprs[i, :] = tpr
        fprs[:, j] = fpr
        mccs[i,j] = mcc


ax = plt.figure().add_subplot(projection='3d')

ax.plot_surface(tprs, fprs, mccs, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                alpha=0.3)
ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(-1, 1),
       xlabel='True Positive Rate', ylabel='False Positive Rate', zlabel='MCC')

plt.show()

