# cython: language_level=3
import pyximport
pyximport.install()
from PSL import PSLSVD
from Bias_PSL import Bias_PSLSVD
from SVDR import SVDR
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import numpy as np
from surprise.model_selection import GridSearchCV

os.environ['SURPRISE_DATA_FOLDER'] = os.getcwd()+"/data/.surprise_data/"
#file_path = os.path.expanduser(os.getcwd()+'/data/surprise/ml-100k/u.data')
#reader = Reader(line_format='user item rating timestamp', sep='\t')
#file_path = os.path.expanduser(os.getcwd()+'/data/.surprise_data/ml-1m/ratings.dat')
#reader = Reader(line_format='user item rating timestamp', sep='::')
#file_path = os.path.expanduser(os.getcwd()+'/data/Jester/jester_ratings.dat')
#reader = Reader(line_format='user item rating', sep='\t\t')
#reader = Reader(name='jester')
file_path = os.path.expanduser(os.getcwd()+'/data/Doban/ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',')

data = Dataset.load_from_file(file_path, reader=reader)

trainset, testset = train_test_split(data, test_size=.20)
#algo = SVDR(biased=False, n_epochs=25, n_factors=10, lr_all=0.005, reg_all=0.5, verbose=True)
#SVDrmse,SVDmae=algo.ffit(trainset,testset)
#algo=PSLSVD(steps=25, K=50, KP=1.44, KI=0.0015, KD=0.01, alpha=0.005, beta=0.05)
#PSLrmse,PSLmae=algo.ffit(trainset,testset)
#jester学习率超过0.005就不动了，不晓得为啥
param_grid = {'KP': [1.44,1.52,1.60,1.64], 'KD': [0.0015,0.0025,0.0035],
              'KI': [0, 0.01,0.02],'alpha':[0.01,0.02,0.03,0.04],'beta':[0.02,0.04]}
gs = GridSearchCV(PSLSVD, param_grid, measures=['rmse', 'mae'], cv=2)
gs.fit(data)
print(gs.best_params['rmse'])
#algo=PSLSVD(steps=40, K=50, KP=1.6, KI=0.01, KD=0.01, alpha=0.04, beta=0.05)
#PSLrmse,PSLmae=algo.ffit(trainset,testset)
#algo = SVDR(biased=True, n_epochs=25, n_factors=50, lr_all=0.04, reg_all=0.05, verbose=True)
#B_SVDrmse,B_SVDmae=algo.ffit(trainset,testset)
"""
algo = SVDR(biased=True, n_epochs=25,n_factors=100, lr_all=0.01, reg_all=0.05, verbose=True)
B_SVDrmse,B_SVDmae=algo.ffit(trainset,testset)
algo=PSLSVD(steps=25, K=100, KP=1.44, KI=0.002, KD=0.001, alpha=0.01, beta=0.05)
PSLrmse,PSLmae=algo.ffit(trainset,testset)
algo=Bias_PSLSVD(steps=25, K=100, KP=1.44, KI=0.002, KD=0.001, alpha=0.01, beta=0.05)
B_PSLrmse,B_PSLmae=algo.ffit(trainset,testset)
SVDrmse[24]=None
SVDmae[24]=None
PSLrmse[17]=PSLrmse[18]=PSLrmse[19]=PSLrmse[20]=PSLrmse[21]=PSLrmse[22]=PSLrmse[23]=PSLrmse[24]=None
PSLmae[17]=PSLmae[18]=PSLmae[19]=PSLmae[20]=PSLmae[21]=PSLmae[22]=PSLmae[23]=PSLmae[24]=None
B_PSLrmse[18]=B_PSLrmse[19]=B_PSLrmse[20]=B_PSLrmse[21]=B_PSLrmse[22]=B_PSLrmse[23]=B_PSLrmse[24]=None
B_PSLmae[18]=B_PSLmae[19]=B_PSLmae[20]=B_PSLmae[21]=B_PSLmae[22]=B_PSLmae[23]=B_PSLmae[24]=None
"""
n=25
x = np.arange(1,1+n).astype(dtype=np.str_)
"""
plt.plot(x, PSLrmse, color='r', linewidth=2, label='PSL')
plt.plot(x, B_PSLrmse, color='b', linewidth=2, label='Bias_PSL')
plt.plot(x, SVDrmse, color='y', linewidth=2, label='SVD')
plt.plot(x, B_SVDrmse, color='orange', linewidth=2, label='Bias_SVD')
"""
plt.xlabel("epoch")
#100k
#plt.ylim((0.90,1))
#1m
#plt.ylim((0.84,0.95))
plt.ylabel("RMSE")
plt.legend()
#plt.savefig('C:\\豆豆\\课件\\科研\\图表矢量图\\1m+100rmse.svg')
plt.show()

"""
plt.plot(x, PSLmae, color='r', linewidth=2, label='PSL')
plt.plot(x, B_PSLmae, color='b', linewidth=2, label='Bias_PSL')
plt.plot(x, SVDmae, color='y', linewidth=2, label='SVD')
plt.plot(x, B_SVDmae, color='orange', linewidth=2,label='Bias_SVD')
"""
plt.xlabel("epoch")
#100k
#plt.ylim((0.70,0.80))
#1m
#plt.ylim((0.66,0.76))
plt.ylabel("MAE")
plt.legend()
#plt.savefig('C:\\豆豆\\课件\\科研\\图表矢量图\\1m+100mae.svg')
plt.show()