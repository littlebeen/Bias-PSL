#cython: language_level=3
from surprise import AlgoBase
cimport numpy as np  # noqa
import numpy as np
from surprise import PredictionImpossible
from surprise import accuracy


class PSLSVD(AlgoBase):

    def __init__(self,steps=20,K=100,KP=1.54,KI=0.0027,KD=0.0012,alpha=0.01,beta=0.05):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)
        self.steps = steps
        self.K = K
        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.alpha = alpha
        self.beta = beta


    def fit(self,trainset):
        AlgoBase.fit(self, trainset)
        cdef np.ndarray[np.double_t] X
        cdef np.ndarray[np.double_t] VV
        cdef np.ndarray[np.double_t, ndim=2] pu
        cdef np.ndarray[np.double_t, ndim=2] qi
        cdef double alpha = self.alpha
        cdef double beta = self.beta
        cdef int K = self.K
        cdef int steps = self.steps
        cdef double KP = self.KP
        cdef double KI = self.KI
        cdef double KD = self.KD
        cdef int u, i, f
        cdef int n=0
        cdef double r, err, dot, puf, qif
        cdef np.ndarray[np.double_t] result
        result =  np.zeros(steps, np.double)
        X = np.zeros(trainset.n_ratings, np.double)
        VV = np.zeros(trainset.n_ratings, np.double)
        rng = np.random.mtrand._rand
        pu = rng.normal(0, .01,
                        (trainset.n_users, K))
        qi = rng.normal(0, .01,
                        (trainset.n_items, K))

        for current_epoch in range(steps):
            print(current_epoch)
            n=0
            e = 0
            for u, i, r in trainset.all_ratings():
                dot = 0
                for f in range(K):
                    dot += qi[i, f] * pu[u, f]
                err = r - dot
                X[n] += err
                T = KP * err + KI * X[n] + KD * (err - VV[n])
                VV[n] = err
                for f in range(K):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += alpha  * (T * qif - beta * puf)
                    qi[i, f] += alpha  * (T * puf - beta * qif)
                n+=1
        self.pu = pu
        self.qi = qi


    def ffit(self,trainset,testset):
        AlgoBase.fit(self, trainset)
        cdef np.ndarray[np.double_t] X
        cdef np.ndarray[np.double_t] VV
        cdef np.ndarray[np.double_t, ndim=2] pu
        cdef np.ndarray[np.double_t, ndim=2] qi
        cdef double alpha = self.alpha
        cdef double beta = self.beta
        cdef int K = self.K
        cdef int steps = self.steps
        cdef double KP = self.KP
        cdef double KI = self.KI
        cdef double KD = self.KD
        cdef int u, i, f
        cdef int n=0
        cdef double r, err, dot, puf, qif
        cdef np.ndarray[np.double_t] rmse
        cdef np.ndarray[np.double_t] mae
        rmse =  np.zeros(self.steps, np.double)
        mae =  np.zeros(self.steps, np.double)
        result =  np.zeros(steps, np.double)
        X = np.zeros(trainset.n_ratings, np.double)
        VV = np.zeros(trainset.n_ratings, np.double)
        rng = np.random.mtrand._rand
        pu = rng.normal(0, .01,
                        (trainset.n_users, K))
        qi = rng.normal(0, .01,
                        (trainset.n_items, K))

        for current_epoch in range(steps):
            print("PSL",current_epoch)
            n=0
            e = 0
            for u, i, r in trainset.all_ratings():
                dot = 0
                for f in range(K):
                    dot += qi[i, f] * pu[u, f]
                err = r - dot
                X[n] += err
                T = KP * err + KI * X[n] + KD * (err - VV[n])
                VV[n] = err
                for f in range(K):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += alpha  * (T * qif - beta * puf)
                    qi[i, f] += alpha  * (T * puf - beta * qif)
                n+=1
            self.pu = pu
            self.qi = qi
            predictions = self.test(testset)
            rmse[current_epoch]=accuracy.rmse(predictions)
            mae[current_epoch]=accuracy.mae(predictions)
        return rmse,mae

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
        if known_user and known_item:
            est = np.dot(self.qi[i], self.pu[u])
        else:
            raise PredictionImpossible('User and item are unknown.')
        return est