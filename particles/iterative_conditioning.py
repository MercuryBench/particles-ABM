import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from DIRTyTorch import TTMap

import numpy as np

import collections
from particles import distributions as dists
from particles import state_space_models as ssms


#################################
# IterativeConditioning filter/smoother class
#################################

class IterativeConditioning:
    """ Particle filter and smoother based on the iterative conditioning method.

    """

    def __init__(self, ssm=None, data=None, sample_size=1, device='cpu', dtype=torch.float32):
        """

        Parameters:
        - ssm: the defined state space models
        - data: observed data
        - sample_size: number of particles
        - state_dim: Dimensionality of the hidden state.
        - observation_dim: Dimensionality of the observation.
        """
        
        #
        self.ssm = ssm
        self.data = data
        self.T = len(self.data)
        self.Ydim = self.data[1].size
        self.N = int(sample_size)
        self.t = 0
        #
        self.Xdim = self.ssm.PX0().rvs(1).size
        #
        # save the filtering states for each time
        self.filtered_states = []
        #
        # initialise the temporary data arrays for each iteration
        self.X, self.Xp = [], []
        self.Xpred, self.Ypred = [], []
        #
        # initialise linear transformations
        self.Xmeans = [np.zeros(self.Xdim) for i in range(self.T)]
        self.Ymeans = [np.zeros(self.Ydim) for i in range(self.T)]
        self.Xstds  = [np.zeros(self.Xdim) for i in range(self.T)]
        self.Ystds  = [np.zeros(self.Ydim) for i in range(self.T)]
        #
        # initialise maps
        grid1 = torch.linspace(-4,4,5)
        rank = 5
        d = self.Ydim + self.Xdim
        self.KRMaps = [TTMap([grid1]*d, [1] + [rank]*(d-2) + [1]) for _ in range(self.T)]
        #
        self.device = device
        self.dtype = dtype

    def __next__(self):
        try:
            yt = self.data[self.t]
        except IndexError:
            raise StopIteration
        
        if self.t == 0:
            self.Xp = self.ssm.PX0().rvs(self.N).reshape(self.N, -1)
        else:
            self.Xp = self.X

        # both in numpy arrays
        self.Xpred = self.ssm.PX(self.t, self.Xp).rvs().reshape(self.N, -1)
        self.Ypred = self.ssm.PY(self.t+1, self.Xp, self.Xpred).rvs().reshape(self.N, -1)

        nan_count_s = np.isnan(self.Xpred).sum()
        inf_count_s = np.isinf(self.Xpred).sum()
        nan_count_o = np.isnan(self.Ypred).sum()
        inf_count_o = np.isinf(self.Ypred).sum()

        self.Xpred[np.isnan(self.Xpred)] = 0.0
        self.Xpred[np.isinf(self.Xpred)] = 0.0
        self.Ypred[np.isnan(self.Ypred)] = 0.0
        self.Ypred[np.isinf(self.Ypred)] = 0.0

        print([nan_count_s, inf_count_s, nan_count_o, inf_count_o])

        # standardisation
        self.Xmeans[self.t] = np.mean(self.Xpred, axis=0)
        self.Xstds[self.t]  = np.std (self.Xpred, axis=0)
        self.Ymeans[self.t] = np.mean(self.Ypred, axis=0)
        self.Ystds[self.t]  = np.std (self.Ypred, axis=0)
        #
        ys = (self.Ypred - self.Ymeans[self.t])/self.Ystds[self.t]
        xs = (self.Xpred - self.Xmeans[self.t])/self.Xstds[self.t]
        #
        # The below are in pytorch
        #
        yt_std = torch.tensor((yt-self.Ymeans[self.t])/self.Ystds[self.t], device=self.device, dtype=self.dtype).detach()
        #
        J = torch.tensor(np.hstack((ys,xs)), device=self.device, dtype=self.dtype).detach()
        W = torch.ones(J.shape[0]).detach()
        optimizer = torch.optim.Adam(self.KRMaps[self.t].parameters(), lr=1e-1)
        for i in range(1000):
            optimizer.zero_grad()
            loss = self.KRMaps[self.t].loss(J,W)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(i, loss.item())
        #
        Jref = self.KRMaps[self.t](J)
        Xref = Jref[:,self.Ydim:]
        #
        yref = self.KRMaps[self.t](yt_std.repeat(self.N,1))
        Xnew, _ = self.KRMaps[self.t].inv(Xref, yref)
        #
        # back to numpy
        self.X = ( Xnew.detach().cpu().numpy() * self.Xstds[self.t] ) + self.Xmeans[self.t]
        #
        self.filtered_states.append(self.X)
        self.t = self.t+1

    def next(self):
        return self.__next__()  # Python 2 compatibility

    def __iter__(self):
        return self

    def filter(self):
        """Forward recursion: compute mean/variance of filter and prediction."""
        for _ in self:
            pass

    def smoother(self):
        """Backward recursion: compute mean/variance of marginal smoother.

        Performs the filter step in a preliminary step if needed.
        """
        