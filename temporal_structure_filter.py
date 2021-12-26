#Reference:https://github.com/piergiaj/tgm-icml19

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class TSF(nn.Module):
    def __init__(self, N=3):            #N is num_f of TGM class
        super(TSF, self).__init__()

        self.N = int(N)

        # create parameteres for center and delta of this super event
        self.center = nn.Parameter(torch.FloatTensor(N))    #size N
        self.gamma = nn.Parameter(torch.FloatTensor(N))     #size N
        self.center=nn.Parameter(torch.normal(2, 3, size=(1,N)))
        self.center.data.normal_(mean=0,std=0.5)
        self.gamma.data.normal_(0, 0.0001)

    def get_filters(self,gamma, center, length, time):
        """
            center (batch,) in [-1, 1]
            gamma (batch,) in [-1, 1]
            length (batch,) of ints
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        center=torch.squeeze(center,dim=0)      #[1,N] ===> [N]

        # scale to length of videos
        centers = (length - 1) * (center + 1) / 2.0         #[N]
        gammas = torch.exp(1.0 - 2.0 * torch.abs(gamma))    #[N]

        x = Variable(torch.zeros(self.N,self.N).to(torch.device(device)))    #[N,N]

        x = centers[:, None] + x                #(N,1)  +  (N,N)  = (N,N)
            # centers[:,None]   [N] ======>(N,1)

        t = Variable(torch.arange(0.0, time).to(torch.device(device)))      #example: [0,1,2,3,4]
            # t => [0.0,1.0,2.0,3.0 ,,,,,  time-1]   for example , time=3 ==>[0.0,1.0,2.0]
            # t size [time]

        f = t - x[:, :, None]
            # a[:,:,None]   (N,N)  ====> (N,N,1)
            # f (N,N, time)

        gammas = gammas.to(torch.device(device))

        f = f / gammas[:, None, None]
            # gammas[:,None, None]  ==> (N,1,1)
            # f (N,N, time)
        
        f = f ** 2.0
        f = np.pi * gammas[:, None, None] * f
        f = 1.0/f
        f = f/(torch.sum(f, dim=2) + 1e-6)[:,:,None]            #torch.Size([N, N, time])

        return f[:,0,:]         #output: torch.Size([N,time])
