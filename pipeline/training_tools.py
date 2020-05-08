import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def get_onehot(seg):
    """
    seg: [slices,dim1,dim2]
    seg has the following labels {[0], [1,11,21],  [2,12,22],  [3,13,23], [10], [20]}
    """
    
    onehot = np.zeros((seg.shape[0],
                       5,
                       seg.shape[1],
                       seg.shape[2]))
    
    onehot[:,0,:,:] = (seg==0)
    onehot[:,1,:,:] = (seg==1)+(seg==11)+(seg==21)
    onehot[:,2,:,:] = (seg==2)+(seg==12)+(seg==22)    
    onehot[:,3,:,:] = (seg==3)+(seg==13)+(seg==23)    
    onehot[:,4,:,:] = (seg==10)+(seg==20)
    
    assert onehot.sum(1).sum() == seg.shape[0]*seg.shape[1]*seg.shape[2]
    return onehot


def dice_loss(pred, target):
    eps = 1
    assert pred.size() == target.size(), 'Input and target are different dim'
    
    if len(target.size())==4:
        n,c,x,y = target.size()
    if len(target.size())==5:
        n,c,x,y,z = target.size()

    target = target.view(n,c,-1)
    pred = pred.view(n,c,-1)
 
    num = torch.sum(2*(target*pred),2) + eps
    den = (pred*pred).sum(2) + (target*target).sum(2) + eps
    dice_loss = 1-num/den
    ind_avg = dice_loss
    total_avg = torch.mean(dice_loss)
    regions_avg = torch.mean(dice_loss, 0)
    
    return total_avg, regions_avg, ind_avg

def hardlabels(input):
    '''
    Pick the most likely class for each pixel
    individual mask: each subjects 
    have different uniformly sample mask
    '''
    input = input.detach().cpu()
    batch_n, chs, xdim, ydim = input.size()

    # Enumarate the chs #
    # enumerate_ch has dimensions [batch_n, chs, xdim, ydim]

    arange = torch.arange(0,chs).view(1,-1, 1, 1)
    arange = arange.repeat(batch_n, 1, 1, 1).float()
    
    enumerate_ch = torch.ones(batch_n, chs, xdim, ydim)
    enumerate_ch = arange*enumerate_ch 

    classes = torch.argmax(input,1).float()
    sample = []
    for c in range(chs):
        _sample = enumerate_ch[:,c,:,:] == classes
        sample += [_sample.unsqueeze(1)]
    sample = torch.cat(sample, 1)
    
    return sample

def augment(input, labels, sigma=0.1, offset=0.1):
    input = F.pad(input, 
                  pad=(1,1,1,1), 
                  mode='constant', 
                  value=-1024)  #For dark background
    
    """Defining Theta"""
    theta = torch.randn(input.shape[0], 2, 3)*sigma
    rand_offset = np.random.uniform(-offset,offset, size=(input.shape[0],2))
    rand_offset = torch.from_numpy(rand_offset).float()
    theta[:,:,2] = rand_offset
    theta[:,0,0] = theta[:,0,0]+1
    theta[:,1,1] = theta[:,1,1]+1
    if input.is_cuda:
        theta = theta.cuda()
    
    """Deforming Inputs"""
    grid_input = F.affine_grid(theta, input.size())  
    grid_labels = F.affine_grid(theta, labels.size()) 
    
    deformed_input = F.grid_sample(input, 
                                   grid_input,
                                   mode='bilinear',
                                   padding_mode='border', 
                                   align_corners = False)
    
    deformed_labels = F.grid_sample(labels, 
                                    grid_labels,
                                    mode='bilinear',
                                    padding_mode='border', 
                                    align_corners = False)
    
    deformed_input = deformed_input[:,:,1:-1,1:-1]
    return deformed_input, deformed_labels

class FocalLoss_Ori(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=[0.25,0.75], gamma=2, balance_index=-1, size_average=True):
        super(FocalLoss_Ori, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6

        if isinstance(self.alpha, (list, tuple)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.Tensor(list(self.alpha))
        elif isinstance(self.alpha, (float,int)):
            assert 0<self.alpha<1.0, 'alpha should be in `(0,1)`)'
            assert balance_index >-1
            alpha = torch.ones((self.num_class))
            alpha *= 1-self.alpha
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        elif isinstance(self.alpha,torch.Tensor):
            self.alpha = self.alpha
        else:
            raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    def forward(self, logit, target):

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous() # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1)) # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1, 1) # [N,d1,d2,...]->[N*d1*d2*...,1]

        # -----------legacy way------------
        #  idx = target.cpu().long()
        # one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # if one_hot_key.device != logit.device:
        #     one_hot_key = one_hot_key.to(logit.device)
        # pt = (one_hot_key * logit).sum(1) + epsilon

        # ----------memory saving way--------
        pt = logit.gather(1, target).view(-1) + self.eps # avoid apply
        logpt = pt.log()

        if self.alpha.device != logpt.device:
            alpha = self.alpha.to(logpt.device)
            alpha_class = alpha.gather(0,target.view(-1))
            logpt = alpha_class*logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
    
def string2float(string): 
    li = list(string.split(",")) 
    return [float(i) for i in li]
