import functools
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import torch
from models.archs.CAMixerSR_arch import CAMixerSR
import numpy as np
import time

class ClassSR_3class_CAMixerSR_net(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(ClassSR_3class_CAMixerSR_net, self).__init__()
        self.upscale=4
        self.classifier=Classifier()
        # self.net1 = CAMixerSR(n_feats=36, scale=4, ratio=0.5)
        # self.net2 = CAMixerSR(n_feats=48, scale=4, ratio=0.5)
        # self.net3 = CAMixerSR(n_feats=60, scale=4, ratio=0.5)

        self.net1 = CAMixerSR(n_feats=60, scale=4, ratio=0.25)
        self.net2 = CAMixerSR(n_feats=60, scale=4, ratio=0.30)
        self.net3 = CAMixerSR(n_feats=60, scale=4, ratio=0.50)

    def forward(self, x,is_train):
        if is_train:
            self.net1.eval()
            self.net2.eval()
            self.net3.eval()
            class_type = self.classifier(x)
            p = F.softmax(class_type, dim=1)
            out1 = self.net1(x)
            out2 = self.net2(x)
            out3 = self.net3(x)

            p1 = p[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            p2 = p[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            p3 = p[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            out = p1 * out1 + p2 * out2 + p3 * out3
            return out, p
        else:

            for i in range(len(x)):
                type = self.classifier(x[i].unsqueeze(0))

                flag = torch.max(type, 1)[1].data.squeeze()
                p = F.softmax(type, dim=1)
                #flag=np.random.randint(0,2)
                #flag=2
                if flag == 0:
                    out = self.net1(x[i].unsqueeze(0))
                elif flag==1:
                    out = self.net2(x[i].unsqueeze(0))
                elif flag==2:
                    out = self.net3(x[i].unsqueeze(0))
                if i == 0:
                    out_res = out
                    type_res = p
                else:
                    out_res = torch.cat((out_res, out), 0)
                    type_res = torch.cat((type_res, p), 0)

            return out_res, type_res

        return out_res, type_res

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    print('Replace to new one...')
                    
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lastOut = nn.Linear(32, 3)

        # Condtion network
        self.CondNet = nn.Sequential(nn.Conv2d(3, 128, 4, 4), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 32, 1))
        arch_util.initialize_weights([self.CondNet], 0.1)
    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AvgPool2d(out.size()[2])(out)
        out = out.view(out.size(0), -1)
        out = self.lastOut(out)
        return out


