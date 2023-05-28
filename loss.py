# loss functions
# a variety of functions to compare.
import torch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss as CELoss
import torch.nn.functional as F
eps = 1e-7

def get_diff_logits(y_pred, y_true):
    y_true_logits = torch.sum( y_pred * y_true, dim=1, keepdim=True)
    return y_pred - y_true_logits


class AULLoss(nn.Module):
    def __init__(self, num_classes=10, a=1.5, q=0.9, eps=eps, scale=1.0):
        super(AULLoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.q = q
        self.eps = eps
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = labels.float() 
        loss = (torch.pow(self.a - torch.sum(label_one_hot * pred, dim=1), self.q) - (self.a-1)**self.q)/ self.q
        return loss.mean() * self.scale


class AGCELoss(nn.Module):
    def __init__(self, num_classes=10, a=1, q=2, eps=eps, scale=1.):
        super(AGCELoss, self).__init__()
        self.a = a
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = labels.float()
        loss = ((self.a+1)**self.q - torch.pow(self.a + torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean() * self.scale

class NCELoss(nn.Module):
    def __init__(self, num_classes=10, scale=1.0):
        super(NCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = labels.float()
        loss = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))
        return self.scale * loss.mean()


class RCELoss(nn.Module):
    def __init__(self, num_classes=10, scale=1.0):
        super(RCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = labels.float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        loss = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * loss.mean()


class NCEandAGCE(torch.nn.Module):
    def __init__(self, num_classes=10, alpha=1., beta = 1., a=3, q=1.5):
        super(NCEandAGCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.agce = AGCELoss(num_classes=num_classes, a=a, q=q, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.agce(pred, labels)


class NCEandRCE(nn.Module):
    def __init__(self, num_classes=10, alpha=1., beta=1.):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.rce = RCELoss(num_classes=num_classes, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)

class NCEandAUL(torch.nn.Module):
    def __init__(self, num_classes=10, alpha=1., beta=1., a=6, q=1.5):
        super(NCEandAUL, self).__init__()
        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.aul = AULLoss(num_classes=num_classes, a=a, q=q, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.aul(pred, labels)

def custom_kl_div(prediction, target):
    output_pos = target * (target.clamp(min=1e-7).log() - prediction)
    zeros = torch.zeros_like(output_pos)
    output = torch.where(target > 0, output_pos, zeros)
    output = torch.sum(output, axis=1)
    return output.mean()

class JSCLoss(torch.nn.Module):
    def __init__(self, num_classes, weights):
        super(JSCLoss, self).__init__()
        print(weights)
        self.num_classes = num_classes
        self.weights = [float(w) for w in weights.split(' ')]
          
        self.scale = -1.0 / ((1.0-self.weights[0]) * np.log((1.0-self.weights[0])))
        assert abs(1.0 - sum(self.weights)) < 0.001

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        distribs = [labels] + [pred]
        assert len(self.weights) == len(distribs)
        mean_distrib = sum([w*d for w,d in zip(self.weights, distribs)])
        mean_distrib_log = mean_distrib.clamp(1e-7, 1.0).log()
        
        jsw = sum([w*custom_kl_div(mean_distrib_log, d) for w,d in zip(self.weights, distribs)])
        return self.scale * jsw



class CSLoss(nn.Module):
    def __init__(self, threshold=1.0):
        super(CSLoss, self).__init__()
        self.threshold = threshold

    def forward(self, y_pred, y_true):
        diff_logits = torch.maximum(self.threshold*(1-y_true) + get_diff_logits(y_pred, y_true), torch.tensor(0))
        loss = torch.mean(torch.max(diff_logits, dim=1).values)
        return loss


class WWLoss(nn.Module):
    def __init__(self, threshold=1.0):
        super(WWLoss, self).__init__()
        self.threshold = threshold

    def forward(self, y_pred, y_true):
        diff_logits = torch.maximum(self.threshold*(1-y_true) + get_diff_logits(y_pred, y_true), torch.tensor(0))
        loss = torch.mean(diff_logits)
        return loss




class CELoss_v1(nn.Module):
    def __init__(self):
        super(CELoss_v1, self).__init__()

    def forward(self, y_pred, y_true):
        diff_logits = get_diff_logits(y_pred, y_true)
        diff_logits = diff_logits
        max_diff = torch.max(diff_logits, dim=1, keepdim=True).values.detach_()
        diff_logits = diff_logits - max_diff
        diff_logits = torch.exp(diff_logits)
        loss = torch.log(torch.mean(diff_logits, dim=1, keepdim=True)) + max_diff
        return torch.mean(loss)



class LDRLoss_V1(nn.Module):
    def __init__(self, threshold=2.0, Lambda=1.0):
        super(LDRLoss_V1, self).__init__()
        self.threshold = threshold
        self.Lambda = Lambda

    def forward(self, y_pred, y_true):
        num_class = y_pred.shape[1]
        y_pred = torch.nn.functional.softplus(y_pred)
        y_denorm = torch.mean(y_pred, dim=1, keepdim=True)
        y_pred = y_pred/y_denorm
        diff_logits = self.threshold*(1-y_true) + get_diff_logits(y_pred, y_true)
        diff_logits = diff_logits/self.Lambda
        max_diff = torch.max(diff_logits, dim=1, keepdim=True).values.detach()
        diff_logits = diff_logits - max_diff
        diff_logits = torch.exp(diff_logits)
        loss = self.Lambda*(torch.log(torch.mean(diff_logits, dim=1, keepdim=True)) + max_diff)
        return loss.mean()
  

 
class ALDRLoss_V1(nn.Module):
    def __init__(self, N, threshold=1.0, Lambda=1.0, alpha = 1e0, softplus=False): # alpha > 1.0
        super(ALDRLoss_V1, self).__init__()
        self.threshold = threshold
        self.Lambda_ref = torch.tensor(Lambda).cuda()
        self.Lambda = torch.tensor([Lambda]*N).view(-1,1).float().cuda()
        self.alpha = alpha
        self.softplus = softplus

    def forward(self, y_pred, y_true, ids):
        lambda_ref = self.Lambda_ref
        num_class = y_pred.shape[1]
        if self.softplus:
          y_pred = torch.nn.functional.softplus(y_pred)
        y_denorm = torch.norm(y_pred, dim=1, p=1.0, keepdim=True)/num_class
        y_pred = y_pred/y_denorm
        pred_prob = torch.nn.functional.softmax(y_pred/self.Lambda[ids], dim=1)+1e-5 # prevent invalid value for log operator
        pred_prob = F.normalize(pred_prob, p=1.0).cuda() 
        kl_reg = torch.sum(pred_prob*torch.log(num_class*pred_prob),dim=1,keepdim=True)
        logK = torch.log(torch.tensor(num_class))
        lambdas = lambda_ref - lambda_ref*kl_reg/(self.alpha*logK)
        self.Lambda[ids]=lambdas.detach()
        extra_loss = -1/2*self.alpha*logK/lambda_ref*(self.Lambda[ids]-lambda_ref)**2
        diff_logits = self.threshold*(1-y_true) + get_diff_logits(y_pred, y_true)
        diff_logits_lam_fix = diff_logits/self.Lambda[ids]
        max_diff_fix = torch.max(diff_logits_lam_fix, dim=1, keepdim=True).values.detach()
        diff_logits_lam_fix = diff_logits_lam_fix - max_diff_fix
        diff_logits_lam_fix = torch.exp(diff_logits_lam_fix)
        loss_lam_fix = self.Lambda[ids]*(torch.log(torch.mean(diff_logits_lam_fix, dim=1, keepdim=True)) + max_diff_fix)+extra_loss
        return loss_lam_fix.mean()#, lambdas.detach()

 




class TGCELoss(nn.Module):
    def __init__(self, q=0.7, k=0.0, N=0): # q = 0.05, 0.7, 0.95; k = 0.1, 0.3, 0.5
        super(TGCELoss, self).__init__()
        self.q = q
        self.k = k
        self.p = torch.tensor([1.0]*N).view(-1,1).float().cuda()

    def prune(self, y_pred, y_true, ids):
        diff_logits = get_diff_logits(y_pred, y_true) 
        diff_logits = torch.exp(diff_logits)
        self.p[ids] = ((1/torch.sum(diff_logits, dim=1, keepdim=True)) > self.k).float().cuda()
        #print(torch.mean(self.p[ids]))

    def forward(self, y_pred, y_true, ids):
        diff_logits = get_diff_logits(y_pred, y_true) 
        diff_logits = torch.exp(diff_logits)
        p = self.p[ids]
        loss = (1-(1/torch.sum(diff_logits, dim=1, keepdim=True))**self.q)/self.q      
        loss = p*loss
        return torch.mean(loss)


class GCELoss(nn.Module):
    def __init__(self, q=0.7, k=0.0): # q = 0.05, 0.7, 0.95; k = 0.1, 0.3, 0.5
        super(GCELoss, self).__init__()
        self.q = q
        self.k = k

    def forward(self, y_pred, y_true):
        diff_logits = get_diff_logits(y_pred, y_true) 
        diff_logits = torch.exp(diff_logits)
        loss = (1-(1/torch.sum(diff_logits, dim=1, keepdim=True))**self.q)/self.q
        trunc_loss = torch.tensor((1-(self.k)**self.q)/self.q).cuda()
        loss = torch.minimum(loss,trunc_loss)
        return torch.mean(loss)


class RLLLoss(nn.Module):
    def __init__(self, threshold=1.0):
        super(RLLLoss, self).__init__()
        self.threshold = threshold

    def forward(self, y_pred, y_true):
        K = y_true.shape[-1]
        diff_logits_1 = get_diff_logits(y_pred, y_true) 
        diff_logits_1 = torch.exp(diff_logits_1)
        loss_1 = -torch.log(self.threshold + 1/torch.sum(diff_logits_1, dim=1, keepdim=True))
        diff_logits_2 = (1-y_true)*y_pred - torch.sum(y_pred, dim=1, keepdim=True)
        diff_logits_2 = torch.exp(diff_logits_2) + self.threshold
        loss_2 = torch.log(torch.sum(diff_logits_2, dim=1, keepdim=True))/(K-1)
        return torch.mean(loss_1 + loss_2)


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, y_pred, y_true):
        probs = self.softmax(y_pred)
        loss = torch.sum(y_true - probs, dim=1)
        return torch.mean(loss)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, y_pred, y_true):
        probs = self.softmax(y_pred)
        loss = torch.sum((y_true - probs)**2, dim=1)
        return torch.mean(loss)



class SCELoss(nn.Module):
    def __init__(self, balance=0.5, A=4):
        super(SCELoss, self).__init__()
        self.balance = balance
        self.A = A

    def forward(self, y_pred, y_true):
        diff_logits = get_diff_logits(y_pred, y_true) 
        diff_logits = torch.exp(diff_logits)
        loss_1 = torch.log(torch.sum(diff_logits, dim=1, keepdim=True)) 
        loss_2 = self.A*(1-(1/torch.sum(diff_logits, dim=1, keepdim=True)))
        loss = (1-self.balance)*loss_1 + self.balance*loss_2
        return torch.mean(loss)

