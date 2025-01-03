import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
from sklearn.metrics import roc_curve
from torch.optim import SGD, Adamax, Adam, RMSprop, AdamW
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, ReduceLROnPlateau, _LRScheduler
import numpy as np

import pandas as pd

from data_loader import *
from models import *

import os
import math
import cv2


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def make_data_loaders(train_df, val_df, test_df, image_dir, batch_size, image_size):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = ChestXRayDataset(data_frame=train_df, image_dir=image_dir, transform=train_transforms)
    val_dataset = ChestXRayDataset(data_frame=val_df, image_dir=image_dir, transform=val_transforms)
    test_dataset = ChestXRayDataset(data_frame=test_df, image_dir=image_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}, {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}, train_dataset.class_count
# Code extracted from MIT licensed code Copyright (c) 2022 Naoki Katsura 
# Implementation from "github pytorch-cosine-annealing-with-warmup"
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
    

# Code extracted from MIT licensed code Copyright (c) 2021 Alinstein Jose
# Implementation of "Weakly supervised Classification and Localization of Chest X-ray images"
def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    -------
    list type, with optimal cutoff value
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold']) # [0]??


# Idea extacted from the [https://arxiv.org/pdf/1901.05555] paper "Class-Balanced Loss Based on Effective Number of Samples"
# We can normalize inverse frequencies but this method is studied to do better in other works
def effective_weights(class_counts, beta=0.999):
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights)  # Normalize the weights -> or other(?)
    return torch.tensor(weights, dtype=torch.float32)  # Convert to tensor


# Code extracted from MIT licensed code Copyright (c) 2020 Alibaba-MIIL
# Original idea in the [https://arxiv.org/pdf/2009.14119] paper "Asymmetric Loss For Multi-Label Classification"
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, average=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.average = average

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        # Really small change introduced by thesis author (me) to try average loss impact
        # As discussed in "https://github.com/Alibaba-MIIL/ASL/issues/22" it's not suposed
        # To have an impact but I want to be able to use mean for ease of comparison.
        if self.average:
            return -loss.mean()
        else:
            return -loss.sum()

# Oirginal function to work with Grad_cam
def get_target_layer(model, model_type):
    if model_type == 'resnet50':
        return model.layer4[-1].conv3
    else:
        raise ValueError(f"Model type {model_type} not supported for GradCAM.")
    
# Code extracted from MIT licensed code Copyright (c) 2021 Alinstein Jose
# Implementation of "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
# From the paper "arXiv:1610.02391"
class Grad_CAM(object):
    """Calculate GradCAM salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        # Modification to the code to make it work with my models
        target_layer = get_target_layer(self.model_arch, model_type)

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 3, *(input_size), device=device))
                print('saliency_map size :', self.activations['value'].shape[2:])


    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()
        torch.set_grad_enabled(True)
        logit = self.model_arch(input)

        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        # score = Variable(score, requires_grad=True)
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)

# Code also extracted from MIT licensed code Copyright (c) 2021 Alinstein Jose
# Implementation of "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
        
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """

    # Squeeze the mask to make it 2D (H, W)
    mask = mask.squeeze().cpu().detach().numpy()  # Convert to numpy and remove extra dimensions

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap + img
    result = result.div(result.max()).squeeze()
    result = result - result.min()
    result = result / result.max()
    return heatmap, result
    
def get_optimizer(params, optimizer='Adam', lr=1e-4, momentum=0.9, weight_decay=0.0):
    """
    Loads and returns the selected optimizer
    """
    if optimizer == 'SGD':
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'SGD_Nesterov':
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    elif optimizer == 'Adamax':
        return Adamax(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'Adam':
        return Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'AdamW':
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'RMSprop':
        return RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise Exception('Unknown optimizer : {}'.format(optimizer))

def get_loss(loss_type='asl1', counts=None, device='cpu', beta=0.99):

    if loss_type == 'bce':
        loss_fn = nn.BCEWithLogitsLoss()
    elif loss_type == 'bce_w':
        if counts is not None:
            class_weights = effective_weights(counts, beta=beta).to(device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            print("You didn't provide weights so we proceed with normal BCE")
            loss_fn = nn.BCEWithLogitsLoss()
    elif loss_type == 'asymmetric':
        loss_fn = AsymmetricLoss()
    elif loss_type == 'asymmetric_avg':
        loss_fn = AsymmetricLoss(average=True)
    elif loss_type == 'asl1':
        loss_fn = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0)
    elif loss_type == 'asl2':
        loss_fn = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.025)
    elif loss_type == 'asl3':
        loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    elif loss_type == 'focal':
        loss_fn = AsymmetricLoss(gamma_neg=2, gamma_pos=2, clip=0)
    
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'.")
    
    return loss_fn


def get_scheduler(optimizer, name='cyclic'):

    if name == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    elif name == 'plateau1':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=6)
    elif name == 'cyclic':
        scheduler = CyclicLR(optimizer, base_lr=0.00005, max_lr=0.006, step_size_up=3200, mode='triangular2')
    elif name == 'cyclic2':
        scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=3200, mode='triangular')
    elif name == 'cosine': #redefine lr for this one since it's biggest point it's its initial lr
        scheduler = CosineAnnealingLR(optimizer, T_max=7, eta_min=0)
    elif name == 'warmupcosine':
        scheduler = CosineAnnealingWarmupRestarts(optimizer, 
                                          first_cycle_steps=10,
                                          cycle_mult=1,
                                          max_lr=0.001,
                                          min_lr=0.00005,
                                          warmup_steps=3,
                                          gamma=0.9)
    elif name == 'chexclusion':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    else:
        raise ValueError(f"Unknown scheduler type '{name}'.")
    
    return scheduler