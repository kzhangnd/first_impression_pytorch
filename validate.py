import itertools
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import os
from time import time
from scipy.stats.mstats import linregress
import sklearn.metrics

from dataset import FirstImpressionsDense
from resnet import ResNet
from torch.utils.data import DataLoader


def validate(
        attribute='Trustworthiness', 
        output_type='distribution', 
        test_fold=0, 
        fold_seed=0, 
        ckpt=None, 
        stage='test'
):

    ###################
    # initialize loss #
    ###################
    if output_type == 'majority_vote':
        criterion = torch.nn.CrossEntropyLoss()
        num_classes = 7
    elif output_type == 'distribution':
        criterion = torch.nn.KLDivLoss()
        num_classes = 7
    elif output_type == 'average':
        criterion = torch.nn.MSELoss().cuda()
        num_classes = 1
    else:
        raise ValueError

    ####################
    # initialize model #
    ####################
    m = ResNet(
            model='18', num_classes=num_classes, 
            strict=True, verbose=True,
            pth=ckpt,
    ).cuda().eval()


    ######################
    # initialize dataset #
    ######################
    ds = FirstImpressionsDense(
            stage=stage, dataset=attribute, augment=False, 
            numpy=False, output_type=output_type, 
            test_fold=test_fold, fold_seed=fold_seed,
            unique=True,
    )
    dl = DataLoader(ds, batch_size=128, num_workers=8, shuffle=False, drop_last=False)

    ys, yhats, face_ids = list(), list(), list()
    pbar = tqdm(total=len(ds), desc='Validate: ')
    for batch_num, fd in enumerate(dl):
        with torch.no_grad():
            x, y, face_id = fd['image'], fd['annotation'], fd['face_id']

            bs = y.size(0)
            x, y = x.cuda().float() / 255., y
            x = x.permute(0, 3, 1, 2) 

            yhat = m(x)

            if output_type in ['distribution', 'majority_vote']:
                yhat = F.softmax(yhat, dim=1)
                yhat = yhat.cpu().data.numpy()
            elif output_type == 'average':
                yhat = yhat.view(bs).cpu().data.numpy()

        ys.extend(y.tolist())
        yhats.extend(yhat.tolist())
        face_ids.extend(face_id.tolist())

        pbar.update(bs)

    ###########################################
    # NOTE: calculate R2 between ys and yhats #
    ###########################################
    return ys, yhats, face_ids


if __name__ == '__main__':
    validate()
