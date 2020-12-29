import itertools
from tqdm import tqdm
import torch
import numpy as np
import os
from time import time
import torch.nn.functional as F
import argparse
from sklearn.metrics import r2_score

from torch.utils.data import DataLoader
from dataset import FirstImpressionsDense
from resnet import ResNet

SAVE_DIR = 'weights'

def train(
        attribute='Trustworthiness', 
        output_type='majority_vote', 
        test_fold=0, 
        time_dir='final_majority_vote', 
        fold_seed=0, 
        num_epochs=3,
        batch_size=32,
):
    ####################
    # load the training dataset #
    ####################
    ds = FirstImpressionsDense(
            stage='train', dataset=attribute, numpy=False, augment=True, 
            output_type=output_type, test_fold=test_fold, fold_seed=fold_seed,
    )
    num_unique_ids = len(np.unique(ds.df['face_id'])) 
    print('Num unique', num_unique_ids)

    dl = DataLoader(
            ds, batch_size=batch_size, 
            num_workers=8,
            shuffle=True, 
            drop_last=True
    )

    ####################
    # load the training (unique) dataset #
    ####################
    us = FirstImpressionsDense(
            stage='train', dataset=attribute, numpy=False, augment=True, 
            output_type=output_type, test_fold=test_fold, fold_seed=fold_seed,
            unique=True,
    )
    print('Num unique (in unique)', num_unique_ids)
    ul = DataLoader(
            us, batch_size=128, 
            num_workers=8,
            shuffle=False, 
            drop_last=False
    )

    ####################
    # load the valid dataset #
    ####################

    ts = FirstImpressionsDense(
            stage='test', dataset=attribute, augment=False, 
            numpy=False, output_type=output_type, 
            test_fold=test_fold, fold_seed=fold_seed,
            unique=True,
    )
    tl = DataLoader(ts, batch_size=128, num_workers=8, shuffle=False, drop_last=False)

    ######################
    # choose the loss fn #
    ######################
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

    ########################
    # initialize the model #
    ########################
    m = ResNet(
            model='18', num_classes=num_classes,
            strict=False, verbose=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    if torch.cuda.device_count() > 1:
        print(f"Model will use {torch.cuda.device_count()} GPUs!")
        m = torch.nn.DataParallel(m)
    

    m.to(device).train()


    ############################
    # initialize the optimizer #
    ############################
    optim = torch.optim.Adam(m.parameters(), lr=0.0001)

    ##################################
    # directories for saving weights #
    ##################################
    # NOTE: CHANGE THIS
    time_dir = str(time()) if time_dir is None else time_dir
    save_dir = os.path.join(SAVE_DIR + '/{}'.format(time_dir))
    save_dir = os.path.join(save_dir, str(fold_seed), str(test_fold), 'checkpoints')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    running_error = list()
    true_epoch = 0
    r2 = None
    r2t = None
    for epoch in range(num_epochs):
        m.train()
        pbar = tqdm(total=len(ds), desc='Train: ')
        for batch_num, fd in enumerate(dl):
            optim.zero_grad()
            x, y = fd['image'], fd['annotation']

            bs = y.size(0)
            x, y = x.cuda().float() / 255., y.cuda()
            x = x.permute(0, 3, 1, 2).contiguous()

            yhat = m(x)

            if output_type == 'majority_vote':
                y = y.long()
            elif output_type == 'distribution':
                y = y.float()
                yhat = F.log_softmax(yhat)
            elif output_type == 'average':
                y = y.float()
                yhat = yhat.view(bs)
            else:
                raise ValueError

            error = criterion(yhat, y)
            error.backward()
            
            optim.step()

            '''
            # if batch_num == 0:
            if ((batch_num*batch_size) % num_unique_ids) < batch_size:
                #####################
                # NOTE: CHANGE THIS #
                #####################
                save_f = os.path.join(save_dir, '{}.pth'.format(true_epoch))
                print('Saving to: {}'.format(save_f))
                torch.save(m.state_dict(), save_f)
                true_epoch += 1
            '''

            running_error.append(float(error.cpu().data.numpy()))
            running_error_display = np.mean(running_error[-100:])
            desc = 'Epoch: {} Batch: {}: Training Error: {} Training R2: {} Validation R2: {} '.format(epoch, batch_num, running_error_display, r2t, r2)
            pbar.set_description(desc)
            pbar.update(bs)

        #####################
        # Saving Weights    #
        #####################
        save_f = os.path.join(save_dir, '{}.pth'.format(epoch))
        print('Saving to: {}'.format(save_f))
        torch.save(m.state_dict(), save_f)

        #####################
        # Training R2        #
        #####################
        m.eval()
        ys, yhats = list(), list()
        for batch_num, fd in enumerate(ul):
            with torch.no_grad():
                x, y = fd['image'], fd['annotation']

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
        r2t = r2_score(ys, yhats)

        #####################
        # Validation        #
        #####################
        ys, yhats = list(), list()
        for batch_num, fd in enumerate(tl):
            with torch.no_grad():
                x, y = fd['image'], fd['annotation']

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
        r2 = r2_score(ys, yhats)


if __name__ == '__main__':
    for fold_seed in range(0, 1):
        print('FOLD SEED', fold_seed)
        for test_fold in range(1):
            print('TEST FOLD', test_fold)
            train(
                    attribute='Trustworthiness', 
                    output_type='average', 
                    test_fold=test_fold, time_dir='incremental_average', 
                    fold_seed=fold_seed,
                    num_epochs=10,
            )
