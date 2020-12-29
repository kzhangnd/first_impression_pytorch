import argparse
from time import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


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

def test(attribute, output_type, ckpt, test_fold=0, fold_seed=0):

    ys, yhats, _ = validate(
        attribute=attribute, 
        output_type=output_type, 
        test_fold=test_fold, 
        fold_seed=fold_seed,
        ckpt=ckpt, 
        stage='test'
    )

    print(ys)
    print(yhats)
    print(len(ys))
    print(len(yhats))

    r2 = r2_score(ys, yhats)
    loss = mean_squared_error(ys, yhats)

    print(f'Test size: {len(ys)} R2: {r2} Loss: {loss}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Match Extracted Features')
    parser.add_argument('-attribute', '-a', help='Attribute of metrics.', default='Trustworthiness')
    parser.add_argument('-output_type', '-t', help='Output type.', default='distribution')
    parser.add_argument('-test_fold', '-tf', help='Testing fold.', type=int, default=0)
    parser.add_argument('-fold_seed', '-fs', help='fold_seed.', type=int, default=0)
    parser.add_argument('-ckpt', '-p', help='Path to weight.', default='/afs/crc.nd.edu/user/k/kzhang4/first_impressions_pytorch/model/incremental_distribution/0/0/checkpoints/116.pth')

    args = parser.parse_args()

    test(args.attribute, args.output_type, args.ckpt, test_fold=args.test_fold, fold_seed=args.fold_seed)
