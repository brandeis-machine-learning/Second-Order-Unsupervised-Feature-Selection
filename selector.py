import numpy as np
from model import GCN_Cluster
import os
import torch
import torch.optim as optim
import time
import clustering
from clustering import ClusterAssignment
import argparse
from util import UnifLabelSampler, normalize, encode2onehotarray, load_dataset
import warnings
warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DATASETS = ['COIL20', 'colon', 'gisette', 'Lung-Cancer', 'madelon', 'Movementlibras', 'nci9', 'ORL', 'Sonar', 'UAV1', 'UAV2', 'waveform-5000']
LEARNING_RATE = 1e-4
EPOCHS = 300
USE_CUDA = True


if not USE_CUDA:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



parser = argparse.ArgumentParser()
parser.add_argument('-a', '--alpha', type=float, required=False, default=1.0)
parser.add_argument('-b', '--beta', type=float, required=False, default=0.001)
args = parser.parse_args()



def train(loader, model, crit, opt, epoch, num_classes):
    model.train()
    for i, (input_tensor, target) in enumerate(loader):
        target_attention = encode2onehotarray(target, num_classes)
        target_attention = torch.as_tensor(torch.from_numpy(target_attention), dtype=torch.float32)
        if USE_CUDA:
            target = target.cuda()
            input_tensor = input_tensor.cuda()
            target_attention = target_attention.cuda()
        input_var = torch.autograd.Variable(input_tensor)
        target_var = torch.autograd.Variable(target)
        target_attention_var = torch.autograd.Variable(target_attention)

        _, _, pseudo_label, _, _, _, pred_att, mask = model(input_var, A)

        loss_cluster_pred = crit(pseudo_label, target_var)# + crit(pred2, target_var)
        loss_att = torch.mean( torch.sum(pred_att*target_attention_var, dim=1) )
        loss_L21 = (torch.sum(torch.sqrt(torch.sum(mask ** 2, dim=1))) + torch.sum(torch.sqrt(torch.sum(mask ** 2, dim=0))))

        loss = loss_cluster_pred + loss_att * args.alpha + loss_L21 * args.beta

        # compute gradient and do SGD step
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss, loss_cluster_pred, loss_att, loss_L21






begin = time.time()
time_cost = []
for DATASET in DATASETS:
    X, y, NUM_CLASSES, num_features = load_dataset(DATASET)
    print(DATASET, "=============START!!", NUM_CLASSES, X.shape)

    model = GCN_Cluster(num_features, NUM_CLASSES, USE_CUDA=USE_CUDA)
    if USE_CUDA:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    A = np.cov(X.T)
    # A = np.corrcoef(X.T)
    A = np.abs(A)
    A[np.isinf(A)] = 1.0
    A[np.isnan(A)] = 1.0
    A = normalize(A)


    start_time = time.time()
    X = torch.as_tensor(torch.from_numpy(X), dtype=torch.float32)   # N*F
    A = torch.as_tensor(torch.from_numpy(A), dtype=torch.float32)   # F*F
    y = torch.as_tensor(torch.from_numpy(y), dtype=torch.long)   # N*C

    deepcluster = ClusterAssignment(NUM_CLASSES, num_features, 1.0)
    criterion = torch.nn.CrossEntropyLoss()

    if USE_CUDA:
        X = X.cuda()
        A = A.cuda()
        y = y.cuda()
        criterion = criterion.cuda()
        deepcluster = deepcluster.cuda()

    for epoch in range(EPOCHS):
        flag, _, _, A_att, _, _, _, mask = model(X, A)

        clustering_loss = deepcluster(flag)
        train_dataset = clustering.cluster_assign(deepcluster.images_lists, X)

        # uniformly sample per target
        sampler = UnifLabelSampler(int(len(train_dataset)),
                                   deepcluster.images_lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=len(train_dataset),
            sampler=sampler,
        )

        loss, loss_cluster_pred, loss_att, loss_L21 = train(train_dataloader, model, criterion, optimizer, epoch, NUM_CLASSES)


        if (epoch+1)%50 == 0 or epoch == 0:
            print('{:.0f} loss: {:.4f} loss_cluster_pred: {:.4f} loss_att: {:.4f} loss_L21: {:.4f} time:{:.4f}'.format(epoch+1, loss, loss_cluster_pred.data, loss_att.data, loss_L21.data, time.time()-start_time))
            sim_save = A_att.detach().cpu().numpy()
            sim_file = 'results/matrix/DEC_' + DATASET + '_' + str(epoch+1) + '_relation.csv'
            np.savetxt(sim_file, sim_save, delimiter = ',')

            mask_save = mask.detach().cpu().numpy()
            mask_file = 'results/matrix/DEC_' + DATASET + '_' + str(epoch+1) + '_mask.csv'
            np.savetxt(mask_file, mask_save, delimiter = ',')

    sim_save = A_att.detach().cpu().numpy()
    sim_file = 'results/DEC_' + DATASET + '_result_' + str(args.alpha) + '_' + str(args.beta) + '.csv'
    np.savetxt(sim_file, sim_save, delimiter = ',')

    mask_save = mask.detach().cpu().numpy()
    mask_file = 'results/DEC_' + DATASET + '_mask_' + str(args.alpha) + '_' + str(args.beta) + '.csv'
    np.savetxt(mask_file, mask_save, delimiter = ',')

    time_cost.append(time.time()-start_time)

print("Total time:", time.time() - begin)
print(time_cost)