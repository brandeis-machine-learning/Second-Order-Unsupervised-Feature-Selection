from skfeature.utility import unsupervised_evaluation
from util import load_dataset
import nxmetis
import numpy as np
import networkx as nx
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

DATASETS = ['COIL20', 'colon', 'gisette', 'Lung-Cancer', 'madelon', 'Movementlibras', 'nci9', 'ORL', 'Sonar', 'UAV1', 'UAV2', 'waveform-5000']
SELECT_PER_CLUSTER = 1
EPOCHS = [1, 50, 100, 150, 200, 250, 300]


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--alpha', type=float, required=False, default=1.0)
parser.add_argument('-b', '--beta', type=float, required=False, default=0.001)
parser.add_argument('-p', '--percent', type=float, required=False, default=0.1)
args = parser.parse_args()

NUM_SELECT = args.percent



def main(DATASET, epoch):
    # load data
    X, y, NUM_CLASSES, num_features = load_dataset(DATASET)

    adj = np.loadtxt('results/matrix/DEC_' + DATASET + '_' + str(epoch) + '_relation.csv', delimiter=",")
    # adj = np.corrcoef(X.T)
    adj = np.abs(adj)
    selected_features = []
    select = np.zeros(num_features)
    count = 0
    num_selected = int(NUM_SELECT * num_features)

    adj = adj - adj * np.eye(adj.shape[0])
    adj[adj <= 0] = 0
    adj_int = adj/np.max(adj)*1000
    adj_int = adj_int.astype(np.int)
    

    # Graph Segmentation =============================================================
    adj_int = nx.from_numpy_array(adj_int)
    (st, parts) = nxmetis.partition(adj_int, int((num_selected+SELECT_PER_CLUSTER-1) / SELECT_PER_CLUSTER), recursive=False)
    cluster_membership = {node: membership for node, membership in enumerate(parts)}

    for c in cluster_membership.keys():
        for _ in range(SELECT_PER_CLUSTER):
            pick, max_weight = 0, 0.0
            for i in range(len(cluster_membership[c])):
                if select[cluster_membership[c][i]] != 0:
                    continue
                cur_weight = 0
                for j in range(len(cluster_membership[c])):
                    cur_weight += adj[cluster_membership[c][i]][cluster_membership[c][j]]
                if max_weight < cur_weight:
                    max_weight = cur_weight
                    pick = i

            if len(cluster_membership[c]) == 0:
                continue
            original_index = cluster_membership[c][pick]
            select[original_index] = 1

    X = np.array(X)
    X = X.T
    count = 0
    for i in range(num_features):
        if select[i] == 1:
            selected_features.append(X[i])
            count += 1
    selected_features = np.array(selected_features)
    selected_features = selected_features.T

    # perform kmeans clustering based on the selected features and repeats 20 times
    nmi_total = []
    acc_total = []
    for i in range(20):
        nmi, acc = unsupervised_evaluation.evaluation(X_selected=selected_features, n_clusters=NUM_CLASSES, y=y)
        nmi_total.append(nmi)
        acc_total.append(acc)

    nmi_total = np.array(nmi_total)
    acc_total = np.array(acc_total)
    np.savetxt('DEC_evaluation/selected/DEC_' + DATASET + '_selected_' + str(NUM_SELECT) + '.csv', select, delimiter = ',')

    return np.mean(nmi_total), np.mean(acc_total), count, np.std(nmi_total), np.std(acc_total)


if __name__ == '__main__':
    res_acc, res_acc_std = [], []
    res_nmi, res_nmi_std = [], []
    start = time.time()
    now = time.time()
    for dataset in DATASETS:
        cur_acc, cur_nmi, cur_acc_std, cur_nmi_std = [], [], [], []
        for epoch in EPOCHS:
            nmi, acc, count, std_nmi, std = main(dataset, epoch)
            cur_acc.append(acc)
            cur_acc_std.append(std)
            cur_nmi.append(nmi)
            cur_nmi_std.append(std_nmi)
        res_acc.append(np.array(cur_acc))
        res_acc_std.append(np.array(cur_acc_std))
        res_nmi.append(np.array(cur_nmi))
        res_nmi_std.append(np.array(cur_nmi_std))
        print(dataset, '----Selected:', count, '----Time:', time.time() - now, res_acc[-1])
        now = time.time()
    res_acc = np.array(res_acc)
    res_acc_std = np.array(res_acc_std)
    res_nmi = np.array(res_nmi)
    res_nmi_std = np.array(res_nmi_std)
    np.savetxt('DEC_evaluation/DEC_acc_' + str(args.alpha) + '_' + str(args.beta) + '_' + str(NUM_SELECT) + '.csv', res_acc, delimiter = ',')
    np.savetxt('DEC_evaluation/DEC_acc_std_' + str(args.alpha) + '_' + str(args.beta) + '_' + str(NUM_SELECT) + '.csv', res_acc_std, delimiter = ',')
    print("TOTAL TIME:", time.time() - start, res_acc)