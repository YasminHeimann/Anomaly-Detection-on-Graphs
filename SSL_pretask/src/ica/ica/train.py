"""" This implementation is largely based on and adapted from:
 https://github.com/sskhandle/Iterative-Classification """
from ica.utils import load_data, pick_aggregator, create_map, build_graph
# from ica.utils import pick_aggregator, create_map, build_graph
# from ica.utils_gcn import load_data
from ica.classifiers import LocalClassifier, RelationalClassifier, ICA

from scipy.stats import sem

from sklearn.metrics import accuracy_score
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', default='citeseer', help='Dataset string.')
parser.add_argument('-classifier', default='sklearn.linear_model.LogisticRegression',
                    help='Underlying classifier.')
parser.add_argument('-seed', type=int, default=42, help='Random seed.')
parser.add_argument('-num_trials', type=int, default=10, help='Number of trials.')
parser.add_argument('-max_iteration', type=int, default=10, help='Number of iterations (iterative classification).')
parser.add_argument('-aggregate', choices=['count', 'prop'], default='count', help='Aggregation operator.')
parser.add_argument('-bootstrap', default=True, action='store_true',
                    help='Bootstrap relational classifier training with local classifier predictions.')
parser.add_argument('-validation', default=False, action='store_true',
                    help='Whether to test on validation set (True) or test set (False).')

args = parser.parse_args()
np.random.seed(args.seed)

# load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
idx_test = np.array([x for x in range(adj.shape[0]) if x not in idx_train])
# from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import encode_onehot, normalize_feature
# data = Dataset(root='/tmp/', name=args.dataset, setting='gcn')
# adj, features, labels = data.adj, data.features, data.labels
# idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

import ipdb
ipdb.set_trace()

# labels = encode_onehot(labels)
#
features = normalize_feature(features)
# idx_train = np.arange(120)
# idx_val = idx_train
# # idx_test = np.arange(120, adj.shape[0])
# idx_test = np.arange(120+500, 120+500+1000)
#
# import ipdb
# ipdb.set_trace()

graph, domain_labels = build_graph(adj, features, labels)

# train / test splits
train = idx_train
if args.validation:
    test = idx_val
else:
    test = idx_test
eval_idx = np.setdiff1d(range(adj.shape[0]), idx_train)

# run training
ica_accuracies = list()
for run in range(args.num_trials):

    t_begin = time.time()

    # random ordering
    np.random.shuffle(eval_idx)

    y_true = [graph.node_list[t].label for t in test]
    local_clf = LocalClassifier(args.classifier)
    agg = pick_aggregator(args.aggregate, domain_labels)
    relational_clf = RelationalClassifier(args.classifier, agg)
    ica = ICA(local_clf, relational_clf, args.bootstrap, max_iteration=args.max_iteration)
    ica.fit(graph, train)
    conditional_node_to_label_map = create_map(graph, train)

    ica_predict = ica.predict(graph, eval_idx, test, conditional_node_to_label_map)
    ica_accuracy = accuracy_score(y_true, ica_predict)
    ica_accuracies.append(ica_accuracy)
    print('Run ' + str(run) + ': \t\t' + str(ica_accuracy) + ', Elapsed time: \t\t' + str(time.time() - t_begin))

print("Final test results: {:.5f} +/- {:.5f} (sem)".format(np.mean(ica_accuracies), sem(ica_accuracies)))


