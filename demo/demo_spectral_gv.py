# This script is adapted from: https://github.com/jac99/Egonn/blob/main/eval/evaluate.py
# This demo script evaluates the EgoNN global and local features for the Place Recognition task on KITTI360 09.
# Outputs results for place recognition both with and without re-ranking using SpectralGV.

import argparse
import numpy as np
import tqdm
import os
import sys
import random
from typing import List
from time import time
import pickle
import copy
import torch

from sgv_utils import *

print('\n' + ' '.join([sys.executable] + sys.argv))

class EvaluationTuple:
    # Tuple describing an evaluation set element
    def __init__(self, timestamp: int, rel_scan_filepath: str, position: np.array, pose: np.array = None):
        # position: x, y position in meters
        # pose: 6 DoF pose (as 4x4 pose matrix)
        assert position.shape == (2,)
        assert pose is None or pose.shape == (4, 4)
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.position = position
        self.pose = pose

    def to_tuple(self):
        return self.timestamp, self.rel_scan_filepath, self.position, self.pose

class EvaluationSet:
    # Evaluation set consisting of map and query elements
    def __init__(self, query_set: List[EvaluationTuple] = None, map_set: List[EvaluationTuple] = None):
        self.query_set = query_set
        self.map_set = map_set

    def save(self, pickle_filepath: str):
        # Pickle the evaluation set

        # Convert data to tuples and save as tuples
        query_l = []
        for e in self.query_set:
            query_l.append(e.to_tuple())

        map_l = []
        for e in self.map_set:
            map_l.append(e.to_tuple())
        pickle.dump([query_l, map_l], open(pickle_filepath, 'wb'))

    def load(self, pickle_filepath: str):
        # Load evaluation set from the pickle
        query_l, map_l = pickle.load(open(pickle_filepath, 'rb'))

        self.query_set = []
        for e in query_l:
            self.query_set.append(EvaluationTuple(e[0], e[1], e[2], e[3]))

        self.map_set = []
        for e in map_l:
            self.map_set.append(EvaluationTuple(e[0], e[1], e[2], e[3]))

    def get_map_positions(self):
        # Get map positions as (N, 2) array
        positions = np.zeros((len(self.map_set), 2), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            positions[ndx] = pos.position
        return positions

    def get_query_positions(self):
        # Get query positions as (N, 2) array
        positions = np.zeros((len(self.query_set), 2), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            positions[ndx] = pos.position
        return positions

class EvaluatorDemo:
    def __init__(self, dataset_type: str, eval_set_pickle: str, device: str,
                 radius: List[float] = (5, 20), k: int = 50, debug: bool = False):
        # radius: list of thresholds (in meters) to consider an element from the map sequence a true positive
        # k: maximum number of nearest neighbours to consider
        # n_samples: number of samples taken from a query sequence (None=all query elements)

        self.dataset_type = dataset_type
        self.eval_set_filepath = os.path.join(os.path.dirname(__file__), 'demo_pickles/', eval_set_pickle)
        self.device = device
        self.radius = radius
        self.k = k
        self.debug = debug

        assert os.path.exists(self.eval_set_filepath), f'Cannot access evaluation set pickle: {self.eval_set_filepath}'
        self.eval_set = EvaluationSet()
        self.eval_set.load(self.eval_set_filepath)

        # self.pc_loader = get_pointcloud_loader(self.dataset_type)

    def evaluate(self, model, *args, **kwargs):
        # This method must be implemented in inheriting classes
        # Must return embedding as a numpy vector
        raise NotImplementedError('Not implemented')

    def compute_embedding(self, pc, model, *args, **kwargs):
        # This method must be implemented in inheriting classes
        # Must return embedding as a numpy vector
        raise NotImplementedError('Not implemented')

    def model2eval(self, model):
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        model.eval()


class MetLocEvaluatorDemo(EvaluatorDemo):
    def __init__(self, dataset_type: str, eval_set_pickle: str, device: str,
                 radius: List[float], k: int = 20):
        super().__init__(dataset_type, eval_set_pickle, device, radius, k)
        self.n_samples = None

    def model2eval(self, models):
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        [model.eval() for model in models]

    def evaluate(self):

        save_path = os.path.dirname(__file__) + '/demo_pickles/ego_nn_' + self.dataset_type + '/'

        with open(save_path + 'query_embeddings.pickle', 'rb') as handle:
            query_embeddings = pickle.load(handle)
        with open(save_path + 'local_query_embeddings.pickle', 'rb') as handle:
            local_query_embeddings = pickle.load(handle)

        with open(save_path + 'map_embeddings.pickle', 'rb') as handle:
            map_embeddings = pickle.load(handle)
        with open(save_path + 'local_map_embeddings.pickle', 'rb') as handle:
            local_map_embeddings = pickle.load(handle)

        local_map_embeddings_keypoints = torch.stack([lme['keypoints'] for lme in local_map_embeddings])
        local_map_embeddings_features = torch.stack([lme['features'] for lme in local_map_embeddings])
        map_positions = self.eval_set.get_map_positions() # Nmap x 2
        query_positions = self.eval_set.get_query_positions() # Nquery x 2

        if self.n_samples is None or len(query_embeddings) <= self.n_samples:
            query_indexes = list(range(len(query_embeddings)))
            self.n_samples = len(query_embeddings)
        else:
            query_indexes = random.sample(range(len(query_embeddings)), self.n_samples)


        # Dictionary to store the number of true positives (for global desc. metrics) for different radius and NN number
        global_metrics = {'tp': {r: [0] * self.k for r in self.radius}}
        global_metrics['tp_rr'] = {r: [0] * self.k for r in self.radius}
        global_metrics['RR'] = {r: [] for r in self.radius}
        global_metrics['RR_rr'] = {r: [] for r in self.radius}
        global_metrics['t_RR'] = []

        for query_ndx in tqdm.tqdm(query_indexes):
            # Check if the query element has a true match within each radius
            query_pos = query_positions[query_ndx]
            query_pose = self.eval_set.query_set[query_ndx].pose

            # Nearest neighbour search in the embedding space
            query_embedding = query_embeddings[query_ndx]
            embed_dist = np.linalg.norm(map_embeddings - query_embedding, axis=1)
            nn_ndx = np.argsort(embed_dist)[:self.k]

            # PLACE RECOGNITION EVALUATION
            # Euclidean distance between the query and nn
            # Here we use non-icp refined poses, but for the global descriptor it's fine
            delta = query_pos - map_positions[nn_ndx]       # (k, 2) array
            euclid_dist = np.linalg.norm(delta, axis=1)     # (k,) array


            # Re-Ranking:
            topk = min(self.k,len(nn_ndx))
            tick = time()
            candidate_lfs = local_map_embeddings_features[nn_ndx]
            candidate_kps = local_map_embeddings_keypoints[nn_ndx]
            fitness_list = sgv_fn(local_query_embeddings[query_ndx], candidate_lfs, candidate_kps,d_thresh=0.4)
            topk_rerank = np.flip(np.asarray(fitness_list).argsort())
            topk_rerank = np.flip(np.asarray(fitness_list).argsort())
            topk_rerank_inds = copy.deepcopy(nn_ndx)
            topk_rerank_inds[:topk] = nn_ndx[topk_rerank]
            t_rerank = time() - tick
            global_metrics['t_RR'].append(t_rerank)

            delta_rerank = query_pos - map_positions[topk_rerank_inds]
            euclid_dist_rr = np.linalg.norm(delta_rerank, axis=1)
                

            # Count true positives for different radius and NN number
            global_metrics['tp'] = {r: [global_metrics['tp'][r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}
            global_metrics['tp_rr'] = {r: [global_metrics['tp_rr'][r][nn] + (1 if (euclid_dist_rr[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}
            global_metrics['RR'] = {r: global_metrics['RR'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist <= r) if x), 0)] for r in self.radius}
            global_metrics['RR_rr'] = {r: global_metrics['RR_rr'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist_rr <= r) if x), 0)] for r in self.radius}


        # Calculate mean metrics
        global_metrics["recall"] = {r: [global_metrics['tp'][r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}
        global_metrics["recall_rr"] = {r: [global_metrics['tp_rr'][r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}
        global_metrics['MRR'] = {r: np.mean(np.asarray(global_metrics['RR'][r])) for r in self.radius}
        global_metrics['MRR_rr'] = {r: np.mean(np.asarray(global_metrics['RR_rr'][r])) for r in self.radius}
        global_metrics['mean_t_RR'] = np.mean(np.asarray(global_metrics['t_RR']))


        return global_metrics


    def print_results(self, global_metrics):
        # Global descriptor results are saved with the last n_k entry
        print('\n','Initial Retrieval:')
        recall = global_metrics['recall']
        for r in recall:
            print(f"Radius: {r} [m] : ")
            print(f"Recall@N : ", end='')
            for x in recall[r]:
                print("{:0.1f}, ".format(x*100.0), end='')
            print("")
            print('MRR: {:0.1f}'.format(global_metrics['MRR'][r]*100.0))
        
        print('\n','Re-Ranking:')
        recall_rr = global_metrics['recall_rr']
        for r_rr in recall_rr:
            print(f"Radius: {r_rr} [m] : ")
            print(f"Recall@N : ", end='')
            for x in recall_rr[r_rr]:
                print("{:0.1f}, ".format(x*100.0), end='')
            print("")
            print('MRR: {:0.1f}'.format(global_metrics['MRR_rr'][r_rr]*100.0))
        print('Re-Ranking Time: {:0.3f}'.format(1000.0 *global_metrics['mean_t_RR']))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate MinkLoc model')
    parser.add_argument('--dataset_type', type=str, required=False, default='kitti360')
    parser.add_argument('--eval_set', type=str, required=False, default='kitti360_09_3.0_eval.pickle', help='File name')
    parser.add_argument('--radius', type=float, nargs='+', default=[5, 20], help='True Positive thresholds in meters')
    parser.add_argument('--n_topk', type=int, default=2, help='Number of keypoints to calculate repeatability')
    args = parser.parse_args()

    print(f'Dataset type: {args.dataset_type}')
    print(f'Evaluation set: {args.eval_set}')
    print(f'Radius: {args.radius} [m]')
    print('')

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))


    evaluator = MetLocEvaluatorDemo(args.dataset_type, args.eval_set, device, radius=args.radius, k=args.n_topk)
    global_metrics= evaluator.evaluate()
    evaluator.print_results(global_metrics)