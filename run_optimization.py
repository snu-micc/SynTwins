from __future__ import print_function

import argparse
import yaml
import os
import sys
sys.path.append(os.path.realpath(__file__))
from tdc import Oracle
from time import time 
import numpy as np 

from syntwins.syntree import SynTree
from syntwins.data_utils import SynTwin_DataLoader

def main(method, oracle_name, syntwins_loader):
    start_time = time() 
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='graph_ga')
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=10000)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--seed', type=int, nargs="+", default=[42])
    parser.add_argument('--oracles', nargs="+", default=[oracle_name]) ### 
    parser.add_argument('--log_results', action='store_true')
    parser.add_argument('--log_code', action='store_true')
    args = parser.parse_args()

    args.method = method
    if method == 'syn_reinvent':
        from MPO.syn_reinvent.run import Syn_REINVENT_Optimizer as Optimizer
    elif method == 'syn_graph_ga':
        from MPO.syn_graph_ga.run import Syn_GraphGA_Optimizer as Optimizer

    path_main = os.path.dirname(os.path.realpath(__file__))
    path_main = os.path.join(path_main, "MPO", args.method)
    sys.path.append(path_main)
    args.output_dir = os.path.join(path_main, "results")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.pickle_directory = path_main
    config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))

    oracle = Oracle(name=oracle_name)
    optimizer = Optimizer(syntwins_loader, args=args)
    for seed in args.seed:
        print('seed', seed)
        optimizer.optimize(oracle=oracle, config=config_default, seed=seed)

    end_time = time()
    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))
    # print('If the program does not exit, press control+c.')


if __name__ == "__main__":
    methods = ['syn_reinvent', 'syn_graph_ga']
    oracle_names = ["Amlodipine_MPO", "Fexofenadine_MPO", "Osimertinib_MPO",  "Perindopril_MPO", 
                 "Ranolazine_MPO",  "Sitagliptin_MPO", "Zaleplon_MPO"]
    syntwins_loader = SynTwin_DataLoader(data_dir='data/', _nBits=256)
    for method in methods:
        for oracle_name in oracle_names:
            main(method, oracle_name, syntwins_loader)

