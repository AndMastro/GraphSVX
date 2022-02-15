""" script_explain.py

    Derive explanations using GraphSVX 
"""

import argparse
import random
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

import configs
from utils.io_utils import fix_seed
from src.data import prepare_data
from src.explainers import GraphSVX
from src.train import evaluate, test
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm

def main():

    sparsity_values = [elem/10 for elem in range(1,10)]

    args = configs.arg_parse()
    fix_seed(args.seed)

    # Load the dataset
    data = prepare_data(args.dataset, args.train_ratio,
                        args.input_dim, args.seed)

    # Load the model
    model_path = 'models/{}_model_{}.pth'.format(args.model, args.dataset)
    model = torch.load(model_path)
    
    # Evaluate the model 
    if args.dataset in ['Cora', 'PubMed']:
        _, test_acc = evaluate(data, model, data.test_mask)
    else: 
        test_acc = test(data, model, data.test_mask)
    print('Test accuracy is {:.4f}'.format(test_acc))

    # Explain it with GraphSVX
    explainer = GraphSVX(data, model, args.gpu)

    # Distinguish graph classfication from node classification
    if args.dataset in ['Mutagenicity', 'syn6']:
        explanations = explainer.explain_graphs(args.indexes,
                                         args.hops,
                                         args.num_samples,
                                         args.info, 
                                         args.multiclass,
                                         args.fullempty,
                                         args.S,
                                         'graph_classification',
                                         args.feat,
                                         args.coal,
                                         args.g,
                                         args.regu,
                                         True)
    else: 
        explanations = explainer.explain(args.indexes,
                                        args.hops,
                                        args.num_samples,
                                        False, #args.info
                                        args.multiclass,
                                        args.fullempty,
                                        args.S,
                                        args.hv,
                                        args.feat,
                                        args.coal,
                                        args.g,
                                        args.regu,
                                        True) #@mastro it was true

    # print("Explanations: ", explanations) #@mastro edit
    # print("Neighbors: ", explainer.neighbours)
    #print('Sum explanations: ', [np.sum(explanation) for explanation in explanations])
    #print('Base value: ', explainer.base_values)

    #Computed explanations given sparsity (computed on the edges)
    sparsity = 0.3
    num_hops = args.hops
    node_index = args.indexes[0] #node to explain
    num_features_explanations = explainer.F #features in explanation, we only consider nodes. The order returned is [f0,..,fn,n0,...nm]. We want to set self.F to 0
    explanations = explanations[0][num_features_explanations:] #only one class, need to use predicted class
    neighbors = explainer.neighbours #k_hop_subgraph_nodes
    
    
    #compute k-hops subgraph since we need edge index for sparsity
    k_hop_subgraph_nodes, k_hop_subgraph_edge_index, _, _ = k_hop_subgraph(
    node_index, num_hops, data.edge_index, relabel_nodes=False,
    num_nodes=None)
    k_hop_subgraph_num_edges = k_hop_subgraph_edge_index.shape[1]
    
    #check that all the ways to compute the k-hop subgraph size are equivalent
    assert(neighbors.shape[0] == explanations.shape[0] == k_hop_subgraph_nodes.shape[0] - 1)
    # print(explanations[num_features_explanations:].shape[0])
    
    k_hop_subgraph_size = k_hop_subgraph_edge_index.shape[1] 
    
    #now sparsity on nodes, change on edges
    num_important_edges = round((1 - sparsity) * k_hop_subgraph_size) #round and not just int
    
    _, idxs = torch.topk(torch.from_numpy(
        np.abs(explanations)), neighbors.shape[0]) #num_important_nodes, with neighbors.shape[0] we take them all in order to remove them to obtain the needed sparsity
 
    vals = [explanations[idx] for idx in idxs]
    influential_nei = {}
    for idx, val in zip(idxs, vals):
        influential_nei[neighbors[idx]] = val
    nodes_and_explanations = [(item[0].item(), item[1].item()) for item in list(influential_nei.items())]

    # print('Most influential neighbours: ', [
    #         (item[0].item(), item[1].item()) for item in list(influential_nei.items())])
    # print("Number of nodes in explanation: ", len(influential_nei))
    
    #print(nodes_and_explanations)
    current_explanation_subgraph_edge_index = k_hop_subgraph_edge_index
    previous_explanation_edge_index = None
    explanation_subgraph_edge_index = None
    for i in tqdm(range(len(nodes_and_explanations) - 1, 0, -1)):
        if current_explanation_subgraph_edge_index.shape[1] <= num_important_edges: #num_important_edges:
            explanation_subgraph_edge_index = current_explanation_subgraph_edge_index
            break
        else:
            node_to_remove = nodes_and_explanations[i][0] #0 is node, 1 is shapley value
            # print("node to remove ", node_to_remove)
            
            indices_to_remove_from = ((current_explanation_subgraph_edge_index[0] == node_to_remove).nonzero())
            indices_to_remove_to = ((current_explanation_subgraph_edge_index[1] == node_to_remove).nonzero())
            # print(indices_to_remove_to)
            # print(indices_to_remove_from)
            if indices_to_remove_from.nelement() != 0:
                indices_to_remove_from_as_set = set(indices_to_remove_from.squeeze().tolist())
            else:
                indices_to_remove_from_as_set = set()

            if indices_to_remove_to.nelement() != 0:
                indices_to_remove_to_as_set = set(indices_to_remove_to.squeeze().tolist())
            else:
                indices_to_remove_to_as_set = set()

            indices_to_remove_as_list = list(indices_to_remove_from_as_set.union(indices_to_remove_to_as_set))
            # print(indices_to_remove_as_list)
            all_indices_as_list = list(range(0,current_explanation_subgraph_edge_index.shape[1]))
            indices_to_keep_as_list = list(set(all_indices_as_list) - set(indices_to_remove_as_list))
             
            # indices_to_keep = torch.cat((indices_to_keep_from, indices_to_keep_to))
            # indices_to_keep = ((current_explanation_subgraph_edge_index != node_to_remove).nonzero())
            indices_to_keep = torch.LongTensor(indices_to_keep_as_list)
            # print(indices_to_remove_from.squeeze())
            # print(indices_to_remove_to.squeeze())
            current_explanation_subgraph_edge_index = torch.index_select(k_hop_subgraph_edge_index, 1, indices_to_keep.squeeze())

            explanation_subgraph_edge_index = current_explanation_subgraph_edge_index
            # current_explanation_subgraph_edge_index_from = current_explanation_subgraph_edge_index[0][current_explanation_subgraph_edge_index[0]!=node_to_remove]
            # current_explanation_subgraph_edge_index_to = current_explanation_subgraph_edge_index[1][current_explanation_subgraph_edge_index[1]!=node_to_remove]
            # current_explanation_subgraph_edge_index = torch.stack((current_explanation_subgraph_edge_index_from, current_explanation_subgraph_edge_index_to))
            
            # print(current_explanation_subgraph_edge_index.shape)
            # # print(current_explanation_subgraph_edge_index)
            # print(k_hop_subgraph_edge_index.shape)
            

    print("Desired sparsity: ", sparsity, "With num of important edge: ", num_important_edges)        
    print("Num edges in explanation: ", explanation_subgraph_edge_index.shape[1])
    print("Num edges in k-hop subgraph: ", k_hop_subgraph_edge_index.shape[1])
    print("Obtained sparsity: ", 1 - explanation_subgraph_edge_index.shape[1]/k_hop_subgraph_edge_index.shape[1])  

if __name__ == "__main__":
    main()
