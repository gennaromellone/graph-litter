import numpy as np
from sklearn.neighbors import kneighbors_graph
import pickle

from utility import utility


'''
	Graph construction, for tracker at CVPR Lab
	Code by Wieke Prummel 
'''

def flatten_features(frame_dict):
    features = []
    for roi_dict in frame_dict:
        features.append(roi_dict["feature_vec"])
    return np.array(features)

def flatten_labels(frame_dict):
    labels = []
    for roi_dict in frame_dict:
        labels.append(roi_dict["labels"])
    
    # Uncomment if graph classification
    #np.unique(labels)
    #labels = max(labels)
    #print(max_labels)
    
    return np.array(labels)

def flatten_rois(frame_dict):
    rois = []
    for roi_dict in frame_dict:
        rois.append(roi_dict["roi_number"])
    return np.array(rois)

# Extract a GCN graph from ROIs and their features. Files are adjacency .npz and graphs .pkl for each image
def exportGraph(rois, path_to_construction, isTrain, knn_param, img_number):
    
    # Create folders if not exist
    
    subf = "train"
    if not isTrain:
        subf = "test"

    npz_path = path_to_construction + "/adjacency_" + subf + "/"
    pkl_path = path_to_construction + "/graphs_" + subf + "/"

    features_matrix = flatten_features(rois)
    if len(features_matrix) < knn_param:
        print("Skipping this ROI. KNN parameters too high")
        pass
    else:
        labels = flatten_labels(rois)
        rois = flatten_rois(rois)
        if np.isnan(features_matrix).any():
            print("Features matrix has NaN values!")
        else:
            # Build a k-nearest neighbors graph using kneighbors_graph
            A = kneighbors_graph(features_matrix, knn_param, mode='distance', include_self=True)
            row, col = A.nonzero()

            # Export NPZ and PKL files
            np.savez(npz_path + 'frame_' + str(img_number), row=row, col=col)
            filename_graph = pkl_path + 'frame_'+ str(img_number) + '.pkl'
            with open(filename_graph, 'wb') as f:
                pickle.dump([rois, features_matrix, labels], f)
                pass


def createGraph(rois, knn_param):
    
    # Create folders if not exist
    
    features_matrix = flatten_features(rois)
    if len(features_matrix) < knn_param:
        print("Skipping this ROI. KNN parameters too high")
        pass
    else:
        labels = flatten_labels(rois)
        rois = flatten_rois(rois)
        if np.isnan(features_matrix).any():
            print("Features matrix has NaN values!")
        else:
            # Build a k-nearest neighbors graph using kneighbors_graph
            A = kneighbors_graph(features_matrix, knn_param, mode='distance', include_self=True)
            row, col = A.nonzero()
            features = [rois, features_matrix, labels]

            return features, row, col
