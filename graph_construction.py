from cProfile import label
import os, re
import pickle
import scipy.io
from sklearn.neighbors import kneighbors_graph
import json
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
import numpy as np
from json import JSONEncoder
from scipy.sparse import csc_matrix
from scipy.io import savemat
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csc_matrix
from scipy import sparse

'''
	Graph construction for tracker at CVPR Lab
	Code by Wieke Prummel 
'''

import json

def get_node_data(node_file):
    # Ouvrir le fichier JSON
    with open(node_file, 'r') as file:
        data = json.load(file)
    json_data = json.dumps(data, indent=4)
    # print ("json data is:", json_data)
    return data

def create_dico(data):
    # Accéder aux données
    threshold_score = data["thresholdScore"]
    frames = data["frames"]
    print("len frames :", data.keys())
    # Initialize an empty list to store the concatenated vectors
    all_roi_vectors = []
    all_labels = []
    dico = {}
    print("threshold score : ", threshold_score)

    # Parcourir les frames et les détections
    for frame in frames:
        for detection in frame["detections"]:
            label = detection["class_id"]
            for roi in detection["rois"]:
                gradient = roi["gradient"]
                intensity = roi["intensity"]
                color = roi["color"]
                texture = roi["texture"]

                # Concaténer les caractéristiques en un seul vecteur
                roi_vector = gradient + intensity + color + texture

                # print("ROI vector:", roi_vector)
                # Append the ROI vector to the list
                all_roi_vectors.append(roi_vector)
                all_labels.append(label)
    # print("all ROI vector:", all_roi_vectors)
    print("all labels:", len(all_labels), len(all_roi_vectors))

    frame_number = 0  # Initialize the frame number
    # roi_info = {"feature_vec": [], "labels": []}
    for frame_number, frame in enumerate(frames):
        dico[frame_number] = {"rois": []}
        print('len frame detections:', len(frame["detections"]))

        for detection in frame["detections"]:
            label = detection["class_id"]

            for roi_number, roi in enumerate(detection["rois"]):
                # Create a new dictionary for each ROI
                roi_dict = {"roi_number": roi_number, "labels": label, "feature_vec": []}

                gradient = roi["gradient"]
                intensity = roi["intensity"]
                color = roi["color"]
                texture = roi["texture"]

                roi_vector = gradient + intensity + color + texture
                roi_dict["feature_vec"] = roi_vector

                dico[frame_number]["rois"].append(roi_dict)

    return dico
# In this example, dico is structured with a list of ROIs for each frame, where each ROI has its own dictionary containing feature vectors and labels. Adjust this structure based on your specific requirements.

# Function to flatten the feature vectors of all ROIs in a frame

def flatten_features(frame_dict):
    features = []
    for roi_dict in frame_dict["rois"]:
        features.append(roi_dict["feature_vec"])
    return np.array(features)

def flatten_labels(frame_dict):
    labels = []
    for roi_dict in frame_dict["rois"]:
        labels.append(roi_dict["labels"])
    return np.array(labels)

def flatten_rois(frame_dict):
    rois = []
    for roi_dict in frame_dict["rois"]:
        rois.append(roi_dict["roi_number"])
    return np.array(rois)


def create_graph(dico, path_to_construction):
    # Number of neighbors for k-nearest neighbors graph
    knn_param = 5
    # Iterate through frames
    for frame_number, frame_dict in dico.items():
        # Flatten the features of all ROIs in the frame
        features_matrix = flatten_features(frame_dict)
        labels = flatten_labels(frame_dict)
        rois = flatten_rois(frame_dict)
        print('len features :', len(features_matrix))
        if len(features_matrix)==0:
            print('frame_number', frame_number)
            print('frame_dict', frame_dict)
            pass
        else:
            
            # Build a k-nearest neighbors graph using kneighbors_graph
            A = kneighbors_graph(features_matrix, knn_param, mode='distance', include_self=True)
            A = A + A.transpose()
            inds_nonzero_A = A.nonzero()
            A[inds_nonzero_A] = 1
            print('Matrix A is:', A, 'A is of type : ', (A))
            np.savez_compressed(path_to_construction + '_frame_' + str(frame_number) + "_train_labeled_nodes_adjacency", A=A)
            
            # N = len(all_roi_vectors) 
            # print('N is of size:', N, 'size of label bin', len(labels))
            nbrs = NearestNeighbors(n_neighbors=knn_param, algorithm='ball_tree').fit(features_matrix)
            Dist, Indx = nbrs.kneighbors(features_matrix)

            sigma = np.mean(np.mean(Dist))

            filename_graph = path_to_construction + '_frame'+ str(frame_number) + '_train_graph.pkl'
            print(filename_graph)
            with open(filename_graph, 'wb') as f:
                # print((dico[frame]["feature_vec"]))
                # pickle.dump([dico[frame], dico[frame]["points"],dico[frame]["labels"], dico[frame]["scores"]], f)
                pickle.dump([rois, features_matrix, labels], f)

def run():
    path_to_construction = '/media/wprumm01/DISK_PhD_WP/Documents/GNN_Explainer/full_graph_construction/'

    isExist = os.path.exists(path_to_construction)
    if not isExist:
        os.mkdir(path_to_construction)

    node_file_path = '/media/wprumm01/DISK_PhD_WP/Documents/GNN_Explainer/training_node_features.json'
    data = get_node_data(node_file_path)

    labels = []
    points = []
    dico = create_dico(data)
    create_graph(dico, path_to_construction)
    # file.close()
run()

