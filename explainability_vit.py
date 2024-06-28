#from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain import Explainer, GNNExplainer
import torch.nn.functional as F
from torch_geometric.explain import unfaithfulness, fidelity, groundtruth_metrics

# Load and save variables
import pickle
import torch
from tqdm import tqdm
import cv2
from skimage import segmentation
from skimage import color
from skimage import morphology
from models.models import GCN_hidden, GCN_graph2, GCN_graph3
import argparse
import numpy as np

from torch_geometric.data import Data
import os
from models.models import ImageSimilarityModel

from utility import utility, gcn, features

def getInformations(file_path):
    objects = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                values = line.strip().split()
                
                # Check if there are at least 5 values in the line
                if len(values) >= 5:
                    class_id = values[0]
                    x_min, y_min, x_max, y_max = map(float, values[1:5])
                    objects.append({'id':class_id,'b_box':[x_min, y_min, x_max, y_max]})
                    #print(f'Class ID: {class_id}, Bounding Box: ({x_min}, {y_min}, {x_max}, {y_max})')
                else:
                    print(f'Invalid line: {line}')

    except FileNotFoundError:
        print(f'File not found: {file_path}')

    return objects


# Arguments to run the experiment
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.') #seed 42
parser.add_argument('--hidden', type=int, default=16, #128
                    help='Number of hidden units.')
parser.add_argument('--classes', type=int, default=10, #128
                    help='Number of classes')
parser.add_argument('--ft', type=int, default=768, #128
                    help='Number of features')
parser.add_argument('--features', type=str, default='vit', #128
                    help='Name of features')
parser.add_argument('--clusters', type=int, default=10, #128
                    help='N. of clusters for segmentation')
parser.add_argument('--histSize', type=int, default=10, #128
                    help='Histogram size (for default features)')
parser.add_argument('--fullImage', action='store_true', default=False,
                    help='Use Full Image or Bounding Boxes')

args = parser.parse_args()
print(args)

args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device("cpu")

device = 'cpu'
def load_graph(graph_file, edge_file):
    with open(graph_file, 'rb') as f:
        features_pickle = pickle.load(f)

    loaded_data = np.load(edge_file)
    
    row, col = loaded_data['row'], loaded_data['col']
    print(len(features_pickle[1]))
    edge_coordinates = np.array([row, col])
    edge_coordinates = torch.tensor(edge_coordinates, dtype=torch.long)

    edges_test = edge_coordinates.view(2, -1).long()
    #edges_normalized = edges_test / edges_test.max().item()
    #edges_normalized = edges_normalized.long()
    y =torch.tensor(features_pickle[2])
    
    #y = y.squeeze()

    features = torch.tensor(features_pickle[1])
    
    _data_test = Data(x=features.float(), edge_index=edges_test, y=y)

    return _data_test, len(features_pickle[0])

@torch.no_grad()
def test(model):
    
    _data_test = _data_test.to(device)
    outputs = model(_data_test)

    predictions = outputs.max(1)[1]

    return outputs, predictions

def my_explainer(model, data, nodes):
    my_features = {'vit':[],'intensity':[],'color':[],'texture':[]} 
    x, edge_index, y = data.x, data.edge_index, data.y
    
    x = x.to(device)
    y = y.unsqueeze(-1).to(device)
    edge_index = edge_index.to(device)

    coeffs = {
        #'edge_size': 0.005,
        'edge_size': 0.9,
        'edge_reduction': 'mean',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 0.9,
        'node_feat_ent': 0.9,
        'EPS': 1e-15,
    }

    
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200, kwargs=coeffs),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='probs',
            ),
        )
 
    S = np.zeros(nodes)
    scores = []
    arr = []
    # Loop on all nodes of the graph
    print("Y",y.shape)
    print("X",x.shape)
    print("edge_index",edge_index.shape)
    explanation = explainer(x, edge_index, index=None, target=y).to(device)
    # Get explanation for the node
    expl = explanation.get_explanation_subgraph().to(device)
    print(expl.node_mask.shape)
    #utility.getFeaturesInformations(expl.node_mask, 768)
    vit, score = utility.getFeaturesNode(expl.node_mask, 768)
    #print(expl.node_mask)
    arr.append(expl.node_mask)
    scores.append(score.tolist())
    

    my_features['vit'].append(vit)
    
    path = 'explain/feature_importance.png'
    expl.visualize_feature_importance(path, top_k=50)

    #path = 'explain/subgraph_explaination.jpg'
    #explanation.visualize_graph(path, "graphviz")

    path = 'explain/subgraph_expl.jpg'
    expl.visualize_graph(path, "graphviz")

    #print(scores)
    with open("output.txt", "w") as f:
        f.write(str(scores))
    return my_features

def main():
    input_path = "../datasets/GLitter/val/"
    #input_path = "./images/"
    model = GCN_graph3(args.ft, args.hidden, args.classes).to(device)

    load_model = torch.load('outputs_vit_glitter/exp01/1GLITTER_VIT.pth')
    model.load_state_dict(load_model['model_state_dict'])
    
    # Initialize model for ViT
    if args.features == "vit":
        modelTransformer = ImageSimilarityModel(device='cuda')
    id_x = 0
    # START
    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.lower().endswith('.jpg')]
    files_sorted = sorted(files)
    for file in tqdm(files_sorted, desc='Extracting ROIs and Features', unit='img'):
        
        
        # Read image informations (Bounding Boxes)
        path = os.path.join(input_path, file)
        info_file = path.replace('.jpg','.txt')
        objects = getInformations(info_file)

        # Iterates on objects in image
        for obj_i in objects:
            
            # Only consider the first N classes
            if int(obj_i['id']) <= 10:
                id_x += 1
                if id_x != 250:
                    continue
                
                print(file)
                frame = cv2.imread(path)
                frame = utility.resizeImage(frame, 800, 600)

                
                b_box = obj_i['b_box']
                x, y, w, h = int(b_box[0]), int(b_box[1]), int(b_box[2]), int(b_box[3])
                
                cropped_region = frame
                if not args.fullImage:
                    cropped_region = frame[y:y + h, x:x + w]

                # Get Class ID for the entire object
                obj_id = int(obj_i['id']) - 1
                

                # Extract ROIs
                utility.drawPatches(cropped_region, 'vit_rois.png')
                rois = []
                extractedFeaturesList = modelTransformer.extractEmbeddings(cropped_region, mainToken=True)
                for roiIdx, extractedFeature in enumerate(extractedFeaturesList):
                    roi_dict = {"roi_number": roiIdx, "labels": obj_id, "feature_vec": []}
                    roi_dict["feature_vec"] = extractedFeature
                    rois.append(roi_dict)
              
                # Export Graphs for each ROI
                graph_features, row, col = gcn.createGraph(rois, knn_param=2)
                edge_coordinates = np.array([row, col])
                edge_coordinates = torch.tensor(edge_coordinates, dtype=torch.long)

                edges_test = edge_coordinates.view(2, -1).long()
                y =torch.tensor(graph_features[2])
                #print(y.size())
                #print(len(graph_features[0]))
    
                _features = torch.tensor(graph_features[1])
                nodes = len(graph_features[0])
                print(nodes)
                data = Data(x=_features.float(), edge_index=edges_test, y=y).to(device)

                outputs = model(data.x, data.edge_index)
                output = torch.argmax(outputs)
                print("Predicted class: ",utility.litter_classes[output.item()])
                print(outputs)
                
                
                _features = my_explainer(model, data.to(device), nodes)
                #f = [0,0,0,1,0,0,0]
                #utility.heatmapROIimage(cropped_region, segments, f, 1, 'vit')
                #utility.exportROIimage(cropped_region, segments, 'roi_new.png')
                #utility.heatmapROIimage(cropped_region, segments, _features['vit'], 1, 'vit')
                
                #print(vettore_normalizzato)
                #print(graph_features)
                

if __name__ == "__main__":
    main()