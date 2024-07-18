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

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def analyze_single_embedding(attentions, embedding_index, discard_ratio, head_fusion, device='cpu'):
    result = torch.eye(attentions[0].size(-1)).to(device)
    print(result.shape)
    print("Attentions", len(attentions))
    with torch.no_grad():
        for attention in attentions:
            attention = attention.to(device)
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"
            B = torch.zeros_like(attention_heads_fused)
            B[:, 0, embedding_index] = 1
            
            attention_to_embedding = attention_heads_fused * B
            #attention_to_embedding = attention_heads_fused[:, embedding_index, :]  

            I = torch.eye(attention_to_embedding.size(-1)).to(device)
            a = (attention_to_embedding + I) / 2
            a = a / a.sum(dim=-1, keepdim=True)

            flat = a.view(a.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * (1 - discard_ratio)), -1, False)
            flat[:, indices] = 0
            a = flat.view_as(a)

            result = torch.matmul(result, a)

    #mask = result[embedding_index, 1:]  
    mask = result[0, 0 , 1 :]
    width = int(np.sqrt(mask.size(0)))  
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)

    return mask

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
    print(y.shape)
    y = y.unsqueeze(-1).to(device)
    print(y.shape)
    print(y.min(), y.max())
    edge_index = edge_index.to(device)

    
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
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
    
    explanation = explainer(x, edge_index, index=11, target=y).to(device)
    # Get explanation for the node
    expl = explanation.get_explanation_subgraph().to(device)
    print("EXPL",expl.node_mask.shape)
    #utility.getFeaturesInformations(expl.node_mask, 768)
    vit, score = utility.getFeaturesNode(expl.node_mask, 768)
    #print(expl.node_mask)
    arr.append(expl.node_mask)
    scores.append(score.tolist())
    

    my_features['vit'].append(vit)
    
    path = 'explain/feature_importance.png'
    expl.visualize_feature_importance(path, top_k=50)

    path = 'explain/subgraph2.jpg'
    expl.visualize_graph(path, "graphviz")

    #print(scores)
    with open("output.txt", "w") as f:
        f.write(str(scores))
    return my_features

def main():
    input_path = "../datasets/GLitter/val/"
    #input_path = "./images/"
    model = GCN_graph3(args.ft, args.hidden, args.classes).to(device)

    load_model = torch.load('outputs_glitter/exp03/1GLITTER_DEF.pth')
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

                cv2.imwrite("vit_output/inpu_img.png", frame)
                b_box = obj_i['b_box']
                x, y, w, h = int(b_box[0]), int(b_box[1]), int(b_box[2]), int(b_box[3])
                
                cropped_region = frame
                if not args.fullImage:
                    cropped_region = frame[y:y + h, x:x + w]
                
                attentions = modelTransformer.extract_features_and_attentions(cropped_region)
                embedding_index = 40 
                discard_ratio = 0.9 
            
                mask = analyze_single_embedding(attentions, embedding_index, discard_ratio, "max", device)
                
                np_img = np.array(cropped_region)[:, :, ::-1]
                mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
                mask = show_mask_on_image(np_img, mask)

                cv2.imwrite("vit_output/input.png", np_img)
                cv2.imwrite("vit_output/outputted.png", mask)
                

if __name__ == "__main__":
    main()

