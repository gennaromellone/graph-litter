'''
This script works with a dataset of images. It search for two folders 'train' and 'test' and extract the bounding boxes 
and then extracts ROIs using SLIC method and for each ROI extract features in order to build a GCN.

Written by Gennaro Mellone

v.1.0.0
'''

import cv2
import torch
from utility import utility, gcn, features
from skimage import segmentation
from skimage import color
from skimage import morphology
import numpy as np
import argparse

import json
import time
import os
from tqdm import tqdm

from models.models import ImageSimilarityModel


# Extract the data from the .txt file (class id and Bounding Boxes)
# Each .jpg image should have a .txt file with annotations class_id, x, y, w, h per object per line.
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

def run(args, isTrain=False):
    
    if isTrain:
        input_path = args.dataset + "/train/"
        maxImages = args.maxImages
    else:
        splitSize = 0.3
        input_path = args.dataset + "/val/"
        maxImages = int(splitSize * args.maxImages)

    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.lower().endswith('.jpg')]
    files_sorted = sorted(files)

    utility.check_and_create_folders(args.output)

    # Initialize model for ViT
    if args.features == "vit":
        modelTransformer = ImageSimilarityModel(device='cuda')
    
    id_x = 0
    classCounter = np.zeros(args.classes)
    
    # Iterates on files in the folder input_path
    for file in tqdm(files_sorted, desc='Extracting ROIs and Features', unit='img'):
        if np.all(classCounter == maxImages):
            break
        
        # Read image informations (Bounding Boxes)
        path = os.path.join(input_path, file)
        info_file = path.replace('.jpg','.txt')
        objects = getInformations(info_file)

        # Iterates on objects in image
        for obj_i in objects:
            
            # Only consider the first N classes
            if int(obj_i['id']) <= args.classes and classCounter[int(obj_i['id'])-1] < maxImages:
                id_x += 1
                classCounter[int(obj_i['id'])-1] += 1
                print(classCounter)
                frame = cv2.imread(path)
                
                b_box = obj_i['b_box']
                x, y, w, h = int(b_box[0]), int(b_box[1]), int(b_box[2]), int(b_box[3])
                
                cropped_region = frame
                if not args.fullImage:
                    cropped_region = frame[y:y + h, x:x + w]

                # Extract ROIs
                lumSLIC = color.rgb2gray(cropped_region)
                maskSLIC = morphology.remove_small_holes(
                    morphology.remove_small_objects(
                        lumSLIC < 0.55, 64),
                    70)
                
                try:
                    # Using maskSLIC
                    segments = segmentation.slic(cropped_region, n_segments=args.clusters, mask=maskSLIC, start_label=1)

                except:
                    # Otherwise use only SLIC
                    print("Mask not working!")
                    segments = segmentation.slic(cropped_region, n_segments=args.clusters, start_label=1)
                
                rois = []
                if args.useBackground:
                    # Get body points (x,y) from the JSON file
                    points = cub_dict[file]

                    # Filter only the ROIs with body points
                    selected_regions = []
                    for point in points:
                        if point[0] != 0 and point[1] != 0:
                            x, y = point[0], point[1]
                            region_number = segments[y, x]
                            selected_regions.append(region_number)
                    
                    # Updates segments with body parts only, the rest will be 0
                    #* NOW CLASS NUMBER WILL BE WITH ONE MORE CLASS !!! *#
                    updated_segments = np.full_like(segments, -1)
                    for region_number in selected_regions:
                        updated_segments[segments == region_number] = region_number

                    segments = updated_segments
                
                #Iterate on each ROI
                roiIdx = 0
                for segment_id in np.unique(segments):
                
                    obj_id = int(obj_i['id']) - 1
                    if args.useBackground:
                        obj_id += 1
                    
                    # TODO Check if for each ROI it considers also black background
                    mask = (segments == segment_id)
                    #roi_image = cv2.bitwise_and(cropped_region, cropped_region, mask=mask)
                    roi = np.zeros_like(cropped_region)
                    roi[mask] = cropped_region[mask]
                    
                    if args.useBackground:
                        segment_classID = 0

                        if segment_id in updated_segments:
                            segment_classID = obj_id
                    else:
                        segment_classID = obj_id

                    # Features Extraction
                    if args.features == "vit":
                        # Extract ViT embeddings
                        with torch.no_grad():
                            extractedFeatures = modelTransformer.extract_embeddings(roi)

                    elif args.features == "standard":
                        # Extract standard features (color, gradient, intensity, texture)
                        gradientFt = features.histogramGradientNormStd(roi, args.histSize)
                        intensityFt= features.histogramIntensityNormStd(roi, args.histSize)
                        colorFt = features.histogramColorNormStd(roi, args.histSize)
                        textureFt = features.histogramGLCMNormStd(roi, args.histSize)

                        extractedFeatures = gradientFt + intensityFt + colorFt + textureFt

                    roi_dict = {"roi_number": roiIdx, "labels": segment_classID, "feature_vec": []}
                    roi_dict["feature_vec"] = extractedFeatures
                    rois.append(roi_dict)

                # To uniform Batch Size == N_CLUSTERS:
                # Use a fixed size of ROIs (the last slots will be Background with 0)
                if args.uniform:
                    rois = utility.fillEmptySlots(rois, args.clusters)
                
                # Export Graphs for each ROI
                gcn.createGraph(rois, args.output, isTrain, knn_param=args.knnParams, img_number=id_x)
    return classCounter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='../coco-dataset/', 
                    help='Main Dataset Folder')
    parser.add_argument('--output', type=str, default='graphs', 
                    help='Output folder where build GCN')
    parser.add_argument('--maxImages', type=int, default=200, 
                    help='Max number of images to extract per each class')
    parser.add_argument('--knnParams', type=int, default=3, 
                    help='K-NN parameter')
    parser.add_argument('--histSize', type=int, default=10, 
                    help='Histogram Size')
    parser.add_argument('--classes', type=int, default=10, 
                    help='Number of Classes')
    parser.add_argument('--clusters', type=int, default=10, 
                    help='Number of Clusters')
    parser.add_argument('--features', type=str, choices=['vit', 'standard'], 
                    help='Select Features to be extracted. "vit"(vision transformer) or "standard"(color,intensity,texture,gradient)')
    
    parser.add_argument('--fullImage', action='store_true', default=False,
                    help='Use Full Image or Bounding Boxes')
    parser.add_argument('--uniform', action='store_true', default=False,
                    help='Uniform Batch size for all ROIs')
    parser.add_argument('--useBackground', action='store_true', default=False,
                    help='Use background (Not working!)')
    args = parser.parse_args()
    print(args)
    
    cub_dict = ""
    # Open the JSON schema for the birds' body parts
    if args.useBackground:
        with open("utility/CUB_parts.json", 'r') as file:
            cub_dict = json.load(file)
    print("Working on Train set")
    trainCount = run(args, isTrain=True)

    print("Working on Test set")
    testCount = run(args, isTrain=False)

    with open(args.output + "info.txt", 'w') as log:
        log.write(f'Main informations: {str(args)}\n Train images per class counter: {str(trainCount)}\n Test images per class counter: {str(testCount)}')

    
