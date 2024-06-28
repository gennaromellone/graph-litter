'''
This script works with a dataset of images. It search for two folders 'train' and 'test' and extract the bounding boxes 
and then extracts ROIs using SLIC method and for each ROI extract features in order to build a GCN.

Written by Gennaro Mellone

v.2.0.0
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
import os
from tqdm import tqdm

from models.models import ImageSimilarityModel
from ultralytics import YOLO


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

def segmentImage(image, clusters=3, useMask=True):
    if useMask:
        lumSLIC = color.rgb2gray(image)
        maskSLIC = morphology.remove_small_holes(
            morphology.remove_small_objects(
                        lumSLIC < 0.7, 500),
                        500)
                #lumSLIC < 0.55, 64),
                #70)
        
        try:
            # Using maskSLIC
            segments = segmentation.slic(image, n_segments=clusters, mask=maskSLIC, start_label=1)

        except:
            # Otherwise use only SLIC
            print("Mask not working!")
            segments = segmentation.slic(image, n_segments=clusters, start_label=1)
    else:
        segments = segmentation.slic(image, n_segments=clusters, start_label=1)
    
    return segments

def segmentsFeaturesExtraction(cropped_region, segments, segment_classID, idxStart = 0):
    idx = idxStart
    rois = []
    for segment_id in np.unique(segments):
        # TODO Check if for each ROI it considers also black background
        mask = (segments == segment_id)
        #roi_image = cv2.bitwise_and(cropped_region, cropped_region, mask=mask)
        roi = np.zeros_like(cropped_region)
        roi[mask] = cropped_region[mask]

        # Features Extraction
        if args.features == "vit":
            print("VIT features NOT IMPLEMENTED FOR YOLO!")
            pass
            # Extract ViT embeddings
            #with torch.no_grad():
            #    extractedFeatures = modelTransformer.extract_embeddings(roi)

        elif args.features == "standard":
            # Extract standard features (color, gradient, intensity, texture)
            gradientFt = features.histogramGradientNormStd(roi, args.histSize)
            intensityFt= features.histogramIntensityNormStd(roi, args.histSize)
            colorFt = features.histogramColorNormStd(roi, args.histSize)
            textureFt = features.histogramGLCMNormStd(roi, args.histSize)

            extractedFeatures = gradientFt + intensityFt + colorFt + textureFt

        roi_dict = {"roi_number": idx, "labels": segment_classID, "feature_vec": []}
        roi_dict["feature_vec"] = extractedFeatures
        rois.append(roi_dict)

        idx += 1

    return rois

def run(args, isTrain=False):
    yoloModel = YOLO("yolov8n-seg.pt")
    notFound = 0
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
                #if id_x != 251:
                #    continue
                classCounter[int(obj_i['id'])-1] += 1
                #frame = cv2.imread(path)

                #print(classCounter)
                # YOLO phase
                # Take only the first element
                res = yoloModel(path)
                

                r = res[0]
                if len(r) == 0:
                    print("No masks found with YOLO!")
                    notFound += 1
                    continue
                
                frame = np.copy(r.orig_img)
                # Iterate on R to get multiple detection, for now just 1
                c = r[0]
                
                # Create binary mask
                b_mask = np.zeros(frame.shape[:2], np.uint8)

                #  Extract contour result
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                
                # Create 3-channel mask
                mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
                isolated = cv2.bitwise_and(mask3ch, frame)

                x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
                # Crop image to object region
                isolated_object = isolated[y1:y2, x1:x2]

                # Create the background mask (inverse of the object mask)
                background_mask = cv2.bitwise_not(mask3ch)

                # Extract the background
                background = cv2.bitwise_and(frame, background_mask)

                
                #frame = utility.resizeImage(frame, 800, 600)            
                # Extract ROIs
                object_segments = segmentImage(isolated_object, clusters=args.clusters)

                background_segments = segmentImage(background, clusters=args.clusters, useMask=False)

                #utility.exportROIimage(cropped_region, segments, 'roi2.png')
                
                full_rois = []
                
                #utility.exportROIimage(background, background_segments, 'roi_background.png')
                
                
                #Iterate on each ROI
                obj_id = int(obj_i['id']) - 1
                
                if args.useBackground:
                    obj_id += 1
                    # CANCEL MASK FROM THE IMAGE
                    full_rois += segmentsFeaturesExtraction(background, background_segments, 0)
                    
                full_rois += segmentsFeaturesExtraction(isolated_object, object_segments, obj_id, len(full_rois))
                

                # To uniform Batch Size == N_CLUSTERS:
                # Use a fixed size of ROIs (the last slots will be Background with 0)
                if args.uniform:
                    full_rois = utility.fillEmptySlots(full_rois, args.clusters)
                with open('graph_bg.txt', 'w') as f:
                    for line in full_rois:
                        f.write(f"Features: {len(line['feature_vec'])}\n{line}\n")
                exit()
                # Export Graphs for each ROI
                gcn.exportGraph(full_rois, args.output, isTrain, knn_param=args.knnParams, img_number=id_x)
    return classCounter, notFound


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

    
    print("Working on Train set")
    trainCount, trainNot = run(args, isTrain=True)

    print("Working on Test set")
    testCount, testNot = run(args, isTrain=False)

    with open(args.output + "info.txt", 'w') as log:
        log.write(f'Main informations: {str(args)}\n Train images per class counter: {str(trainCount)}\n Train images not found: {str(trainNot)}\n Test images per class counter: {str(testCount)}\n Test images not found: {str(testNot)}')

    