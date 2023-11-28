import cv2
import torch
from util import util
from skimage import segmentation
from skimage import color
from skimage import morphology
import numpy as np

import json
import matplotlib.pyplot as plt
import time

output_video = True
DETECTION_THRESHOLD = 0.75
HISTOGRAM_SIZE = 10
#from optimization import histogramCSLTP_fast

model = torch.hub.load('ultralytics/yolov5', 'custom', 'birds_yolox_20cls.pt')
model.eval()

video_path = 'video/birds_1.mp4'
cap = cv2.VideoCapture(video_path)

if output_video:
    output_path = 'video_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

objects = []
idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    start = time.time()
    results = model(frame) 

    # Do prediction
    pred = results.pred[0]

    boxes = pred[:, :4] 
    confidences = pred[:, 4]
    obj = []

    # Loop on detections
    for box, confidence, class_id in zip(boxes, confidences, pred[:, 5]):
        # Extract the score
        score = util.float_formatter(confidence.item())

        # Consider only detections greater than a Threshold 
        if float(score) >= DETECTION_THRESHOLD:
            x, y, w, h = box.tolist()
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            cropped_region = frame[y:y + h, x:x + w]
            lum = color.rgb2gray(cropped_region)
            
            # ROI
            
            mask = morphology.remove_small_holes(
                morphology.remove_small_objects(
                    lum < 0.7, 500),
                500)
            #slic = segmentation.slic(cropped_region, n_segments=50, start_label=1)
            segments = segmentation.slic(cropped_region, n_segments=50, mask=mask, start_label=1)

            #segmented_image = color.label2rgb(segments, image=cropped_region)
            #cv2.imshow('FullImage', cropped_region)
            
            rois = []
            #Iterate on each ROI
            for segment_id in np.unique(segments):
                mask = (segments == segment_id)
                roi = np.zeros_like(cropped_region)
                roi[mask] = cropped_region[mask]

                rois.append({
                    "gradient": util.histogramGradient(roi, HISTOGRAM_SIZE),
                    "intensity": util.histogramIntensity(roi,HISTOGRAM_SIZE),
                    "color": util.histogramColor(roi, HISTOGRAM_SIZE),
                    #"texture": histogramCSLTP_fast(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), HISTOGRAM_SIZE)
                    "texture": util.histogramGLCM(roi, HISTOGRAM_SIZE)
                })

            obj.append({
                "class_id": int(class_id),
                "class": util.bird_classes[int(class_id)],
                "score": score,
                "rois": rois
            })
    if output_video:
        output_frame = results.render()[0]
        out.write(output_frame)

    if obj:
        pass
        # PASS obj TO GRAPH NEURAL NETWORK!!
        #cv2.imshow('YOLOv5 Object Detection', output_frame)
    objects.append({
        "frame": str(idx),
        "detections":obj
    })
    idx += 1
    print("External Time:", time.time()-start)
        
    
dictionary = {
    "thresholdScore": DETECTION_THRESHOLD,
    "frames":objects
}
json_object = json.dumps(dictionary, indent=2)
with open("birds1_training_1.json", "w") as outfile:
    outfile.write(json_object)
cap.release()
out.release()
