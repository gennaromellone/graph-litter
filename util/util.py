import cv2
import numpy as np

from skimage.feature import graycomatrix, graycoprops

classes = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]
float_formatter = "{:.2f}".format

def extractFrames(pred):
    boxes = pred[:, :4]  # Estrai le coordinate x, y, w, h
    confidences = pred[:, 4]
    obj = []
    for box, confidence, class_id in zip(boxes, confidences, pred[:, 5]):
        x, y, w, h = box.tolist()
        x, y, w, h = int(x), int(y), int(w), int(h)
        score = float_formatter(confidence.item())
        
        if float(score) >= 0.6:
            obj.append({
                'class_id': int(class_id),
                'class': classes[int(class_id)],
                'score': score,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })
    return obj

def sobel_gradient(image):
    # Applica il filtro di Sobel per il gradiente lungo l'asse x
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

    # Applica il filtro di Sobel per il gradiente lungo l'asse y
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calcola il modulo del gradiente
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Calcola la direzione del gradiente
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    return gradient_magnitude.flatten(), gradient_direction.flatten()

def histogramGradient(image, hist_size):
    hist = cv2.calcHist([image], [0], None, [hist_size], [0, 256])
    hist = hist.flatten()
    gradient = np.gradient(hist)

    return gradient.tolist()

def histogramIntensity(image, hist_size):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten()

    # Calcola la media dei livelli di intensità
    intensity_mean = np.sum(hist * np.arange(256)) / np.sum(hist)

    # Calcola la deviazione standard dei livelli di intensità
    intensity_stddev = np.sqrt(np.sum(hist * (np.arange(256) - intensity_mean) ** 2) / np.sum(hist))

    # Trova il valore massimo e il valore minimo dei livelli di intensità
    intensity_max = np.max(np.arange(256)[hist > 0])
    intensity_min = np.min(np.arange(256)[hist > 0])

    arr = np.zeros(hist_size)
    arr[0] = intensity_min
    arr[1] = intensity_max
    arr[2] = intensity_mean
    arr[3] = intensity_stddev

    return arr.tolist()

def histogramColor(image, hist_size):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calcola l'istogramma dei colori (ad esempio, per il canale rosso)
    hist = cv2.calcHist([image_rgb], [0], None, [hist_size], [0, 256])
    hist = hist.flatten()

    return hist.tolist()
import time 

def histogramCSLTP(image, hist_size):

    def sigmoid(x):
        if x > 3:
            return 2
        elif x < -3:
            return 1
        else:
            return 0 
    
    
    image_height = image.shape[0]
    image_width = image.shape[1]

    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    zeroHorizontal = np.zeros(image_width + 2).reshape(1, image_width + 2)
    zeroVertical = np.zeros(image_height).reshape(image_height, 1)

    img_grey = np.concatenate((img_grey, zeroVertical), axis = 1)
    img_grey = np.concatenate((zeroVertical, img_grey), axis = 1)
    img_grey = np.concatenate((zeroHorizontal, img_grey), axis = 0)
    img_grey = np.concatenate((img_grey, zeroHorizontal), axis = 0)

    pattern_img = np.zeros((image_height + 1, image_width + 1))
    
    
    for x in range(1, image_height -2):
        for y in range(1, image_width -2):
            
            s1 = sigmoid(img_grey[x-2, y-2] - img_grey[x+2, y+2])
            s3 = sigmoid(img_grey[x-2, y+2] - img_grey[x+2, y-2])*3
    
            s = s1 + s3
        
            pattern_img[x, y] = s
    start = time.time()
    pattern_img = pattern_img[1:(image_height+1), 1:(image_width+1)].astype(int)
    
    histogram = np.histogram(pattern_img, bins = np.arange(hist_size +1))[0]
    histogram = histogram.reshape(1, -1)
    
    print("Time elapsed:", time.time() - start)

    return histogram[0].tolist()
    
def histogramGLCM(image, hist_size):
    #props: {‘contrast’, ‘dissimilarity’, ‘homogeneity’, ‘energy’, ‘correlation’, ‘ASM’}
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(img_grey, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
    
    arr = np.zeros(hist_size)
    arr[0] = graycoprops(glcm, 'contrast')[0, 0]
    arr[1] = graycoprops(glcm, 'dissimilarity')[0, 0]
    arr[2] = graycoprops(glcm, 'homogeneity')[0, 0]
    arr[3] = graycoprops(glcm, 'energy')[0, 0]
    arr[4] = graycoprops(glcm, 'correlation')[0, 0]
    arr[5] = graycoprops(glcm, 'ASM')[0, 0]
    
    #print(arr.tolist())
    return arr.tolist()
