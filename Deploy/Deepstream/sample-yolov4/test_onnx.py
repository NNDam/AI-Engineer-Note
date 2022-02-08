import cv2
import numpy as np
from exec_backends.trt_backend import TrtModel


def preprocess(img, input_size = (416, 416)):
    resized_img = cv2.resize(img, (input_size[1], input_size[0]))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    resized_img = np.expand_dims(resized_img, 0)
    resized_img = resized_img.astype('float32') / 255.0
    resized_img = np.transpose(resized_img, (0, 3, 1, 2))
    return resized_img

def visualize(img, bboxes):
    height, width, _ = img.shape
    bboxes[:, 0] *= width
    bboxes[:, 1] *= height
    bboxes[:, 2] *= width
    bboxes[:, 3] *= height
    for x1, y1, x2, y2 in bboxes:
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img

if __name__ == '__main__':
    model_path = 'weights/model-1x3x416x416-fp16.engine'
    img_path = 'test_images/test.png'

    model = TrtModel(model_path, max_size = 416)
    img = cv2.imread(img_path)
    batch = preprocess(img)

    num_detections, bboxes, confs, classes = model.run(batch)
    print(num_detections.shape, bboxes.shape, confs.shape, classes.shape)
    bboxes  = bboxes[0][:num_detections[0][0]]
    confs   = confs[0][:num_detections[0][0]]
    classes = classes[0][:num_detections[0][0]]
    print(bboxes)
    vis = visualize(img.copy(), bboxes)
    cv2.imshow('vis.jpg', vis)
    cv2.waitKey(0)
