import sys
import pyds
import ctypes
import numpy as np

def layer_finder(output_layer_info, name):
    """ Return the layer contained in output_layer_info which corresponds
        to the given name.
    """
    for layer in output_layer_info:
        # dataType == 0 <=> dataType == FLOAT
        # print(layer.layerName)
        if layer.dataType == 0 and layer.layerName == name:
            return layer
    return None


def clip(x):
    return min(max(0.0, x), 1.0)

def make_object(index, layers, default_classId = 1):
    """ Creates a NvDsInferObjectDetectionInfo object from one layer of SSD.
        Return None if the class Id is invalid, if the detection confidence
        is under the threshold or if the width/height of the bounding box is
        null/negative.
        Return the created NvDsInferObjectDetectionInfo object otherwise.
    """
    box_layer, score_layer = layers
    res = pyds.NvDsInferObjectDetectionInfo()
    res.detectionConfidence = score_layer[index]
    res.classId = default_classId

    rect_x1_f = box_layer[index][0]
    rect_y1_f = box_layer[index][1]
    rect_x2_f = box_layer[index][2]
    rect_y2_f = box_layer[index][3]
    res.left = clip(rect_x1_f)
    res.top = clip(rect_y1_f)
    res.width = clip(rect_x2_f - rect_x1_f)
    res.height = clip(rect_y2_f - rect_y1_f)

    return res

def nvds_infer_parse_scrfd(output_layer_info, input_size):
    """ Get data from output_layer_info and fill object_list
        num_detections: [1]
        nmsed_bboxes:   [200, 4]
        nmsed_scores:   [200]
        nmsed_classes:  [200]
        nmsed_landmarks:[200, 10]
    """
    num_detection_layer = output_layer_info[0]
    box_layer           = output_layer_info[1]
    score_layer         = output_layer_info[2]
    class_layer         = output_layer_info[3]
    landmark_layer      = output_layer_info[4]

    # if not num_detection_layer or not score_layer or not class_layer or not box_layer or not landmark_layer:
    #     sys.stderr.write("ERROR: some layers missing in output tensors\n")
    #     return []
    
    ptr = ctypes.cast(pyds.get_ptr(num_detection_layer.buffer), ctypes.POINTER(ctypes.c_int32))
    num_detection = np.ctypeslib.as_array(ptr, shape=(1,))[0]
    object_list = []
    landmark_list = []

    if num_detection > 0:
        ptr = ctypes.cast(pyds.get_ptr(box_layer.buffer), ctypes.POINTER(ctypes.c_float))
        box_result = np.ctypeslib.as_array(ptr, shape=(200,4))

        # Normalize 
        box_result = box_result.astype('float32')
        box_result[:, 0] /= input_size[0]
        box_result[:, 1] /= input_size[1]
        box_result[:, 2] /= input_size[0]
        box_result[:, 3] /= input_size[1]

        ptr = ctypes.cast(pyds.get_ptr(score_layer.buffer), ctypes.POINTER(ctypes.c_float))
        score_result = np.ctypeslib.as_array(ptr, shape=(200,))
        ptr = ctypes.cast(pyds.get_ptr(landmark_layer.buffer), ctypes.POINTER(ctypes.c_float))
        landmark_result = np.ctypeslib.as_array(ptr, shape=(200,10))
        x3_layers = box_result, score_result
        for i in range(num_detection):
            obj = make_object(i, x3_layers)
            if obj:
                object_list.append(obj)
                landmark_list.append(landmark_result[i])
    # print(landmark_list)
    return object_list, landmark_list