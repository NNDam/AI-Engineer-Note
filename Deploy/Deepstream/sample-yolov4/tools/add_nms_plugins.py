#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#!/usr/bin/env python3
import onnx_graphsurgeon as gs
import argparse
import onnx
import numpy as np

def create_and_add_plugin_node(graph, topK, keepTopK):
    
    batch_size = graph.inputs[0].shape[0]
    input_h = graph.inputs[0].shape[2]
    input_w = graph.inputs[0].shape[3]
    print('batch_size', batch_size)

    tensors = graph.tensors()
    boxes_tensor = tensors["boxes"]
    confs_tensor = tensors["confs"]

    num_detections = gs.Variable(name="num_detections").to_variable(dtype=np.int32, shape=[-1, 1])
    nmsed_boxes = gs.Variable(name="nmsed_boxes").to_variable(dtype=np.float32, shape=[-1, keepTopK, 4])
    nmsed_scores = gs.Variable(name="nmsed_scores").to_variable(dtype=np.float32, shape=[-1, keepTopK])
    nmsed_classes = gs.Variable(name="nmsed_classes").to_variable(dtype=np.float32, shape=[-1, keepTopK])

    new_outputs = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]

    mns_node = gs.Node(
        op="BatchedNMSDynamic_TRT",
        attrs=create_attrs(input_h, input_w, topK, keepTopK),
        inputs=[boxes_tensor, confs_tensor],
        outputs=new_outputs)

    graph.nodes.append(mns_node)
    graph.outputs = new_outputs

    return graph.cleanup().toposort()




def create_attrs(input_h, input_w, topK, keepTopK):

    num_anchors = 3

    h1 = input_h // 8
    h2 = input_h // 16
    h3 = input_h // 32

    w1 = input_w // 8
    w2 = input_w // 16
    w3 = input_w // 32

    num_boxes = num_anchors * (h1 * w1 + h2 * w2 + h3 * w3)

    attrs = {}

    attrs["shareLocation"] = 1
    attrs["backgroundLabelId"] = -1
    attrs["numClasses"] = 80
    attrs["topK"] = topK
    attrs["keepTopK"] = keepTopK
    attrs["scoreThreshold"] = 0.4
    attrs["iouThreshold"] = 0.6
    attrs["isNormalized"] = 1
    attrs["clipBoxes"] = 1

    # 001 is the default plugin version the parser will search for, and therefore can be omitted,
    # but we include it here for illustrative purposes.
    attrs["plugin_version"] = "1"

    return attrs


def main():
    parser = argparse.ArgumentParser(description="Add batchedNMSPlugin")
    parser.add_argument("-f", "--model", help="Path to the ONNX model generated by export_model.py", default="yolov4_1_3_416_416.onnx")
    parser.add_argument("-t", "--topK", help="number of bounding boxes for nms", default=2000)
    parser.add_argument("-k", "--keepTopK", help="bounding boxes to be kept per image", default=1000)

    args, _ = parser.parse_known_args()

    graph = gs.import_onnx(onnx.load(args.model))
    
    graph = create_and_add_plugin_node(graph, int(args.topK), int(args.keepTopK))
    
    onnx.save(gs.export_onnx(graph), args.model + ".nms.onnx")


if __name__ == '__main__':
    main()
