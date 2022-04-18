/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"
#include "gstnvdsmeta.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))
#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)

static bool dict_ready=false;
static const int NUM_CLASSES_YOLO = 80;
std::vector<std::string> dict_table;

void *set_metadata_ptr(std::array<float, 10> & arr)
{
    gfloat *user_metadata = (gfloat*)g_malloc0(10*sizeof(gfloat));

    for(int i = 0; i < 10; i++) {
       user_metadata[i] = arr[i];
    }
    return (void *)user_metadata;
}

float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

struct ObjectPoint{
   float ctx;
   float cty;
   float width;
   float height;
   float confidence;
   int classId;
};

extern "C" bool NvDsInferParseCustomYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloV4TLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList);

static NvDsInferParseObjectInfo convertBBoxFaceDetection(const float& bx1, const float& by1, const float& bx2,
                                     const float& by2, const uint& netW, const uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution

    float x1 = clamp(bx1, 0, netW);
    float y1 = clamp(by1, 0, netH);
    float x2 = clamp(bx2, 0, netW);
    float y2 = clamp(by2, 0, netH);

    b.left = x1;
    b.width = clamp(x2 - x1, 0, netW);
    b.top = y1;
    b.height = clamp(y2 - y1, 0, netH);
    std::cout << " left " << b.left << " width " << b.width << " top " << b.top << " height " << b.height << std::endl;
    return b;
}

/* YOLOv4 implementations */
static NvDsInferParseObjectInfo convertBBoxYoloV4(const float& bx1, const float& by1, const float& bx2,
                                     const float& by2, const uint& netW, const uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution

    float x1 = bx1 * netW;
    float y1 = by1 * netH;
    float x2 = bx2 * netW;
    float y2 = by2 * netH;

    x1 = clamp(x1, 0, netW);
    y1 = clamp(y1, 0, netH);
    x2 = clamp(x2, 0, netW);
    y2 = clamp(y2, 0, netH);

    b.left = x1;
    b.width = clamp(x2 - x1, 0, netW);
    b.top = y1;
    b.height = clamp(y2 - y1, 0, netH);
    std::cout << " left " << b.left << " width " << b.width << " top " << b.top << " height " << b.height << std::endl;
    return b;
}

static void addBBoxProposalFaceDetection(const float bx, const float by, const float bw, const float bh,
                     const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, 
                     std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBoxFaceDetection(bx, by, bw, bh, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    // bbi.landmarks = face_landmarks;
    binfo.push_back(bbi);
}


static void addBBoxProposalYoloV4(const float bx, const float by, const float bw, const float bh,
                     const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBoxYoloV4(bx, by, bw, bh, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo> decodeFaceDetectionTensor(
    const float* boxes, const float* scores, const float* classes, const float* landmarks,
    const uint num_bboxes, NvDsInferParseDetectionParams const& detectionParams,
    const uint& netW, const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;
    std::vector<std::array<float, 10>> blmk;

    NvDsUserMetaList *obj_user_meta_list = NULL;

    std::cout << "go here " << std::endl;
    uint bbox_location = 0;
    uint score_location = 0;
    uint lmk_location = 0;
    for (uint b = 0; b < num_bboxes; ++b)
    {
        float bx1 = boxes[bbox_location];
        float by1 = boxes[bbox_location + 1];
        float bx2 = boxes[bbox_location + 2];
        float by2 = boxes[bbox_location + 3];
        float maxProb = scores[score_location];
        std::array<float, 10> face_landmarks;
        face_landmarks[0] = landmarks[lmk_location + 0];
        face_landmarks[1] = landmarks[lmk_location + 1];
        face_landmarks[2] = landmarks[lmk_location + 2];
        face_landmarks[3] = landmarks[lmk_location + 3];
        face_landmarks[4] = landmarks[lmk_location + 4];
        face_landmarks[5] = landmarks[lmk_location + 5];
        face_landmarks[6] = landmarks[lmk_location + 6];
        face_landmarks[7] = landmarks[lmk_location + 7];
        face_landmarks[8] = landmarks[lmk_location + 8];
        face_landmarks[9] = landmarks[lmk_location + 9];
        // int maxIndex = (int) classes[score_location];
        // Only have face ID = 0
        int maxIndex = 0;

        if (maxProb > detectionParams.perClassPreclusterThreshold[maxIndex])
        {
            addBBoxProposalFaceDetection(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
        }
        blmk.push_back(face_landmarks);
        bbox_location += 4;
        score_location += 1;
        lmk_location += 10;
    }
    assert( binfo.size() == blmk.size());
    for (uint m = 0; m < blmk.size(); ++m)
    {
        NvDsInferParseObjectInfo item = binfo[m];
        std::array<float, 10> lmks = blmk[m];

        NvDsUserMeta* um;
        um->user_meta_data = set_metadata_ptr(lmks);

        // std::cout << "BBox Cords: " << item.left << " " << item.top << " " << (item.left+item.width) << " " << (item.top+item.height) << " " <<  item.detectionConfidence <<  std::endl;
        // std::cout << "LNMs Cords: " << std::endl;
        // for(auto i = lmk.begin(); i != lmk.end(); ++i)
        // {
        //     std::cout << *i << std::endl;
        // }
        obj_user_meta_list = g_list_append(obj_user_meta_list, um);

    }
    return binfo;
}

static std::vector<NvDsInferParseObjectInfo> decodeYoloV4Tensor(
    const float* boxes, const float* scores, const float* classes,
    const uint num_bboxes, NvDsInferParseDetectionParams const& detectionParams,
    const uint& netW, const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;


    uint bbox_location = 0;
    uint score_location = 0;
    for (uint b = 0; b < num_bboxes; ++b)
    {
        float bx1 = boxes[bbox_location];
        float by1 = boxes[bbox_location + 1];
        float bx2 = boxes[bbox_location + 2];
        float by2 = boxes[bbox_location + 3];
        float maxProb = scores[score_location];
        int maxIndex = (int) classes[score_location];

        if (maxProb > detectionParams.perClassPreclusterThreshold[maxIndex])
        {
            addBBoxProposalYoloV4(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
        }

        bbox_location += 4;
        score_location += 1;
    }

    return binfo;
}

bool compareObject(ObjectPoint obj1, ObjectPoint obj2)
{
    return (obj1.ctx < obj2.ctx);
}

static std::string mergeDetectionResult(const std::vector<ObjectPoint> &objectList){
    //Divide to 2 lines
    std::vector <ObjectPoint> objectList1;
    std::vector <ObjectPoint> objectList2;
    const float max_dis = 0.3;
    float min_y = 1.;
    for (ObjectPoint obj : objectList){
        if (obj.cty < min_y)
            {
                min_y =  obj.cty;
            }
    }
    for (ObjectPoint obj : objectList){
        if (obj.cty - min_y < max_dis)
            {
                objectList1.push_back(obj);
            }
        else objectList2.push_back(obj);
    } 
    // Sort each line
    std::string licensePlate = "";
    if (objectList1.size() > 0){
        sort(objectList1.begin(), objectList1.end(), compareObject);
        for (ObjectPoint obj: objectList1){
            licensePlate += dict_table[obj.classId];
            }
        }
    if (objectList2.size() > 0){
        sort(objectList2.begin(), objectList2.end(), compareObject);
        for (ObjectPoint obj: objectList2){
            licensePlate += dict_table[obj.classId];
            }
        }

    return licensePlate;

}

extern "C" bool NvDsInferParseCustomFaceDetection(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (detectionParams.numClassesConfigured != 1)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    std::vector<NvDsInferParseObjectInfo> objects;
    const NvDsInferLayerInfo &num_detections     = outputLayersInfo[0];
    const NvDsInferLayerInfo &boxes        = outputLayersInfo[1]; // (num_boxes, 4)
    const NvDsInferLayerInfo &scores       = outputLayersInfo[2]; // (num_boxes, )
    const NvDsInferLayerInfo &classes      = outputLayersInfo[3]; // (num_boxes, )
    const NvDsInferLayerInfo &landmarks    = outputLayersInfo[4]; // (num_boxes, 10)

    int num_bboxes = *(const int*)(num_detections.buffer);
    std::cout << "number of boxes: " << num_bboxes << std::endl;

    assert(boxes.inferDims.numDims == 2);
    assert(scores.inferDims.numDims == 1);
    assert(classes.inferDims.numDims == 1);
    assert(landmarks.inferDims.numDims == 2);

    std::cout << "Network Info: " << networkInfo.height << "  " << networkInfo.width << std::endl;

    std::vector<NvDsInferParseObjectInfo> outObjs =
        decodeFaceDetectionTensor(
            (const float*)(boxes.buffer), (const float*)(scores.buffer), (const float*)(classes.buffer), (const float*)(landmarks.buffer), num_bboxes, detectionParams,
            networkInfo.width, networkInfo.height);

    objects.insert(objects.end(), outObjs.begin(), outObjs.end());

    objectList = objects;

    return true;
}

extern "C" bool NvDsInferParseCustomYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    // {
    //     std::cerr << "WARNING: Num classes mismatch. Configured:"
    //               << detectionParams.numClassesConfigured
    //               << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    // }

    std::vector<NvDsInferParseObjectInfo> objects;
    const NvDsInferLayerInfo &n_bboxes   = outputLayersInfo[0];
    const NvDsInferLayerInfo &boxes      = outputLayersInfo[1]; // (num_boxes, 4)
    const NvDsInferLayerInfo &scores     = outputLayersInfo[2]; // (num_boxes, )
    const NvDsInferLayerInfo &classes    = outputLayersInfo[3]; // (num_boxes, )
    

    int num_bboxes = *(const int*)(n_bboxes.buffer);


    assert(boxes.inferDims.numDims == 2);
    assert(scores.inferDims.numDims == 1);
    assert(classes.inferDims.numDims == 1);

    // std::cout << "Network Info: " << networkInfo.height << "  " << networkInfo.width << std::endl;

    std::vector<NvDsInferParseObjectInfo> outObjs =
        decodeYoloV4Tensor(
            (const float*)(boxes.buffer), (const float*)(scores.buffer), (const float*)(classes.buffer), num_bboxes, detectionParams,
            networkInfo.width, networkInfo.height);

    objects.insert(objects.end(), outObjs.begin(), outObjs.end());

    objectList = objects;

    return true;
}

/* YOLOv4 TLT with Padding*/
extern "C" bool NvDsInferParseCustomYoloV4TLT(
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList) {

     if(outputLayersInfo.size() != 4)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 4 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    /* Host memory for "BatchedNMS"
       BatchedNMS has 4 output bindings, the order is:
       keepCount, bboxes, scores, classes
    */
    int* p_keep_count = (int *) outputLayersInfo[0].buffer;
    float* p_bboxes = (float *) outputLayersInfo[1].buffer;
    float* p_scores = (float *) outputLayersInfo[2].buffer;
    float* p_classes = (float *) outputLayersInfo[3].buffer;

    const float threshold = detectionParams.perClassThreshold[0];
    const float pad_ratio_width = 0.01;
    const float pad_ratio_height = 0.04;

    const int keep_top_k = 200;
    const char* log_enable = std::getenv("ENABLE_DEBUG");

    // if(log_enable != NULL && std::stoi(log_enable)) {
    //     std::cout <<"keep cout"
    //           <<p_keep_count[0] << std::endl;
    // }

    for (int i = 0; i < p_keep_count[0] && objectList.size() <= keep_top_k; i++) {

        if ( p_scores[i] < threshold) continue;

        // if(log_enable != NULL && std::stoi(log_enable)) {
        //     std::cout << "label/conf/ x/y x/y -- "
        //               << p_classes[i] << " " << p_scores[i] << " "
        //               << p_bboxes[4*i] << " " << p_bboxes[4*i+1] << " " << p_bboxes[4*i+2] << " "<< p_bboxes[4*i+3] << " " << std::endl;
        // }

        if((unsigned int) p_classes[i] >= detectionParams.numClassesConfigured) continue;
        if(p_bboxes[4*i+2] < p_bboxes[4*i] || p_bboxes[4*i+3] < p_bboxes[4*i+1]) continue;

        NvDsInferObjectDetectionInfo object;
        object.classId = (int) p_classes[i];
        object.detectionConfidence = p_scores[i];
        /* Clip object box co-ordinates to network resolution */
        float obj_left = CLIP(p_bboxes[4*i] * networkInfo.width, 0, networkInfo.width - 1);
        float obj_top = CLIP(p_bboxes[4*i+1] * networkInfo.height, 0, networkInfo.height - 1);
        float obj_width = CLIP(p_bboxes[4*i+2] * networkInfo.width, 0, networkInfo.width - 1) - obj_left;
        float obj_height = CLIP(p_bboxes[4*i+3] * networkInfo.height, 0, networkInfo.height - 1) - obj_top;
        
        /* Add padding*/
        float pad_width = pad_ratio_width * obj_width;
        float pad_height = pad_ratio_height * obj_height;
        object.left = CLIP(p_bboxes[4*i] * networkInfo.width - pad_width, 0, networkInfo.width - 1);
        object.top = CLIP(p_bboxes[4*i+1] * networkInfo.height - pad_height, 0, networkInfo.height - 1);
        object.width = CLIP(p_bboxes[4*i+2] * networkInfo.width + 2.0*pad_width, 0, networkInfo.width - 1) - object.left;
        object.height = CLIP(p_bboxes[4*i+3] * networkInfo.height + 2.0*pad_height, 0, networkInfo.height - 1) - object.top;

        if(object.height < 0 || object.width < 0)
            continue;
        objectList.push_back(object);
    }
    return true;
}


/* YOLOv4 TLT from detection to recognition*/
extern "C" bool NvDsInferParseCustomYoloV4LPR(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                 NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                 std::vector<NvDsInferAttribute> &attrList, std::string &attrString) {

     if(outputLayersInfo.size() != 4)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 4 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    /* Host memory for "BatchedNMS"
       BatchedNMS has 4 output bindings, the order is:
       keepCount, bboxes, scores, classes
    */
    int* p_keep_count = (int *) outputLayersInfo[0].buffer;
    float* p_bboxes = (float *) outputLayersInfo[1].buffer;
    float* p_scores = (float *) outputLayersInfo[2].buffer;
    float* p_classes = (float *) outputLayersInfo[3].buffer;

    const int max_length = 10;
    const char* log_enable = std::getenv("ENABLE_DEBUG");

    // if(log_enable != NULL && std::stoi(log_enable)) {
    //     std::cout <<"keep cout"
    //           <<p_keep_count[0] << std::endl;
    // }
    // Read dict
    std::ifstream fdict;
    setlocale(LC_CTYPE, "");

    if(!dict_ready) {
        fdict.open("dict.txt");
        if(!fdict.is_open())
        {
            std::cout << "open dictionary file failed." << std::endl;
            return false;
        }
        while(!fdict.eof()) {
            std::string strLineAnsi;
            if (getline(fdict, strLineAnsi) ) {
                if (strLineAnsi.length() > 1) {
                    strLineAnsi.erase(1);
                }
                dict_table.push_back(strLineAnsi);
            }
        }
        dict_ready=true;
        fdict.close();
    }

    // Empty list of object
    std::vector <ObjectPoint> objectList;
    // Append detection result
    for (int i = 0; i < p_keep_count[0] && objectList.size() <= max_length; i++) {

        if ( (float)p_scores[i] < classifierThreshold) continue;

        if(log_enable != NULL && std::stoi(log_enable)) {
            std::cout << "label/conf/ x/y x/y -- "
                      << p_classes[i] << " " << p_scores[i] << " "
                      << p_bboxes[4*i] << " " << p_bboxes[4*i+1] << " " << p_bboxes[4*i+2] << " "<< p_bboxes[4*i+3] << " " << std::endl;
        }

        if(p_bboxes[4*i+2] < p_bboxes[4*i] || p_bboxes[4*i+3] < p_bboxes[4*i+1]) continue;

        ObjectPoint obj;
        obj.ctx = (float)(p_bboxes[4*i] + p_bboxes[4*i+2])/2.0;
        obj.cty = (float)(p_bboxes[4*i+1] + p_bboxes[4*i+3])/2.0;
        obj.width = (float)p_bboxes[4*i+2] - p_bboxes[4*i];
        obj.height = (float)p_bboxes[4*i+3] - p_bboxes[4*i+1];
        obj.confidence = (float)p_scores[i];
        obj.classId = (int) p_classes[i];

        if(obj.height < 0 || obj.width < 0)
            continue;
        objectList.push_back(obj);
    }
    // Add to metadata
    NvDsInferAttribute LPR_attr;
    // LPR_attr.attributeConfidence = sumConfidence / objectListConfidence.size();
    attrString = mergeDetectionResult(objectList);
    if (objectList.size() >=  3) {
        LPR_attr.attributeIndex = 0;
        LPR_attr.attributeValue = 1;
        LPR_attr.attributeConfidence = 1.0;
        LPR_attr.attributeLabel = strdup(attrString.c_str());
        for (ObjectPoint obj: objectList)
            LPR_attr.attributeConfidence *= obj.confidence;
        attrList.push_back(LPR_attr);
        // std::cout << "License plate: " << attrString << "  -  Confidence: " << LPR_attr.attributeConfidence << std::endl;
    }

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV4);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV4TLT);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomFaceDetection);
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV4LPR);