/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#ifndef TRT_BATCHED_NMS_PLUGIN_H
#define TRT_BATCHED_NMS_PLUGIN_H
#include "gatherNMSCustomOutputs.h"
#include "kernel.h"
#include "nmsUtils.h"
#include "plugin.h"
#include <string>
#include <vector>

using namespace nvinfer1::plugin;
namespace nvinfer1
{
namespace plugin
{

class BatchedNMSCustomPlugin : public IPluginV2Ext
{
public:
    BatchedNMSCustomPlugin(NMSParameters param);
    BatchedNMSCustomPlugin(const void* data, size_t length);
    ~BatchedNMSCustomPlugin() override = default;

    // IPluginV2 methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;
    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override;
    int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* libNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    void setClipParam(bool clip) noexcept;
    void setScoreBits(int32_t scoreBits) noexcept;

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
        noexcept override;
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
        noexcept override;
    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;
    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override;
    IPluginV2Ext* clone() const noexcept override;

private:
    NMSParameters param{};
    int boxesSize{};
    int scoresSize{};
    int landmarksSize{};
    int numPriors{};
    std::string mNamespace;
    bool mClipBoxes{};
    DataType mPrecision;
    int32_t mScoreBits;
    pluginStatus_t mPluginStatus{};
};

class BatchedNMSCustomDynamicPlugin : public IPluginV2DynamicExt
{
public:
    BatchedNMSCustomDynamicPlugin(NMSParameters param);
    BatchedNMSCustomDynamicPlugin(const void* data, size_t length);
    ~BatchedNMSCustomDynamicPlugin() override = default;

    // IPluginV2 methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* libNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    void setClipParam(bool clip) noexcept;
    void setScoreBits(int32_t scoreBits) noexcept;

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputType, int nbInputs) const noexcept override;

    // IPluginV2DynamicExt methods
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(
        int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(
        const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(
        const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

private:
    NMSParameters param{};
    int boxesSize{};
    int scoresSize{};
    int landmarksSize{};
    int numPriors{};
    std::string mNamespace;
    bool mClipBoxes{};
    DataType mPrecision;
    int32_t mScoreBits;
    pluginStatus_t mPluginStatus{};
};

class BatchedNMSCustomBasePluginCreator : public BaseCreator
{
public:
    BatchedNMSCustomBasePluginCreator();
    ~BatchedNMSCustomBasePluginCreator() override = default;

    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;

protected:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

class BatchedNMSCustomPluginCreator : public BatchedNMSCustomBasePluginCreator
{
public:
    const char* getPluginName() const noexcept override;
    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
};

class BatchedNMSCustomDynamicPluginCreator : public BatchedNMSCustomBasePluginCreator
{
public:
    const char* getPluginName() const noexcept override;
    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2DynamicExt* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;
};

} // namespace plugin

REGISTER_TENSORRT_PLUGIN(BatchedNMSCustomPluginCreator);
REGISTER_TENSORRT_PLUGIN(BatchedNMSCustomDynamicPluginCreator);
} // namespace nvinfer1

#endif // TRT_BATCHED_NMS_PLUGIN_H
