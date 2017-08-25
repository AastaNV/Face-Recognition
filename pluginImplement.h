#ifndef __PLUGIN_LAYER_H__
#define __PLUGIN_LAYER_H__

#include <cassert>
#include <iostream>
#include <cudnn.h>
#include <cstring>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"

#define CHECK(status)                                                                                           \
    {                                                                                                                           \
        if (status != 0)                                                                                                \
        {                                                                                                                               \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) \
                      << " at line " << __LINE__                                                        \
                      << std::endl;                                                                     \
            abort();                                                                                                    \
        }                                                                                                                               \
    }

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

enum FunctionType
{
    SELECT=0,
    SUMMARY
};

class bboxProfile {
public:
    bboxProfile(float4& p, int idx): pos(p), bboxNum(idx) {}

    float4 pos;
    int bboxNum = -1;
    int labelID = -1;
};

class tagProfile {
public:
    tagProfile(int b, int l): bboxID(b), label(l) {}
    int bboxID;
    int label;
};

class BboxMergeLayer : public IPlugin
{
public:
    BboxMergeLayer() {};
    BboxMergeLayer(const void* buffer, size_t size);

    inline int getNbOutputs() const override { return 1; };
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;
    inline void terminate() override { ; }

    inline size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override;

    size_t getSerializationSize() override;
    void serialize(void* buffer) override;

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override;

protected:
    void mergeRect(std::vector<float4>& rects, const float4& rect);

    DimsCHW dimsData;
    DimsCHW dimsConf;
    DimsCHW dimsBbox;

    int ow;
    int oh;
    int owh;
    int cls;
    float cell_width;
    float cell_height;
};

class RecognitionLayer : public IPlugin
{
public:
    RecognitionLayer(FunctionType t) { type = t; };
    RecognitionLayer(FunctionType t, const void* buffer, size_t size);

    int getNbOutputs() const override;
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    inline int initialize() override { return 0; }
    inline void terminate() override { ; }

    inline size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override;

    size_t getSerializationSize() override;
    void serialize(void* buffer) override;

    inline void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override { ; };

protected:
    int select(const void*const *inputs, void** outputs);
    int summary(const void*const *inputs, void** outputs);
    int bboxExist(const float4& pos, const int idx);
    int tagExist(int label, int idx);

    size_t classNum;   
    FunctionType type;
    static std::vector<bboxProfile*> bboxTable;
    static std::vector<tagProfile*> tagTable;
};

class DataRoiLayer : public IPlugin
{
public:
    DataRoiLayer() {};
    DataRoiLayer(const void* buffer, size_t size);

    inline int getNbOutputs() const override { return 1; };
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;
    inline void terminate() override { ; }

    inline size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override;

    size_t getSerializationSize() override;
    void serialize(void* buffer) override;

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override;

protected:
    DimsCHW dimsData;
    DimsCHW dimsRoi;
};

class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override;
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;

    bool isPlugin(const char* name) override;
    void destroyPlugin();

    std::unique_ptr<BboxMergeLayer> mBboxMergeLayer{ nullptr };
    std::unique_ptr<DataRoiLayer> mDataRoiLayer{ nullptr };
    std::unique_ptr<RecognitionLayer> mSelectLayer{ nullptr };
    std::unique_ptr<RecognitionLayer> mSummaryLayer{ nullptr };
};

#endif
