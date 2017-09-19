#include <pluginImplement.h>

std::vector<bboxProfile*> RecognitionLayer::bboxTable;
std::vector<tagProfile*> RecognitionLayer::tagTable;

bool bboxOverlap(const float4& r1, const float4& r2)
{
    float unionSize = (std::max(r1.z, r2.z)-std::min(r1.x, r2.x)) * (std::max(r1.w, r2.w)-std::min(r1.y, r2.y));
    float interSize = (std::min(r1.z, r2.z)-std::max(r1.x, r2.x)) * (std::min(r1.w, r2.w)-std::max(r1.y, r2.y));
    if( unionSize == 0 ) return true;
    else return (interSize/unionSize) > 0.5;
}

/******************************/
// PluginFactory
/******************************/
nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights)
{
    assert(isPlugin(layerName));
    if (!strcmp(layerName, "bboxMerge"))
    {
        assert(mBboxMergeLayer.get() == nullptr);
        mBboxMergeLayer = std::unique_ptr<BboxMergeLayer>(new BboxMergeLayer());
        return mBboxMergeLayer.get();
    }
    else if (!strcmp(layerName, "dataRoi"))
    {
        assert(mDataRoiLayer.get() == nullptr);
        mDataRoiLayer = std::unique_ptr<DataRoiLayer>(new DataRoiLayer());
        return mDataRoiLayer.get();
    }
    else if (!strcmp(layerName, "selectBbox"))
    {
        assert(mSelectLayer.get() == nullptr);
        mSelectLayer = std::unique_ptr<RecognitionLayer>(new RecognitionLayer(FunctionType::SELECT));
        return mSelectLayer.get();
    }
    else if (!strcmp(layerName, "summaryLabel"))
    {
        assert(mSummaryLayer.get() == nullptr);
        mSummaryLayer = std::unique_ptr<RecognitionLayer>(new RecognitionLayer(FunctionType::SUMMARY));
        return mSummaryLayer.get();
    }
    else
    {
        assert(0);
        return nullptr;
    }
}

IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength)
{
    assert(isPlugin(layerName));
    if (!strcmp(layerName, "bboxMerge"))
    {
        assert(mBboxMergeLayer.get() == nullptr);
        mBboxMergeLayer = std::unique_ptr<BboxMergeLayer>(new BboxMergeLayer(serialData, serialLength));
        return mBboxMergeLayer.get();
    }
    else if (!strcmp(layerName, "dataRoi"))
    {
        assert(mDataRoiLayer.get() == nullptr);
        mDataRoiLayer = std::unique_ptr<DataRoiLayer>(new DataRoiLayer(serialData, serialLength));
        return mDataRoiLayer.get();
    }
    else if (!strcmp(layerName, "selectBbox"))
    {
        assert(mSelectLayer.get() == nullptr);
        mSelectLayer = std::unique_ptr<RecognitionLayer>(new RecognitionLayer(FunctionType::SELECT, serialData, serialLength));
        return mSelectLayer.get();
    }
    else if (!strcmp(layerName, "summaryLabel"))
    {
        assert(mSummaryLayer.get() == nullptr);
        mSummaryLayer = std::unique_ptr<RecognitionLayer>(new RecognitionLayer(FunctionType::SUMMARY, serialData, serialLength));
        return mSummaryLayer.get();
    }
    else
    {
        assert(0);
        return nullptr;
    }
}

bool PluginFactory::isPlugin(const char* name)
{
    return (!strcmp(name, "bboxMerge")
         || !strcmp(name, "dataRoi")
         || !strcmp(name, "selectBbox")
         || !strcmp(name, "summaryLabel"));
}

void PluginFactory::destroyPlugin()
{
    mBboxMergeLayer.release();
    mBboxMergeLayer = nullptr;
    mDataRoiLayer.release();
    mDataRoiLayer = nullptr;
    mSelectLayer.release();
    mSelectLayer = nullptr;
    mSummaryLayer.release();
    mSummaryLayer = nullptr;
}



/******************************/
// BboxMerge Plugin Layer
/******************************/
BboxMergeLayer::BboxMergeLayer(const void* buffer, size_t size)
{
    assert(size==(9*sizeof(int)));
    const int* d = reinterpret_cast<const int*>(buffer);

    dimsData = DimsCHW{d[0], d[1], d[2]};
    dimsConf = DimsCHW{d[3], d[4], d[5]};
    dimsBbox = DimsCHW{d[6], d[7], d[8]};
}

Dims BboxMergeLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims==3);
    return DimsCHW(1, 1, 1);
}

int BboxMergeLayer::initialize()
{
    ow  = dimsBbox.w();
    oh  = dimsBbox.h();
    owh = ow * oh;
    cls = dimsConf.c();

    cell_width  = dimsData.w() / ow;
    cell_height = dimsData.h() / oh;
    return 0;
}

int BboxMergeLayer::enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream)
{
    CHECK(cudaThreadSynchronize());
    std::vector< std::vector<float4> > rects;
    rects.resize(cls);

    float* conf = (float*)inputs[1];
    float* bbox = (float*)inputs[2];

    for( uint32_t z=0; z < cls; z++ )
    {
        rects[z].reserve(owh);
        for( uint32_t y=0; y < oh; y++ )
        {
            for( uint32_t x=0; x < ow; x++)
            {
                const float coverage = conf[z * owh + y * ow + x];
                if( coverage > 0.5 )
                {
                    const float mx = x * cell_width;
                    const float my = y * cell_height;

                    const float x1 = (bbox[0 * owh + y * ow + x] + mx);
                    const float y1 = (bbox[1 * owh + y * ow + x] + my);
                    const float x2 = (bbox[2 * owh + y * ow + x] + mx);
                    const float y2 = (bbox[3 * owh + y * ow + x] + my);
                    mergeRect( rects[z], make_float4(x1, y1, x2, y2) );
                }
            }
        }
    }

    int n = 0;
    int numMax = dimsBbox.c() * dimsBbox.h() * dimsBbox.w();
    for( uint32_t z = 0; z < cls; z++ )
    {
        const uint32_t numBox = rects[z].size();

        for( uint32_t b = 0; b < numBox && n < numMax; b++ )
        {
            const float4 r = rects[z][b];

            bbox[n * 4 + 0] = r.x;
            bbox[n * 4 + 1] = r.y;
            bbox[n * 4 + 2] = r.z;
            bbox[n * 4 + 3] = r.w;
            n++;
        }
    }

    float* count = (float*)outputs[0];
    count[0] = float(n);
    return 0;
}

size_t BboxMergeLayer::getSerializationSize()
{
    return 9*sizeof(int);
}

void BboxMergeLayer::serialize(void* buffer)
{
    int* d = reinterpret_cast<int*>(buffer);
    d[0] = dimsData.c(); d[1] = dimsData.h(); d[2] = dimsData.w();
    d[3] = dimsConf.c(); d[4] = dimsConf.h(); d[5] = dimsConf.w();
    d[6] = dimsBbox.c(); d[7] = dimsBbox.h(); d[8] = dimsBbox.w();
}

void BboxMergeLayer::configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)
{
    dimsData = DimsCHW{inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]};
    dimsConf = DimsCHW{inputs[1].d[0], inputs[1].d[1], inputs[1].d[2]};
    dimsBbox = DimsCHW{inputs[2].d[0], inputs[2].d[1], inputs[2].d[2]};
}

void BboxMergeLayer::mergeRect(std::vector<float4>& rects, const float4& rect)
{
    const uint32_t num_rects = rects.size();
    bool intersects = false;

    for( uint32_t r=0; r < num_rects; r++ )
    {
        if( bboxOverlap(rects[r], rect) )
        {
            intersects = true;
            if( rect.x < rects[r].x )    rects[r].x = rect.x;
            if( rect.y < rects[r].y )    rects[r].y = rect.y;
            if( rect.z > rects[r].z )    rects[r].z = rect.z;
            if( rect.w > rects[r].w )    rects[r].w = rect.w;

            break;
        }
    }
    if( !intersects ) rects.push_back(rect);
}



/******************************/
// DataRoi Plugin Layer
/******************************/
void convertROI(float* input, float* output, char* mean, const int* srcSize, const int* dstSize, const int* roi, cudaStream_t stream);

DataRoiLayer::DataRoiLayer(const void* buffer, size_t size)
{
    assert(size==(6*sizeof(int)));
    const int* d = reinterpret_cast<const int*>(buffer);

    dimsData  = DimsCHW{d[0], d[1], d[2]};
    dimsRoi   = DimsCHW{d[3], d[4], d[5]};
}

Dims DataRoiLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims==2);
    return DimsCHW(3, 224, 224);
}

int DataRoiLayer::initialize()
{
    return 0;
}

int DataRoiLayer::enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream)
{
    float* bbox = (float*)inputs[1];

    int srcSize[] {dimsData.c(), dimsData.h(), dimsData.w()};
    int dstSize[] {dimsRoi.c(), dimsRoi.h(), dimsRoi.w()};
    int roi[] = { int(bbox[0]+0.5), int(bbox[1]+0.5), int(bbox[2]+0.5), int(bbox[3]+0.5)}; //rounding  
    convertROI((float*)inputs[0], (float*)outputs[0], nullptr, srcSize, dstSize, roi, stream);

    return 0;
}

size_t DataRoiLayer::getSerializationSize()
{
    return 6*sizeof(int);
}

void DataRoiLayer::serialize(void* buffer)
{
    int* d = reinterpret_cast<int*>(buffer);
    d[0] = dimsData.c(); d[1] = dimsData.h(); d[2] = dimsData.w();
    d[3] = dimsRoi.c();  d[4] = dimsRoi.h();  d[5] = dimsRoi.w();
}

void DataRoiLayer::configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)
{
    dimsData  = DimsCHW{inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]};
    dimsRoi   = DimsCHW{3, 224, 224};
}



/******************************/
// Recognition Plugin Layer
/******************************/
RecognitionLayer::RecognitionLayer(FunctionType t, const void* buffer, size_t size)
{
    assert(size==(sizeof(int)));
    const int* d = reinterpret_cast<const int*>(buffer);

    classNum = d[0];
    type = t;
}

int RecognitionLayer::getNbOutputs() const
{
    if( type==FunctionType::SELECT ) return 2;
    else if( type==FunctionType::SUMMARY ) return 1;
}

Dims RecognitionLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    if( type==FunctionType::SELECT )
    {
        assert(nbInputDims==2);
        return index==0 ? DimsCHW(4, 1, 1):DimsCHW(1, 1, 1);
    }
    else if( type==FunctionType::SUMMARY )
    {
        assert(nbInputDims==4);
        classNum = inputs[3].d[0];
        return DimsCHW(1, inputs[0].d[1], inputs[0].d[2]);
    }
}

int RecognitionLayer::enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream)
{
    CHECK(cudaThreadSynchronize());

    if( type==FunctionType::SELECT ) return select(inputs, outputs);
    else if( type==FunctionType::SUMMARY ) return summary(inputs, outputs);
}

size_t RecognitionLayer::getSerializationSize()
{
    return sizeof(int);
}

void RecognitionLayer::serialize(void* buffer)
{
    int* d = reinterpret_cast<int*>(buffer);
    d[0] = classNum;
}

int RecognitionLayer::select(const void*const *inputs, void** outputs)
{
    float* bbox = (float*)inputs[0];
    float* count = (float*)inputs[1];
    float* select = (float*)outputs[0];
    float* index = (float*)outputs[1];

    int queryIdx = -1;
    int bboxNum = static_cast<int>(count[0]);

    for( size_t i=0,id=0; i < bboxNum; i++,id+=4 ) {
        float4 p = make_float4(bbox[id+0], bbox[id+1], bbox[id+2], bbox[id+3]);
        if( bboxExist(p, i)<0 ) {
            if( queryIdx < 0 ) {
                bboxTable.push_back(new bboxProfile(p, i));
                queryIdx = bboxTable.size()-1;
            }
        }
    }

    if( queryIdx < 0 && bboxTable.size() > 0 ) queryIdx = rand() % bboxTable.size();
    if( queryIdx > -1 ) {
        int queryNum = bboxTable[queryIdx]->bboxNum;
        if( queryNum > -1 ) {
            select[0] = bboxTable[queryIdx]->pos.x;
            select[1] = bboxTable[queryIdx]->pos.y;
            select[2] = bboxTable[queryIdx]->pos.z;
            select[3] = bboxTable[queryIdx]->pos.w;
            index[0] = queryIdx;
            std::cout << "pass "<< queryIdx << " to trt" << std::endl;
            std::cout << select[0] << " " << select[1] << " " << select[2] << " " << select[3] << " " << std::endl;
        }
    }
    else index[0] = -1;
    return 0;
}

int RecognitionLayer::summary(const void*const *inputs, void** outputs)
{
    float* count = (float*)inputs[1];
    float* index = (float*)inputs[2];
    float* res = (float*)inputs[3];
    float* label = (float*)outputs[0];

    int bboxNum = static_cast<int>(count[0]);
    int queryIdx = static_cast<int>(index[0]);
    if( queryIdx > -1 ) {
        int classIndex = -1;
        float classMax = -1.0f;

        for( size_t n=0; n < classNum; n++ )
        {
            const float value = res[n];
            if( value > classMax )
            {
                classIndex = n;
                classMax   = value;
            }
        }
        bboxTable[queryIdx]->labelID = classIndex;
        std::cout << "ID=" <<queryIdx << ", label=" << classIndex << std::endl;
/*
       if( tagExist(classIndex, queryIdx) < 0 ) {
           tagTable.push_back(new tagProfile(queryIdx,classIndex));
           bboxTable[queryIdx]->labelID = tagTable.size()-1;
        }
*/

    }

    for( int i=0; i<bboxNum; i++ )
    {
        label[i] = -1;
        for( int j=0; j<bboxTable.size(); j++)
        {
            if( bboxTable[j]->bboxNum==i )
                label[i] = bboxTable[j]->labelID;
        }
/*
        for( int j=0; j<bboxTable.size(); j++)
        {
            if( bboxTable[j]->bboxNum==i )
            {
                if( bboxTable[j]->labelID>-1 && tagTable[bboxTable[j]->labelID]->bboxID==i ) label[i] = tagTable[bboxTable[j]->labelID]->label;
                break;
            }
        }
*/
    }

    for( int i=bboxTable.size()-1; i>=0; i-- )
        if( bboxTable[i]->bboxNum==-1 ) bboxTable.erase(bboxTable.begin()+i);
    for( int i=0; i<bboxTable.size(); i++) bboxTable[i]->bboxNum = -1;

    return 0;
}

int RecognitionLayer::bboxExist(const float4& p, const int idx)
{
    for( size_t i = 0; i < bboxTable.size(); i++ )
    {
        if( bboxOverlap(bboxTable[i]->pos,p) )
        {
            bboxTable[i]->pos = p;
            bboxTable[i]->bboxNum = idx;
            return 0;
        }
    }
    return -1;
}

int RecognitionLayer::tagExist(int label, int idx)
{
    for( size_t i = 0; i < tagTable.size(); i++ )
    {
        if( label == tagTable[i]->label ) {
            if( tagTable[i]->bboxID>-1 )  bboxTable[tagTable[i]->bboxID]->labelID = -1;

            tagTable[i]->bboxID = idx;
            bboxTable[idx]->labelID = i;
            return 0;
        }
    }

    return -1;
}
