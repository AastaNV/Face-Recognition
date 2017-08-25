#include <algorithm>
#include "gstCamera.h"
#include "glDisplay.h"
#include "glTexture.h"

#include "cudaNormalize.h"
#include "cudaOverlay.h"
#include "cudaFont.h"
#include "tensorNet.h"


using namespace nvinfer1;
using namespace nvcaffeparser1;

static const int BATCH_SIZE = 1;
static const int TIMING_ITERATIONS = 100;

const char* model  = "/home/nvidia/face-recognition/data/deploy.prototxt";
const char* weight = "/home/nvidia/face-recognition/data/merge.caffemodel";
const char* label  = "/home/nvidia/face-recognition/data/labels.txt";

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_COV = "coverage_fd";
const char* OUTPUT_BLOB_BOX = "bboxes_fd";
const char* OUTPUT_BLOB_NUM = "count_fd";
const char* OUTPUT_BLOB_SEL = "bbox_fr";
const char* OUTPUT_BLOB_IDX = "bbox_id";
const char* OUTPUT_BLOB_RES = "softmax_fr";
const char* OUTPUT_BLOB_LAB = "label";

#define DEFAULT_CAMERA -1        // -1 for onboard camera, or change to index of /dev/video V4L2 camera (>=0)    

cudaError_t cudaPreImageNetMean( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value );



bool signal_recieved = false;

void sig_handler(int signo)
{
    if( signo == SIGINT )
    {
        printf("received SIGINT\n");
        signal_recieved = true;
    }
}

float* allocateMemory(DimsCHW dims, char* info)
{
    float* ptr;
    size_t size;
    std::cout << "Allocate memory: " << info << std::endl;
    size = BATCH_SIZE * dims.c() * dims.h() * dims.w();
    assert(!cudaMallocManaged(&ptr, size*sizeof(float)));
    return ptr;
}

// load label info
std::vector<std::string> loadLabelInfo(const char* filename)
{   
    assert(filename);
    std::vector<std::string> labelInfo;

    FILE* f = fopen(filename, "r");
    if( !f )
    {   
        printf("failed to open %s\n", filename);
        assert(0);
    }
    
    char str[512];
    while( fgets(str, 512, f) != NULL )
    {   
        const int syn = 9;  // length of synset prefix (in characters)
        const int len = strlen(str);
        
        if( len > syn && str[0] == 'n' && str[syn] == ' ' )
        {   
            str[syn]   = 0;
            str[len-1] = 0;
            
            const std::string b = (str + syn + 1);
            labelInfo.push_back(b);
        }
        else if( len > 0 )      // no 9-character synset prefix (i.e. from DIGITS snapshot)
        {   
            if( str[len-1] == '\n' ) str[len-1] = 0;
            labelInfo.push_back(str);
        }
    }
    fclose(f);
    return labelInfo;
}

bool DrawBoxes(float* input, float* output, uint32_t width, uint32_t height, const float scale_x, const float scale_y, float* conf, float* bbox, const int numBoundingBoxes)
{
    // Only handle single class here
    const float4 color = make_float4( 0.0f, 255.0f, 175.0f, 100.0f);

    printf("%i bounding boxes detected\n", numBoundingBoxes);
    for( int n=0; n < numBoundingBoxes; n++ )
    {
        float* bb = bbox + (n * 4);
        bb[0] *= scale_x;
        bb[1] *= scale_y;
        bb[2] *= scale_x;
        bb[3] *= scale_y;
        printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n,  bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]);
    }

    if( numBoundingBoxes>0 )
    {
        if( CUDA_FAILED(cudaRectOutlineOverlay((float4*)input, (float4*)output, width, height, (float4*)bbox, numBoundingBoxes, color)))
            printf("failed to draw boxes\n");
        CUDA(cudaThreadSynchronize());
    }
}

void ShowClassification(cudaFont* font, void* input, void* output, uint32_t width, uint32_t height,
                        float* lab, float* bbox, std::vector<std::string> &labelInfo, const int numBoundingBoxes)
{
    char str[512];

    if( font != NULL )
    {
        for( int i=0; i<numBoundingBoxes; i++)
        {
            sprintf(str, "%s", (lab[i]>-1)?labelInfo[int(lab[i])].c_str():"NAN");
            std::cout << "bbox=" << i << " class=" << lab[i] << " label=" << str << std::endl;

            float* bb = bbox + (i * 4);
            font->RenderOverlay((float4*)input, (float4*)output, width, height, (const char*)str, bb[0], bb[3], make_float4(255.0f, 255.0f, 255.0f, 255.0f));
            CUDA(cudaThreadSynchronize());
        }
    }
}



int main(int argc, char** argv)
{
    std::cout << "Building and running a GPU inference engine for " << model << ", N=" << BATCH_SIZE << "..." << std::endl;


    /* camera */
    if( signal(SIGINT, sig_handler) == SIG_ERR )
        printf("\ncan't catch SIGINT\n");

    gstCamera* camera = gstCamera::Create(DEFAULT_CAMERA);

    if( !camera )
    {
        printf("failed to initialize video device\n");
        return 0;
    }

    printf("successfully initialized video device\n");
    printf("    width:  %u\n", camera->GetWidth());
    printf("   height:  %u\n", camera->GetHeight());
    printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());


    /* create networks */
    TensorNet tensorNet;
    std::vector<std::string> labelInfo = loadLabelInfo(label);
    tensorNet.caffeToTRTModel(model, weight, std::vector < std::string > {OUTPUT_BLOB_COV, OUTPUT_BLOB_BOX, OUTPUT_BLOB_NUM, OUTPUT_BLOB_SEL, OUTPUT_BLOB_IDX, OUTPUT_BLOB_RES, OUTPUT_BLOB_LAB}, BATCH_SIZE);
    tensorNet.createInference();


    /* openGL window */
    cudaFont* font = cudaFont::Create();
    glDisplay* display = glDisplay::Create();
    glTexture* texture = NULL;

    if( !display ) {
        printf("failed to create openGL display\n");
    }
    else
    {
        texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);
        if( !texture ) printf("failed to create openGL texture\n");
    }


    /* open camera */
    if( !camera->Open() )
    {
        printf("failed to open camera for streaming\n");
        return 0;
    }


    /* prepare tensor */
    DimsCHW dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
    DimsCHW dimsConf = tensorNet.getTensorDims(OUTPUT_BLOB_COV);
    DimsCHW dimsBbox = tensorNet.getTensorDims(OUTPUT_BLOB_BOX);
    DimsCHW dimsNum  = tensorNet.getTensorDims(OUTPUT_BLOB_NUM);
    DimsCHW dimsSel  = tensorNet.getTensorDims(OUTPUT_BLOB_SEL);
    DimsCHW dimsIdx  = tensorNet.getTensorDims(OUTPUT_BLOB_IDX);
    DimsCHW dimsRes  = tensorNet.getTensorDims(OUTPUT_BLOB_RES);
    DimsCHW dimsLab  = tensorNet.getTensorDims(OUTPUT_BLOB_LAB);

    float* data = allocateMemory(dimsData, (char*)"input blob");
    float* conf = allocateMemory(dimsConf, (char*)"coverage");     // for cpu plugin layer
    float* bbox = allocateMemory(dimsBbox, (char*)"box");          // for cpu plugin layer
    float* num = allocateMemory(dimsNum, (char*)"count");
    float* sel = allocateMemory(dimsSel, (char*)"selected bbox");  // for cpu plugin layer
    float* idx = allocateMemory(dimsIdx, (char*)"selected index"); // for cpu plugin layer
    float* res = allocateMemory(dimsRes, (char*)"softmax");        // for cpu plugin layer
    float* lab = allocateMemory(dimsLab, (char*)"label");


    /* main loop */
    while( !signal_recieved )
    {
        void* imgCPU  = NULL;
        void* imgCUDA = NULL;
        void* imgRGBA = NULL;

        if( !camera->Capture(&imgCPU, &imgCUDA, 1000) ) printf("failed to capture frame\n");
        if( !camera->ConvertRGBA(imgCUDA, &imgRGBA) ) printf("failed to convert from NV12 to RGBA\n");

        if( CUDA_FAILED(cudaPreImageNetMean((float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(), data, dimsData.w(), dimsData.h(), make_float3(127.0f, 127.0f, 127.0f))) )
        {
            printf("cudaPreImageNetMean failed\n");
            return 0;
        }


        void* buffers[] = {data, conf, bbox, num, sel, idx, res, lab};
        tensorNet.imageInference(buffers, 8, BATCH_SIZE);

        const float scale_x = float(camera->GetWidth())  / float(dimsData.w());
        const float scale_y = float(camera->GetHeight()) / float(dimsData.h());

        int numBoundingBoxes = int(num[0]);
        DrawBoxes((float*)imgRGBA, (float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), scale_x, scale_y, conf, bbox, numBoundingBoxes);
        ShowClassification(font, imgRGBA, imgRGBA, camera->GetWidth(), camera->GetHeight(), lab, bbox, labelInfo, numBoundingBoxes);

        if( display != NULL )
        {
            char str[256];
            sprintf(str, "TensorRT build %x | %4.1f FPS", NV_GIE_VERSION, display->GetFPS());
            display->SetTitle(str);
        }

        if( display != NULL )
        {
            display->UserEvents();
            display->BeginRender();

            if( texture != NULL )
            {
                CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f),
                                       (float4*)imgRGBA, make_float2(0.0f, 1.0f),
                                       camera->GetWidth(), camera->GetHeight()));

                void* tex_map = texture->MapCUDA();
                if( tex_map != NULL )
                {
                    cudaMemcpy(tex_map, imgRGBA, texture->GetSize(), cudaMemcpyDeviceToDevice);
                    texture->Unmap();
                }
                texture->Render(100,100);
             }
             display->EndRender();
        }

    }


    /* destory */
    tensorNet.destroy();
    tensorNet.printTimes(TIMING_ITERATIONS);

    if( camera != NULL )
    {
        delete camera;
        camera = NULL;
    }

    if( display != NULL )
    {
        delete display;
        display = NULL;
    }

    std::cout << "Done." << std::endl;
    return 0;
}
