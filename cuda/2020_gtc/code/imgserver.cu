/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  #define WINDOWS_LEAN_AND_MEAN
  #define NOMINMAX
  #include <windows.h>
#endif

#ifdef __linux__
  #include <sys/syscall.h>
#endif

#if !defined(NO_GL)
#include <GL/glew.h>
#include <GL/gl.h>
#ifdef __linux__
  #include <GL/glx.h>
#endif
#include <GL/freeglut.h>
#endif

#include <cuda_runtime.h>
#if !defined(NO_GL)
#include <cuda_gl_interop.h>
#endif
#include <curand_kernel.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <queue>
#include <stdio.h>
#include <string.h>
#include <thread>

#include <nvtx3/nvToolsExt.h>

using std::chrono::system_clock;

#if defined(KERNEL_OPT_1)
    #define FLOAT_T double
#else
    #define FLOAT_T float
#endif

// ****************** Global variables ***************************************

#define REFRESH_DELAY     1 //ms

#if !defined(NO_GL)
GLuint tex_cudaResult;  // where we will copy the CUDA result

unsigned int window_width = 1000;
unsigned int window_height = 500;

GLuint shDrawTex;
cudaGraphicsResource_t cudaTextureResultResource;
#endif // !NO_GL

constexpr unsigned int FeatureVectorSizeInts = 10;
constexpr unsigned int FeatureVectorSizeBitsPerInt = sizeof(unsigned int);
constexpr unsigned int MaxFeatureVectors = 1024;
// EXERCISE: allocate one too few in this array
constexpr size_t FeaturesMemSize = sizeof(unsigned int) * (FeatureVectorSizeInts * MaxFeatureVectors + 1);

unsigned int ImgLateThreshold = 200;
size_t MaxImages = 1000000000;

class CudaRenderBuffer;
std::unique_ptr<CudaRenderBuffer> g_pCudaRenderBuffer(nullptr);

volatile std::sig_atomic_t gExitFlag = 0;
volatile std::sig_atomic_t gNextImgFlag = 1;

void
handle_sigint(int)
{
    gExitFlag = 1;
}

// ****************** Forward declarations *************************************

#if !defined(NO_GL)
void display();
void reshape(int w, int h);
#endif

// ****************** CUDA helper functions ************************************

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        const char* errStr = cudaGetErrorString(err);
        printf("checkCudaErrors() API error = %04d (%s) from file <%s>, line %i.\n",
              err, errStr ? errStr : "unknown", file, line);
        exit(EXIT_FAILURE);
    }
}

// ****************** Helper code to load BMP files ***************************

#pragma pack(1)

typedef struct
{
    short type;
    int size;
    short reserved1;
    short reserved2;
    int offset;
} BMPHeader;

typedef struct
{
    int size;
    int width;
    int height;
    short planes;
    short bitsPerPixel;
    unsigned compression;
    unsigned imageSize;
    int xPelsPerMeter;
    int yPelsPerMeter;
    int clrUsed;
    int clrImportant;
} BMPInfoHeader;

template<typename T>
T* LoadBmpFile(unsigned int& width, unsigned int& height, const char* fileName)
{
    T* dst = nullptr;
    BMPHeader hdr;
    BMPInfoHeader infoHdr;
    int x, y;

    FILE* fd = nullptr;

    printf("Loading %s...\n", fileName);

    if (!(fd = fopen(fileName, "rb")))
    {
        printf("***BMP load error: file access denied***\n");
        return nullptr;
    }

    fread(&hdr, sizeof(hdr), 1, fd);

    if (hdr.type != 0x4D42)
    {
        printf("***BMP load error: bad file format***\n");
        return nullptr;
    }

    fread(&infoHdr, sizeof(infoHdr), 1, fd);

    if (infoHdr.bitsPerPixel != 24)
    {
        printf("***BMP load error: invalid color depth***\n");
        return nullptr;
    }

    if (infoHdr.compression)
    {
        printf("***BMP load error: compressed image***\n");
        return nullptr;
    }

    width  = infoHdr.width;
    height = infoHdr.height;
    dst = new T[infoHdr.width * infoHdr.height];

    fseek(fd, hdr.offset - sizeof(hdr) - sizeof(infoHdr), SEEK_CUR);

    for (y = 0; y < infoHdr.height; y++)
    {
        int real_y = infoHdr.height - y - 1;
        for (x = 0; x < infoHdr.width; x++)
        {
            (dst)[(real_y * infoHdr.width + x)].w = 0.;
            (dst)[(real_y * infoHdr.width + x)].z = (float)fgetc(fd);
            (dst)[(real_y * infoHdr.width + x)].y = (float)fgetc(fd);
            (dst)[(real_y * infoHdr.width + x)].x = (float)fgetc(fd);
        }

        for (x = 0; x < (4 - (3 * infoHdr.width) % 4) % 4; x++)
        {
            fgetc(fd);
        }
    }


    if (ferror(fd))
    {
        printf("***Unknown BMP load error.***\n");
        delete[] dst;
        return nullptr;
    }

    fclose(fd);
    return dst;
}

// ****************** Utility and image classes *************************************

template<typename T>
class LockedQueue
{
public:
    LockedQueue(size_t max = 0)
    : m_max(max)
    {
    }

    virtual ~LockedQueue()
    {
        Clear();
    }

    void Clear()
    {
        std::lock_guard<std::mutex> guard(dataMutex);
        while (!data.empty())
        {
            data.pop();
        }
    }

    // thread-safe put
    void Put(std::function<T()> imgFn)
    {
        while (!gExitFlag)
        {
            if (!m_max || data.size() < m_max)
            {
                std::lock_guard<std::mutex> guard(dataMutex);
                if (!m_max || data.size() < m_max)
                {
                    data.push(imgFn());
                    return;
                }
            }
        }
    }

    // thread-safe get
    T Get()
    {
        T entry = {};

        while (!gExitFlag)
        {
            if (!data.empty())
            {
                std::lock_guard<std::mutex> guard(dataMutex);
                if (!data.empty())
                {
                    entry = data.front();
                    data.pop();
                    break;
                }
            }
        }

        return entry;
    }

    bool Empty() const
    {
        return data.empty();
    }

protected:
    size_t m_max;
    std::queue<T> data;
    std::mutex dataMutex;
};

// base class for host and device image classes
// image buffers are of template type T
template<typename T>
class BaseImage
{
public:
    BaseImage(int imgId, system_clock::time_point creationTime, size_t instanceId, nvtxRangeId_t lifetimeRange)
    : m_imgId(imgId)
    , m_creationTime(creationTime)
    , m_instanceId(instanceId)
    , m_lifetimeRange(lifetimeRange)
    {}

    virtual bool IsValid() const = 0;

    virtual T* GetPtr() const = 0;

    // image creation time
    system_clock::time_point GetTime() const
    {
        return m_creationTime;
    }

    // reference ID of the original image
    int GetId() const
    {
        return m_imgId;
    }

    // unique ID of this image instance
    size_t GetInstanceId() const
    {
        return m_instanceId;
    }

    // NVTX range ID to track lifetime of this image
    nvtxRangeId_t GetLifetimeRange() const
    {
        return m_lifetimeRange;
    }

private:
    int m_imgId;
    system_clock::time_point m_creationTime;
    size_t m_instanceId;
    nvtxRangeId_t m_lifetimeRange;
};

// class for host (CPU) image objects
template<typename T>
class HostImage : public BaseImage<T>
{
public:
    HostImage()
    : BaseImage<T>(-1, {}, 0, {})
    , m_ptr(nullptr)
    {
    }

    HostImage(T* ptr, int imgId, system_clock::time_point creationTime, size_t instanceId, nvtxRangeId_t lifetimeRange)
    : BaseImage<T>(imgId, creationTime, instanceId, lifetimeRange)
    , m_ptr(ptr)
    {
    }

    virtual bool IsValid() const override
    {
        return m_ptr != nullptr;
    }

    virtual T* GetPtr() const override
    {
        return m_ptr;
    }

private:
    T* m_ptr;
};

#if !defined(REUSE_DEVICE_BUFFERS)

// class for cuda/device (GPU) image objects
// device buffers are free'd upon object destruction
template<typename T>
class CudaImage : public BaseImage<T>
{
public:
    CudaImage()
    : BaseImage<T>(-1, {}, 0, {})
    , m_ptr(nullptr)
    {
    }

    CudaImage(T* ptr, int imgId, system_clock::time_point creationTime, size_t instanceId, nvtxRangeId_t lifetimeRange)
    : BaseImage<T>(imgId, creationTime, instanceId, lifetimeRange)
    , m_ptr(ptr, [](T* devPtr){
        nvtxRangePush("CudaImage destructor");
        checkCudaErrors(cudaFree(devPtr));
        nvtxRangePop();
    })
    {
    }

    virtual bool IsValid() const override
    {
        return m_ptr != nullptr && this->GetId() != -1;
    }

    virtual T* GetPtr() const override
    {
        return m_ptr.get();
    }

private:
    std::shared_ptr<T> m_ptr;
};

#else

// class for cuda/device (GPU) image objects
// device buffers are returned to a pool upon object destruction
template<typename T>
class CudaImage : public BaseImage<T>
{
public:
    CudaImage()
    : BaseImage<T>(-1, {}, 0, {})
    , m_ptr(nullptr)
    , m_pQueue(nullptr)
    {
    }

    CudaImage(T* ptr, int imgId, system_clock::time_point creationTime, size_t instanceId, nvtxRangeId_t lifetimeRange)
    : BaseImage<T>(imgId, creationTime, instanceId, lifetimeRange)
    , m_ptr(ptr)
    , m_pQueue(nullptr)
    {
    }

    virtual ~CudaImage()
    {
        if (m_ptr && m_pQueue)
        {
            T* ptr = m_ptr;
            m_pQueue->Put([ptr](){ return ptr; });
        }
    }

    void SetQueue(LockedQueue<T*>* pQueue)
    {
        m_pQueue = pQueue;
    }

    virtual bool IsValid() const override
    {
        return m_ptr != nullptr && this->GetId() != -1;
    }

    virtual T* GetPtr() const override
    {
        return m_ptr;
    }

private:
    T* m_ptr;
    LockedQueue<T*>* m_pQueue;
};

#endif

using HostQueue = LockedQueue<HostImage<uchar4>>;
using CudaQueue = LockedQueue<CudaImage<uchar4>>;

// ImageDescriptors describe an image by its ID and its feature set.
// Features are computed using the ExtractFeatures CUDA kernel.
struct ImageDescriptor
{
    int id;
    unsigned int* pFeaturesDevPtr;

    ImageDescriptor()
    : id(-1)
    , pFeaturesDevPtr(nullptr)
    {
    }

    ImageDescriptor(int _id, unsigned int* _pDevPtr)
    : id(_id)
    , pFeaturesDevPtr(_pDevPtr)
    {
    }
};

std::vector<ImageDescriptor> g_imgDescriptors;

// class that handles rendering the output grid using CUDA/OpenGL interop.
// If an image couldn't be matched, the grid cell is rendered black.
// If an image could be matched but wasn't analyzed fast enough, a saf smiley is shown.
// Otherwise, the gray-scale version of the image with its marked features is rendered.
class CudaRenderBuffer
{
public:
    CudaRenderBuffer(int imgWidth, int imgHeight, int columns, int rows, CudaQueue& imgQueue, uchar4* pSmiley)
    : m_imgWidth(imgWidth)
    , m_imgHeight(imgHeight)
    , m_columns(columns)
    , m_rows(rows)
    , m_pTexture(nullptr)
    , m_pZeroBuffer(nullptr)
    , m_pSmileyBuffer(nullptr)
    , m_pSmiley(pSmiley)
    , m_queue(imgQueue)
    , m_currentIndex(0)
    , m_firstCall(true)
    {
#if defined(REUSE_DEVICE_BUFFERS)
        const auto bufferSize = m_imgWidth * m_imgHeight * sizeof(uchar4);
        checkCudaErrors(cudaMalloc((void**)&m_pZeroBuffer, bufferSize));
        checkCudaErrors(cudaMemset(m_pZeroBuffer, 0, bufferSize));
        checkCudaErrors(cudaMalloc((void**)&m_pSmileyBuffer, bufferSize));
        checkCudaErrors(cudaMemcpy(m_pSmileyBuffer, pSmiley, bufferSize, cudaMemcpyHostToDevice));
#endif // REUSE_DEVICE_BUFFERS
    }

    virtual ~CudaRenderBuffer()
    {
#if defined(REUSE_DEVICE_BUFFERS)
        checkCudaErrors(cudaFree(m_pZeroBuffer));
        checkCudaErrors(cudaFree(m_pSmileyBuffer));
#endif // REUSE_DEVICE_BUFFERS
    }

    bool Render()
    {
#if defined(RENDER_ON_DEMAND)
        if (m_queue.Empty())
        {
            // nothing new to render in the queue, exit early
            return false;
        }
#endif // RENDER_ON_DEMAND

        bool renderResult = false;

        std::string message("Render: ");
        uint32_t color = 0xFFFF0000;
        nvtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.colorType = NVTX_COLOR_ARGB;

        std::lock_guard<std::mutex> guard(m_mutex);

#if !defined(REUSE_DEVICE_BUFFERS)
        const auto bufferSize = m_imgWidth * m_imgHeight * sizeof(uchar4);
        checkCudaErrors(cudaMalloc((void**)&m_pZeroBuffer, bufferSize));
        checkCudaErrors(cudaMemset(m_pZeroBuffer, 0, bufferSize));
        checkCudaErrors(cudaMalloc((void**)&m_pSmileyBuffer, bufferSize));
        checkCudaErrors(cudaMemcpy(m_pSmileyBuffer, m_pSmiley, bufferSize, cudaMemcpyHostToDevice));
#endif // !REUSE_DEVICE_BUFFERS

#if !defined(NO_GL)
        checkCudaErrors(cudaGraphicsMapResources(1, &cudaTextureResultResource));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&m_pTexture, cudaTextureResultResource, 0, 0));

        if (m_firstCall)
        {
            // on the very first call, zero out the complete grid so everything is shown black
            Update(m_pZeroBuffer);
            m_currentIndex = 0;
        }
#endif // !NO_GL

        // decide what should be rendered in the current grid cell
        uchar4* devPtr = nullptr;
        CudaImage<uchar4> image;

        if (!m_queue.Empty())
        {
            image = m_queue.Get();

            if (!image.IsValid())
            {
                devPtr = m_pZeroBuffer;
                message += "Invalid";
            }
            else
            {
                nvtxRangeEnd(image.GetLifetimeRange());

                const auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - image.GetTime());
                if (latency.count() > ImgLateThreshold)
                {
                    devPtr = m_pSmileyBuffer;
                    color = 0xFFD2930B;
                    message += std::to_string(image.GetInstanceId()) + " is late (" + std::to_string(latency.count()) + " ms)";
                }
                else
                {
                    devPtr = image.GetPtr();
                    color = 0xFF00FF00;
                    message += std::to_string(image.GetInstanceId()) + " is ok";
                }

                printf("Latency: %llu ms\n", (long long unsigned)latency.count());
            }

            renderResult = true;
        }

        eventAttrib.message.ascii = message.c_str();
        eventAttrib.color = color;
        nvtxRangePushEx(&eventAttrib);

#if !defined(NO_GL)

        if (devPtr)
        {
            Update(devPtr);
        }

        checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaTextureResultResource));
        m_pTexture = nullptr;
#endif

#if !defined(REUSE_DEVICE_BUFFERS)
        checkCudaErrors(cudaFree(m_pZeroBuffer));
        checkCudaErrors(cudaFree(m_pSmileyBuffer));
#endif

        nvtxRangePop();
        return renderResult;
    }

    int ImgWidth() const
    {
        return m_imgWidth;
    }

    int ImgHeight() const
    {
        return m_imgHeight;
    }

    int NumColumns() const
    {
        return m_columns;
    }

    int NumRows() const
    {
        return m_rows;
    }

private:
    // update a single grid cell
    void UpdateCell(const uchar4* pDevPtr, int posX, int posY)
    {
        checkCudaErrors(cudaMemcpy2DToArray(
            m_pTexture,
            posX * m_imgWidth * sizeof(uchar4),
            posY * m_imgHeight,
            pDevPtr,
            m_imgWidth * sizeof(uchar4),
            m_imgWidth * sizeof(uchar4),
            m_imgHeight,
            cudaMemcpyDeviceToDevice));
    }

    // update all grid cells as necessary
    // for the very first call, this means all cells
    // for all other calls, this means only the changed cell
    void Update(const uchar4* pDevPtr)
    {
        const uint32_t startIndex = m_firstCall ? 0 : m_currentIndex;
        const uint32_t endIndex = m_firstCall ? m_columns * m_rows - 1 : m_currentIndex;

        for (uint32_t idx = startIndex; idx <= endIndex; ++idx)
        {
            const int posX = idx % m_columns;
            const int posY = idx / m_columns;

            UpdateCell(pDevPtr, posX, posY);
        }

        // advance to the next cell for the next render call
        m_currentIndex = (m_currentIndex + 1) % (m_columns * m_rows);
        m_firstCall = false;
    }

private:
    int m_imgWidth;
    int m_imgHeight;
    int m_columns;
    int m_rows;
    CudaQueue& m_queue;

    // CUDA generated data in cuda memory or in a mapped PBO made of BGRA 8 bits
    // map the texture and blit the result thanks to CUDA API
    // We want to copy cudaRenderResource data to the texture
    // map buffer objects to get CUDA device pointers
    cudaArray* m_pTexture;
    std::mutex m_mutex;

    uchar4* m_pSmiley;
    uchar4* m_pZeroBuffer;
    uchar4* m_pSmileyBuffer;

    bool m_firstCall;
    uint32_t m_currentIndex;
};


// ****************** OpenGL rendering code and helper functions ***************************

#if !defined(NO_GL)

static inline const char* glErrorToString(GLenum err)
{
#define CASE_RETURN_MACRO(arg) case arg: return #arg
    switch(err)
    {
        CASE_RETURN_MACRO(GL_NO_ERROR);
        CASE_RETURN_MACRO(GL_INVALID_ENUM);
        CASE_RETURN_MACRO(GL_INVALID_VALUE);
        CASE_RETURN_MACRO(GL_INVALID_OPERATION);
        CASE_RETURN_MACRO(GL_OUT_OF_MEMORY);
        CASE_RETURN_MACRO(GL_STACK_UNDERFLOW);
        CASE_RETURN_MACRO(GL_STACK_OVERFLOW);
#ifdef GL_INVALID_FRAMEBUFFER_OPERATION
        CASE_RETURN_MACRO(GL_INVALID_FRAMEBUFFER_OPERATION);
#endif
        default: break;
    }
#undef CASE_RETURN_MACRO
    return "*UNKNOWN*";
}

inline bool sdkCheckErrorGL(const char *file, const int line)
{
    bool ret_val = true;

    // check for error
    GLenum gl_error = glGetError();

    if (gl_error != GL_NO_ERROR)
    {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        char tmpStr[512];
        // NOTE: "%s(%i) : " allows Visual Studio to directly jump to the file at the right line
        // when the user double clicks on the error line in the Output pane. Like any compile error.
        sprintf_s(tmpStr, 255, "\n%s(%i) : GL Error : %s\n\n", file, line, glErrorToString(gl_error));
        fprintf(stderr, "%s", tmpStr);
#endif
        fprintf(stderr, "GL Error in file '%s' in line %d :\n", file, line);
        fprintf(stderr, "%s\n", glErrorToString(gl_error));
        ret_val = false;
    }

    return ret_val;
}

#define SDK_CHECK_ERROR_GL()                                              \
    if( false == sdkCheckErrorGL( __FILE__, __LINE__)) {                  \
        cudaDeviceReset();                                                \
        exit(EXIT_FAILURE);                                               \
    }

static const char *glsl_drawtex_vertshader_src =
    "void main(void)\n"
    "{\n"
    "	gl_Position = gl_Vertex;\n"
    "	gl_TexCoord[0].xy = gl_MultiTexCoord0.xy;\n"
    "}\n";

static const char *glsl_drawtex_fragshader_src =
    "#version 130\n"
    "uniform usampler2D texImage;\n"
    "void main()\n"
    "{\n"
    "   vec4 c = texture(texImage, gl_TexCoord[0].xy);\n"
    "	gl_FragColor = c / 255.0;\n"
    "}\n";

// display image to the screen as textured quad
void displayImage(GLuint texture)
{
    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, window_width, window_height);

    // if the texture is a 8 bits UI, scale the fetch with a GLSL shader
    glUseProgram(shDrawTex);
    GLint id = glGetUniformLocation(shDrawTex, "texImage");
    glUniform1i(id, 0); // texture unit 0 to "texImage"
    SDK_CHECK_ERROR_GL();

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(-1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 0.0);
    glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 1.0);
    glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(0.0, 1.0);
    glVertex3f(-1.0, 1.0, 0.5);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);

    glUseProgram(0);
}

void display()
{
    if (gExitFlag)
    {
        glutLeaveMainLoop();
        return;
    }

    nvtxRangePush("display");
    bool success = g_pCudaRenderBuffer->Render();

#if defined(RENDER_ON_DEMAND)
    if (success)
#endif
    {
        // display image and flip backbuffer
        displayImage(tex_cudaResult);
        glutSwapBuffers();
    }

    if (success)
    {
        // unlock next source image
        gNextImgFlag = 1;
    }

    nvtxRangePop();
}

void reshape(int w, int h)
{
    window_width = w;
    window_height = h;
}

void deleteTexture(GLuint *tex)
{
    glDeleteTextures(1, tex);
    SDK_CHECK_ERROR_GL();
    *tex = 0;
}

void cleanup()
{
    // free gl resources
    checkCudaErrors(cudaGraphicsUnregisterResource(cudaTextureResultResource));
    deleteTexture(&tex_cudaResult);
}

GLuint compileGLSLprogram(const char *vertex_shader_src, const char *fragment_shader_src)
{
    GLuint v, f, p = 0;

    p = glCreateProgram();

    if (vertex_shader_src)
    {
        v = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(v, 1, &vertex_shader_src, NULL);
        glCompileShader(v);

        // check if shader compiled
        GLint compiled = 0;
        glGetShaderiv(v, GL_COMPILE_STATUS, &compiled);

        if (!compiled)
        {
            //#ifdef NV_REPORT_COMPILE_ERRORS
            char temp[256] = "";
            glGetShaderInfoLog(v, 256, NULL, temp);
            printf("Vtx Compile failed:\n%s\n", temp);
            //#endif
            glDeleteShader(v);
            return 0;
        }
        else
        {
            glAttachShader(p,v);
        }
    }

    if (fragment_shader_src)
    {
        f = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(f, 1, &fragment_shader_src, NULL);
        glCompileShader(f);

        // check if shader compiled
        GLint compiled = 0;
        glGetShaderiv(f, GL_COMPILE_STATUS, &compiled);

        if (!compiled)
        {
            //#ifdef NV_REPORT_COMPILE_ERRORS
            char temp[256] = "";
            glGetShaderInfoLog(f, 256, NULL, temp);
            printf("frag Compile failed:\n%s\n", temp);
            //#endif
            glDeleteShader(f);
            return 0;
        }
        else
        {
            glAttachShader(p,f);
        }
    }

    glLinkProgram(p);

    int infologLength = 0;
    int charsWritten  = 0;

    GLint linked = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &linked);

    if (linked == 0)
    {
        glGetProgramiv(p, GL_INFO_LOG_LENGTH, (GLint *)&infologLength);

        if (infologLength > 0)
        {
            char *infoLog = (char *)malloc(infologLength);
            glGetProgramInfoLog(p, infologLength, (GLsizei *)&charsWritten, infoLog);
            printf("Shader compilation error: %s\n", infoLog);
            free(infoLog);
        }
    }

    return p;
}

void initGLBuffers()
{
    const auto renderImgWidth = g_pCudaRenderBuffer->ImgWidth() * g_pCudaRenderBuffer->NumColumns();
    const auto renderImgHeight = g_pCudaRenderBuffer->ImgHeight() * g_pCudaRenderBuffer->NumRows();

    // create texture that will receive the result of CUDA
    // create a texture
    glGenTextures(1, &tex_cudaResult);
    glBindTexture(GL_TEXTURE_2D, tex_cudaResult);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, renderImgWidth, renderImgHeight, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
    SDK_CHECK_ERROR_GL();
    // register this texture with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaTextureResultResource, tex_cudaResult,
                                                GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

    glBindTexture(GL_TEXTURE_2D, 0);

    shDrawTex = compileGLSLprogram(glsl_drawtex_vertshader_src, glsl_drawtex_fragshader_src);

    SDK_CHECK_ERROR_GL();
}

void timerEvent(int value)
{
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

bool initGL(int *argc, char **argv)
{
    // Create GL context
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutCreateWindow("CUDA ImgServer");

    // register callbacks
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    glutCloseFunc(cleanup);

    if (glewInit() != GLEW_OK)
    {
        std::cerr << "glewInit() failed!" << std::endl;
        return false;
    }

    // default initialization
    glClearColor(0.5, 0.5, 0.5, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    SDK_CHECK_ERROR_GL();

    return true;
}

#endif // !NO_GL

// ****************** Worker (server) thread *************************************

// convert an RGB(A) color to its luminance value (gray-scale)
__device__ unsigned char RGBtoLuminance(
    const uchar4& color
)
{
    return (FLOAT_T)0.299 * color.x + (FLOAT_T)0.587 * color.y + (FLOAT_T)0.144 * color.z;
}

__device__ uchar4 GetPixel(
    uchar4* pImg,
    int x,
    int y,
    int imgw,
    int imgh)
{
    if (x >= 0 && y >= 0 && x < imgw && y < imgh)
    {
        return pImg[y * imgw + x];
    }

    return {0, 0, 0, 0};
}

#if defined(KERNEL_OPT_1) || defined(KERNEL_OPT_2)

__device__ unsigned char GetLuminance(
    uchar4* pImg,
    int x,
    int y,
    int imgw,
    int imgh,
    bool applyBlur
)
{
    if (!applyBlur)
    {
        return RGBtoLuminance(GetPixel(pImg, x, y, imgw, imgh));
    }
    else
    {
        unsigned int value = 0;
        const int radius = 3;
        const auto num = (radius + 1) * (radius + 1);
        for (int dy = -radius; dy <= radius; ++dy)
        {
            for (int dx = -radius; dx <= radius; ++dx)
            {
                value += RGBtoLuminance(GetPixel(pImg, x + dx, y + dy, imgw, imgh)) * ((FLOAT_T)1.0 / num);
            }
        }
        return (unsigned char)value;
    }
}

#elif defined(KERNEL_OPT_3)

__device__ unsigned char GetLuminance(
    uchar4* pImg,
    int x,
    int y,
    int imgw,
    int imgh,
    bool /*applyBlur*/
)
{
    return GetPixel(pImg, x, y, imgw, imgh).x;
}

// gray-scale conversion kernel
// uses pImg as input and writes the result to both pImg and pOut
__global__ void ConvertToLuminance(
    uchar4* pOut,
    uchar4* pImg,
    const int imgw
)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = ty * imgw + tx;

    const auto pixel = pImg[idx];
    const auto luminance = RGBtoLuminance(pixel);
    const auto color = make_uchar4(luminance, luminance, luminance, 255);
    pImg[idx] = color;
    pOut[idx] = color;
}

// 9x9 gaussian blur kernel
// uses pImg as input and writes the result to pOut
__global__ void ApplyBlur(
    uchar4* pOut,
    uchar4* pImg,
    const int imgw,
    const int imgh
)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = ty * imgw + tx;

    unsigned int value = 0;
    const int radius = 3;
    const auto num = (radius + 1) * (radius + 1);
    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int x = tx + dx;
            const int y = ty + dy;
            value += GetPixel(pImg, x, y, imgw, imgh).x * ((FLOAT_T)1.0 / num);
        }
    }

    pOut[idx].x = (unsigned char)value;
}

#endif // KERNEL_OPT_1 || KERNEL_OPT_2

// luminance difference direction for feature detection algorithm
enum LuminanceDiff
{
    DiffUnknown = 0,
    DiffDarker = 1,
    DiffBrighter = 2
};

// random number function using the cuRand library
// returns a value in the range [-1,+1]
__device__ float RandomOffset(curandState_t* pState)
{
    const int num = (curand(pState) % 200) - 100;
    return (float)num / (FLOAT_T)100;
}

// feature detection and extraction kernel
__global__ void ExtractFeatures(
    uchar4* pOut,
    uchar4* pImg,
    const int imgw,
    const int imgh,
    unsigned int* pFeatureVectors,
    bool applyBlur
)
{
    // very stupid circle
    const int LUX[] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};
    const int LUY[] = {0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1};

    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
#if defined(KERNEL_OPT_3)
    const int outIdx = ty * imgw + tx;
#else
    const int outIdx = (imgh - ty - 1) * imgw + tx;
#endif // KERNEL_OPT_3

    const auto luminance = GetLuminance(pImg, tx, ty, imgw, imgh, applyBlur);
    constexpr int threshold = 50;
    constexpr int minContPixels = 12;

    // keypoint extraction:
    // get number of contiguous pixels above or below threshold
    // current difference direction (brighter, darker) is stored as LuminanceDiff enum
    int numPixels = 0;
    LuminanceDiff diff = DiffUnknown;

    // iterate over all pixels of the 'circle'
    for (int p = 0; p < 16; ++p)
    {
        const auto currentLuminance = GetLuminance(pImg, tx + LUX[p], ty + LUY[p], imgw, imgh, applyBlur);

        if (currentLuminance < luminance - threshold)
        {
            if (diff != DiffBrighter)
            {
                diff = DiffDarker;
                ++numPixels;
                if (numPixels == minContPixels)
                {
                    // once we found enough pixels with the same difference direction
                    // at a time, we can stop the check
                    break;
                }
                continue;
            }
        }
        else if (currentLuminance > luminance + threshold)
        {
            if (diff != DiffDarker)
            {
                diff = DiffBrighter;
                ++numPixels;
                if (numPixels == minContPixels)
                {
                    // once we found enough pixels with the same difference direction
                    // at a time, we can stop the check
                    break;
                }
                continue;
            }
        }

        numPixels = 0;
        diff = DiffUnknown;
    }

    // if we have enough contiguous pixels in the circle, we found a feature
    if (numPixels >= minContPixels)
    {
        // feature extraction:
        // atomically select the next feature vector index (fvi)
        const unsigned int fvi = atomicAdd(&pFeatureVectors[0], 1);
        if (fvi >= MaxFeatureVectors)
        {
            // no more features available, mark this in the output image
            pOut[outIdx] = make_uchar4(0, 255, 0, 255);
        }
        else
        {
            // compute the feature vector for this keypoint

            // we want to use the same pseudo-random number sequence for each keypoint,
            // so we initialize the cuRand state every time with the same seeds
            curandState_t randState;
            curand_init(0, 0, 0, &randState);

            // keypoints are randomly selected in a 65x65 search are
            constexpr int patchRadius = 32;

            // from the vector, select the feature assigned to this block
            //               fvi
            //              vvvvv
            // [<num>|..0..|..1..|..2..|..3..|..numTestFeatures-1..]
            //  <--->                   <--->
            //  sizeof(unsigned int)    sizeof(int) * FeatureVectorSizeInts
            unsigned int* pFeature = &pFeatureVectors[1 + FeatureVectorSizeInts * fvi];

            pFeature[0] = tx;
            pFeature[1] = ty;
            for (unsigned int fi = 2; fi < FeatureVectorSizeInts; ++fi)
            {
                unsigned int value = 0;
                for (unsigned int fb = 0; fb < FeatureVectorSizeBitsPerInt; ++fb)
                {
                    const int pxA = patchRadius * RandomOffset(&randState);
                    const int pyA = patchRadius * RandomOffset(&randState);
                    const int pxB = patchRadius * RandomOffset(&randState);
                    const int pyB = patchRadius * RandomOffset(&randState);

                    const auto lumA = GetLuminance(pImg, tx + pxA, ty + pyA, imgw, imgh, applyBlur);
                    const auto lumB = GetLuminance(pImg, tx + pxB, ty + pyB, imgw, imgh, applyBlur);

                    // binary feature function:
                    // if value of keypoint A > keypoint B, feature bit is 1, else 0
                    if (lumA > lumB)
                    {
                        value |= (1 << fb);
                    }
                }

                pFeature[fi] = value;
            }

            // mark it in output image
            pOut[outIdx] = make_uchar4(255, 0, 0, 255);
        }
    }
#if !defined(KERNEL_OPT_3)
    else
    {
        const auto outColor = GetLuminance(pImg, tx, ty, imgw, imgh, false);
        pOut[outIdx] = make_uchar4(outColor, outColor, outColor, 255);
    }
#endif // !KERNEL_OPT_3
}

// feature vector comparison kernel
__global__ void
MatchFeatures(
    const unsigned int* pTest,
    const unsigned int* pBase,
    unsigned int* pResult)
{
    const unsigned int numTestFeatures = pTest[0];
    const unsigned int numBaseFeatures = pBase[0] <= MaxFeatureVectors ? pBase[0] : MaxFeatureVectors;

    // we use one CUDA block per feature
    // if there aren't enough features in the vector, the block exits early
    if (blockIdx.x < numTestFeatures)
    {
        // from the vector, select the feature assigned to this block
        //              vvvvv
        // [<num>|..0..|..1..|..2..|..3..|..numTestFeatures-1..]
        //  <--->                   <--->
        //  sizeof(unsigned int)    sizeof(int) * FeatureVectorSizeInts
        const unsigned int* pTestFeature = &pTest[1 + FeatureVectorSizeInts * blockIdx.x];
        __shared__ unsigned int bestBaseFeature[FeatureVectorSizeInts - 2];

        // the first thread in the block searches the best-matching feature in the base features vector
        if (threadIdx.x == 0)
        {
            const unsigned int* pBaseFeature = nullptr;
            unsigned int testX = pTestFeature[0];
            unsigned int testY = pTestFeature[1];

#if defined(GDB_EXERCISE)
            unsigned int minDiff = 0;
#else
            unsigned int minDiff = UINT_MAX;
#endif // GDB_EXERCISE

            for (unsigned int i = 0; i < numBaseFeatures; ++i)
            {
                const unsigned int baseX = pBase[1 + FeatureVectorSizeInts * i + 0];
                const unsigned int baseY = pBase[1 + FeatureVectorSizeInts * i + 1];

                const unsigned int diff = __usad(baseX, testX, 0) + __usad(baseY, testY, 0);
                if (diff < minDiff)
                {
                    minDiff = diff;
                    pBaseFeature = &pBase[1 + FeatureVectorSizeInts * i + 2];
                }
            }

            for (unsigned int i = 0; i < FeatureVectorSizeInts - 2; ++i)
            {
                bestBaseFeature[i] = pBaseFeature[i];
            }
        }

        // all threads wait for the first thread to finish
        __syncthreads();

        // all threads compute the difference between the two selected features
        const unsigned int valueTest = pTestFeature[2 + threadIdx.x];
        const unsigned int valueBase = bestBaseFeature[threadIdx.x];
        const unsigned int diff = valueTest ^ valueBase;
        const int numBitsDiff = __popc(diff);

        // the number of difference bits is added to the output result
        atomicAdd(pResult, numBitsDiff);
    }
}

struct ProcessResult
{
    uchar4* pImgDevPtr = nullptr;
    unsigned int* pFeaturesDevPtr = nullptr;
};

#if defined(REUSE_DEVICE_BUFFERS)
LockedQueue<uchar4*> g_bufferPool;
#endif //REUSE_DEVICE_BUFFERS

ProcessResult ProcessImage(const uchar4* pHostSrcImage, uchar4* pImgBuffer, unsigned int* pFeaturesBuffer, int width, int height, bool applyBlur = true)
{
    nvtxRangePush("ProcessImage");
    nvtxRangePush("Prepare device");

    const auto imageSize = sizeof(uchar4) * width * height;

    uchar4* pDevSrcImage = pImgBuffer;
    if (!pImgBuffer)
    {
        checkCudaErrors(cudaMalloc((void**)&pDevSrcImage, imageSize));
    }
    checkCudaErrors(cudaMemcpy(pDevSrcImage, pHostSrcImage, imageSize, cudaMemcpyHostToDevice));

    ProcessResult result;
#if defined(REUSE_DEVICE_BUFFERS)
    // try to get one from the buffer pool
    // otherwise, simply allocate (will be put into the pool later)
    if (!g_bufferPool.Empty())
    {
        result.pImgDevPtr = g_bufferPool.Get();
    }

    if (!result.pImgDevPtr)
#endif //REUSE_DEVICE_BUFFERS
    {
        checkCudaErrors(cudaMalloc((void **)&result.pImgDevPtr, imageSize));
    }

    result.pFeaturesDevPtr = pFeaturesBuffer;
    if (!pFeaturesBuffer)
    {
        checkCudaErrors(cudaMalloc((void**)&result.pFeaturesDevPtr, FeaturesMemSize));
    }

    checkCudaErrors(cudaMemset(result.pFeaturesDevPtr, 0, sizeof(unsigned int)));

    nvtxRangePop();

    nvtxRangePush("Run kernels");

    dim3 block(16, 16, 1);
    dim3 grid(width / block.x, height / block.y, 1);

#if defined(KERNEL_OPT_3)
    ConvertToLuminance<<<grid, block>>>(
        result.pImgDevPtr, // out (luminance)
        pDevSrcImage, // in (color), out (luminance)
        width
    );
    checkCudaErrors(cudaGetLastError());

    if (applyBlur)
    {
        ApplyBlur<<<grid, block>>>(
            pDevSrcImage, // out (luminance + blur)
            result.pImgDevPtr, // in (luminance)
            width,
            height
        );
        checkCudaErrors(cudaGetLastError());
    }
#endif // KERNEL_OPT_3

    ExtractFeatures<<<grid, block>>>(
        result.pImgDevPtr,
        pDevSrcImage,
        width,
        height,
        result.pFeaturesDevPtr,
        applyBlur
    );
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaDeviceSynchronize());

    nvtxRangePop();

    nvtxRangePush("Cleanup device");

    if (!pImgBuffer)
    {
        checkCudaErrors(cudaFree(pDevSrcImage));
    }

    nvtxRangePop();
    nvtxRangePop();

    return result;
}

struct WorkerArgs
{
#if defined(CUDA_GRAPHS_2)
    cudaGraph_t graph = 0;
    cudaGraphExec_t graphExec = 0;
    char errorBuffer[256];
#endif

    cudaStream_t comparisonStream;
    unsigned int* pResultsDev = nullptr;
    unsigned int* pResultsHost = nullptr;
};

int CompareFeatures(WorkerArgs& args, unsigned int* pFeaturesDevPtr)
{
    nvtxRangePush("CompareFeatures");

    dim3 grid(MaxFeatureVectors, 1);
    dim3 block(FeatureVectorSizeInts - 2, 1);

    const auto numFeatures = g_imgDescriptors.size();
    const auto resultsSize = sizeof(unsigned int) * numFeatures;

    unsigned int bestResult = UINT_MAX;
    int resultId = -1;

    // dbComplexity is used to simulate a larger feature DB
    // without actually requiring as many reference images
#if defined(CUDA_GRAPHS_1) || defined(CUDA_GRAPHS_2)
    int dbComplexity = 100;
#else
    int dbComplexity = 1;
#endif

    if (!args.pResultsHost)
    {
        args.pResultsHost = new unsigned int[numFeatures];
        checkCudaErrors(cudaMalloc((void**)&args.pResultsDev, resultsSize));
    }

#if defined(CUDA_GRAPHS_2)
    if (!args.graph)
    {
        // first time this is called, create the CUDA graph using the stream capture API
        // we need to have a user-created CUDA stream for this
        cudaStreamCreate(&args.comparisonStream);
        checkCudaErrors(cudaStreamBeginCapture(args.comparisonStream, cudaStreamCaptureModeGlobal));

        for (int c = 0; c < dbComplexity; ++c)
        {
            // for each image in the DB, clear the device result memory, compute the comparison result, and copy it back
            for (size_t i = 0; i < numFeatures; ++i)
            {
                const auto& imgDescr = g_imgDescriptors[i];
                checkCudaErrors(cudaMemcpyAsync(args.pResultsDev + i, args.pResultsHost + i, sizeof(unsigned int), cudaMemcpyHostToDevice, args.comparisonStream));
                MatchFeatures<<<grid, block, 0, args.comparisonStream>>>(pFeaturesDevPtr, imgDescr.pFeaturesDevPtr, args.pResultsDev + i);
                checkCudaErrors(cudaGetLastError());
                checkCudaErrors(cudaMemcpyAsync(args.pResultsHost + i, args.pResultsDev + i, sizeof(unsigned int), cudaMemcpyDeviceToHost, args.comparisonStream));
            }
        }

        checkCudaErrors(cudaStreamEndCapture(args.comparisonStream, &args.graph));

        auto cudaStatus = cudaGraphInstantiate(&args.graphExec, args.graph, nullptr, args.errorBuffer, sizeof(args.errorBuffer));
        if (cudaStatus != cudaSuccess)
        {
            printf("Error instantiating graph: %s\n", args.errorBuffer);
            exit(1);
        }
    }

    // clear results and launch CUDA graph
    memset(args.pResultsHost, 0, resultsSize);
    checkCudaErrors(cudaGraphLaunch(args.graphExec, args.comparisonStream));
    checkCudaErrors(cudaStreamSynchronize(args.comparisonStream));

#else
    // clear result and launch CUDA API calls
    memset(args.pResultsHost, 0, resultsSize);

    for (int c = 0; c < dbComplexity; ++c)
    {
        // for each image in the DB, clear the device result memory, compute the comparison result, and copy it back
        for (size_t i = 0; i < numFeatures; ++i)
        {
            checkCudaErrors(cudaMemcpy(args.pResultsDev + i, args.pResultsHost + i, sizeof(unsigned int), cudaMemcpyHostToDevice));
            const auto& imgDescr = g_imgDescriptors[i];
            MatchFeatures<<<grid, block>>>(pFeaturesDevPtr, imgDescr.pFeaturesDevPtr, args.pResultsDev + i);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaMemcpy(args.pResultsHost + i, args.pResultsDev + i, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
    }
#endif

    for (size_t i = 0; i < numFeatures; ++i)
    {
        const auto result = args.pResultsHost[i];
        if (result < bestResult)
        {
            resultId = (int)i;
            bestResult = result;
        }
    }

    nvtxRangePop();

    return resultId;
}

void WorkerThread(
    HostQueue& srcQueue,
    CudaQueue& dstQueue,
    WorkerArgs& args)
{
#ifdef __linux__
    nvtxNameOsThread(syscall(SYS_gettid), "Worker");
#endif

    const auto width = g_pCudaRenderBuffer->ImgWidth();
    const auto height = g_pCudaRenderBuffer->ImgHeight();

#if !defined(NO_EXTRACTION)
    uchar4* pTmpImage = nullptr;
    unsigned int* pTmpFeatures = nullptr;
#endif //NO_EXTRACTION

#if defined(REUSE_DEVICE_BUFFERS)
    checkCudaErrors(cudaMalloc((void**)&pTmpImage, sizeof(uchar4) * width * height));
    checkCudaErrors(cudaMalloc((void**)&pTmpFeatures, FeaturesMemSize));
#endif // REUSE_DEVICE_BUFFERS

    while (!gExitFlag)
    {
        // pick the next source image
        auto hostImage = srcQueue.Get();
        if (hostImage.IsValid())
        {
            nvtxRangePush(("Process & Copy Image: " + std::to_string(hostImage.GetInstanceId())).c_str());
#if defined(NO_EXTRACTION)
            int dstImgId = hostImage.GetId();
            uchar4* pDstImage = nullptr;
            const auto imgSize = sizeof(uchar4) * width * height;
            checkCudaErrors(cudaMalloc((void**)&pDstImage, imgSize));
            checkCudaErrors(cudaMemcpy(pDstImage, hostImage.GetPtr(), imgSize, cudaMemcpyHostToDevice));

            printf("Received %d\n", dstImgId);
#else
            auto result = ProcessImage(hostImage.GetPtr(), pTmpImage, pTmpFeatures, width, height);

            int dstImgId = CompareFeatures(args, result.pFeaturesDevPtr);
            auto* pDstImage = result.pImgDevPtr;

#if !defined(REUSE_DEVICE_BUFFERS)
            checkCudaErrors(cudaFree(result.pFeaturesDevPtr));
#endif // REUSE_DEVICE_BUFFERS

            const int expectedId = hostImage.GetId();
            printf("%llu: Received %d, Classified as %d - %s\n",
                (long long unsigned)hostImage.GetInstanceId(),
                expectedId,
                dstImgId,
                expectedId == dstImgId
                ? "MATCH"
                : "FAIL");

            if (dstImgId != expectedId)
            {
                dstImgId = -1;
            }
#endif // NO_EXTRACTION

            nvtxRangePop();
            nvtxRangePush("Pass to Render");

            dstQueue.Put([pDstImage, dstImgId, &hostImage]() {
                auto cudaImage = CudaImage<uchar4>(pDstImage, dstImgId, hostImage.GetTime(), hostImage.GetInstanceId(), hostImage.GetLifetimeRange());
#if defined(REUSE_DEVICE_BUFFERS)
                cudaImage.SetQueue(&g_bufferPool);
#endif //REUSE_DEVICE_BUFFERS
                return cudaImage;
            });

            nvtxRangePop();
        }
    }

#if defined(REUSE_DEVICE_BUFFERS)
        checkCudaErrors(cudaFree(pTmpImage));
        checkCudaErrors(cudaFree(pTmpFeatures));
#endif //REUSE_DEVICE_BUFFERS
}


// ****************** Source (client) thread *************************************

void SourceThread(std::atomic<size_t>& instanceId, std::vector<uchar4*>& images, LockedQueue<HostImage<uchar4>>& imgQueue)
{
#ifdef __linux__
    nvtxNameOsThread(syscall(SYS_gettid), "Source");
#endif

    while (!gExitFlag)
    {
        size_t currentInstanceId = instanceId.fetch_add(1);
        if (currentInstanceId > MaxImages)
        {
            gExitFlag = 1;
            return;
        }

        while (!gExitFlag && !gNextImgFlag)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        if (gExitFlag)
        {
            return;
        }

        gNextImgFlag = 0;
        nvtxRangePush(("New Image: " + std::to_string(currentInstanceId)).c_str());

        const int imgId = std::rand() % images.size();
        auto* pImg = images[imgId];
        imgQueue.Put([=](){
            nvtxRangeId_t lifetimeRange = nvtxRangeStart(("Image Lifetime: " + std::to_string(currentInstanceId)).c_str());
            auto startTime = system_clock::now();
            return HostImage<uchar4>(pImg, imgId, startTime, currentInstanceId, lifetimeRange);
        });

        nvtxRangePop();
    }
}


// ****************** Main program *************************************

void SetBlockingWaitIfNeeded()
{
    char *value = std::getenv("BLOCKING_WAIT");
    if (value && std::string(value) == "1")
    {
        unsigned int deviceFlags = 0;
        checkCudaErrors(cudaGetDeviceFlags(&deviceFlags));
        checkCudaErrors(cudaSetDeviceFlags(deviceFlags | cudaDeviceScheduleBlockingSync));
    }
}

void RunProgram(int argc, char **argv)
{
#if !defined(NO_GL)
    initGLBuffers();

    signal(SIGINT, handle_sigint);

    // start rendering mainloop
    nvtxRangePush("OpenGL Main Loop");
    glutMainLoop();
    nvtxRangePop();
#else
    signal(SIGINT, handle_sigint);
    while (!gExitFlag)
    {
        g_pCudaRenderBuffer->Render();
        std::this_thread::sleep_for(std::chrono::milliseconds(REFRESH_DELAY));
    }
#endif //!NO_GL
}

int main(int argc, char** argv)
{
#ifdef __linux__
    // main thread is also used for OpenGL rendering
    nvtxNameOsThread(syscall(SYS_gettid), "Render");
#endif

#if !defined(STEP)
    #error Select exercise step with "step=x"
#else
    printf("Exercise step %d\n", STEP);
#endif

    std::srand(0);

    if (argc > 1)
    {
        ImgLateThreshold = std::atoi(argv[1]);

        if (argc > 2)
        {
            MaxImages = std::atoi(argv[2]);
        }
    }

    unsigned int imgWidth = 0;
    unsigned int imgHeight = 0;

    nvtxRangePush("Loading images from disk");
    constexpr unsigned int numImages = 10;
    std::vector<uchar4*> files;
    char fileName[256];
    files.reserve(numImages);
    for (unsigned int i = 1; i <= numImages; ++i)
    {
        std::snprintf(fileName, sizeof(fileName) - 1, "../img/image_%05u.bmp", i);
        auto* pImage = LoadBmpFile<uchar4>(imgWidth, imgHeight, fileName);
        if (pImage)
        {
            files.push_back(pImage);
        }
    }

    if (files.empty())
    {
        printf("Failed to load any input images\n");
        return 1;
    }

    auto* pSmiley = LoadBmpFile<uchar4>(imgWidth, imgHeight, "../img/smiley.bmp");
    nvtxRangePop();

#if !defined(NO_GL)
    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    if (false == initGL(&argc, argv))
    {
        return 1;
    }
#endif // !NO_GL

    // Now initialize CUDA context
    cudaSetDevice(0);

    SetBlockingWaitIfNeeded();

    HostQueue srcQueue(1);
    CudaQueue dstQueue(1);

    g_pCudaRenderBuffer.reset(new CudaRenderBuffer(imgWidth, imgHeight, 8, 4, dstQueue, pSmiley));

    nvtxRangePush("Compute Features DB");
    g_imgDescriptors.reserve(files.size());

    uchar4* pTmpImage = nullptr;
#if defined(REUSE_DEVICE_BUFFERS)
    checkCudaErrors(cudaMalloc((void**)&pTmpImage, sizeof(uchar4) * imgWidth * imgHeight));
#endif // REUSE_DEVICE_BUFFERS

    for (int i = 0; i < files.size(); ++i)
    {
        ProcessResult result = ProcessImage(files[i], pTmpImage, nullptr, imgWidth, imgHeight);
        g_imgDescriptors.emplace_back(ImageDescriptor(i, result.pFeaturesDevPtr));

        // introduce some white noise into the client data
        const float noiseRatio = (std::rand() % 5) / 100.;
        for (int n = 0; n < noiseRatio * imgWidth * imgHeight; ++n)
        {
            const int noiseIdx = std::rand() % (imgWidth * imgHeight);
            uchar4& color = files[i][noiseIdx];
            color.x = 255;
            color.y = 255;
            color.z = 255;
        }
    }

#if defined(REUSE_DEVICE_BUFFERS)
    checkCudaErrors(cudaFree(pTmpImage));
#endif // REUSE_DEVICE_BUFFERS
    checkCudaErrors(cudaDeviceSynchronize());
    nvtxRangePop();

    std::atomic<size_t> instanceId(0);

    std::thread sourceThread(
        SourceThread,
        std::ref(instanceId),
        std::ref(files),
        std::ref(srcQueue));

    WorkerArgs workerArgs;
    std::thread workerThread(
        WorkerThread,
        std::ref(srcQueue),
        std::ref(dstQueue),
        std::ref(workerArgs));

    RunProgram(argc, argv);
    gExitFlag = 1;

    sourceThread.join();
    workerThread.join();

    for (auto* pImg : files)
    {
        delete[] pImg;
    }
    delete[] pSmiley;

    g_pCudaRenderBuffer.reset();

    for (auto& descr : g_imgDescriptors)
    {
        checkCudaErrors(cudaFree(descr.pFeaturesDevPtr));
    }

#if defined(REUSE_DEVICE_BUFFERS)
    g_bufferPool.Clear();
#endif // REUSE_DEVICE_BUFFERS

    return 0;
}
