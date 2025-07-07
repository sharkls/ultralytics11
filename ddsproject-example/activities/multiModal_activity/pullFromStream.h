#pragma once

#include <string>
#include <functional>

extern "C"
{
#define __STDC_CONSTANT_MACROS

#include <libavutil/log.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavformat/avformat.h>

#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

class PullFromStream
{
public:
    PullFromStream();
    ~PullFromStream();
    
    bool init(const std::string& url, const bool GPUAccel = false);
    void final();
    void pull(std::function<void(char* data, const int width, const int height)> callback);

private:
    int decode(std::function<void(char* data, const int width, const int height)> callback);

    AVFormatContext *pFctx {nullptr};
    AVCodecContext *ctx {nullptr};
    AVFrame *hwFrame {nullptr};
    AVFrame *frame {nullptr};
    AVPacket *pkt {nullptr};

    int idx{0};
    bool m_GPUAccel {false};
};


