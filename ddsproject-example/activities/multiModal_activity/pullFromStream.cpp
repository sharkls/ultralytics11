#include "pullFromStream.h"
#include <iostream>
#include <chrono>

long long getTimeStamp()
{
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

int count = 0;
long long tmpTimeStamp = 0;

int PullFromStream::decode(std::function<void(char* data, const int width, const int height)> callback)
{
    int ret;
    char buf[1024];
    //把数据包发送到解码器上下文
    ret = avcodec_send_packet(ctx, pkt);

    if (ret < 0)
    {
        av_log(NULL, AV_LOG_ERROR, "Failed to send pkt to decoder\n");
        return ret;
    }

    while (ret >= 0)
    {
        //解码后的数据存储到frame中
        ret = avcodec_receive_frame(ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
        {
            return 0;
        }
        else if (ret < 0)
        {
            av_log(NULL, AV_LOG_ERROR, "Error during decoding\n");
            return ret;
        }



        SwsContext *sws_ctx = sws_getContext(frame->width, frame->height, AV_PIX_FMT_YUV420P,  frame->width, frame->height, AV_PIX_FMT_BGR24, SWS_BICUBIC, nullptr, nullptr, nullptr);




        AVFrame *rgb_frame  = av_frame_alloc();

        int i = av_image_get_buffer_size(AV_PIX_FMT_BGR24, frame->width, frame->height, 1);
        uint8_t *framebuf = new uint8_t[i];

        av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize, framebuf, AV_PIX_FMT_BGR24, frame->width, frame->height, 1);


        rgb_frame->width = frame->width;
        rgb_frame->height = frame->height;
        rgb_frame->format = (int)AV_PIX_FMT_BGR24;

        sws_scale(
            sws_ctx,
            frame->data, frame->linesize,
            0, frame->height,
            rgb_frame->data, rgb_frame->linesize
        );

        
        callback((char *)rgb_frame->data[0], rgb_frame->width, rgb_frame->height);





        //生成图片文件名字
        // snprintf(buf, sizeof(buf), "%s-%d.pgm", filename, ctx->frame_number);
        //保存图片
        // savepic(frame->data[0], frame->linesize[0], frame->width, frame->height, buf);

        // savepic_bgr(rgb_frame->data[0], rgb_frame->linesize[0], rgb_frame->width, rgb_frame->height, buf);

        // count++;
        // if (getTimeStamp() - tmpTimeStamp > 1000)
        // {
        //     std::cout << "================fps: " << count << std::endl;

        //     tmpTimeStamp = getTimeStamp();
        //     count = 0;
        // }

        sws_freeContext(sws_ctx);
        av_frame_free(&rgb_frame);
        free(framebuf);
    }
    return 0;
}

PullFromStream::PullFromStream()
{
}

PullFromStream::~PullFromStream()
{
}

bool PullFromStream::init(const std::string& url)
{
    av_register_all();
    
    
    
    
    
    
    
    avformat_network_init();

    
    // AVFormatContext *pFctx = nullptr;
    // AVCodecContext *ctx = NULL;
    // AVFrame *frame = NULL;
    // AVPacket *pkt = NULL;

    AVDictionary* options = nullptr;
    av_dict_set(&options, "stimeout", "30000000", 0);
    av_dict_set(&options, "max_delay", "500000", 0);  // 设置最大延迟
    av_dict_set(&options, "buffer_size", "10240000", 0);  // 设置缓冲区大小
    

    av_log_set_level(AV_LOG_DEBUG);

    int ret;
    //打开源路径
    ret = avformat_open_input(&pFctx, url.c_str(), nullptr, &options);
    if (ret < 0)
    {
        std::cout << "avformat_open_input returns " << ret << std::endl;
        av_log(NULL, AV_LOG_ERROR, "%s\n", (ret));

        return false;
    }
    //查看是否存在流信息
    ret = avformat_find_stream_info(pFctx, NULL);
    if (ret < 0)
    {
        av_log(NULL, AV_LOG_ERROR, "Failed to retrieve input stream information\n");
        return false;
    }
    //寻找视频流
    idx = av_find_best_stream(pFctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (idx < 0)
    {
        av_log(NULL, AV_LOG_ERROR, "Don't find best video stream\n");
        return false;
    }
    //寻找视频流的编码器
    AVStream *inStream = pFctx->streams[idx];
    //设置视频流的解码器
    AVCodec *codec = (AVCodec *)avcodec_find_decoder(inStream->codecpar->codec_id);
    if (!codec)
    {
        av_log(NULL, AV_LOG_ERROR, "could not find codec\n");
        return false;
    }
    //定义解码器上下文
    ctx = avcodec_alloc_context3(codec);
    if (!ctx)
    {
        av_log(NULL, AV_LOG_ERROR, "NO MEMORY\n");
        return false;
    }


    //对解码器上下文初始化
    ret = avcodec_parameters_to_context(ctx, inStream->codecpar);
    if (ret < 0)
    {
        av_log(NULL, AV_LOG_ERROR, "Failed to copy codec parameters to decoder context\n");
        return false;
    }


    // ctx->pix_fmt = AV_PIX_FMT_RGB24; //AV_PIX_FMT_YUV420P;


    //把解码器上下文和解码器绑定，正式启动解码器
    ret = avcodec_open2(ctx, codec, NULL);
    if (ret < 0)
    {
        av_log(NULL, AV_LOG_ERROR, "Could not open codec\n");
        return false;
    }
    //定义AVFrame
    frame = av_frame_alloc();
    if (!frame)
    {
        av_log(NULL, AV_LOG_ERROR, "no memory\n");
        return false;
    }
    //定义AVPacket
    pkt = av_packet_alloc();
    if (!pkt)
    {
        av_log(NULL, AV_LOG_ERROR, "no memory\n");
        return false;
    }

    return true;
}

void PullFromStream::pull(std::function<void(char* data, const int width, const int height)> callback)
{
    //从输入数据中读取压缩数据包
    while (av_read_frame(pFctx, pkt) >= 0)
    {
        //查看是否是视频流的id
        if (pkt->stream_index == idx)
        {
            //解码数据包
            int ret = decode(/*ctx, frame, pkt,*/ callback);
            if (ret < 0)
            {
                break;
            }
        }
        //释放pkt当中的数据，并且重置以便重用
        av_packet_unref(pkt);
    }
    //确保所有已解码但尚未输出的帧都被正确处理
    // decode(ctx, frame, NULL, callback);
}

void PullFromStream::final()
{
    if (pFctx)
    {
        avformat_close_input(&pFctx);
    }

    if (ctx)
    {
        avcodec_free_context(&ctx);
    }

    if (frame)
    {
        av_frame_free(&frame);
    }

    if (pkt)
    {
        av_packet_free(&pkt);
    }
}

