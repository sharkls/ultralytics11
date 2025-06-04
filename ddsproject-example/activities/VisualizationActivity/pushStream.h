#pragma once

#include <iostream>
#include <vector>
// #include <opencv2/highgui.hpp>
// #include <opencv2/opencv.hpp>
// #include <opencv2/video.hpp>
// #include "PutText.h"

extern "C"
{
// #include </usr/include/x86_64-linux-gnu/libavformat/avformat.h>
// #include </usr/include/x86_64-linux-gnu/libavcodec/avcodec.h>
// #include </usr/include/x86_64-linux-gnu/libavutil/imgutils.h>
// #include </usr/include/x86_64-linux-gnu/libswscale/swscale.h>

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

class PushStream
{
public:
	PushStream(int width, int height);
	~PushStream();

	void init(int bitrate, const std::string& codec_profile, const std::string& target_rtmp, int fps);
	void final();
	void push(uint8_t *pData, const int _stride);

	int m_width;
	int m_height;
private:
	void initialize_io_context(AVFormatContext *&fctx, const char *output);

	void initialize_avformat_context(AVFormatContext*& fctx, const char* format_name, const char * url);
	void set_codec_params(AVFormatContext*& fctx, AVCodecContext*& codec_ctx, int fps, int bitrate);
	void initialize_codec_stream(AVStream*& stream, AVCodecContext*& codec_ctx, const AVCodec*& codec, std::string codec_profile);
	SwsContext* initialize_sample_scaler(AVCodecContext* codec_ctx);
	AVFrame* allocate_frame_buffer(AVCodecContext* codec_ctx);
	void write_frame(AVCodecContext* codec_ctx, AVFormatContext* fmt_ctx, AVFrame* frame);


private:
	SwsContext *swsctx;
	AVFrame *frame;
	AVFormatContext *ofmt_ctx;
	AVCodecContext *out_codec_ctx;
	AVStream *out_stream;


};

