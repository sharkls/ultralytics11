#include "pushStream.h"


PushStream::PushStream(int width, int height):
    m_width(width), m_height(height)
{
}

PushStream::~PushStream()
{
}

void PushStream::initialize_avformat_context(AVFormatContext *&fctx, const char *format_name, const char * url)
{

  // fctx = avformat_alloc_context();
  // AVOutputFormat *output_format = nullptr;
    
  //   // 分配输出上下文（旧方法）
  //   output_format = av_guess_format(nullptr, url, nullptr);
  //   if (!output_format) {
  //       std::cerr << "无法猜测输出格式" << std::endl;
  //       return;
  //   }

  //   fctx->oformat = output_format;


    int ret = avformat_alloc_output_context2(&fctx, nullptr, format_name, url);
    if (ret < 0)
    {
        std::cout << "Could not allocate output format context!" << std::endl;
        exit(1);
    }
}

void PushStream::initialize_io_context(AVFormatContext *&fctx, const char *output)
{
  if (!(fctx->oformat->flags & AVFMT_NOFILE))
  {
    int ret = avio_open2(&fctx->pb, output, AVIO_FLAG_WRITE, nullptr, nullptr);
    if (ret < 0)
    {
      std::cout << "Could not open output IO context!" << std::endl;
      exit(1);
    }
  }
}

void PushStream::set_codec_params(AVFormatContext *&fctx, AVCodecContext *&codec_ctx, int fps, int bitrate)
{
  const AVRational dst_fps = {fps, 1};

  codec_ctx->codec_tag = 0;
  codec_ctx->codec_id = AV_CODEC_ID_H264;
  codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
  codec_ctx->width = m_width;
  codec_ctx->height = m_height;
  codec_ctx->gop_size = 12;
  codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  codec_ctx->framerate = dst_fps;
  codec_ctx->time_base = av_inv_q(dst_fps);
  codec_ctx->bit_rate = bitrate;
  if (fctx->oformat->flags & AVFMT_GLOBALHEADER)
  {
    codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }
}

void PushStream::initialize_codec_stream(AVStream *&stream, AVCodecContext *&codec_ctx, const AVCodec *&codec, std::string codec_profile)
{
  int ret = avcodec_parameters_from_context(stream->codecpar, codec_ctx);
  if (ret < 0)
  {
    std::cout << "Could not initialize stream codec parameters!" << std::endl;
    exit(1);
  }

  AVDictionary *codec_options = nullptr;
  av_dict_set(&codec_options, "profile", codec_profile.c_str(), 0);
  av_dict_set(&codec_options, "preset",
              "medium"
              // "superfast"
              , 0);
  av_dict_set(&codec_options, "tune", "zerolatency", 0);

  // open video encoder
  ret = avcodec_open2(codec_ctx, codec, &codec_options);
  if (ret < 0)
  {
    std::cout << "Could not open video encoder!" << std::endl;
    exit(1);
  }
}

SwsContext *PushStream::initialize_sample_scaler(AVCodecContext *codec_ctx)
{
  SwsContext *swsctx = sws_getContext(m_width, m_height, AV_PIX_FMT_BGR24, m_width, m_height, codec_ctx->pix_fmt, SWS_BICUBIC, nullptr, nullptr, nullptr);
  if (!swsctx)
  {
    std::cout << "Could not initialize sample scaler!" << std::endl;
    exit(1);
  }

  return swsctx;
}

AVFrame *PushStream::allocate_frame_buffer(AVCodecContext *codec_ctx)
{
  AVFrame *frame = av_frame_alloc();
  int i = av_image_get_buffer_size(codec_ctx->pix_fmt, m_width, m_height, 1);
  uint8_t *framebuf = new uint8_t[i];

  av_image_fill_arrays(frame->data, frame->linesize, framebuf, codec_ctx->pix_fmt, m_width, m_height, 1);
  frame->width = m_width;
  frame->height = m_height;
  frame->format = static_cast<int>(codec_ctx->pix_fmt);

  return frame;
}

void PushStream::write_frame(AVCodecContext *codec_ctx, AVFormatContext *fmt_ctx, AVFrame *frame)
{
  AVPacket pkt = {0};
  av_new_packet(&pkt, 0);

  int ret = avcodec_send_frame(codec_ctx, frame);
  if (ret < 0)
  {
    std::cout << "Error sending frame to codec context!" << std::endl;
    exit(1);
  }

  ret = avcodec_receive_packet(codec_ctx, &pkt);
  if (ret < 0)
  {
    std::cout << "Error receiving packet from codec context!" << std::endl;
    exit(1);
  }

  av_interleaved_write_frame(fmt_ctx, &pkt);
  av_packet_unref(&pkt);
}

// int64_t rescale_q(int64_t src_ts, 
//                   int src_num, int src_den, 
//                   int dst_num, int dst_den) {
//     if (src_ts == AV_NOPTS_VALUE) {
//         return AV_NOPTS_VALUE;
//     }
//     // 避免整数溢出的安全计算方式
//     return (src_ts * (int64_t)src_num * dst_den) / (src_den * (int64_t)dst_num);
// }

void PushStream::push(uint8_t *pData, const int _stride) // (cv::Mat mat)
{
    clock_t start, end;
    start = clock();

    const int stride[] = {_stride};
    sws_scale(swsctx, &pData, stride, 0, m_height, frame->data, frame->linesize);
    frame->pts += av_rescale_q(1, out_codec_ctx->time_base, out_stream->time_base);
    write_frame(out_codec_ctx, ofmt_ctx, frame);
    end = clock();
    std::cout << "one frame time:" << double(end - start)/ CLOCKS_PER_SEC << "\r" << std::endl;
}

void PushStream::init(int bitrate, const std::string& codec_profile, const std::string& target_rtmp, int fps) //, int width, int height)
{
  std::cout << "start push stream\n";
  #if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 9, 100)
      av_register_all();
  #endif
      avformat_network_init();
  
      const char *output = target_rtmp.c_str();
      int ret;
    
      std::vector<uint8_t> imgbuf(m_height * m_width * 3 + 16);
      
      ofmt_ctx = nullptr;
      const AVCodec *out_codec = nullptr;
      out_stream = nullptr;
      out_codec_ctx = nullptr;
    
      // initialize_avformat_context(ofmt_ctx, "flv");
      initialize_avformat_context(ofmt_ctx, "rtsp", output);
      // initialize_io_context(ofmt_ctx, output);
    
      out_codec = avcodec_find_encoder(AV_CODEC_ID_H264);
      out_stream = avformat_new_stream(ofmt_ctx, out_codec);
      out_codec_ctx = avcodec_alloc_context3(out_codec);
    
      set_codec_params(ofmt_ctx, out_codec_ctx, fps, bitrate);
      initialize_codec_stream(out_stream, out_codec_ctx, out_codec, codec_profile);
    
      out_stream->codecpar->extradata = out_codec_ctx->extradata;
      out_stream->codecpar->extradata_size = out_codec_ctx->extradata_size;
    
      av_dump_format(ofmt_ctx, 0, output, 1);
    
      swsctx = initialize_sample_scaler(out_codec_ctx);//, width, height);
      frame = allocate_frame_buffer(out_codec_ctx);//, width, height);
    
      int cur_size;
      uint8_t *cur_ptr;
    
      ret = avformat_write_header(ofmt_ctx, nullptr);
      if (ret < 0)
      {
        std::cout << "Could not write header!" << std::endl;
        exit(1);
      }
      clock_t start, end;
}

void PushStream::final()
{
    av_write_trailer(ofmt_ctx);
    av_frame_free(&frame);
    avcodec_close(out_codec_ctx);
    avio_close(ofmt_ctx->pb);
    avformat_free_context(ofmt_ctx);
}

