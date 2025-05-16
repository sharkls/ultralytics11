/*******************************************************
 文件名：BaseTrack.cpp
 作者：sharkls
 描述：跟踪模块基类
 版本：v1.0
 日期：2025-05-14
 *******************************************************/

#include "BaseTrack.h"

int BaseTrack::count = 0;

BaseTrack::BaseTrack() : track_id(0), is_activated(false), state(TrackState::New), start_frame(0), frame_id(0) {}

int BaseTrack::next_id() { return ++count; }

void BaseTrack::mark_lost() { state = TrackState::Lost; }

void BaseTrack::mark_removed() { state = TrackState::Removed; }

void BaseTrack::reset_id() { count = 0; } 
