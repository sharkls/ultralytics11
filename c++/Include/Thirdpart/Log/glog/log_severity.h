// Copyright (c) 2024, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef BASE_LOG_SEVERITY_H__
#define BASE_LOG_SEVERITY_H__

#if defined(GLOG_USE_GLOG_EXPORT)
// #  include "glog/export.h"
#  include "export.h"
#endif

#if !defined(GLOG_EXPORT)
#  error <log_severity.h> was not included correctly. See the documentation for how to consume the library.
// #  error <glog/log_severity.h> was not included correctly. See the documentation for how to consume the library.
#endif

namespace google {

// The recommended semantics of the log levels are as follows:
//
// INFO:
//   Use for state changes or other major events, or to aid debugging.
// WARNING:
//   Use for undesired but relatively expected events, which may indicate a
//   problem
// ERROR:
//   Use for undesired and unexpected events that the program can recover from.
//   All ERRORs should be actionable - it should be appropriate to file a bug
//   whenever an ERROR occurs in production.
// FATAL:
//   Use for undesired and unexpected events that the program cannot recover
//   from.

// Variables of type LogSeverity are widely taken to lie in the range
// [0, NUM_SEVERITIES-1].  Be careful to preserve this assumption if
// you ever need to change their values or add a new severity.

enum LogSeverity {
  GLOG_INFO = 0,
  GLOG_WARNING = 1,
  GLOG_ERROR = 2,
  GLOG_FATAL = 3,
#ifndef GLOG_NO_ABBREVIATED_SEVERITIES
#  ifdef ERROR
#  error ERROR macro is defined. Define GLOG_NO_ABBREVIATED_SEVERITIES before including logging.h. See the document for detail.
#  endif
  INFO = GLOG_INFO,
  WARNING = GLOG_WARNING,
  ERROR = GLOG_ERROR,
  FATAL = GLOG_FATAL
#endif
};

#if defined(__cpp_inline_variables)
#  if (__cpp_inline_variables >= 201606L)
#    define GLOG_INLINE_VARIABLE inline
#  endif  // (__cpp_inline_variables >= 201606L)
#endif    // defined(__cpp_inline_variables)

#if !defined(GLOG_INLINE_VARIABLE)
#  define GLOG_INLINE_VARIABLE
#endif  // !defined(GLOG_INLINE_VARIABLE)

GLOG_INLINE_VARIABLE
constexpr int NUM_SEVERITIES = 4;

// DFATAL is FATAL in debug mode, ERROR in normal mode
#ifdef NDEBUG
#  define DFATAL_LEVEL ERROR
#else
#  define DFATAL_LEVEL FATAL
#endif

// NDEBUG usage helpers related to (RAW_)DCHECK:
//
// DEBUG_MODE is for small !NDEBUG uses like
//   if (DEBUG_MODE) foo.CheckThatFoo();
// instead of substantially more verbose
//   #ifndef NDEBUG
//     foo.CheckThatFoo();
//   #endif
//
// IF_DEBUG_MODE is for small !NDEBUG uses like
//   IF_DEBUG_MODE( string error; )
//   DCHECK(Foo(&error)) << error;
// instead of substantially more verbose
//   #ifndef NDEBUG
//     string error;
//     DCHECK(Foo(&error)) << error;
//   #endif
//
#ifdef NDEBUG
enum { DEBUG_MODE = 0 };
#  define IF_DEBUG_MODE(x)
#else
enum { DEBUG_MODE = 1 };
#  define IF_DEBUG_MODE(x) x
#endif

} // namespace google

#endif  // BASE_LOG_SEVERITY_H__
