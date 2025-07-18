// Generated by the protocol buffer compiler.  DO NOT EDIT!
// NO CHECKED-IN PROTOBUF GENCODE
// source: ObjectDetectionActivity.proto
// Protobuf C++ Version: 5.30.0-dev

#ifndef ObjectDetectionActivity_2eproto_2epb_2eh
#define ObjectDetectionActivity_2eproto_2epb_2eh

#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include "google/protobuf/runtime_version.h"
#if PROTOBUF_VERSION != 5030000
#error "Protobuf C++ gencode is built with an incompatible version of"
#error "Protobuf C++ headers/runtime. See"
#error "https://protobuf.dev/support/cross-version-runtime-guarantee/#cpp"
#endif
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/arena.h"
#include "google/protobuf/arenastring.h"
#include "google/protobuf/generated_message_tctable_decl.h"
#include "google/protobuf/generated_message_util.h"
#include "google/protobuf/metadata_lite.h"
#include "google/protobuf/generated_message_reflection.h"
#include "google/protobuf/message.h"
#include "google/protobuf/message_lite.h"
#include "google/protobuf/repeated_field.h"  // IWYU pragma: export
#include "google/protobuf/extension_set.h"  // IWYU pragma: export
#include "google/protobuf/unknown_field_set.h"
// @@protoc_insertion_point(includes)

// Must be included last.
#include "google/protobuf/port_def.inc"

#define PROTOBUF_INTERNAL_EXPORT_ObjectDetectionActivity_2eproto

namespace google {
namespace protobuf {
namespace internal {
template <typename T>
::absl::string_view GetAnyMessageName();
}  // namespace internal
}  // namespace protobuf
}  // namespace google

// Internal implementation detail -- do not use these members.
struct TableStruct_ObjectDetectionActivity_2eproto {
  static const ::uint32_t offsets[];
};
extern "C" {
extern const ::google::protobuf::internal::DescriptorTable descriptor_table_ObjectDetectionActivity_2eproto;
}  // extern "C"
class Config;
struct ConfigDefaultTypeInternal;
extern ConfigDefaultTypeInternal _Config_default_instance_;
extern const ::google::protobuf::internal::ClassDataFull Config_class_data_;
class TopicConfig;
struct TopicConfigDefaultTypeInternal;
extern TopicConfigDefaultTypeInternal _TopicConfig_default_instance_;
extern const ::google::protobuf::internal::ClassDataFull TopicConfig_class_data_;
namespace google {
namespace protobuf {
}  // namespace protobuf
}  // namespace google


// ===================================================================


// -------------------------------------------------------------------

class TopicConfig final : public ::google::protobuf::Message
/* @@protoc_insertion_point(class_definition:TopicConfig) */ {
 public:
  inline TopicConfig() : TopicConfig(nullptr) {}
  ~TopicConfig() PROTOBUF_FINAL;

#if defined(PROTOBUF_CUSTOM_VTABLE)
  void operator delete(TopicConfig* PROTOBUF_NONNULL msg, std::destroying_delete_t) {
    SharedDtor(*msg);
    ::google::protobuf::internal::SizedDelete(msg, sizeof(TopicConfig));
  }
#endif

  template <typename = void>
  explicit PROTOBUF_CONSTEXPR TopicConfig(::google::protobuf::internal::ConstantInitialized);

  inline TopicConfig(const TopicConfig& from) : TopicConfig(nullptr, from) {}
  inline TopicConfig(TopicConfig&& from) noexcept
      : TopicConfig(nullptr, std::move(from)) {}
  inline TopicConfig& operator=(const TopicConfig& from) {
    CopyFrom(from);
    return *this;
  }
  inline TopicConfig& operator=(TopicConfig&& from) noexcept {
    if (this == &from) return *this;
    if (::google::protobuf::internal::CanMoveWithInternalSwap(GetArena(), from.GetArena())) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return _internal_metadata_.unknown_fields<::google::protobuf::UnknownFieldSet>(::google::protobuf::UnknownFieldSet::default_instance);
  }
  inline ::google::protobuf::UnknownFieldSet* PROTOBUF_NONNULL mutable_unknown_fields()
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return _internal_metadata_.mutable_unknown_fields<::google::protobuf::UnknownFieldSet>();
  }

  static const ::google::protobuf::Descriptor* PROTOBUF_NONNULL descriptor() {
    return GetDescriptor();
  }
  static const ::google::protobuf::Descriptor* PROTOBUF_NONNULL GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::google::protobuf::Reflection* PROTOBUF_NONNULL GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const TopicConfig& default_instance() {
    return *reinterpret_cast<const TopicConfig*>(
        &_TopicConfig_default_instance_);
  }
  static constexpr int kIndexInFileMessages = 0;
  friend void swap(TopicConfig& a, TopicConfig& b) { a.Swap(&b); }
  inline void Swap(TopicConfig* PROTOBUF_NONNULL other) {
    if (other == this) return;
    if (::google::protobuf::internal::CanUseInternalSwap(GetArena(), other->GetArena())) {
      InternalSwap(other);
    } else {
      ::google::protobuf::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(TopicConfig* PROTOBUF_NONNULL other) {
    if (other == this) return;
    ABSL_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  TopicConfig* PROTOBUF_NONNULL New(::google::protobuf::Arena* PROTOBUF_NULLABLE arena = nullptr) const {
    return ::google::protobuf::Message::DefaultConstruct<TopicConfig>(arena);
  }
  using ::google::protobuf::Message::CopyFrom;
  void CopyFrom(const TopicConfig& from);
  using ::google::protobuf::Message::MergeFrom;
  void MergeFrom(const TopicConfig& from) { TopicConfig::MergeImpl(*this, from); }

  private:
  static void MergeImpl(::google::protobuf::MessageLite& to_msg,
                        const ::google::protobuf::MessageLite& from_msg);

  public:
  bool IsInitialized() const {
    return true;
  }
  ABSL_ATTRIBUTE_REINITIALIZES void Clear() PROTOBUF_FINAL;
  #if defined(PROTOBUF_CUSTOM_VTABLE)
  private:
  static ::size_t ByteSizeLong(const ::google::protobuf::MessageLite& msg);
  static ::uint8_t* PROTOBUF_NONNULL _InternalSerialize(
      const ::google::protobuf::MessageLite& msg, ::uint8_t* PROTOBUF_NONNULL target,
      ::google::protobuf::io::EpsCopyOutputStream* PROTOBUF_NONNULL stream);

  public:
  ::size_t ByteSizeLong() const { return ByteSizeLong(*this); }
  ::uint8_t* PROTOBUF_NONNULL _InternalSerialize(
      ::uint8_t* PROTOBUF_NONNULL target,
      ::google::protobuf::io::EpsCopyOutputStream* PROTOBUF_NONNULL stream) const {
    return _InternalSerialize(*this, target, stream);
  }
  #else   // PROTOBUF_CUSTOM_VTABLE
  ::size_t ByteSizeLong() const final;
  ::uint8_t* PROTOBUF_NONNULL _InternalSerialize(
      ::uint8_t* PROTOBUF_NONNULL target,
      ::google::protobuf::io::EpsCopyOutputStream* PROTOBUF_NONNULL stream) const final;
  #endif  // PROTOBUF_CUSTOM_VTABLE
  int GetCachedSize() const { return _impl_._cached_size_.Get(); }

  private:
  void SharedCtor(::google::protobuf::Arena* PROTOBUF_NULLABLE arena);
  static void SharedDtor(MessageLite& self);
  void InternalSwap(TopicConfig* PROTOBUF_NONNULL other);
 private:
  template <typename T>
  friend ::absl::string_view(::google::protobuf::internal::GetAnyMessageName)();
  static ::absl::string_view FullMessageName() { return "TopicConfig"; }

 protected:
  explicit TopicConfig(::google::protobuf::Arena* PROTOBUF_NULLABLE arena);
  TopicConfig(::google::protobuf::Arena* PROTOBUF_NULLABLE arena, const TopicConfig& from);
  TopicConfig(
      ::google::protobuf::Arena* PROTOBUF_NULLABLE arena, TopicConfig&& from) noexcept
      : TopicConfig(arena) {
    *this = ::std::move(from);
  }
  const ::google::protobuf::internal::ClassData* PROTOBUF_NONNULL GetClassData() const PROTOBUF_FINAL;
  static void* PROTOBUF_NONNULL PlacementNew_(
      const void* PROTOBUF_NONNULL, void* PROTOBUF_NONNULL mem,
      ::google::protobuf::Arena* PROTOBUF_NULLABLE arena);
  static constexpr auto InternalNewImpl_();

 public:
  static constexpr auto InternalGenerateClassData_();

  ::google::protobuf::Metadata GetMetadata() const;
  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------
  enum : int {
    kCameraMergedTopicFieldNumber = 3,
    kObjectDetectionResultTopicFieldNumber = 4,
  };
  // optional string camera_merged_topic = 3;
  bool has_camera_merged_topic() const;
  void clear_camera_merged_topic() ;
  const std::string& camera_merged_topic() const;
  template <typename Arg_ = const std::string&, typename... Args_>
  void set_camera_merged_topic(Arg_&& arg, Args_... args);
  std::string* PROTOBUF_NONNULL mutable_camera_merged_topic();
  [[nodiscard]] std::string* PROTOBUF_NULLABLE release_camera_merged_topic();
  void set_allocated_camera_merged_topic(std::string* PROTOBUF_NULLABLE value);

  private:
  const std::string& _internal_camera_merged_topic() const;
  PROTOBUF_ALWAYS_INLINE void _internal_set_camera_merged_topic(const std::string& value);
  std::string* PROTOBUF_NONNULL _internal_mutable_camera_merged_topic();

  public:
  // optional string object_detection_result_topic = 4;
  bool has_object_detection_result_topic() const;
  void clear_object_detection_result_topic() ;
  const std::string& object_detection_result_topic() const;
  template <typename Arg_ = const std::string&, typename... Args_>
  void set_object_detection_result_topic(Arg_&& arg, Args_... args);
  std::string* PROTOBUF_NONNULL mutable_object_detection_result_topic();
  [[nodiscard]] std::string* PROTOBUF_NULLABLE release_object_detection_result_topic();
  void set_allocated_object_detection_result_topic(std::string* PROTOBUF_NULLABLE value);

  private:
  const std::string& _internal_object_detection_result_topic() const;
  PROTOBUF_ALWAYS_INLINE void _internal_set_object_detection_result_topic(const std::string& value);
  std::string* PROTOBUF_NONNULL _internal_mutable_object_detection_result_topic();

  public:
  // @@protoc_insertion_point(class_scope:TopicConfig)
 private:
  class _Internal;
  friend class ::google::protobuf::internal::TcParser;
  static const ::google::protobuf::internal::TcParseTable<1, 2,
                                   0, 68,
                                   2>
      _table_;

  friend class ::google::protobuf::MessageLite;
  friend class ::google::protobuf::Arena;
  template <typename T>
  friend class ::google::protobuf::Arena::InternalHelper;
  using InternalArenaConstructable_ = void;
  using DestructorSkippable_ = void;
  struct Impl_ {
    inline explicit constexpr Impl_(::google::protobuf::internal::ConstantInitialized) noexcept;
    inline explicit Impl_(
        ::google::protobuf::internal::InternalVisibility visibility,
        ::google::protobuf::Arena* PROTOBUF_NULLABLE arena);
    inline explicit Impl_(
        ::google::protobuf::internal::InternalVisibility visibility,
        ::google::protobuf::Arena* PROTOBUF_NULLABLE arena, const Impl_& from,
        const TopicConfig& from_msg);
    ::google::protobuf::internal::HasBits<1> _has_bits_;
    ::google::protobuf::internal::CachedSize _cached_size_;
    ::google::protobuf::internal::ArenaStringPtr camera_merged_topic_;
    ::google::protobuf::internal::ArenaStringPtr object_detection_result_topic_;
    PROTOBUF_TSAN_DECLARE_MEMBER
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_ObjectDetectionActivity_2eproto;
};

extern const ::google::protobuf::internal::ClassDataFull TopicConfig_class_data_;
// -------------------------------------------------------------------

class Config final : public ::google::protobuf::Message
/* @@protoc_insertion_point(class_definition:Config) */ {
 public:
  inline Config() : Config(nullptr) {}
  ~Config() PROTOBUF_FINAL;

#if defined(PROTOBUF_CUSTOM_VTABLE)
  void operator delete(Config* PROTOBUF_NONNULL msg, std::destroying_delete_t) {
    SharedDtor(*msg);
    ::google::protobuf::internal::SizedDelete(msg, sizeof(Config));
  }
#endif

  template <typename = void>
  explicit PROTOBUF_CONSTEXPR Config(::google::protobuf::internal::ConstantInitialized);

  inline Config(const Config& from) : Config(nullptr, from) {}
  inline Config(Config&& from) noexcept
      : Config(nullptr, std::move(from)) {}
  inline Config& operator=(const Config& from) {
    CopyFrom(from);
    return *this;
  }
  inline Config& operator=(Config&& from) noexcept {
    if (this == &from) return *this;
    if (::google::protobuf::internal::CanMoveWithInternalSwap(GetArena(), from.GetArena())) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return _internal_metadata_.unknown_fields<::google::protobuf::UnknownFieldSet>(::google::protobuf::UnknownFieldSet::default_instance);
  }
  inline ::google::protobuf::UnknownFieldSet* PROTOBUF_NONNULL mutable_unknown_fields()
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return _internal_metadata_.mutable_unknown_fields<::google::protobuf::UnknownFieldSet>();
  }

  static const ::google::protobuf::Descriptor* PROTOBUF_NONNULL descriptor() {
    return GetDescriptor();
  }
  static const ::google::protobuf::Descriptor* PROTOBUF_NONNULL GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::google::protobuf::Reflection* PROTOBUF_NONNULL GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const Config& default_instance() {
    return *reinterpret_cast<const Config*>(
        &_Config_default_instance_);
  }
  static constexpr int kIndexInFileMessages = 1;
  friend void swap(Config& a, Config& b) { a.Swap(&b); }
  inline void Swap(Config* PROTOBUF_NONNULL other) {
    if (other == this) return;
    if (::google::protobuf::internal::CanUseInternalSwap(GetArena(), other->GetArena())) {
      InternalSwap(other);
    } else {
      ::google::protobuf::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(Config* PROTOBUF_NONNULL other) {
    if (other == this) return;
    ABSL_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  Config* PROTOBUF_NONNULL New(::google::protobuf::Arena* PROTOBUF_NULLABLE arena = nullptr) const {
    return ::google::protobuf::Message::DefaultConstruct<Config>(arena);
  }
  using ::google::protobuf::Message::CopyFrom;
  void CopyFrom(const Config& from);
  using ::google::protobuf::Message::MergeFrom;
  void MergeFrom(const Config& from) { Config::MergeImpl(*this, from); }

  private:
  static void MergeImpl(::google::protobuf::MessageLite& to_msg,
                        const ::google::protobuf::MessageLite& from_msg);

  public:
  bool IsInitialized() const {
    return true;
  }
  ABSL_ATTRIBUTE_REINITIALIZES void Clear() PROTOBUF_FINAL;
  #if defined(PROTOBUF_CUSTOM_VTABLE)
  private:
  static ::size_t ByteSizeLong(const ::google::protobuf::MessageLite& msg);
  static ::uint8_t* PROTOBUF_NONNULL _InternalSerialize(
      const ::google::protobuf::MessageLite& msg, ::uint8_t* PROTOBUF_NONNULL target,
      ::google::protobuf::io::EpsCopyOutputStream* PROTOBUF_NONNULL stream);

  public:
  ::size_t ByteSizeLong() const { return ByteSizeLong(*this); }
  ::uint8_t* PROTOBUF_NONNULL _InternalSerialize(
      ::uint8_t* PROTOBUF_NONNULL target,
      ::google::protobuf::io::EpsCopyOutputStream* PROTOBUF_NONNULL stream) const {
    return _InternalSerialize(*this, target, stream);
  }
  #else   // PROTOBUF_CUSTOM_VTABLE
  ::size_t ByteSizeLong() const final;
  ::uint8_t* PROTOBUF_NONNULL _InternalSerialize(
      ::uint8_t* PROTOBUF_NONNULL target,
      ::google::protobuf::io::EpsCopyOutputStream* PROTOBUF_NONNULL stream) const final;
  #endif  // PROTOBUF_CUSTOM_VTABLE
  int GetCachedSize() const { return _impl_._cached_size_.Get(); }

  private:
  void SharedCtor(::google::protobuf::Arena* PROTOBUF_NULLABLE arena);
  static void SharedDtor(MessageLite& self);
  void InternalSwap(Config* PROTOBUF_NONNULL other);
 private:
  template <typename T>
  friend ::absl::string_view(::google::protobuf::internal::GetAnyMessageName)();
  static ::absl::string_view FullMessageName() { return "Config"; }

 protected:
  explicit Config(::google::protobuf::Arena* PROTOBUF_NULLABLE arena);
  Config(::google::protobuf::Arena* PROTOBUF_NULLABLE arena, const Config& from);
  Config(
      ::google::protobuf::Arena* PROTOBUF_NULLABLE arena, Config&& from) noexcept
      : Config(arena) {
    *this = ::std::move(from);
  }
  const ::google::protobuf::internal::ClassData* PROTOBUF_NONNULL GetClassData() const PROTOBUF_FINAL;
  static void* PROTOBUF_NONNULL PlacementNew_(
      const void* PROTOBUF_NONNULL, void* PROTOBUF_NONNULL mem,
      ::google::protobuf::Arena* PROTOBUF_NULLABLE arena);
  static constexpr auto InternalNewImpl_();

 public:
  static constexpr auto InternalGenerateClassData_();

  ::google::protobuf::Metadata GetMetadata() const;
  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------
  enum : int {
    kTopicConfigFieldNumber = 1,
  };
  // optional .TopicConfig topic_config = 1;
  bool has_topic_config() const;
  void clear_topic_config() ;
  const ::TopicConfig& topic_config() const;
  [[nodiscard]] ::TopicConfig* PROTOBUF_NULLABLE release_topic_config();
  ::TopicConfig* PROTOBUF_NONNULL mutable_topic_config();
  void set_allocated_topic_config(::TopicConfig* PROTOBUF_NULLABLE value);
  void unsafe_arena_set_allocated_topic_config(::TopicConfig* PROTOBUF_NULLABLE value);
  ::TopicConfig* PROTOBUF_NULLABLE unsafe_arena_release_topic_config();

  private:
  const ::TopicConfig& _internal_topic_config() const;
  ::TopicConfig* PROTOBUF_NONNULL _internal_mutable_topic_config();

  public:
  // @@protoc_insertion_point(class_scope:Config)
 private:
  class _Internal;
  friend class ::google::protobuf::internal::TcParser;
  static const ::google::protobuf::internal::TcParseTable<0, 1,
                                   1, 0,
                                   2>
      _table_;

  friend class ::google::protobuf::MessageLite;
  friend class ::google::protobuf::Arena;
  template <typename T>
  friend class ::google::protobuf::Arena::InternalHelper;
  using InternalArenaConstructable_ = void;
  using DestructorSkippable_ = void;
  struct Impl_ {
    inline explicit constexpr Impl_(::google::protobuf::internal::ConstantInitialized) noexcept;
    inline explicit Impl_(
        ::google::protobuf::internal::InternalVisibility visibility,
        ::google::protobuf::Arena* PROTOBUF_NULLABLE arena);
    inline explicit Impl_(
        ::google::protobuf::internal::InternalVisibility visibility,
        ::google::protobuf::Arena* PROTOBUF_NULLABLE arena, const Impl_& from,
        const Config& from_msg);
    ::google::protobuf::internal::HasBits<1> _has_bits_;
    ::google::protobuf::internal::CachedSize _cached_size_;
    ::TopicConfig* PROTOBUF_NULLABLE topic_config_;
    PROTOBUF_TSAN_DECLARE_MEMBER
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_ObjectDetectionActivity_2eproto;
};

extern const ::google::protobuf::internal::ClassDataFull Config_class_data_;

// ===================================================================




// ===================================================================


#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// -------------------------------------------------------------------

// TopicConfig

// optional string camera_merged_topic = 3;
inline bool TopicConfig::has_camera_merged_topic() const {
  bool value = (_impl_._has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline void TopicConfig::clear_camera_merged_topic() {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  _impl_.camera_merged_topic_.ClearToEmpty();
  _impl_._has_bits_[0] &= ~0x00000001u;
}
inline const std::string& TopicConfig::camera_merged_topic() const
    ABSL_ATTRIBUTE_LIFETIME_BOUND {
  // @@protoc_insertion_point(field_get:TopicConfig.camera_merged_topic)
  return _internal_camera_merged_topic();
}
template <typename Arg_, typename... Args_>
PROTOBUF_ALWAYS_INLINE void TopicConfig::set_camera_merged_topic(Arg_&& arg, Args_... args) {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  _impl_._has_bits_[0] |= 0x00000001u;
  _impl_.camera_merged_topic_.Set(static_cast<Arg_&&>(arg), args..., GetArena());
  // @@protoc_insertion_point(field_set:TopicConfig.camera_merged_topic)
}
inline std::string* PROTOBUF_NONNULL TopicConfig::mutable_camera_merged_topic()
    ABSL_ATTRIBUTE_LIFETIME_BOUND {
  std::string* _s = _internal_mutable_camera_merged_topic();
  // @@protoc_insertion_point(field_mutable:TopicConfig.camera_merged_topic)
  return _s;
}
inline const std::string& TopicConfig::_internal_camera_merged_topic() const {
  ::google::protobuf::internal::TSanRead(&_impl_);
  return _impl_.camera_merged_topic_.Get();
}
inline void TopicConfig::_internal_set_camera_merged_topic(const std::string& value) {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  _impl_._has_bits_[0] |= 0x00000001u;
  _impl_.camera_merged_topic_.Set(value, GetArena());
}
inline std::string* PROTOBUF_NONNULL TopicConfig::_internal_mutable_camera_merged_topic() {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  _impl_._has_bits_[0] |= 0x00000001u;
  return _impl_.camera_merged_topic_.Mutable( GetArena());
}
inline std::string* PROTOBUF_NULLABLE TopicConfig::release_camera_merged_topic() {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  // @@protoc_insertion_point(field_release:TopicConfig.camera_merged_topic)
  if ((_impl_._has_bits_[0] & 0x00000001u) == 0) {
    return nullptr;
  }
  _impl_._has_bits_[0] &= ~0x00000001u;
  auto* released = _impl_.camera_merged_topic_.Release();
  if (::google::protobuf::internal::DebugHardenForceCopyDefaultString()) {
    _impl_.camera_merged_topic_.Set("", GetArena());
  }
  return released;
}
inline void TopicConfig::set_allocated_camera_merged_topic(std::string* PROTOBUF_NULLABLE value) {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  if (value != nullptr) {
    _impl_._has_bits_[0] |= 0x00000001u;
  } else {
    _impl_._has_bits_[0] &= ~0x00000001u;
  }
  _impl_.camera_merged_topic_.SetAllocated(value, GetArena());
  if (::google::protobuf::internal::DebugHardenForceCopyDefaultString() && _impl_.camera_merged_topic_.IsDefault()) {
    _impl_.camera_merged_topic_.Set("", GetArena());
  }
  // @@protoc_insertion_point(field_set_allocated:TopicConfig.camera_merged_topic)
}

// optional string object_detection_result_topic = 4;
inline bool TopicConfig::has_object_detection_result_topic() const {
  bool value = (_impl_._has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline void TopicConfig::clear_object_detection_result_topic() {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  _impl_.object_detection_result_topic_.ClearToEmpty();
  _impl_._has_bits_[0] &= ~0x00000002u;
}
inline const std::string& TopicConfig::object_detection_result_topic() const
    ABSL_ATTRIBUTE_LIFETIME_BOUND {
  // @@protoc_insertion_point(field_get:TopicConfig.object_detection_result_topic)
  return _internal_object_detection_result_topic();
}
template <typename Arg_, typename... Args_>
PROTOBUF_ALWAYS_INLINE void TopicConfig::set_object_detection_result_topic(Arg_&& arg, Args_... args) {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  _impl_._has_bits_[0] |= 0x00000002u;
  _impl_.object_detection_result_topic_.Set(static_cast<Arg_&&>(arg), args..., GetArena());
  // @@protoc_insertion_point(field_set:TopicConfig.object_detection_result_topic)
}
inline std::string* PROTOBUF_NONNULL TopicConfig::mutable_object_detection_result_topic()
    ABSL_ATTRIBUTE_LIFETIME_BOUND {
  std::string* _s = _internal_mutable_object_detection_result_topic();
  // @@protoc_insertion_point(field_mutable:TopicConfig.object_detection_result_topic)
  return _s;
}
inline const std::string& TopicConfig::_internal_object_detection_result_topic() const {
  ::google::protobuf::internal::TSanRead(&_impl_);
  return _impl_.object_detection_result_topic_.Get();
}
inline void TopicConfig::_internal_set_object_detection_result_topic(const std::string& value) {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  _impl_._has_bits_[0] |= 0x00000002u;
  _impl_.object_detection_result_topic_.Set(value, GetArena());
}
inline std::string* PROTOBUF_NONNULL TopicConfig::_internal_mutable_object_detection_result_topic() {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  _impl_._has_bits_[0] |= 0x00000002u;
  return _impl_.object_detection_result_topic_.Mutable( GetArena());
}
inline std::string* PROTOBUF_NULLABLE TopicConfig::release_object_detection_result_topic() {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  // @@protoc_insertion_point(field_release:TopicConfig.object_detection_result_topic)
  if ((_impl_._has_bits_[0] & 0x00000002u) == 0) {
    return nullptr;
  }
  _impl_._has_bits_[0] &= ~0x00000002u;
  auto* released = _impl_.object_detection_result_topic_.Release();
  if (::google::protobuf::internal::DebugHardenForceCopyDefaultString()) {
    _impl_.object_detection_result_topic_.Set("", GetArena());
  }
  return released;
}
inline void TopicConfig::set_allocated_object_detection_result_topic(std::string* PROTOBUF_NULLABLE value) {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  if (value != nullptr) {
    _impl_._has_bits_[0] |= 0x00000002u;
  } else {
    _impl_._has_bits_[0] &= ~0x00000002u;
  }
  _impl_.object_detection_result_topic_.SetAllocated(value, GetArena());
  if (::google::protobuf::internal::DebugHardenForceCopyDefaultString() && _impl_.object_detection_result_topic_.IsDefault()) {
    _impl_.object_detection_result_topic_.Set("", GetArena());
  }
  // @@protoc_insertion_point(field_set_allocated:TopicConfig.object_detection_result_topic)
}

// -------------------------------------------------------------------

// Config

// optional .TopicConfig topic_config = 1;
inline bool Config::has_topic_config() const {
  bool value = (_impl_._has_bits_[0] & 0x00000001u) != 0;
  PROTOBUF_ASSUME(!value || _impl_.topic_config_ != nullptr);
  return value;
}
inline void Config::clear_topic_config() {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  if (_impl_.topic_config_ != nullptr) _impl_.topic_config_->Clear();
  _impl_._has_bits_[0] &= ~0x00000001u;
}
inline const ::TopicConfig& Config::_internal_topic_config() const {
  ::google::protobuf::internal::TSanRead(&_impl_);
  const ::TopicConfig* p = _impl_.topic_config_;
  return p != nullptr ? *p : reinterpret_cast<const ::TopicConfig&>(::_TopicConfig_default_instance_);
}
inline const ::TopicConfig& Config::topic_config() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
  // @@protoc_insertion_point(field_get:Config.topic_config)
  return _internal_topic_config();
}
inline void Config::unsafe_arena_set_allocated_topic_config(
    ::TopicConfig* PROTOBUF_NULLABLE value) {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  if (GetArena() == nullptr) {
    delete reinterpret_cast<::google::protobuf::MessageLite*>(_impl_.topic_config_);
  }
  _impl_.topic_config_ = reinterpret_cast<::TopicConfig*>(value);
  if (value != nullptr) {
    _impl_._has_bits_[0] |= 0x00000001u;
  } else {
    _impl_._has_bits_[0] &= ~0x00000001u;
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:Config.topic_config)
}
inline ::TopicConfig* PROTOBUF_NULLABLE Config::release_topic_config() {
  ::google::protobuf::internal::TSanWrite(&_impl_);

  _impl_._has_bits_[0] &= ~0x00000001u;
  ::TopicConfig* released = _impl_.topic_config_;
  _impl_.topic_config_ = nullptr;
  if (::google::protobuf::internal::DebugHardenForceCopyInRelease()) {
    auto* old = reinterpret_cast<::google::protobuf::MessageLite*>(released);
    released = ::google::protobuf::internal::DuplicateIfNonNull(released);
    if (GetArena() == nullptr) {
      delete old;
    }
  } else {
    if (GetArena() != nullptr) {
      released = ::google::protobuf::internal::DuplicateIfNonNull(released);
    }
  }
  return released;
}
inline ::TopicConfig* PROTOBUF_NULLABLE Config::unsafe_arena_release_topic_config() {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  // @@protoc_insertion_point(field_release:Config.topic_config)

  _impl_._has_bits_[0] &= ~0x00000001u;
  ::TopicConfig* temp = _impl_.topic_config_;
  _impl_.topic_config_ = nullptr;
  return temp;
}
inline ::TopicConfig* PROTOBUF_NONNULL Config::_internal_mutable_topic_config() {
  ::google::protobuf::internal::TSanWrite(&_impl_);
  if (_impl_.topic_config_ == nullptr) {
    auto* p = ::google::protobuf::Message::DefaultConstruct<::TopicConfig>(GetArena());
    _impl_.topic_config_ = reinterpret_cast<::TopicConfig*>(p);
  }
  return _impl_.topic_config_;
}
inline ::TopicConfig* PROTOBUF_NONNULL Config::mutable_topic_config()
    ABSL_ATTRIBUTE_LIFETIME_BOUND {
  _impl_._has_bits_[0] |= 0x00000001u;
  ::TopicConfig* _msg = _internal_mutable_topic_config();
  // @@protoc_insertion_point(field_mutable:Config.topic_config)
  return _msg;
}
inline void Config::set_allocated_topic_config(::TopicConfig* PROTOBUF_NULLABLE value) {
  ::google::protobuf::Arena* message_arena = GetArena();
  ::google::protobuf::internal::TSanWrite(&_impl_);
  if (message_arena == nullptr) {
    delete reinterpret_cast<::google::protobuf::MessageLite*>(_impl_.topic_config_);
  }

  if (value != nullptr) {
    ::google::protobuf::Arena* submessage_arena = value->GetArena();
    if (message_arena != submessage_arena) {
      value = ::google::protobuf::internal::GetOwnedMessage(message_arena, value, submessage_arena);
    }
    _impl_._has_bits_[0] |= 0x00000001u;
  } else {
    _impl_._has_bits_[0] &= ~0x00000001u;
  }

  _impl_.topic_config_ = reinterpret_cast<::TopicConfig*>(value);
  // @@protoc_insertion_point(field_set_allocated:Config.topic_config)
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)


// @@protoc_insertion_point(global_scope)

#include "google/protobuf/port_undef.inc"

#endif  // ObjectDetectionActivity_2eproto_2epb_2eh
