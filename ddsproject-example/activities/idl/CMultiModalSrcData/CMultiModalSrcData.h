// Copyright 2016 Proyectos y Sistemas de Mantenimiento SL (eProsima).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*!
 * @file CMultiModalSrcData.h
 * This header file contains the declaration of the described types in the IDL file.
 *
 * This file was generated by the tool fastddsgen.
 */

#ifndef _FAST_DDS_GENERATED_CMULTIMODALSRCDATA_H_
#define _FAST_DDS_GENERATED_CMULTIMODALSRCDATA_H_

#include <array>
#include <bitset>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <fastcdr/cdr/fixed_size_string.hpp>
#include <fastcdr/xcdr/external.hpp>
#include <fastcdr/xcdr/optional.hpp>

#include "../CDataBase/CDataBase.h"


#if defined(_WIN32)
#if defined(EPROSIMA_USER_DLL_EXPORT)
#define eProsima_user_DllExport __declspec( dllexport )
#else
#define eProsima_user_DllExport
#endif  // EPROSIMA_USER_DLL_EXPORT
#else
#define eProsima_user_DllExport
#endif  // _WIN32

#if defined(_WIN32)
#if defined(EPROSIMA_USER_DLL_EXPORT)
#if defined(CMULTIMODALSRCDATA_SOURCE)
#define CMULTIMODALSRCDATA_DllAPI __declspec( dllexport )
#else
#define CMULTIMODALSRCDATA_DllAPI __declspec( dllimport )
#endif // CMULTIMODALSRCDATA_SOURCE
#else
#define CMULTIMODALSRCDATA_DllAPI
#endif  // EPROSIMA_USER_DLL_EXPORT
#else
#define CMULTIMODALSRCDATA_DllAPI
#endif // _WIN32

namespace eprosima {
namespace fastcdr {
class Cdr;
class CdrSizeCalculator;
} // namespace fastcdr
} // namespace eprosima







/*!
 * @brief This class represents the structure CVideoSrcData defined by the user in the IDL file.
 * @ingroup CMultiModalSrcData
 */
class CVideoSrcData : public CDataBase
{
public:

    /*!
     * @brief Default constructor.
     */
    eProsima_user_DllExport CVideoSrcData();

    /*!
     * @brief Default destructor.
     */
    eProsima_user_DllExport ~CVideoSrcData();

    /*!
     * @brief Copy constructor.
     * @param x Reference to the object CVideoSrcData that will be copied.
     */
    eProsima_user_DllExport CVideoSrcData(
            const CVideoSrcData& x);

    /*!
     * @brief Move constructor.
     * @param x Reference to the object CVideoSrcData that will be copied.
     */
    eProsima_user_DllExport CVideoSrcData(
            CVideoSrcData&& x) noexcept;

    /*!
     * @brief Copy assignment.
     * @param x Reference to the object CVideoSrcData that will be copied.
     */
    eProsima_user_DllExport CVideoSrcData& operator =(
            const CVideoSrcData& x);

    /*!
     * @brief Move assignment.
     * @param x Reference to the object CVideoSrcData that will be copied.
     */
    eProsima_user_DllExport CVideoSrcData& operator =(
            CVideoSrcData&& x) noexcept;

    /*!
     * @brief Comparison operator.
     * @param x CVideoSrcData object to compare.
     */
    eProsima_user_DllExport bool operator ==(
            const CVideoSrcData& x) const;

    /*!
     * @brief Comparison operator.
     * @param x CVideoSrcData object to compare.
     */
    eProsima_user_DllExport bool operator !=(
            const CVideoSrcData& x) const;

    /*!
     * @brief This function sets a value in member ucCameraId
     * @param _ucCameraId New value for member ucCameraId
     */
    eProsima_user_DllExport void ucCameraId(
            uint8_t _ucCameraId);

    /*!
     * @brief This function returns the value of member ucCameraId
     * @return Value of member ucCameraId
     */
    eProsima_user_DllExport uint8_t ucCameraId() const;

    /*!
     * @brief This function returns a reference to member ucCameraId
     * @return Reference to member ucCameraId
     */
    eProsima_user_DllExport uint8_t& ucCameraId();


    /*!
     * @brief This function sets a value in member usBmpLength
     * @param _usBmpLength New value for member usBmpLength
     */
    eProsima_user_DllExport void usBmpLength(
            uint16_t _usBmpLength);

    /*!
     * @brief This function returns the value of member usBmpLength
     * @return Value of member usBmpLength
     */
    eProsima_user_DllExport uint16_t usBmpLength() const;

    /*!
     * @brief This function returns a reference to member usBmpLength
     * @return Reference to member usBmpLength
     */
    eProsima_user_DllExport uint16_t& usBmpLength();


    /*!
     * @brief This function sets a value in member usBmpWidth
     * @param _usBmpWidth New value for member usBmpWidth
     */
    eProsima_user_DllExport void usBmpWidth(
            uint16_t _usBmpWidth);

    /*!
     * @brief This function returns the value of member usBmpWidth
     * @return Value of member usBmpWidth
     */
    eProsima_user_DllExport uint16_t usBmpWidth() const;

    /*!
     * @brief This function returns a reference to member usBmpWidth
     * @return Reference to member usBmpWidth
     */
    eProsima_user_DllExport uint16_t& usBmpWidth();


    /*!
     * @brief This function sets a value in member unBmpBytes
     * @param _unBmpBytes New value for member unBmpBytes
     */
    eProsima_user_DllExport void unBmpBytes(
            uint32_t _unBmpBytes);

    /*!
     * @brief This function returns the value of member unBmpBytes
     * @return Value of member unBmpBytes
     */
    eProsima_user_DllExport uint32_t unBmpBytes() const;

    /*!
     * @brief This function returns a reference to member unBmpBytes
     * @return Reference to member unBmpBytes
     */
    eProsima_user_DllExport uint32_t& unBmpBytes();


    /*!
     * @brief This function copies the value in member vecImageBuf
     * @param _vecImageBuf New value to be copied in member vecImageBuf
     */
    eProsima_user_DllExport void vecImageBuf(
            const std::vector<uint8_t>& _vecImageBuf);

    /*!
     * @brief This function moves the value in member vecImageBuf
     * @param _vecImageBuf New value to be moved in member vecImageBuf
     */
    eProsima_user_DllExport void vecImageBuf(
            std::vector<uint8_t>&& _vecImageBuf);

    /*!
     * @brief This function returns a constant reference to member vecImageBuf
     * @return Constant reference to member vecImageBuf
     */
    eProsima_user_DllExport const std::vector<uint8_t>& vecImageBuf() const;

    /*!
     * @brief This function returns a reference to member vecImageBuf
     * @return Reference to member vecImageBuf
     */
    eProsima_user_DllExport std::vector<uint8_t>& vecImageBuf();

private:

    uint8_t m_ucCameraId{0};
    uint16_t m_usBmpLength{0};
    uint16_t m_usBmpWidth{0};
    uint32_t m_unBmpBytes{0};
    std::vector<uint8_t> m_vecImageBuf;

};




/*!
 * @brief This class represents the structure CDisparityResult defined by the user in the IDL file.
 * @ingroup CMultiModalSrcData
 */
class CDisparityResult
{
public:

    /*!
     * @brief Default constructor.
     */
    eProsima_user_DllExport CDisparityResult();

    /*!
     * @brief Default destructor.
     */
    eProsima_user_DllExport ~CDisparityResult();

    /*!
     * @brief Copy constructor.
     * @param x Reference to the object CDisparityResult that will be copied.
     */
    eProsima_user_DllExport CDisparityResult(
            const CDisparityResult& x);

    /*!
     * @brief Move constructor.
     * @param x Reference to the object CDisparityResult that will be copied.
     */
    eProsima_user_DllExport CDisparityResult(
            CDisparityResult&& x) noexcept;

    /*!
     * @brief Copy assignment.
     * @param x Reference to the object CDisparityResult that will be copied.
     */
    eProsima_user_DllExport CDisparityResult& operator =(
            const CDisparityResult& x);

    /*!
     * @brief Move assignment.
     * @param x Reference to the object CDisparityResult that will be copied.
     */
    eProsima_user_DllExport CDisparityResult& operator =(
            CDisparityResult&& x) noexcept;

    /*!
     * @brief Comparison operator.
     * @param x CDisparityResult object to compare.
     */
    eProsima_user_DllExport bool operator ==(
            const CDisparityResult& x) const;

    /*!
     * @brief Comparison operator.
     * @param x CDisparityResult object to compare.
     */
    eProsima_user_DllExport bool operator !=(
            const CDisparityResult& x) const;

    /*!
     * @brief This function sets a value in member usWidth
     * @param _usWidth New value for member usWidth
     */
    eProsima_user_DllExport void usWidth(
            uint16_t _usWidth);

    /*!
     * @brief This function returns the value of member usWidth
     * @return Value of member usWidth
     */
    eProsima_user_DllExport uint16_t usWidth() const;

    /*!
     * @brief This function returns a reference to member usWidth
     * @return Reference to member usWidth
     */
    eProsima_user_DllExport uint16_t& usWidth();


    /*!
     * @brief This function sets a value in member usHeight
     * @param _usHeight New value for member usHeight
     */
    eProsima_user_DllExport void usHeight(
            uint16_t _usHeight);

    /*!
     * @brief This function returns the value of member usHeight
     * @return Value of member usHeight
     */
    eProsima_user_DllExport uint16_t usHeight() const;

    /*!
     * @brief This function returns a reference to member usHeight
     * @return Reference to member usHeight
     */
    eProsima_user_DllExport uint16_t& usHeight();


    /*!
     * @brief This function copies the value in member vecDistanceInfo
     * @param _vecDistanceInfo New value to be copied in member vecDistanceInfo
     */
    eProsima_user_DllExport void vecDistanceInfo(
            const std::vector<int32_t>& _vecDistanceInfo);

    /*!
     * @brief This function moves the value in member vecDistanceInfo
     * @param _vecDistanceInfo New value to be moved in member vecDistanceInfo
     */
    eProsima_user_DllExport void vecDistanceInfo(
            std::vector<int32_t>&& _vecDistanceInfo);

    /*!
     * @brief This function returns a constant reference to member vecDistanceInfo
     * @return Constant reference to member vecDistanceInfo
     */
    eProsima_user_DllExport const std::vector<int32_t>& vecDistanceInfo() const;

    /*!
     * @brief This function returns a reference to member vecDistanceInfo
     * @return Reference to member vecDistanceInfo
     */
    eProsima_user_DllExport std::vector<int32_t>& vecDistanceInfo();

private:

    uint16_t m_usWidth{0};
    uint16_t m_usHeight{0};
    std::vector<int32_t> m_vecDistanceInfo;

};


/*!
 * @brief This class represents the structure CMultiModalSrcData defined by the user in the IDL file.
 * @ingroup CMultiModalSrcData
 */
class CMultiModalSrcData : public CDataBase
{
public:

    /*!
     * @brief Default constructor.
     */
    eProsima_user_DllExport CMultiModalSrcData();

    /*!
     * @brief Default destructor.
     */
    eProsima_user_DllExport ~CMultiModalSrcData();

    /*!
     * @brief Copy constructor.
     * @param x Reference to the object CMultiModalSrcData that will be copied.
     */
    eProsima_user_DllExport CMultiModalSrcData(
            const CMultiModalSrcData& x);

    /*!
     * @brief Move constructor.
     * @param x Reference to the object CMultiModalSrcData that will be copied.
     */
    eProsima_user_DllExport CMultiModalSrcData(
            CMultiModalSrcData&& x) noexcept;

    /*!
     * @brief Copy assignment.
     * @param x Reference to the object CMultiModalSrcData that will be copied.
     */
    eProsima_user_DllExport CMultiModalSrcData& operator =(
            const CMultiModalSrcData& x);

    /*!
     * @brief Move assignment.
     * @param x Reference to the object CMultiModalSrcData that will be copied.
     */
    eProsima_user_DllExport CMultiModalSrcData& operator =(
            CMultiModalSrcData&& x) noexcept;

    /*!
     * @brief Comparison operator.
     * @param x CMultiModalSrcData object to compare.
     */
    eProsima_user_DllExport bool operator ==(
            const CMultiModalSrcData& x) const;

    /*!
     * @brief Comparison operator.
     * @param x CMultiModalSrcData object to compare.
     */
    eProsima_user_DllExport bool operator !=(
            const CMultiModalSrcData& x) const;

    /*!
     * @brief This function copies the value in member vecVideoSrcData
     * @param _vecVideoSrcData New value to be copied in member vecVideoSrcData
     */
    eProsima_user_DllExport void vecVideoSrcData(
            const std::vector<CVideoSrcData>& _vecVideoSrcData);

    /*!
     * @brief This function moves the value in member vecVideoSrcData
     * @param _vecVideoSrcData New value to be moved in member vecVideoSrcData
     */
    eProsima_user_DllExport void vecVideoSrcData(
            std::vector<CVideoSrcData>&& _vecVideoSrcData);

    /*!
     * @brief This function returns a constant reference to member vecVideoSrcData
     * @return Constant reference to member vecVideoSrcData
     */
    eProsima_user_DllExport const std::vector<CVideoSrcData>& vecVideoSrcData() const;

    /*!
     * @brief This function returns a reference to member vecVideoSrcData
     * @return Reference to member vecVideoSrcData
     */
    eProsima_user_DllExport std::vector<CVideoSrcData>& vecVideoSrcData();


    /*!
     * @brief This function copies the value in member vecfHomography
     * @param _vecfHomography New value to be copied in member vecfHomography
     */
    eProsima_user_DllExport void vecfHomography(
            const std::vector<float>& _vecfHomography);

    /*!
     * @brief This function moves the value in member vecfHomography
     * @param _vecfHomography New value to be moved in member vecfHomography
     */
    eProsima_user_DllExport void vecfHomography(
            std::vector<float>&& _vecfHomography);

    /*!
     * @brief This function returns a constant reference to member vecfHomography
     * @return Constant reference to member vecfHomography
     */
    eProsima_user_DllExport const std::vector<float>& vecfHomography() const;

    /*!
     * @brief This function returns a reference to member vecfHomography
     * @return Reference to member vecfHomography
     */
    eProsima_user_DllExport std::vector<float>& vecfHomography();


    /*!
     * @brief This function copies the value in member tDisparityResult
     * @param _tDisparityResult New value to be copied in member tDisparityResult
     */
    eProsima_user_DllExport void tDisparityResult(
            const CDisparityResult& _tDisparityResult);

    /*!
     * @brief This function moves the value in member tDisparityResult
     * @param _tDisparityResult New value to be moved in member tDisparityResult
     */
    eProsima_user_DllExport void tDisparityResult(
            CDisparityResult&& _tDisparityResult);

    /*!
     * @brief This function returns a constant reference to member tDisparityResult
     * @return Constant reference to member tDisparityResult
     */
    eProsima_user_DllExport const CDisparityResult& tDisparityResult() const;

    /*!
     * @brief This function returns a reference to member tDisparityResult
     * @return Reference to member tDisparityResult
     */
    eProsima_user_DllExport CDisparityResult& tDisparityResult();

private:

    std::vector<CVideoSrcData> m_vecVideoSrcData;
    std::vector<float> m_vecfHomography;
    CDisparityResult m_tDisparityResult;

};

#endif // _FAST_DDS_GENERATED_CMULTIMODALSRCDATA_H_



