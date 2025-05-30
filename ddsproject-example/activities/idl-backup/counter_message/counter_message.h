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
 * @file counter_message.h
 * This header file contains the declaration of the described types in the IDL file.
 *
 * This file was generated by the tool fastddsgen.
 */

#ifndef _FAST_DDS_GENERATED_COUNTER_MESSAGE_H_
#define _FAST_DDS_GENERATED_COUNTER_MESSAGE_H_

#include <array>
#include <bitset>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <fastcdr/cdr/fixed_size_string.hpp>
#include <fastcdr/xcdr/external.hpp>
#include <fastcdr/xcdr/optional.hpp>



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
#if defined(COUNTER_MESSAGE_SOURCE)
#define COUNTER_MESSAGE_DllAPI __declspec( dllexport )
#else
#define COUNTER_MESSAGE_DllAPI __declspec( dllimport )
#endif // COUNTER_MESSAGE_SOURCE
#else
#define COUNTER_MESSAGE_DllAPI
#endif  // EPROSIMA_USER_DLL_EXPORT
#else
#define COUNTER_MESSAGE_DllAPI
#endif // _WIN32

namespace eprosima {
namespace fastcdr {
class Cdr;
class CdrSizeCalculator;
} // namespace fastcdr
} // namespace eprosima





/*!
 * @brief This class represents the structure CounterMessage defined by the user in the IDL file.
 * @ingroup counter_message
 */
class CounterMessage
{
public:

    /*!
     * @brief Default constructor.
     */
    eProsima_user_DllExport CounterMessage();

    /*!
     * @brief Default destructor.
     */
    eProsima_user_DllExport ~CounterMessage();

    /*!
     * @brief Copy constructor.
     * @param x Reference to the object CounterMessage that will be copied.
     */
    eProsima_user_DllExport CounterMessage(
            const CounterMessage& x);

    /*!
     * @brief Move constructor.
     * @param x Reference to the object CounterMessage that will be copied.
     */
    eProsima_user_DllExport CounterMessage(
            CounterMessage&& x) noexcept;

    /*!
     * @brief Copy assignment.
     * @param x Reference to the object CounterMessage that will be copied.
     */
    eProsima_user_DllExport CounterMessage& operator =(
            const CounterMessage& x);

    /*!
     * @brief Move assignment.
     * @param x Reference to the object CounterMessage that will be copied.
     */
    eProsima_user_DllExport CounterMessage& operator =(
            CounterMessage&& x) noexcept;

    /*!
     * @brief Comparison operator.
     * @param x CounterMessage object to compare.
     */
    eProsima_user_DllExport bool operator ==(
            const CounterMessage& x) const;

    /*!
     * @brief Comparison operator.
     * @param x CounterMessage object to compare.
     */
    eProsima_user_DllExport bool operator !=(
            const CounterMessage& x) const;

    /*!
     * @brief This function sets a value in member cnt
     * @param _cnt New value for member cnt
     */
    eProsima_user_DllExport void cnt(
            int32_t _cnt);

    /*!
     * @brief This function returns the value of member cnt
     * @return Value of member cnt
     */
    eProsima_user_DllExport int32_t cnt() const;

    /*!
     * @brief This function returns a reference to member cnt
     * @return Reference to member cnt
     */
    eProsima_user_DllExport int32_t& cnt();


    /*!
     * @brief This function copies the value in member tip
     * @param _tip New value to be copied in member tip
     */
    eProsima_user_DllExport void tip(
            const std::string& _tip);

    /*!
     * @brief This function moves the value in member tip
     * @param _tip New value to be moved in member tip
     */
    eProsima_user_DllExport void tip(
            std::string&& _tip);

    /*!
     * @brief This function returns a constant reference to member tip
     * @return Constant reference to member tip
     */
    eProsima_user_DllExport const std::string& tip() const;

    /*!
     * @brief This function returns a reference to member tip
     * @return Reference to member tip
     */
    eProsima_user_DllExport std::string& tip();

private:

    int32_t m_cnt{0};
    std::string m_tip;

};

#endif // _FAST_DDS_GENERATED_COUNTER_MESSAGE_H_



