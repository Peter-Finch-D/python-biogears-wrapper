// Copyright (c) 2005-2014 Code Synthesis Tools CC
//
// This program was generated by CodeSynthesis XSD, an XML Schema to
// C++ data binding compiler.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA
//
// In addition, as a special exception, Code Synthesis Tools CC gives
// permission to link this program with the Xerces-C++ library (or with
// modified versions of Xerces-C++ that use the same license as Xerces-C++),
// and distribute linked combinations including the two. You must obey
// the GNU General Public License version 2 in all respects for all of
// the code used other than Xerces-C++. If you modify this copy of the
// program, you may extend this exception to your version of the program,
// but you are not obligated to do so. If you do not wish to do so, delete
// this exception statement from your version.
//
// Furthermore, Code Synthesis Tools CC makes a special exception for
// the Free/Libre and Open Source Software (FLOSS) which is described
// in the accompanying FLOSSE file.
//

/**
 * @file
 * @brief Generated from PatientNutrition.xsd.
 */

#ifndef PATIENT_NUTRITION_HXX
#define PATIENT_NUTRITION_HXX

#ifndef XSD_CXX11
#define XSD_CXX11
#endif

#ifndef XSD_USE_CHAR
#define XSD_USE_CHAR
#endif

#ifndef XSD_CXX_TREE_USE_CHAR
#define XSD_CXX_TREE_USE_CHAR
#endif

// Begin prologue.
//
#include "Properties.hxx"

//
// End prologue.

#include <xsd/cxx/config.hxx>

#if (XSD_INT_VERSION != 4000000L)
#error XSD runtime version mismatch
#endif

#include <xsd/cxx/pre.hxx>

#include "data-model-schema.hxx"

// Forward declarations.
//
namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
      }
    }
  }
}


#include <memory>    // ::std::unique_ptr
#include <limits>    // std::numeric_limits
#include <algorithm> // std::binary_search
#include <utility>   // std::move

#include <xsd/cxx/xml/char-utf8.hxx>

#include <xsd/cxx/tree/exceptions.hxx>
#include <xsd/cxx/tree/elements.hxx>
#include <xsd/cxx/tree/containers.hxx>
#include <xsd/cxx/tree/list.hxx>

#include <xsd/cxx/xml/dom/parsing-header.hxx>

#include "Properties.hxx"

#include "NutritionData.hxx"

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      /**
       * @brief C++ namespace for the %uri:/mil/tatrc/physiology/datamodel
       * schema namespace.
       */
      namespace datamodel
      {
      }
    }
  }
}

#include <iosfwd>

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
      }
    }
  }
}

#include <iosfwd>

#include <xercesc/sax/InputSource.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMErrorHandler.hpp>

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        /**
         * @name Parsing functions for the %Nutrition document root.
         */
        //@{

        /**
         * @brief Parse a URI or a local file.
         *
         * @param uri A URI or a local file name.
         * @param f Parsing flags.
         * @param p Parsing properties. 
         * @return A pointer to the root of the object model.
         *
         * This function uses exceptions to report parsing errors.
         */
        BIOGEARS_CDM_API
        ::std::unique_ptr< ::mil::tatrc::physiology::datamodel::NutritionData >
        Nutrition (const ::std::string& uri,
                   ::xml_schema::flags f = 0,
                   const ::xml_schema::properties& p = ::xml_schema::properties ());

        /**
         * @brief Parse a URI or a local file with an error handler.
         *
         * @param uri A URI or a local file name.
         * @param eh An error handler.
         * @param f Parsing flags.
         * @param p Parsing properties. 
         * @return A pointer to the root of the object model.
         *
         * This function reports parsing errors by calling the error handler.
         */
        BIOGEARS_CDM_API
        ::std::unique_ptr< ::mil::tatrc::physiology::datamodel::NutritionData >
        Nutrition (const ::std::string& uri,
                   ::xml_schema::error_handler& eh,
                   ::xml_schema::flags f = 0,
                   const ::xml_schema::properties& p = ::xml_schema::properties ());

        /**
         * @brief Parse a URI or a local file with a Xerces-C++ DOM error
         * handler.
         *
         * @param uri A URI or a local file name.
         * @param eh A Xerces-C++ DOM error handler.
         * @param f Parsing flags.
         * @param p Parsing properties. 
         * @return A pointer to the root of the object model.
         *
         * This function reports parsing errors by calling the error handler.
         */
        BIOGEARS_CDM_API
        ::std::unique_ptr< ::mil::tatrc::physiology::datamodel::NutritionData >
        Nutrition (const ::std::string& uri,
                   ::xercesc::DOMErrorHandler& eh,
                   ::xml_schema::flags f = 0,
                   const ::xml_schema::properties& p = ::xml_schema::properties ());

        /**
         * @brief Parse a standard input stream.
         *
         * @param is A standrad input stream.
         * @param f Parsing flags.
         * @param p Parsing properties. 
         * @return A pointer to the root of the object model.
         *
         * This function uses exceptions to report parsing errors.
         */
        BIOGEARS_CDM_API
        ::std::unique_ptr< ::mil::tatrc::physiology::datamodel::NutritionData >
        Nutrition (::std::istream& is,
                   ::xml_schema::flags f = 0,
                   const ::xml_schema::properties& p = ::xml_schema::properties ());

        /**
         * @brief Parse a standard input stream with an error handler.
         *
         * @param is A standrad input stream.
         * @param eh An error handler.
         * @param f Parsing flags.
         * @param p Parsing properties. 
         * @return A pointer to the root of the object model.
         *
         * This function reports parsing errors by calling the error handler.
         */
        BIOGEARS_CDM_API
        ::std::unique_ptr< ::mil::tatrc::physiology::datamodel::NutritionData >
        Nutrition (::std::istream& is,
                   ::xml_schema::error_handler& eh,
                   ::xml_schema::flags f = 0,
                   const ::xml_schema::properties& p = ::xml_schema::properties ());

        /**
         * @brief Parse a standard input stream with a Xerces-C++ DOM error
         * handler.
         *
         * @param is A standrad input stream.
         * @param eh A Xerces-C++ DOM error handler.
         * @param f Parsing flags.
         * @param p Parsing properties. 
         * @return A pointer to the root of the object model.
         *
         * This function reports parsing errors by calling the error handler.
         */
        BIOGEARS_CDM_API
        ::std::unique_ptr< ::mil::tatrc::physiology::datamodel::NutritionData >
        Nutrition (::std::istream& is,
                   ::xercesc::DOMErrorHandler& eh,
                   ::xml_schema::flags f = 0,
                   const ::xml_schema::properties& p = ::xml_schema::properties ());

        /**
         * @brief Parse a standard input stream with a resource id.
         *
         * @param is A standrad input stream.
         * @param id A resource id.
         * @param f Parsing flags.
         * @param p Parsing properties. 
         * @return A pointer to the root of the object model.
         *
         * The resource id is used to identify the document being parsed in
         * diagnostics as well as to resolve relative paths.
         *
         * This function uses exceptions to report parsing errors.
         */
        BIOGEARS_CDM_API
        ::std::unique_ptr< ::mil::tatrc::physiology::datamodel::NutritionData >
        Nutrition (::std::istream& is,
                   const ::std::string& id,
                   ::xml_schema::flags f = 0,
                   const ::xml_schema::properties& p = ::xml_schema::properties ());

        /**
         * @brief Parse a standard input stream with a resource id and an
         * error handler.
         *
         * @param is A standrad input stream.
         * @param id A resource id.
         * @param eh An error handler.
         * @param f Parsing flags.
         * @param p Parsing properties. 
         * @return A pointer to the root of the object model.
         *
         * The resource id is used to identify the document being parsed in
         * diagnostics as well as to resolve relative paths.
         *
         * This function reports parsing errors by calling the error handler.
         */
        BIOGEARS_CDM_API
        ::std::unique_ptr< ::mil::tatrc::physiology::datamodel::NutritionData >
        Nutrition (::std::istream& is,
                   const ::std::string& id,
                   ::xml_schema::error_handler& eh,
                   ::xml_schema::flags f = 0,
                   const ::xml_schema::properties& p = ::xml_schema::properties ());

        /**
         * @brief Parse a standard input stream with a resource id and a
         * Xerces-C++ DOM error handler.
         *
         * @param is A standrad input stream.
         * @param id A resource id.
         * @param eh A Xerces-C++ DOM error handler.
         * @param f Parsing flags.
         * @param p Parsing properties. 
         * @return A pointer to the root of the object model.
         *
         * The resource id is used to identify the document being parsed in
         * diagnostics as well as to resolve relative paths.
         *
         * This function reports parsing errors by calling the error handler.
         */
        BIOGEARS_CDM_API
        ::std::unique_ptr< ::mil::tatrc::physiology::datamodel::NutritionData >
        Nutrition (::std::istream& is,
                   const ::std::string& id,
                   ::xercesc::DOMErrorHandler& eh,
                   ::xml_schema::flags f = 0,
                   const ::xml_schema::properties& p = ::xml_schema::properties ());

        /**
         * @brief Parse a Xerces-C++ input source.
         *
         * @param is A Xerces-C++ input source.
         * @param f Parsing flags.
         * @param p Parsing properties. 
         * @return A pointer to the root of the object model.
         *
         * This function uses exceptions to report parsing errors.
         */
        BIOGEARS_CDM_API
        ::std::unique_ptr< ::mil::tatrc::physiology::datamodel::NutritionData >
        Nutrition (::xercesc::InputSource& is,
                   ::xml_schema::flags f = 0,
                   const ::xml_schema::properties& p = ::xml_schema::properties ());

        /**
         * @brief Parse a Xerces-C++ input source with an error handler.
         *
         * @param is A Xerces-C++ input source.
         * @param eh An error handler.
         * @param f Parsing flags.
         * @param p Parsing properties. 
         * @return A pointer to the root of the object model.
         *
         * This function reports parsing errors by calling the error handler.
         */
        BIOGEARS_CDM_API
        ::std::unique_ptr< ::mil::tatrc::physiology::datamodel::NutritionData >
        Nutrition (::xercesc::InputSource& is,
                   ::xml_schema::error_handler& eh,
                   ::xml_schema::flags f = 0,
                   const ::xml_schema::properties& p = ::xml_schema::properties ());

        /**
         * @brief Parse a Xerces-C++ input source with a Xerces-C++ DOM
         * error handler.
         *
         * @param is A Xerces-C++ input source.
         * @param eh A Xerces-C++ DOM error handler.
         * @param f Parsing flags.
         * @param p Parsing properties. 
         * @return A pointer to the root of the object model.
         *
         * This function reports parsing errors by calling the error handler.
         */
        BIOGEARS_CDM_API
        ::std::unique_ptr< ::mil::tatrc::physiology::datamodel::NutritionData >
        Nutrition (::xercesc::InputSource& is,
                   ::xercesc::DOMErrorHandler& eh,
                   ::xml_schema::flags f = 0,
                   const ::xml_schema::properties& p = ::xml_schema::properties ());

        /**
         * @brief Parse a Xerces-C++ DOM document.
         *
         * @param d A Xerces-C++ DOM document.
         * @param f Parsing flags.
         * @param p Parsing properties. 
         * @return A pointer to the root of the object model.
         */
        BIOGEARS_CDM_API
        ::std::unique_ptr< ::mil::tatrc::physiology::datamodel::NutritionData >
        Nutrition (const ::xercesc::DOMDocument& d,
                   ::xml_schema::flags f = 0,
                   const ::xml_schema::properties& p = ::xml_schema::properties ());

        /**
         * @brief Parse a Xerces-C++ DOM document.
         *
         * @param d A pointer to the Xerces-C++ DOM document.
         * @param f Parsing flags.
         * @param p Parsing properties. 
         * @return A pointer to the root of the object model.
         *
         * This function is normally used together with the keep_dom and
         * own_dom parsing flags to assign ownership of the DOM document
         * to the object model.
         */
        BIOGEARS_CDM_API
        ::std::unique_ptr< ::mil::tatrc::physiology::datamodel::NutritionData >
        Nutrition (::xml_schema::dom::unique_ptr< ::xercesc::DOMDocument > d,
                   ::xml_schema::flags f = 0,
                   const ::xml_schema::properties& p = ::xml_schema::properties ());

        //@}
      }
    }
  }
}

#include <iosfwd>

#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMErrorHandler.hpp>
#include <xercesc/framework/XMLFormatter.hpp>

#include <xsd/cxx/xml/dom/auto-ptr.hxx>

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        /**
         * @name Serialization functions for the %Nutrition document root.
         */
        //@{

        /**
         * @brief Serialize to a standard output stream.
         *
         * @param os A standrad output stream.
         * @param x An object model to serialize.
         * @param m A namespace information map.
         * @param e A character encoding to produce XML in.
         * @param f Serialization flags.
         *
         * This function uses exceptions to report serialization errors.
         */
        BIOGEARS_CDM_API
        void
        Nutrition (::std::ostream& os,
                   const ::mil::tatrc::physiology::datamodel::NutritionData& x, 
                   const ::xml_schema::namespace_infomap& m = ::xml_schema::namespace_infomap (),
                   const ::std::string& e = "UTF-8",
                   ::xml_schema::flags f = 0);

        /**
         * @brief Serialize to a standard output stream with an error handler.
         *
         * @param os A standrad output stream.
         * @param x An object model to serialize.
         * @param eh An error handler.
         * @param m A namespace information map.
         * @param e A character encoding to produce XML in.
         * @param f Serialization flags.
         *
         * This function reports serialization errors by calling the error
         * handler.
         */
        BIOGEARS_CDM_API
        void
        Nutrition (::std::ostream& os,
                   const ::mil::tatrc::physiology::datamodel::NutritionData& x, 
                   ::xml_schema::error_handler& eh,
                   const ::xml_schema::namespace_infomap& m = ::xml_schema::namespace_infomap (),
                   const ::std::string& e = "UTF-8",
                   ::xml_schema::flags f = 0);

        /**
         * @brief Serialize to a standard output stream with a Xerces-C++ DOM
         * error handler.
         *
         * @param os A standrad output stream.
         * @param x An object model to serialize.
         * @param eh A Xerces-C++ DOM error handler.
         * @param m A namespace information map.
         * @param e A character encoding to produce XML in.
         * @param f Serialization flags.
         *
         * This function reports serialization errors by calling the error
         * handler.
         */
        BIOGEARS_CDM_API
        void
        Nutrition (::std::ostream& os,
                   const ::mil::tatrc::physiology::datamodel::NutritionData& x, 
                   ::xercesc::DOMErrorHandler& eh,
                   const ::xml_schema::namespace_infomap& m = ::xml_schema::namespace_infomap (),
                   const ::std::string& e = "UTF-8",
                   ::xml_schema::flags f = 0);

        /**
         * @brief Serialize to a Xerces-C++ XML format target.
         *
         * @param ft A Xerces-C++ XML format target.
         * @param x An object model to serialize.
         * @param m A namespace information map.
         * @param e A character encoding to produce XML in.
         * @param f Serialization flags.
         *
         * This function uses exceptions to report serialization errors.
         */
        BIOGEARS_CDM_API
        void
        Nutrition (::xercesc::XMLFormatTarget& ft,
                   const ::mil::tatrc::physiology::datamodel::NutritionData& x, 
                   const ::xml_schema::namespace_infomap& m = ::xml_schema::namespace_infomap (),
                   const ::std::string& e = "UTF-8",
                   ::xml_schema::flags f = 0);

        /**
         * @brief Serialize to a Xerces-C++ XML format target with an error
         * handler.
         *
         * @param ft A Xerces-C++ XML format target.
         * @param x An object model to serialize.
         * @param eh An error handler.
         * @param m A namespace information map.
         * @param e A character encoding to produce XML in.
         * @param f Serialization flags.
         *
         * This function reports serialization errors by calling the error
         * handler.
         */
        BIOGEARS_CDM_API
        void
        Nutrition (::xercesc::XMLFormatTarget& ft,
                   const ::mil::tatrc::physiology::datamodel::NutritionData& x, 
                   ::xml_schema::error_handler& eh,
                   const ::xml_schema::namespace_infomap& m = ::xml_schema::namespace_infomap (),
                   const ::std::string& e = "UTF-8",
                   ::xml_schema::flags f = 0);

        /**
         * @brief Serialize to a Xerces-C++ XML format target with a
         * Xerces-C++ DOM error handler.
         *
         * @param ft A Xerces-C++ XML format target.
         * @param x An object model to serialize.
         * @param eh A Xerces-C++ DOM error handler.
         * @param m A namespace information map.
         * @param e A character encoding to produce XML in.
         * @param f Serialization flags.
         *
         * This function reports serialization errors by calling the error
         * handler.
         */
        BIOGEARS_CDM_API
        void
        Nutrition (::xercesc::XMLFormatTarget& ft,
                   const ::mil::tatrc::physiology::datamodel::NutritionData& x, 
                   ::xercesc::DOMErrorHandler& eh,
                   const ::xml_schema::namespace_infomap& m = ::xml_schema::namespace_infomap (),
                   const ::std::string& e = "UTF-8",
                   ::xml_schema::flags f = 0);

        /**
         * @brief Serialize to an existing Xerces-C++ DOM document.
         *
         * @param d A Xerces-C++ DOM document.
         * @param x An object model to serialize.
         * @param f Serialization flags.
         *
         * Note that it is your responsibility to create the DOM document
         * with the correct root element as well as set the necessary
         * namespace mapping attributes.
         */
        BIOGEARS_CDM_API
        void
        Nutrition (::xercesc::DOMDocument& d,
                   const ::mil::tatrc::physiology::datamodel::NutritionData& x,
                   ::xml_schema::flags f = 0);

        /**
         * @brief Serialize to a new Xerces-C++ DOM document.
         *
         * @param x An object model to serialize.
         * @param m A namespace information map.
         * @param f Serialization flags.
         * @return A pointer to the new Xerces-C++ DOM document.
         */
        BIOGEARS_CDM_API
        ::xml_schema::dom::unique_ptr< ::xercesc::DOMDocument >
        Nutrition (const ::mil::tatrc::physiology::datamodel::NutritionData& x, 
                   const ::xml_schema::namespace_infomap& m = ::xml_schema::namespace_infomap (),
                   ::xml_schema::flags f = 0);

        //@}
      }
    }
  }
}

#include <xsd/cxx/post.hxx>

// Begin epilogue.
//
//
// End epilogue.

#endif // PATIENT_NUTRITION_HXX
