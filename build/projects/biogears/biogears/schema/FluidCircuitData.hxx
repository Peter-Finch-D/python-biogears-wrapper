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
 * @brief Generated from FluidCircuitData.xsd.
 */

#ifndef FLUID_CIRCUIT_DATA_HXX
#define FLUID_CIRCUIT_DATA_HXX

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
        class FluidCircuitData;
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

#include "CircuitData.hxx"

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
        /**
         * @brief Class corresponding to the %FluidCircuitData schema type.
         *
         * @nosubgrouping
         */
        class BIOGEARS_CDM_API FluidCircuitData: public ::mil::tatrc::physiology::datamodel::CircuitData
        {
          public:
          /**
           * @name Node
           *
           * @brief Accessor and modifier functions for the %Node
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::xml_schema::string Node_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< Node_type > Node_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef Node_sequence::iterator Node_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef Node_sequence::const_iterator Node_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< Node_type, char > Node_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const Node_sequence&
          Node () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          Node_sequence&
          Node ();

          /**
           * @brief Copy elements from a given sequence.
           *
           * @param s A sequence to copy elements from.
           *
           * For each element in @a s this function makes a copy and adds it 
           * to the sequence. Note that this operation completely changes the 
           * sequence and all old elements will be lost.
           */
          void
          Node (const Node_sequence& s);

          //@}

          /**
           * @name Path
           *
           * @brief Accessor and modifier functions for the %Path
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::xml_schema::string Path_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< Path_type > Path_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef Path_sequence::iterator Path_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef Path_sequence::const_iterator Path_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< Path_type, char > Path_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const Path_sequence&
          Path () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          Path_sequence&
          Path ();

          /**
           * @brief Copy elements from a given sequence.
           *
           * @param s A sequence to copy elements from.
           *
           * For each element in @a s this function makes a copy and adds it 
           * to the sequence. Note that this operation completely changes the 
           * sequence and all old elements will be lost.
           */
          void
          Path (const Path_sequence& s);

          //@}

          /**
           * @name Constructors
           */
          //@{

          /**
           * @brief Default constructor.
           *
           * Note that this constructor leaves required elements and
           * attributes uninitialized.
           */
          FluidCircuitData ();

          /**
           * @brief Create an instance from the ultimate base and
           * initializers for required elements and attributes.
           */
          FluidCircuitData (const Name_type&);

          /**
           * @brief Create an instance from the ultimate base and
           * initializers for required elements and attributes
           * (::std::unique_ptr version).
           *
           * This constructor will try to use the passed values directly
           * instead of making copies.
           */
          FluidCircuitData (::std::unique_ptr< Name_type >);

          /**
           * @brief Create an instance from a DOM element.
           *
           * @param e A DOM element to extract the data from.
           * @param f Flags to create the new instance with.
           * @param c A pointer to the object that will contain the new
           * instance.
           */
          FluidCircuitData (const ::xercesc::DOMElement& e,
                            ::xml_schema::flags f = 0,
                            ::xml_schema::container* c = 0);

          /**
           * @brief Copy constructor.
           *
           * @param x An instance to make a copy of.
           * @param f Flags to create the copy with.
           * @param c A pointer to the object that will contain the copy.
           *
           * For polymorphic object models use the @c _clone function instead.
           */
          FluidCircuitData (const FluidCircuitData& x,
                            ::xml_schema::flags f = 0,
                            ::xml_schema::container* c = 0);

          /**
           * @brief Copy the instance polymorphically.
           *
           * @param f Flags to create the copy with.
           * @param c A pointer to the object that will contain the copy.
           * @return A pointer to the dynamically allocated copy.
           *
           * This function ensures that the dynamic type of the instance is
           * used for copying and should be used for polymorphic object
           * models instead of the copy constructor.
           */
          virtual FluidCircuitData*
          _clone (::xml_schema::flags f = 0,
                  ::xml_schema::container* c = 0) const;

          /**
           * @brief Copy assignment operator.
           *
           * @param x An instance to make a copy of.
           * @return A reference to itself.
           *
           * For polymorphic object models use the @c _clone function instead.
           */
          FluidCircuitData&
          operator= (const FluidCircuitData& x);

          //@}

          /**
           * @brief Destructor.
           */
          virtual 
          ~FluidCircuitData ();

          // Implementation.
          //

          //@cond

          protected:
          void
          parse (::xsd::cxx::xml::dom::parser< char >&,
                 ::xml_schema::flags);

          protected:
          Node_sequence Node_;
          Path_sequence Path_;

          //@endcond
        };
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
        BIOGEARS_CDM_API
        ::std::ostream&
        operator<< (::std::ostream&, const FluidCircuitData&);
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
        BIOGEARS_CDM_API
        void
        operator<< (::xercesc::DOMElement&, const FluidCircuitData&);
      }
    }
  }
}

#include <xsd/cxx/post.hxx>

// Begin epilogue.
//
//
// End epilogue.

#endif // FLUID_CIRCUIT_DATA_HXX
