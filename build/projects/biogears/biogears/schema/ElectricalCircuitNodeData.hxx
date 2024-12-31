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
 * @brief Generated from ElectricalCircuitNodeData.xsd.
 */

#ifndef ELECTRICAL_CIRCUIT_NODE_DATA_HXX
#define ELECTRICAL_CIRCUIT_NODE_DATA_HXX

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
        class ElectricalCircuitNodeData;
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

#include "CircuitNodeData.hxx"

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        class ScalarElectricPotentialData;
      }
    }
  }
}

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        class ScalarElectricChargeData;
      }
    }
  }
}

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
         * @brief Class corresponding to the %ElectricalCircuitNodeData schema type.
         *
         * @nosubgrouping
         */
        class BIOGEARS_CDM_API ElectricalCircuitNodeData: public ::mil::tatrc::physiology::datamodel::CircuitNodeData
        {
          public:
          /**
           * @name Voltage
           *
           * @brief Accessor and modifier functions for the %Voltage
           * optional element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::ScalarElectricPotentialData Voltage_type;

          /**
           * @brief Element optional container type.
           */
          typedef ::xsd::cxx::tree::optional< Voltage_type > Voltage_optional;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< Voltage_type, char > Voltage_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * container.
           *
           * @return A constant reference to the optional container.
           */
          const Voltage_optional&
          Voltage () const;

          /**
           * @brief Return a read-write reference to the element container.
           *
           * @return A reference to the optional container.
           */
          Voltage_optional&
          Voltage ();

          /**
           * @brief Set the element value.
           *
           * @param x A new value to set.
           *
           * This function makes a copy of its argument and sets it as
           * the new value of the element.
           */
          void
          Voltage (const Voltage_type& x);

          /**
           * @brief Set the element value.
           *
           * @param x An optional container with the new value to set.
           *
           * If the value is present in @a x then this function makes a copy 
           * of this value and sets it as the new value of the element.
           * Otherwise the element container is set the 'not present' state.
           */
          void
          Voltage (const Voltage_optional& x);

          /**
           * @brief Set the element value without copying.
           *
           * @param p A new value to use.
           *
           * This function will try to use the passed value directly instead
           * of making a copy.
           */
          void
          Voltage (::std::unique_ptr< Voltage_type > p);

          //@}

          /**
           * @name NextVoltage
           *
           * @brief Accessor and modifier functions for the %NextVoltage
           * optional element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::ScalarElectricPotentialData NextVoltage_type;

          /**
           * @brief Element optional container type.
           */
          typedef ::xsd::cxx::tree::optional< NextVoltage_type > NextVoltage_optional;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< NextVoltage_type, char > NextVoltage_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * container.
           *
           * @return A constant reference to the optional container.
           */
          const NextVoltage_optional&
          NextVoltage () const;

          /**
           * @brief Return a read-write reference to the element container.
           *
           * @return A reference to the optional container.
           */
          NextVoltage_optional&
          NextVoltage ();

          /**
           * @brief Set the element value.
           *
           * @param x A new value to set.
           *
           * This function makes a copy of its argument and sets it as
           * the new value of the element.
           */
          void
          NextVoltage (const NextVoltage_type& x);

          /**
           * @brief Set the element value.
           *
           * @param x An optional container with the new value to set.
           *
           * If the value is present in @a x then this function makes a copy 
           * of this value and sets it as the new value of the element.
           * Otherwise the element container is set the 'not present' state.
           */
          void
          NextVoltage (const NextVoltage_optional& x);

          /**
           * @brief Set the element value without copying.
           *
           * @param p A new value to use.
           *
           * This function will try to use the passed value directly instead
           * of making a copy.
           */
          void
          NextVoltage (::std::unique_ptr< NextVoltage_type > p);

          //@}

          /**
           * @name Charge
           *
           * @brief Accessor and modifier functions for the %Charge
           * optional element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::ScalarElectricChargeData Charge_type;

          /**
           * @brief Element optional container type.
           */
          typedef ::xsd::cxx::tree::optional< Charge_type > Charge_optional;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< Charge_type, char > Charge_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * container.
           *
           * @return A constant reference to the optional container.
           */
          const Charge_optional&
          Charge () const;

          /**
           * @brief Return a read-write reference to the element container.
           *
           * @return A reference to the optional container.
           */
          Charge_optional&
          Charge ();

          /**
           * @brief Set the element value.
           *
           * @param x A new value to set.
           *
           * This function makes a copy of its argument and sets it as
           * the new value of the element.
           */
          void
          Charge (const Charge_type& x);

          /**
           * @brief Set the element value.
           *
           * @param x An optional container with the new value to set.
           *
           * If the value is present in @a x then this function makes a copy 
           * of this value and sets it as the new value of the element.
           * Otherwise the element container is set the 'not present' state.
           */
          void
          Charge (const Charge_optional& x);

          /**
           * @brief Set the element value without copying.
           *
           * @param p A new value to use.
           *
           * This function will try to use the passed value directly instead
           * of making a copy.
           */
          void
          Charge (::std::unique_ptr< Charge_type > p);

          //@}

          /**
           * @name NextCharge
           *
           * @brief Accessor and modifier functions for the %NextCharge
           * optional element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::ScalarElectricChargeData NextCharge_type;

          /**
           * @brief Element optional container type.
           */
          typedef ::xsd::cxx::tree::optional< NextCharge_type > NextCharge_optional;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< NextCharge_type, char > NextCharge_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * container.
           *
           * @return A constant reference to the optional container.
           */
          const NextCharge_optional&
          NextCharge () const;

          /**
           * @brief Return a read-write reference to the element container.
           *
           * @return A reference to the optional container.
           */
          NextCharge_optional&
          NextCharge ();

          /**
           * @brief Set the element value.
           *
           * @param x A new value to set.
           *
           * This function makes a copy of its argument and sets it as
           * the new value of the element.
           */
          void
          NextCharge (const NextCharge_type& x);

          /**
           * @brief Set the element value.
           *
           * @param x An optional container with the new value to set.
           *
           * If the value is present in @a x then this function makes a copy 
           * of this value and sets it as the new value of the element.
           * Otherwise the element container is set the 'not present' state.
           */
          void
          NextCharge (const NextCharge_optional& x);

          /**
           * @brief Set the element value without copying.
           *
           * @param p A new value to use.
           *
           * This function will try to use the passed value directly instead
           * of making a copy.
           */
          void
          NextCharge (::std::unique_ptr< NextCharge_type > p);

          //@}

          /**
           * @name ChargeBaseline
           *
           * @brief Accessor and modifier functions for the %ChargeBaseline
           * optional element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::ScalarElectricChargeData ChargeBaseline_type;

          /**
           * @brief Element optional container type.
           */
          typedef ::xsd::cxx::tree::optional< ChargeBaseline_type > ChargeBaseline_optional;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< ChargeBaseline_type, char > ChargeBaseline_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * container.
           *
           * @return A constant reference to the optional container.
           */
          const ChargeBaseline_optional&
          ChargeBaseline () const;

          /**
           * @brief Return a read-write reference to the element container.
           *
           * @return A reference to the optional container.
           */
          ChargeBaseline_optional&
          ChargeBaseline ();

          /**
           * @brief Set the element value.
           *
           * @param x A new value to set.
           *
           * This function makes a copy of its argument and sets it as
           * the new value of the element.
           */
          void
          ChargeBaseline (const ChargeBaseline_type& x);

          /**
           * @brief Set the element value.
           *
           * @param x An optional container with the new value to set.
           *
           * If the value is present in @a x then this function makes a copy 
           * of this value and sets it as the new value of the element.
           * Otherwise the element container is set the 'not present' state.
           */
          void
          ChargeBaseline (const ChargeBaseline_optional& x);

          /**
           * @brief Set the element value without copying.
           *
           * @param p A new value to use.
           *
           * This function will try to use the passed value directly instead
           * of making a copy.
           */
          void
          ChargeBaseline (::std::unique_ptr< ChargeBaseline_type > p);

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
          ElectricalCircuitNodeData ();

          /**
           * @brief Create an instance from the ultimate base and
           * initializers for required elements and attributes.
           */
          ElectricalCircuitNodeData (const Name_type&);

          /**
           * @brief Create an instance from the ultimate base and
           * initializers for required elements and attributes
           * (::std::unique_ptr version).
           *
           * This constructor will try to use the passed values directly
           * instead of making copies.
           */
          ElectricalCircuitNodeData (::std::unique_ptr< Name_type >);

          /**
           * @brief Create an instance from a DOM element.
           *
           * @param e A DOM element to extract the data from.
           * @param f Flags to create the new instance with.
           * @param c A pointer to the object that will contain the new
           * instance.
           */
          ElectricalCircuitNodeData (const ::xercesc::DOMElement& e,
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
          ElectricalCircuitNodeData (const ElectricalCircuitNodeData& x,
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
          virtual ElectricalCircuitNodeData*
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
          ElectricalCircuitNodeData&
          operator= (const ElectricalCircuitNodeData& x);

          //@}

          /**
           * @brief Destructor.
           */
          virtual 
          ~ElectricalCircuitNodeData ();

          // Implementation.
          //

          //@cond

          protected:
          void
          parse (::xsd::cxx::xml::dom::parser< char >&,
                 ::xml_schema::flags);

          protected:
          Voltage_optional Voltage_;
          NextVoltage_optional NextVoltage_;
          Charge_optional Charge_;
          NextCharge_optional NextCharge_;
          ChargeBaseline_optional ChargeBaseline_;

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
        operator<< (::std::ostream&, const ElectricalCircuitNodeData&);
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
        operator<< (::xercesc::DOMElement&, const ElectricalCircuitNodeData&);
      }
    }
  }
}

#include <xsd/cxx/post.hxx>

// Begin epilogue.
//
//
// End epilogue.

#endif // ELECTRICAL_CIRCUIT_NODE_DATA_HXX
