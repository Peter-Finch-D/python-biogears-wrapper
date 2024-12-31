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
 * @brief Generated from PhysiologyEngineStateData.xsd.
 */

#ifndef PHYSIOLOGY_ENGINE_STATE_DATA_HXX
#define PHYSIOLOGY_ENGINE_STATE_DATA_HXX

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
        class PhysiologyEngineStateData;
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

#include "ObjectData.hxx"

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        class ScalarTimeData;
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
        class PatientData;
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
        class ConditionData;
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
        class ActionData;
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
        class SubstanceData;
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
        class SubstanceCompoundData;
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
        class SystemData;
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
        class CompartmentManagerData;
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
        class PhysiologyEngineConfigurationData;
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
        class CircuitManagerData;
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
        class DataRequestsData;
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
         * @brief Class corresponding to the %PhysiologyEngineStateData schema type.
         *
         * @nosubgrouping
         */
        class BIOGEARS_CDM_API PhysiologyEngineStateData: public ::mil::tatrc::physiology::datamodel::ObjectData
        {
          public:
          /**
           * @name SimulationTime
           *
           * @brief Accessor and modifier functions for the %SimulationTime
           * optional element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::ScalarTimeData SimulationTime_type;

          /**
           * @brief Element optional container type.
           */
          typedef ::xsd::cxx::tree::optional< SimulationTime_type > SimulationTime_optional;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< SimulationTime_type, char > SimulationTime_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * container.
           *
           * @return A constant reference to the optional container.
           */
          const SimulationTime_optional&
          SimulationTime () const;

          /**
           * @brief Return a read-write reference to the element container.
           *
           * @return A reference to the optional container.
           */
          SimulationTime_optional&
          SimulationTime ();

          /**
           * @brief Set the element value.
           *
           * @param x A new value to set.
           *
           * This function makes a copy of its argument and sets it as
           * the new value of the element.
           */
          void
          SimulationTime (const SimulationTime_type& x);

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
          SimulationTime (const SimulationTime_optional& x);

          /**
           * @brief Set the element value without copying.
           *
           * @param p A new value to use.
           *
           * This function will try to use the passed value directly instead
           * of making a copy.
           */
          void
          SimulationTime (::std::unique_ptr< SimulationTime_type > p);

          //@}

          /**
           * @name Patient
           *
           * @brief Accessor and modifier functions for the %Patient
           * optional element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::PatientData Patient_type;

          /**
           * @brief Element optional container type.
           */
          typedef ::xsd::cxx::tree::optional< Patient_type > Patient_optional;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< Patient_type, char > Patient_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * container.
           *
           * @return A constant reference to the optional container.
           */
          const Patient_optional&
          Patient () const;

          /**
           * @brief Return a read-write reference to the element container.
           *
           * @return A reference to the optional container.
           */
          Patient_optional&
          Patient ();

          /**
           * @brief Set the element value.
           *
           * @param x A new value to set.
           *
           * This function makes a copy of its argument and sets it as
           * the new value of the element.
           */
          void
          Patient (const Patient_type& x);

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
          Patient (const Patient_optional& x);

          /**
           * @brief Set the element value without copying.
           *
           * @param p A new value to use.
           *
           * This function will try to use the passed value directly instead
           * of making a copy.
           */
          void
          Patient (::std::unique_ptr< Patient_type > p);

          //@}

          /**
           * @name Condition
           *
           * @brief Accessor and modifier functions for the %Condition
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::ConditionData Condition_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< Condition_type > Condition_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef Condition_sequence::iterator Condition_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef Condition_sequence::const_iterator Condition_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< Condition_type, char > Condition_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const Condition_sequence&
          Condition () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          Condition_sequence&
          Condition ();

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
          Condition (const Condition_sequence& s);

          //@}

          /**
           * @name ActiveAction
           *
           * @brief Accessor and modifier functions for the %ActiveAction
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::ActionData ActiveAction_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< ActiveAction_type > ActiveAction_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef ActiveAction_sequence::iterator ActiveAction_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef ActiveAction_sequence::const_iterator ActiveAction_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< ActiveAction_type, char > ActiveAction_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const ActiveAction_sequence&
          ActiveAction () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          ActiveAction_sequence&
          ActiveAction ();

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
          ActiveAction (const ActiveAction_sequence& s);

          //@}

          /**
           * @name ActiveSubstance
           *
           * @brief Accessor and modifier functions for the %ActiveSubstance
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::SubstanceData ActiveSubstance_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< ActiveSubstance_type > ActiveSubstance_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef ActiveSubstance_sequence::iterator ActiveSubstance_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef ActiveSubstance_sequence::const_iterator ActiveSubstance_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< ActiveSubstance_type, char > ActiveSubstance_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const ActiveSubstance_sequence&
          ActiveSubstance () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          ActiveSubstance_sequence&
          ActiveSubstance ();

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
          ActiveSubstance (const ActiveSubstance_sequence& s);

          //@}

          /**
           * @name ActiveSubstanceCompound
           *
           * @brief Accessor and modifier functions for the %ActiveSubstanceCompound
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::SubstanceCompoundData ActiveSubstanceCompound_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< ActiveSubstanceCompound_type > ActiveSubstanceCompound_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef ActiveSubstanceCompound_sequence::iterator ActiveSubstanceCompound_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef ActiveSubstanceCompound_sequence::const_iterator ActiveSubstanceCompound_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< ActiveSubstanceCompound_type, char > ActiveSubstanceCompound_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const ActiveSubstanceCompound_sequence&
          ActiveSubstanceCompound () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          ActiveSubstanceCompound_sequence&
          ActiveSubstanceCompound ();

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
          ActiveSubstanceCompound (const ActiveSubstanceCompound_sequence& s);

          //@}

          /**
           * @name System
           *
           * @brief Accessor and modifier functions for the %System
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::SystemData System_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< System_type > System_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef System_sequence::iterator System_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef System_sequence::const_iterator System_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< System_type, char > System_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const System_sequence&
          System () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          System_sequence&
          System ();

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
          System (const System_sequence& s);

          //@}

          /**
           * @name CompartmentManager
           *
           * @brief Accessor and modifier functions for the %CompartmentManager
           * optional element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::CompartmentManagerData CompartmentManager_type;

          /**
           * @brief Element optional container type.
           */
          typedef ::xsd::cxx::tree::optional< CompartmentManager_type > CompartmentManager_optional;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< CompartmentManager_type, char > CompartmentManager_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * container.
           *
           * @return A constant reference to the optional container.
           */
          const CompartmentManager_optional&
          CompartmentManager () const;

          /**
           * @brief Return a read-write reference to the element container.
           *
           * @return A reference to the optional container.
           */
          CompartmentManager_optional&
          CompartmentManager ();

          /**
           * @brief Set the element value.
           *
           * @param x A new value to set.
           *
           * This function makes a copy of its argument and sets it as
           * the new value of the element.
           */
          void
          CompartmentManager (const CompartmentManager_type& x);

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
          CompartmentManager (const CompartmentManager_optional& x);

          /**
           * @brief Set the element value without copying.
           *
           * @param p A new value to use.
           *
           * This function will try to use the passed value directly instead
           * of making a copy.
           */
          void
          CompartmentManager (::std::unique_ptr< CompartmentManager_type > p);

          //@}

          /**
           * @name Configuration
           *
           * @brief Accessor and modifier functions for the %Configuration
           * optional element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::PhysiologyEngineConfigurationData Configuration_type;

          /**
           * @brief Element optional container type.
           */
          typedef ::xsd::cxx::tree::optional< Configuration_type > Configuration_optional;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< Configuration_type, char > Configuration_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * container.
           *
           * @return A constant reference to the optional container.
           */
          const Configuration_optional&
          Configuration () const;

          /**
           * @brief Return a read-write reference to the element container.
           *
           * @return A reference to the optional container.
           */
          Configuration_optional&
          Configuration ();

          /**
           * @brief Set the element value.
           *
           * @param x A new value to set.
           *
           * This function makes a copy of its argument and sets it as
           * the new value of the element.
           */
          void
          Configuration (const Configuration_type& x);

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
          Configuration (const Configuration_optional& x);

          /**
           * @brief Set the element value without copying.
           *
           * @param p A new value to use.
           *
           * This function will try to use the passed value directly instead
           * of making a copy.
           */
          void
          Configuration (::std::unique_ptr< Configuration_type > p);

          //@}

          /**
           * @name CircuitManager
           *
           * @brief Accessor and modifier functions for the %CircuitManager
           * optional element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::CircuitManagerData CircuitManager_type;

          /**
           * @brief Element optional container type.
           */
          typedef ::xsd::cxx::tree::optional< CircuitManager_type > CircuitManager_optional;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< CircuitManager_type, char > CircuitManager_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * container.
           *
           * @return A constant reference to the optional container.
           */
          const CircuitManager_optional&
          CircuitManager () const;

          /**
           * @brief Return a read-write reference to the element container.
           *
           * @return A reference to the optional container.
           */
          CircuitManager_optional&
          CircuitManager ();

          /**
           * @brief Set the element value.
           *
           * @param x A new value to set.
           *
           * This function makes a copy of its argument and sets it as
           * the new value of the element.
           */
          void
          CircuitManager (const CircuitManager_type& x);

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
          CircuitManager (const CircuitManager_optional& x);

          /**
           * @brief Set the element value without copying.
           *
           * @param p A new value to use.
           *
           * This function will try to use the passed value directly instead
           * of making a copy.
           */
          void
          CircuitManager (::std::unique_ptr< CircuitManager_type > p);

          //@}

          /**
           * @name DataRequests
           *
           * @brief Accessor and modifier functions for the %DataRequests
           * optional element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::DataRequestsData DataRequests_type;

          /**
           * @brief Element optional container type.
           */
          typedef ::xsd::cxx::tree::optional< DataRequests_type > DataRequests_optional;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< DataRequests_type, char > DataRequests_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * container.
           *
           * @return A constant reference to the optional container.
           */
          const DataRequests_optional&
          DataRequests () const;

          /**
           * @brief Return a read-write reference to the element container.
           *
           * @return A reference to the optional container.
           */
          DataRequests_optional&
          DataRequests ();

          /**
           * @brief Set the element value.
           *
           * @param x A new value to set.
           *
           * This function makes a copy of its argument and sets it as
           * the new value of the element.
           */
          void
          DataRequests (const DataRequests_type& x);

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
          DataRequests (const DataRequests_optional& x);

          /**
           * @brief Set the element value without copying.
           *
           * @param p A new value to use.
           *
           * This function will try to use the passed value directly instead
           * of making a copy.
           */
          void
          DataRequests (::std::unique_ptr< DataRequests_type > p);

          //@}

          /**
           * @name Constructors
           */
          //@{

          /**
           * @brief Create an instance from the ultimate base and
           * initializers for required elements and attributes.
           */
          PhysiologyEngineStateData ();

          /**
           * @brief Create an instance from a DOM element.
           *
           * @param e A DOM element to extract the data from.
           * @param f Flags to create the new instance with.
           * @param c A pointer to the object that will contain the new
           * instance.
           */
          PhysiologyEngineStateData (const ::xercesc::DOMElement& e,
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
          PhysiologyEngineStateData (const PhysiologyEngineStateData& x,
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
          virtual PhysiologyEngineStateData*
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
          PhysiologyEngineStateData&
          operator= (const PhysiologyEngineStateData& x);

          //@}

          /**
           * @brief Destructor.
           */
          virtual 
          ~PhysiologyEngineStateData ();

          // Implementation.
          //

          //@cond

          protected:
          void
          parse (::xsd::cxx::xml::dom::parser< char >&,
                 ::xml_schema::flags);

          protected:
          SimulationTime_optional SimulationTime_;
          Patient_optional Patient_;
          Condition_sequence Condition_;
          ActiveAction_sequence ActiveAction_;
          ActiveSubstance_sequence ActiveSubstance_;
          ActiveSubstanceCompound_sequence ActiveSubstanceCompound_;
          System_sequence System_;
          CompartmentManager_optional CompartmentManager_;
          Configuration_optional Configuration_;
          CircuitManager_optional CircuitManager_;
          DataRequests_optional DataRequests_;

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
        operator<< (::std::ostream&, const PhysiologyEngineStateData&);
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
        operator<< (::xercesc::DOMElement&, const PhysiologyEngineStateData&);
      }
    }
  }
}

#include <xsd/cxx/post.hxx>

// Begin epilogue.
//
//
// End epilogue.

#endif // PHYSIOLOGY_ENGINE_STATE_DATA_HXX
