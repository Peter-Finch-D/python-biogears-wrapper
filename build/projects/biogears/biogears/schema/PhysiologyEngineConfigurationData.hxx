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
 * @brief Generated from PhysiologyEngineConfigurationData.xsd.
 */

#ifndef PHYSIOLOGY_ENGINE_CONFIGURATION_DATA_HXX
#define PHYSIOLOGY_ENGINE_CONFIGURATION_DATA_HXX

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
        class PhysiologyEngineConfigurationData;
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
        class enumOnOff;
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
        class ElectroCardioGramWaveformInterpolatorData;
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
        class PhysiologyEngineStabilizationData;
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
         * @brief Class corresponding to the %PhysiologyEngineConfigurationData schema type.
         *
         * @nosubgrouping
         */
        class BIOGEARS_CDM_API PhysiologyEngineConfigurationData: public ::mil::tatrc::physiology::datamodel::ObjectData
        {
          public:
          /**
           * @name TimeStep
           *
           * @brief Accessor and modifier functions for the %TimeStep
           * optional element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::ScalarTimeData TimeStep_type;

          /**
           * @brief Element optional container type.
           */
          typedef ::xsd::cxx::tree::optional< TimeStep_type > TimeStep_optional;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< TimeStep_type, char > TimeStep_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * container.
           *
           * @return A constant reference to the optional container.
           */
          const TimeStep_optional&
          TimeStep () const;

          /**
           * @brief Return a read-write reference to the element container.
           *
           * @return A reference to the optional container.
           */
          TimeStep_optional&
          TimeStep ();

          /**
           * @brief Set the element value.
           *
           * @param x A new value to set.
           *
           * This function makes a copy of its argument and sets it as
           * the new value of the element.
           */
          void
          TimeStep (const TimeStep_type& x);

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
          TimeStep (const TimeStep_optional& x);

          /**
           * @brief Set the element value without copying.
           *
           * @param p A new value to use.
           *
           * This function will try to use the passed value directly instead
           * of making a copy.
           */
          void
          TimeStep (::std::unique_ptr< TimeStep_type > p);

          //@}

          /**
           * @name WritePatientBaselineFile
           *
           * @brief Accessor and modifier functions for the %WritePatientBaselineFile
           * optional element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::enumOnOff WritePatientBaselineFile_type;

          /**
           * @brief Element optional container type.
           */
          typedef ::xsd::cxx::tree::optional< WritePatientBaselineFile_type > WritePatientBaselineFile_optional;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< WritePatientBaselineFile_type, char > WritePatientBaselineFile_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * container.
           *
           * @return A constant reference to the optional container.
           */
          const WritePatientBaselineFile_optional&
          WritePatientBaselineFile () const;

          /**
           * @brief Return a read-write reference to the element container.
           *
           * @return A reference to the optional container.
           */
          WritePatientBaselineFile_optional&
          WritePatientBaselineFile ();

          /**
           * @brief Set the element value.
           *
           * @param x A new value to set.
           *
           * This function makes a copy of its argument and sets it as
           * the new value of the element.
           */
          void
          WritePatientBaselineFile (const WritePatientBaselineFile_type& x);

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
          WritePatientBaselineFile (const WritePatientBaselineFile_optional& x);

          /**
           * @brief Set the element value without copying.
           *
           * @param p A new value to use.
           *
           * This function will try to use the passed value directly instead
           * of making a copy.
           */
          void
          WritePatientBaselineFile (::std::unique_ptr< WritePatientBaselineFile_type > p);

          //@}

          /**
           * @name ElectroCardioGramInterpolator
           *
           * @brief Accessor and modifier functions for the %ElectroCardioGramInterpolator
           * optional element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::ElectroCardioGramWaveformInterpolatorData ElectroCardioGramInterpolator_type;

          /**
           * @brief Element optional container type.
           */
          typedef ::xsd::cxx::tree::optional< ElectroCardioGramInterpolator_type > ElectroCardioGramInterpolator_optional;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< ElectroCardioGramInterpolator_type, char > ElectroCardioGramInterpolator_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * container.
           *
           * @return A constant reference to the optional container.
           */
          const ElectroCardioGramInterpolator_optional&
          ElectroCardioGramInterpolator () const;

          /**
           * @brief Return a read-write reference to the element container.
           *
           * @return A reference to the optional container.
           */
          ElectroCardioGramInterpolator_optional&
          ElectroCardioGramInterpolator ();

          /**
           * @brief Set the element value.
           *
           * @param x A new value to set.
           *
           * This function makes a copy of its argument and sets it as
           * the new value of the element.
           */
          void
          ElectroCardioGramInterpolator (const ElectroCardioGramInterpolator_type& x);

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
          ElectroCardioGramInterpolator (const ElectroCardioGramInterpolator_optional& x);

          /**
           * @brief Set the element value without copying.
           *
           * @param p A new value to use.
           *
           * This function will try to use the passed value directly instead
           * of making a copy.
           */
          void
          ElectroCardioGramInterpolator (::std::unique_ptr< ElectroCardioGramInterpolator_type > p);

          //@}

          /**
           * @name ElectroCardioGramInterpolatorFile
           *
           * @brief Accessor and modifier functions for the %ElectroCardioGramInterpolatorFile
           * optional element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::xml_schema::string ElectroCardioGramInterpolatorFile_type;

          /**
           * @brief Element optional container type.
           */
          typedef ::xsd::cxx::tree::optional< ElectroCardioGramInterpolatorFile_type > ElectroCardioGramInterpolatorFile_optional;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< ElectroCardioGramInterpolatorFile_type, char > ElectroCardioGramInterpolatorFile_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * container.
           *
           * @return A constant reference to the optional container.
           */
          const ElectroCardioGramInterpolatorFile_optional&
          ElectroCardioGramInterpolatorFile () const;

          /**
           * @brief Return a read-write reference to the element container.
           *
           * @return A reference to the optional container.
           */
          ElectroCardioGramInterpolatorFile_optional&
          ElectroCardioGramInterpolatorFile ();

          /**
           * @brief Set the element value.
           *
           * @param x A new value to set.
           *
           * This function makes a copy of its argument and sets it as
           * the new value of the element.
           */
          void
          ElectroCardioGramInterpolatorFile (const ElectroCardioGramInterpolatorFile_type& x);

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
          ElectroCardioGramInterpolatorFile (const ElectroCardioGramInterpolatorFile_optional& x);

          /**
           * @brief Set the element value without copying.
           *
           * @param p A new value to use.
           *
           * This function will try to use the passed value directly instead
           * of making a copy.
           */
          void
          ElectroCardioGramInterpolatorFile (::std::unique_ptr< ElectroCardioGramInterpolatorFile_type > p);

          //@}

          /**
           * @name StabilizationCriteria
           *
           * @brief Accessor and modifier functions for the %StabilizationCriteria
           * optional element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::PhysiologyEngineStabilizationData StabilizationCriteria_type;

          /**
           * @brief Element optional container type.
           */
          typedef ::xsd::cxx::tree::optional< StabilizationCriteria_type > StabilizationCriteria_optional;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< StabilizationCriteria_type, char > StabilizationCriteria_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * container.
           *
           * @return A constant reference to the optional container.
           */
          const StabilizationCriteria_optional&
          StabilizationCriteria () const;

          /**
           * @brief Return a read-write reference to the element container.
           *
           * @return A reference to the optional container.
           */
          StabilizationCriteria_optional&
          StabilizationCriteria ();

          /**
           * @brief Set the element value.
           *
           * @param x A new value to set.
           *
           * This function makes a copy of its argument and sets it as
           * the new value of the element.
           */
          void
          StabilizationCriteria (const StabilizationCriteria_type& x);

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
          StabilizationCriteria (const StabilizationCriteria_optional& x);

          /**
           * @brief Set the element value without copying.
           *
           * @param p A new value to use.
           *
           * This function will try to use the passed value directly instead
           * of making a copy.
           */
          void
          StabilizationCriteria (::std::unique_ptr< StabilizationCriteria_type > p);

          //@}

          /**
           * @name StabilizationCriteriaFile
           *
           * @brief Accessor and modifier functions for the %StabilizationCriteriaFile
           * optional element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::xml_schema::string StabilizationCriteriaFile_type;

          /**
           * @brief Element optional container type.
           */
          typedef ::xsd::cxx::tree::optional< StabilizationCriteriaFile_type > StabilizationCriteriaFile_optional;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< StabilizationCriteriaFile_type, char > StabilizationCriteriaFile_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * container.
           *
           * @return A constant reference to the optional container.
           */
          const StabilizationCriteriaFile_optional&
          StabilizationCriteriaFile () const;

          /**
           * @brief Return a read-write reference to the element container.
           *
           * @return A reference to the optional container.
           */
          StabilizationCriteriaFile_optional&
          StabilizationCriteriaFile ();

          /**
           * @brief Set the element value.
           *
           * @param x A new value to set.
           *
           * This function makes a copy of its argument and sets it as
           * the new value of the element.
           */
          void
          StabilizationCriteriaFile (const StabilizationCriteriaFile_type& x);

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
          StabilizationCriteriaFile (const StabilizationCriteriaFile_optional& x);

          /**
           * @brief Set the element value without copying.
           *
           * @param p A new value to use.
           *
           * This function will try to use the passed value directly instead
           * of making a copy.
           */
          void
          StabilizationCriteriaFile (::std::unique_ptr< StabilizationCriteriaFile_type > p);

          //@}

          /**
           * @name Constructors
           */
          //@{

          /**
           * @brief Create an instance from the ultimate base and
           * initializers for required elements and attributes.
           */
          PhysiologyEngineConfigurationData ();

          /**
           * @brief Create an instance from a DOM element.
           *
           * @param e A DOM element to extract the data from.
           * @param f Flags to create the new instance with.
           * @param c A pointer to the object that will contain the new
           * instance.
           */
          PhysiologyEngineConfigurationData (const ::xercesc::DOMElement& e,
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
          PhysiologyEngineConfigurationData (const PhysiologyEngineConfigurationData& x,
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
          virtual PhysiologyEngineConfigurationData*
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
          PhysiologyEngineConfigurationData&
          operator= (const PhysiologyEngineConfigurationData& x);

          //@}

          /**
           * @brief Destructor.
           */
          virtual 
          ~PhysiologyEngineConfigurationData ();

          // Implementation.
          //

          //@cond

          protected:
          void
          parse (::xsd::cxx::xml::dom::parser< char >&,
                 ::xml_schema::flags);

          protected:
          TimeStep_optional TimeStep_;
          WritePatientBaselineFile_optional WritePatientBaselineFile_;
          ElectroCardioGramInterpolator_optional ElectroCardioGramInterpolator_;
          ElectroCardioGramInterpolatorFile_optional ElectroCardioGramInterpolatorFile_;
          StabilizationCriteria_optional StabilizationCriteria_;
          StabilizationCriteriaFile_optional StabilizationCriteriaFile_;

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
        operator<< (::std::ostream&, const PhysiologyEngineConfigurationData&);
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
        operator<< (::xercesc::DOMElement&, const PhysiologyEngineConfigurationData&);
      }
    }
  }
}

#include <xsd/cxx/post.hxx>

// Begin epilogue.
//
//
// End epilogue.

#endif // PHYSIOLOGY_ENGINE_CONFIGURATION_DATA_HXX
