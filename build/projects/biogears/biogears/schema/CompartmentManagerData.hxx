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
 * @brief Generated from CompartmentManagerData.xsd.
 */

#ifndef COMPARTMENT_MANAGER_DATA_HXX
#define COMPARTMENT_MANAGER_DATA_HXX

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
        class CompartmentManagerData;
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
        class ElectricalCompartmentData;
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
        class ElectricalCompartmentLinkData;
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
        class LiquidCompartmentData;
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
        class LiquidCompartmentLinkData;
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
        class LiquidCompartmentGraphData;
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
        class GasCompartmentData;
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
        class GasCompartmentLinkData;
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
        class GasCompartmentGraphData;
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
        class ThermalCompartmentData;
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
        class ThermalCompartmentLinkData;
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
        class TissueCompartmentData;
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
         * @brief Class corresponding to the %CompartmentManagerData schema type.
         *
         * @nosubgrouping
         */
        class BIOGEARS_CDM_API CompartmentManagerData: public ::mil::tatrc::physiology::datamodel::ObjectData
        {
          public:
          /**
           * @name ElectricalCompartment
           *
           * @brief Accessor and modifier functions for the %ElectricalCompartment
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::ElectricalCompartmentData ElectricalCompartment_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< ElectricalCompartment_type > ElectricalCompartment_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef ElectricalCompartment_sequence::iterator ElectricalCompartment_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef ElectricalCompartment_sequence::const_iterator ElectricalCompartment_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< ElectricalCompartment_type, char > ElectricalCompartment_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const ElectricalCompartment_sequence&
          ElectricalCompartment () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          ElectricalCompartment_sequence&
          ElectricalCompartment ();

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
          ElectricalCompartment (const ElectricalCompartment_sequence& s);

          //@}

          /**
           * @name ElectricalLink
           *
           * @brief Accessor and modifier functions for the %ElectricalLink
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::ElectricalCompartmentLinkData ElectricalLink_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< ElectricalLink_type > ElectricalLink_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef ElectricalLink_sequence::iterator ElectricalLink_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef ElectricalLink_sequence::const_iterator ElectricalLink_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< ElectricalLink_type, char > ElectricalLink_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const ElectricalLink_sequence&
          ElectricalLink () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          ElectricalLink_sequence&
          ElectricalLink ();

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
          ElectricalLink (const ElectricalLink_sequence& s);

          //@}

          /**
           * @name LiquidCompartment
           *
           * @brief Accessor and modifier functions for the %LiquidCompartment
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::LiquidCompartmentData LiquidCompartment_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< LiquidCompartment_type > LiquidCompartment_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef LiquidCompartment_sequence::iterator LiquidCompartment_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef LiquidCompartment_sequence::const_iterator LiquidCompartment_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< LiquidCompartment_type, char > LiquidCompartment_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const LiquidCompartment_sequence&
          LiquidCompartment () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          LiquidCompartment_sequence&
          LiquidCompartment ();

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
          LiquidCompartment (const LiquidCompartment_sequence& s);

          //@}

          /**
           * @name LiquidLink
           *
           * @brief Accessor and modifier functions for the %LiquidLink
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::LiquidCompartmentLinkData LiquidLink_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< LiquidLink_type > LiquidLink_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef LiquidLink_sequence::iterator LiquidLink_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef LiquidLink_sequence::const_iterator LiquidLink_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< LiquidLink_type, char > LiquidLink_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const LiquidLink_sequence&
          LiquidLink () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          LiquidLink_sequence&
          LiquidLink ();

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
          LiquidLink (const LiquidLink_sequence& s);

          //@}

          /**
           * @name LiquidSubstance
           *
           * @brief Accessor and modifier functions for the %LiquidSubstance
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::xml_schema::string LiquidSubstance_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< LiquidSubstance_type > LiquidSubstance_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef LiquidSubstance_sequence::iterator LiquidSubstance_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef LiquidSubstance_sequence::const_iterator LiquidSubstance_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< LiquidSubstance_type, char > LiquidSubstance_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const LiquidSubstance_sequence&
          LiquidSubstance () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          LiquidSubstance_sequence&
          LiquidSubstance ();

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
          LiquidSubstance (const LiquidSubstance_sequence& s);

          //@}

          /**
           * @name LiquidGraph
           *
           * @brief Accessor and modifier functions for the %LiquidGraph
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::LiquidCompartmentGraphData LiquidGraph_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< LiquidGraph_type > LiquidGraph_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef LiquidGraph_sequence::iterator LiquidGraph_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef LiquidGraph_sequence::const_iterator LiquidGraph_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< LiquidGraph_type, char > LiquidGraph_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const LiquidGraph_sequence&
          LiquidGraph () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          LiquidGraph_sequence&
          LiquidGraph ();

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
          LiquidGraph (const LiquidGraph_sequence& s);

          //@}

          /**
           * @name GasCompartment
           *
           * @brief Accessor and modifier functions for the %GasCompartment
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::GasCompartmentData GasCompartment_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< GasCompartment_type > GasCompartment_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef GasCompartment_sequence::iterator GasCompartment_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef GasCompartment_sequence::const_iterator GasCompartment_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< GasCompartment_type, char > GasCompartment_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const GasCompartment_sequence&
          GasCompartment () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          GasCompartment_sequence&
          GasCompartment ();

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
          GasCompartment (const GasCompartment_sequence& s);

          //@}

          /**
           * @name GasLink
           *
           * @brief Accessor and modifier functions for the %GasLink
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::GasCompartmentLinkData GasLink_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< GasLink_type > GasLink_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef GasLink_sequence::iterator GasLink_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef GasLink_sequence::const_iterator GasLink_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< GasLink_type, char > GasLink_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const GasLink_sequence&
          GasLink () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          GasLink_sequence&
          GasLink ();

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
          GasLink (const GasLink_sequence& s);

          //@}

          /**
           * @name GasSubstance
           *
           * @brief Accessor and modifier functions for the %GasSubstance
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::xml_schema::string GasSubstance_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< GasSubstance_type > GasSubstance_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef GasSubstance_sequence::iterator GasSubstance_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef GasSubstance_sequence::const_iterator GasSubstance_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< GasSubstance_type, char > GasSubstance_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const GasSubstance_sequence&
          GasSubstance () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          GasSubstance_sequence&
          GasSubstance ();

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
          GasSubstance (const GasSubstance_sequence& s);

          //@}

          /**
           * @name GasGraph
           *
           * @brief Accessor and modifier functions for the %GasGraph
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::GasCompartmentGraphData GasGraph_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< GasGraph_type > GasGraph_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef GasGraph_sequence::iterator GasGraph_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef GasGraph_sequence::const_iterator GasGraph_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< GasGraph_type, char > GasGraph_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const GasGraph_sequence&
          GasGraph () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          GasGraph_sequence&
          GasGraph ();

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
          GasGraph (const GasGraph_sequence& s);

          //@}

          /**
           * @name ThermalCompartment
           *
           * @brief Accessor and modifier functions for the %ThermalCompartment
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::ThermalCompartmentData ThermalCompartment_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< ThermalCompartment_type > ThermalCompartment_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef ThermalCompartment_sequence::iterator ThermalCompartment_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef ThermalCompartment_sequence::const_iterator ThermalCompartment_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< ThermalCompartment_type, char > ThermalCompartment_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const ThermalCompartment_sequence&
          ThermalCompartment () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          ThermalCompartment_sequence&
          ThermalCompartment ();

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
          ThermalCompartment (const ThermalCompartment_sequence& s);

          //@}

          /**
           * @name ThermalLink
           *
           * @brief Accessor and modifier functions for the %ThermalLink
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::ThermalCompartmentLinkData ThermalLink_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< ThermalLink_type > ThermalLink_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef ThermalLink_sequence::iterator ThermalLink_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef ThermalLink_sequence::const_iterator ThermalLink_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< ThermalLink_type, char > ThermalLink_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const ThermalLink_sequence&
          ThermalLink () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          ThermalLink_sequence&
          ThermalLink ();

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
          ThermalLink (const ThermalLink_sequence& s);

          //@}

          /**
           * @name TissueCompartment
           *
           * @brief Accessor and modifier functions for the %TissueCompartment
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::mil::tatrc::physiology::datamodel::TissueCompartmentData TissueCompartment_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< TissueCompartment_type > TissueCompartment_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef TissueCompartment_sequence::iterator TissueCompartment_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef TissueCompartment_sequence::const_iterator TissueCompartment_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< TissueCompartment_type, char > TissueCompartment_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const TissueCompartment_sequence&
          TissueCompartment () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          TissueCompartment_sequence&
          TissueCompartment ();

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
          TissueCompartment (const TissueCompartment_sequence& s);

          //@}

          /**
           * @name TissueSubstance
           *
           * @brief Accessor and modifier functions for the %TissueSubstance
           * sequence element.
           */
          //@{

          /**
           * @brief Element type.
           */
          typedef ::xml_schema::string TissueSubstance_type;

          /**
           * @brief Element sequence container type.
           */
          typedef ::xsd::cxx::tree::sequence< TissueSubstance_type > TissueSubstance_sequence;

          /**
           * @brief Element iterator type.
           */
          typedef TissueSubstance_sequence::iterator TissueSubstance_iterator;

          /**
           * @brief Element constant iterator type.
           */
          typedef TissueSubstance_sequence::const_iterator TissueSubstance_const_iterator;

          /**
           * @brief Element traits type.
           */
          typedef ::xsd::cxx::tree::traits< TissueSubstance_type, char > TissueSubstance_traits;

          /**
           * @brief Return a read-only (constant) reference to the element
           * sequence.
           *
           * @return A constant reference to the sequence container.
           */
          const TissueSubstance_sequence&
          TissueSubstance () const;

          /**
           * @brief Return a read-write reference to the element sequence.
           *
           * @return A reference to the sequence container.
           */
          TissueSubstance_sequence&
          TissueSubstance ();

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
          TissueSubstance (const TissueSubstance_sequence& s);

          //@}

          /**
           * @name Constructors
           */
          //@{

          /**
           * @brief Create an instance from the ultimate base and
           * initializers for required elements and attributes.
           */
          CompartmentManagerData ();

          /**
           * @brief Create an instance from a DOM element.
           *
           * @param e A DOM element to extract the data from.
           * @param f Flags to create the new instance with.
           * @param c A pointer to the object that will contain the new
           * instance.
           */
          CompartmentManagerData (const ::xercesc::DOMElement& e,
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
          CompartmentManagerData (const CompartmentManagerData& x,
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
          virtual CompartmentManagerData*
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
          CompartmentManagerData&
          operator= (const CompartmentManagerData& x);

          //@}

          /**
           * @brief Destructor.
           */
          virtual 
          ~CompartmentManagerData ();

          // Implementation.
          //

          //@cond

          protected:
          void
          parse (::xsd::cxx::xml::dom::parser< char >&,
                 ::xml_schema::flags);

          protected:
          ElectricalCompartment_sequence ElectricalCompartment_;
          ElectricalLink_sequence ElectricalLink_;
          LiquidCompartment_sequence LiquidCompartment_;
          LiquidLink_sequence LiquidLink_;
          LiquidSubstance_sequence LiquidSubstance_;
          LiquidGraph_sequence LiquidGraph_;
          GasCompartment_sequence GasCompartment_;
          GasLink_sequence GasLink_;
          GasSubstance_sequence GasSubstance_;
          GasGraph_sequence GasGraph_;
          ThermalCompartment_sequence ThermalCompartment_;
          ThermalLink_sequence ThermalLink_;
          TissueCompartment_sequence TissueCompartment_;
          TissueSubstance_sequence TissueSubstance_;

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
        operator<< (::std::ostream&, const CompartmentManagerData&);
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
        operator<< (::xercesc::DOMElement&, const CompartmentManagerData&);
      }
    }
  }
}

#include <xsd/cxx/post.hxx>

// Begin epilogue.
//
//
// End epilogue.

#endif // COMPARTMENT_MANAGER_DATA_HXX
