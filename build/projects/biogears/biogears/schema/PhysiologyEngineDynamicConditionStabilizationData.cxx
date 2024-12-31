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

// Begin prologue.
//
#include "Properties.hxx"

//
// End prologue.

#include <xsd/cxx/pre.hxx>

#include "PhysiologyEngineDynamicConditionStabilizationData.hxx"

#include "PhysiologyEngineDynamicStabilizationCriteriaData.hxx"

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        // PhysiologyEngineDynamicConditionStabilizationData
        // 

        const PhysiologyEngineDynamicConditionStabilizationData::Criteria_type& PhysiologyEngineDynamicConditionStabilizationData::
        Criteria () const
        {
          return this->Criteria_.get ();
        }

        PhysiologyEngineDynamicConditionStabilizationData::Criteria_type& PhysiologyEngineDynamicConditionStabilizationData::
        Criteria ()
        {
          return this->Criteria_.get ();
        }

        void PhysiologyEngineDynamicConditionStabilizationData::
        Criteria (const Criteria_type& x)
        {
          this->Criteria_.set (x);
        }

        void PhysiologyEngineDynamicConditionStabilizationData::
        Criteria (::std::unique_ptr< Criteria_type > x)
        {
          this->Criteria_.set (std::move (x));
        }
      }
    }
  }
}

#include <xsd/cxx/xml/dom/parsing-source.hxx>

#include <xsd/cxx/tree/type-factory-map.hxx>

namespace _xsd
{
  static
  const ::xsd::cxx::tree::type_factory_plate< 0, char >
  type_factory_plate_init;
}

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        // PhysiologyEngineDynamicConditionStabilizationData
        //

        PhysiologyEngineDynamicConditionStabilizationData::
        PhysiologyEngineDynamicConditionStabilizationData ()
        : ::mil::tatrc::physiology::datamodel::PhysiologyEngineConditionStabilizationData (),
          Criteria_ (this)
        {
        }

        PhysiologyEngineDynamicConditionStabilizationData::
        PhysiologyEngineDynamicConditionStabilizationData (const Name_type& Name,
                                                           const Criteria_type& Criteria)
        : ::mil::tatrc::physiology::datamodel::PhysiologyEngineConditionStabilizationData (Name),
          Criteria_ (Criteria, this)
        {
        }

        PhysiologyEngineDynamicConditionStabilizationData::
        PhysiologyEngineDynamicConditionStabilizationData (const Name_type& Name,
                                                           ::std::unique_ptr< Criteria_type > Criteria)
        : ::mil::tatrc::physiology::datamodel::PhysiologyEngineConditionStabilizationData (Name),
          Criteria_ (std::move (Criteria), this)
        {
        }

        PhysiologyEngineDynamicConditionStabilizationData::
        PhysiologyEngineDynamicConditionStabilizationData (const PhysiologyEngineDynamicConditionStabilizationData& x,
                                                           ::xml_schema::flags f,
                                                           ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::PhysiologyEngineConditionStabilizationData (x, f, c),
          Criteria_ (x.Criteria_, f, this)
        {
        }

        PhysiologyEngineDynamicConditionStabilizationData::
        PhysiologyEngineDynamicConditionStabilizationData (const ::xercesc::DOMElement& e,
                                                           ::xml_schema::flags f,
                                                           ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::PhysiologyEngineConditionStabilizationData (e, f | ::xml_schema::flags::base, c),
          Criteria_ (this)
        {
          if ((f & ::xml_schema::flags::base) == 0)
          {
            ::xsd::cxx::xml::dom::parser< char > p (e, true, false, true);
            this->parse (p, f);
          }
        }

        void PhysiologyEngineDynamicConditionStabilizationData::
        parse (::xsd::cxx::xml::dom::parser< char >& p,
               ::xml_schema::flags f)
        {
          this->::mil::tatrc::physiology::datamodel::PhysiologyEngineConditionStabilizationData::parse (p, f);

          for (; p.more_content (); p.next_content (false))
          {
            const ::xercesc::DOMElement& i (p.cur_element ());
            const ::xsd::cxx::xml::qualified_name< char > n (
              ::xsd::cxx::xml::dom::name< char > (i));

            // Criteria
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "Criteria",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< Criteria_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                if (!Criteria_.present ())
                {
                  ::std::unique_ptr< Criteria_type > r (
                    dynamic_cast< Criteria_type* > (tmp.get ()));

                  if (r.get ())
                    tmp.release ();
                  else
                    throw ::xsd::cxx::tree::not_derived< char > ();

                  this->Criteria_.set (::std::move (r));
                  continue;
                }
              }
            }

            break;
          }

          if (!Criteria_.present ())
          {
            throw ::xsd::cxx::tree::expected_element< char > (
              "Criteria",
              "uri:/mil/tatrc/physiology/datamodel");
          }
        }

        PhysiologyEngineDynamicConditionStabilizationData* PhysiologyEngineDynamicConditionStabilizationData::
        _clone (::xml_schema::flags f,
                ::xml_schema::container* c) const
        {
          return new class PhysiologyEngineDynamicConditionStabilizationData (*this, f, c);
        }

        PhysiologyEngineDynamicConditionStabilizationData& PhysiologyEngineDynamicConditionStabilizationData::
        operator= (const PhysiologyEngineDynamicConditionStabilizationData& x)
        {
          if (this != &x)
          {
            static_cast< ::mil::tatrc::physiology::datamodel::PhysiologyEngineConditionStabilizationData& > (*this) = x;
            this->Criteria_ = x.Criteria_;
          }

          return *this;
        }

        PhysiologyEngineDynamicConditionStabilizationData::
        ~PhysiologyEngineDynamicConditionStabilizationData ()
        {
        }

        static
        const ::xsd::cxx::tree::type_factory_initializer< 0, char, PhysiologyEngineDynamicConditionStabilizationData >
        _xsd_PhysiologyEngineDynamicConditionStabilizationData_type_factory_init (
          "PhysiologyEngineDynamicConditionStabilizationData",
          "uri:/mil/tatrc/physiology/datamodel");
      }
    }
  }
}

#include <ostream>

#include <xsd/cxx/tree/std-ostream-map.hxx>

namespace _xsd
{
  static
  const ::xsd::cxx::tree::std_ostream_plate< 0, char >
  std_ostream_plate_init;
}

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        ::std::ostream&
        operator<< (::std::ostream& o, const PhysiologyEngineDynamicConditionStabilizationData& i)
        {
          o << static_cast< const ::mil::tatrc::physiology::datamodel::PhysiologyEngineConditionStabilizationData& > (i);

          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            o << ::std::endl << "Criteria: ";
            om.insert (o, i.Criteria ());
          }

          return o;
        }

        static
        const ::xsd::cxx::tree::std_ostream_initializer< 0, char, PhysiologyEngineDynamicConditionStabilizationData >
        _xsd_PhysiologyEngineDynamicConditionStabilizationData_std_ostream_init;
      }
    }
  }
}

#include <istream>
#include <xsd/cxx/xml/sax/std-input-source.hxx>
#include <xsd/cxx/tree/error-handler.hxx>

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

#include <ostream>
#include <xsd/cxx/tree/error-handler.hxx>
#include <xsd/cxx/xml/dom/serialization-source.hxx>

#include <xsd/cxx/tree/type-serializer-map.hxx>

namespace _xsd
{
  static
  const ::xsd::cxx::tree::type_serializer_plate< 0, char >
  type_serializer_plate_init;
}

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        void
        operator<< (::xercesc::DOMElement& e, const PhysiologyEngineDynamicConditionStabilizationData& i)
        {
          e << static_cast< const ::mil::tatrc::physiology::datamodel::PhysiologyEngineConditionStabilizationData& > (i);

          // Criteria
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            const PhysiologyEngineDynamicConditionStabilizationData::Criteria_type& x (i.Criteria ());
            if (typeid (PhysiologyEngineDynamicConditionStabilizationData::Criteria_type) == typeid (x))
            {
              ::xercesc::DOMElement& s (
                ::xsd::cxx::xml::dom::create_element (
                  "Criteria",
                  "uri:/mil/tatrc/physiology/datamodel",
                  e));

              s << x;
            }
            else
              tsm.serialize (
                "Criteria",
                "uri:/mil/tatrc/physiology/datamodel",
                false, true, e, x);
          }
        }

        static
        const ::xsd::cxx::tree::type_serializer_initializer< 0, char, PhysiologyEngineDynamicConditionStabilizationData >
        _xsd_PhysiologyEngineDynamicConditionStabilizationData_type_serializer_init (
          "PhysiologyEngineDynamicConditionStabilizationData",
          "uri:/mil/tatrc/physiology/datamodel");
      }
    }
  }
}

#include <xsd/cxx/post.hxx>

// Begin epilogue.
//
//
// End epilogue.

