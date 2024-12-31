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

#include "HemorrhageData.hxx"

#include "ScalarVolumePerTimeData.hxx"

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        // HemorrhageData
        // 

        const HemorrhageData::InitialRate_type& HemorrhageData::
        InitialRate () const
        {
          return this->InitialRate_.get ();
        }

        HemorrhageData::InitialRate_type& HemorrhageData::
        InitialRate ()
        {
          return this->InitialRate_.get ();
        }

        void HemorrhageData::
        InitialRate (const InitialRate_type& x)
        {
          this->InitialRate_.set (x);
        }

        void HemorrhageData::
        InitialRate (::std::unique_ptr< InitialRate_type > x)
        {
          this->InitialRate_.set (std::move (x));
        }

        const HemorrhageData::Compartment_type& HemorrhageData::
        Compartment () const
        {
          return this->Compartment_.get ();
        }

        HemorrhageData::Compartment_type& HemorrhageData::
        Compartment ()
        {
          return this->Compartment_.get ();
        }

        void HemorrhageData::
        Compartment (const Compartment_type& x)
        {
          this->Compartment_.set (x);
        }

        void HemorrhageData::
        Compartment (::std::unique_ptr< Compartment_type > x)
        {
          this->Compartment_.set (std::move (x));
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
        // HemorrhageData
        //

        HemorrhageData::
        HemorrhageData ()
        : ::mil::tatrc::physiology::datamodel::PatientActionData (),
          InitialRate_ (this),
          Compartment_ (this)
        {
        }

        HemorrhageData::
        HemorrhageData (const InitialRate_type& InitialRate,
                        const Compartment_type& Compartment)
        : ::mil::tatrc::physiology::datamodel::PatientActionData (),
          InitialRate_ (InitialRate, this),
          Compartment_ (Compartment, this)
        {
        }

        HemorrhageData::
        HemorrhageData (::std::unique_ptr< InitialRate_type > InitialRate,
                        const Compartment_type& Compartment)
        : ::mil::tatrc::physiology::datamodel::PatientActionData (),
          InitialRate_ (std::move (InitialRate), this),
          Compartment_ (Compartment, this)
        {
        }

        HemorrhageData::
        HemorrhageData (const HemorrhageData& x,
                        ::xml_schema::flags f,
                        ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::PatientActionData (x, f, c),
          InitialRate_ (x.InitialRate_, f, this),
          Compartment_ (x.Compartment_, f, this)
        {
        }

        HemorrhageData::
        HemorrhageData (const ::xercesc::DOMElement& e,
                        ::xml_schema::flags f,
                        ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::PatientActionData (e, f | ::xml_schema::flags::base, c),
          InitialRate_ (this),
          Compartment_ (this)
        {
          if ((f & ::xml_schema::flags::base) == 0)
          {
            ::xsd::cxx::xml::dom::parser< char > p (e, true, false, true);
            this->parse (p, f);
          }
        }

        void HemorrhageData::
        parse (::xsd::cxx::xml::dom::parser< char >& p,
               ::xml_schema::flags f)
        {
          this->::mil::tatrc::physiology::datamodel::PatientActionData::parse (p, f);

          for (; p.more_content (); p.next_content (false))
          {
            const ::xercesc::DOMElement& i (p.cur_element ());
            const ::xsd::cxx::xml::qualified_name< char > n (
              ::xsd::cxx::xml::dom::name< char > (i));

            // InitialRate
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "InitialRate",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< InitialRate_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                if (!InitialRate_.present ())
                {
                  ::std::unique_ptr< InitialRate_type > r (
                    dynamic_cast< InitialRate_type* > (tmp.get ()));

                  if (r.get ())
                    tmp.release ();
                  else
                    throw ::xsd::cxx::tree::not_derived< char > ();

                  this->InitialRate_.set (::std::move (r));
                  continue;
                }
              }
            }

            break;
          }

          if (!InitialRate_.present ())
          {
            throw ::xsd::cxx::tree::expected_element< char > (
              "InitialRate",
              "uri:/mil/tatrc/physiology/datamodel");
          }

          while (p.more_attributes ())
          {
            const ::xercesc::DOMAttr& i (p.next_attribute ());
            const ::xsd::cxx::xml::qualified_name< char > n (
              ::xsd::cxx::xml::dom::name< char > (i));

            if (n.name () == "Compartment" && n.namespace_ ().empty ())
            {
              this->Compartment_.set (Compartment_traits::create (i, f, this));
              continue;
            }
          }

          if (!Compartment_.present ())
          {
            throw ::xsd::cxx::tree::expected_attribute< char > (
              "Compartment",
              "");
          }
        }

        HemorrhageData* HemorrhageData::
        _clone (::xml_schema::flags f,
                ::xml_schema::container* c) const
        {
          return new class HemorrhageData (*this, f, c);
        }

        HemorrhageData& HemorrhageData::
        operator= (const HemorrhageData& x)
        {
          if (this != &x)
          {
            static_cast< ::mil::tatrc::physiology::datamodel::PatientActionData& > (*this) = x;
            this->InitialRate_ = x.InitialRate_;
            this->Compartment_ = x.Compartment_;
          }

          return *this;
        }

        HemorrhageData::
        ~HemorrhageData ()
        {
        }

        static
        const ::xsd::cxx::tree::type_factory_initializer< 0, char, HemorrhageData >
        _xsd_HemorrhageData_type_factory_init (
          "HemorrhageData",
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
        operator<< (::std::ostream& o, const HemorrhageData& i)
        {
          o << static_cast< const ::mil::tatrc::physiology::datamodel::PatientActionData& > (i);

          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            o << ::std::endl << "InitialRate: ";
            om.insert (o, i.InitialRate ());
          }

          o << ::std::endl << "Compartment: " << i.Compartment ();
          return o;
        }

        static
        const ::xsd::cxx::tree::std_ostream_initializer< 0, char, HemorrhageData >
        _xsd_HemorrhageData_std_ostream_init;
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
        operator<< (::xercesc::DOMElement& e, const HemorrhageData& i)
        {
          e << static_cast< const ::mil::tatrc::physiology::datamodel::PatientActionData& > (i);

          // InitialRate
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            const HemorrhageData::InitialRate_type& x (i.InitialRate ());
            if (typeid (HemorrhageData::InitialRate_type) == typeid (x))
            {
              ::xercesc::DOMElement& s (
                ::xsd::cxx::xml::dom::create_element (
                  "InitialRate",
                  "uri:/mil/tatrc/physiology/datamodel",
                  e));

              s << x;
            }
            else
              tsm.serialize (
                "InitialRate",
                "uri:/mil/tatrc/physiology/datamodel",
                false, true, e, x);
          }

          // Compartment
          //
          {
            ::xercesc::DOMAttr& a (
              ::xsd::cxx::xml::dom::create_attribute (
                "Compartment",
                e));

            a << i.Compartment ();
          }
        }

        static
        const ::xsd::cxx::tree::type_serializer_initializer< 0, char, HemorrhageData >
        _xsd_HemorrhageData_type_serializer_init (
          "HemorrhageData",
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

