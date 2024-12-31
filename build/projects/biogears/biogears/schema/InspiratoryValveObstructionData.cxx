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

#include "InspiratoryValveObstructionData.hxx"

#include "Scalar0To1Data.hxx"

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        // InspiratoryValveObstructionData
        // 

        const InspiratoryValveObstructionData::Severity_type& InspiratoryValveObstructionData::
        Severity () const
        {
          return this->Severity_.get ();
        }

        InspiratoryValveObstructionData::Severity_type& InspiratoryValveObstructionData::
        Severity ()
        {
          return this->Severity_.get ();
        }

        void InspiratoryValveObstructionData::
        Severity (const Severity_type& x)
        {
          this->Severity_.set (x);
        }

        void InspiratoryValveObstructionData::
        Severity (::std::unique_ptr< Severity_type > x)
        {
          this->Severity_.set (std::move (x));
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
        // InspiratoryValveObstructionData
        //

        InspiratoryValveObstructionData::
        InspiratoryValveObstructionData ()
        : ::mil::tatrc::physiology::datamodel::AnesthesiaMachineActionData (),
          Severity_ (this)
        {
        }

        InspiratoryValveObstructionData::
        InspiratoryValveObstructionData (const Severity_type& Severity)
        : ::mil::tatrc::physiology::datamodel::AnesthesiaMachineActionData (),
          Severity_ (Severity, this)
        {
        }

        InspiratoryValveObstructionData::
        InspiratoryValveObstructionData (::std::unique_ptr< Severity_type > Severity)
        : ::mil::tatrc::physiology::datamodel::AnesthesiaMachineActionData (),
          Severity_ (std::move (Severity), this)
        {
        }

        InspiratoryValveObstructionData::
        InspiratoryValveObstructionData (const InspiratoryValveObstructionData& x,
                                         ::xml_schema::flags f,
                                         ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::AnesthesiaMachineActionData (x, f, c),
          Severity_ (x.Severity_, f, this)
        {
        }

        InspiratoryValveObstructionData::
        InspiratoryValveObstructionData (const ::xercesc::DOMElement& e,
                                         ::xml_schema::flags f,
                                         ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::AnesthesiaMachineActionData (e, f | ::xml_schema::flags::base, c),
          Severity_ (this)
        {
          if ((f & ::xml_schema::flags::base) == 0)
          {
            ::xsd::cxx::xml::dom::parser< char > p (e, true, false, false);
            this->parse (p, f);
          }
        }

        void InspiratoryValveObstructionData::
        parse (::xsd::cxx::xml::dom::parser< char >& p,
               ::xml_schema::flags f)
        {
          this->::mil::tatrc::physiology::datamodel::AnesthesiaMachineActionData::parse (p, f);

          for (; p.more_content (); p.next_content (false))
          {
            const ::xercesc::DOMElement& i (p.cur_element ());
            const ::xsd::cxx::xml::qualified_name< char > n (
              ::xsd::cxx::xml::dom::name< char > (i));

            // Severity
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "Severity",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< Severity_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                if (!Severity_.present ())
                {
                  ::std::unique_ptr< Severity_type > r (
                    dynamic_cast< Severity_type* > (tmp.get ()));

                  if (r.get ())
                    tmp.release ();
                  else
                    throw ::xsd::cxx::tree::not_derived< char > ();

                  this->Severity_.set (::std::move (r));
                  continue;
                }
              }
            }

            break;
          }

          if (!Severity_.present ())
          {
            throw ::xsd::cxx::tree::expected_element< char > (
              "Severity",
              "uri:/mil/tatrc/physiology/datamodel");
          }
        }

        InspiratoryValveObstructionData* InspiratoryValveObstructionData::
        _clone (::xml_schema::flags f,
                ::xml_schema::container* c) const
        {
          return new class InspiratoryValveObstructionData (*this, f, c);
        }

        InspiratoryValveObstructionData& InspiratoryValveObstructionData::
        operator= (const InspiratoryValveObstructionData& x)
        {
          if (this != &x)
          {
            static_cast< ::mil::tatrc::physiology::datamodel::AnesthesiaMachineActionData& > (*this) = x;
            this->Severity_ = x.Severity_;
          }

          return *this;
        }

        InspiratoryValveObstructionData::
        ~InspiratoryValveObstructionData ()
        {
        }

        static
        const ::xsd::cxx::tree::type_factory_initializer< 0, char, InspiratoryValveObstructionData >
        _xsd_InspiratoryValveObstructionData_type_factory_init (
          "InspiratoryValveObstructionData",
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
        operator<< (::std::ostream& o, const InspiratoryValveObstructionData& i)
        {
          o << static_cast< const ::mil::tatrc::physiology::datamodel::AnesthesiaMachineActionData& > (i);

          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            o << ::std::endl << "Severity: ";
            om.insert (o, i.Severity ());
          }

          return o;
        }

        static
        const ::xsd::cxx::tree::std_ostream_initializer< 0, char, InspiratoryValveObstructionData >
        _xsd_InspiratoryValveObstructionData_std_ostream_init;
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
        operator<< (::xercesc::DOMElement& e, const InspiratoryValveObstructionData& i)
        {
          e << static_cast< const ::mil::tatrc::physiology::datamodel::AnesthesiaMachineActionData& > (i);

          // Severity
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            const InspiratoryValveObstructionData::Severity_type& x (i.Severity ());
            if (typeid (InspiratoryValveObstructionData::Severity_type) == typeid (x))
            {
              ::xercesc::DOMElement& s (
                ::xsd::cxx::xml::dom::create_element (
                  "Severity",
                  "uri:/mil/tatrc/physiology/datamodel",
                  e));

              s << x;
            }
            else
              tsm.serialize (
                "Severity",
                "uri:/mil/tatrc/physiology/datamodel",
                false, true, e, x);
          }
        }

        static
        const ::xsd::cxx::tree::type_serializer_initializer< 0, char, InspiratoryValveObstructionData >
        _xsd_InspiratoryValveObstructionData_type_serializer_init (
          "InspiratoryValveObstructionData",
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

