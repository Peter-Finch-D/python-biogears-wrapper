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

#include "OxygenTankPressureLossData.hxx"

#include "enumOnOff.hxx"

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        // OxygenTankPressureLossData
        // 

        const OxygenTankPressureLossData::State_type& OxygenTankPressureLossData::
        State () const
        {
          return this->State_.get ();
        }

        OxygenTankPressureLossData::State_type& OxygenTankPressureLossData::
        State ()
        {
          return this->State_.get ();
        }

        void OxygenTankPressureLossData::
        State (const State_type& x)
        {
          this->State_.set (x);
        }

        void OxygenTankPressureLossData::
        State (::std::unique_ptr< State_type > x)
        {
          this->State_.set (std::move (x));
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
        // OxygenTankPressureLossData
        //

        OxygenTankPressureLossData::
        OxygenTankPressureLossData ()
        : ::mil::tatrc::physiology::datamodel::AnesthesiaMachineActionData (),
          State_ (this)
        {
        }

        OxygenTankPressureLossData::
        OxygenTankPressureLossData (const State_type& State)
        : ::mil::tatrc::physiology::datamodel::AnesthesiaMachineActionData (),
          State_ (State, this)
        {
        }

        OxygenTankPressureLossData::
        OxygenTankPressureLossData (const OxygenTankPressureLossData& x,
                                    ::xml_schema::flags f,
                                    ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::AnesthesiaMachineActionData (x, f, c),
          State_ (x.State_, f, this)
        {
        }

        OxygenTankPressureLossData::
        OxygenTankPressureLossData (const ::xercesc::DOMElement& e,
                                    ::xml_schema::flags f,
                                    ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::AnesthesiaMachineActionData (e, f | ::xml_schema::flags::base, c),
          State_ (this)
        {
          if ((f & ::xml_schema::flags::base) == 0)
          {
            ::xsd::cxx::xml::dom::parser< char > p (e, true, false, true);
            this->parse (p, f);
          }
        }

        void OxygenTankPressureLossData::
        parse (::xsd::cxx::xml::dom::parser< char >& p,
               ::xml_schema::flags f)
        {
          this->::mil::tatrc::physiology::datamodel::AnesthesiaMachineActionData::parse (p, f);

          while (p.more_attributes ())
          {
            const ::xercesc::DOMAttr& i (p.next_attribute ());
            const ::xsd::cxx::xml::qualified_name< char > n (
              ::xsd::cxx::xml::dom::name< char > (i));

            if (n.name () == "State" && n.namespace_ ().empty ())
            {
              this->State_.set (State_traits::create (i, f, this));
              continue;
            }
          }

          if (!State_.present ())
          {
            throw ::xsd::cxx::tree::expected_attribute< char > (
              "State",
              "");
          }
        }

        OxygenTankPressureLossData* OxygenTankPressureLossData::
        _clone (::xml_schema::flags f,
                ::xml_schema::container* c) const
        {
          return new class OxygenTankPressureLossData (*this, f, c);
        }

        OxygenTankPressureLossData& OxygenTankPressureLossData::
        operator= (const OxygenTankPressureLossData& x)
        {
          if (this != &x)
          {
            static_cast< ::mil::tatrc::physiology::datamodel::AnesthesiaMachineActionData& > (*this) = x;
            this->State_ = x.State_;
          }

          return *this;
        }

        OxygenTankPressureLossData::
        ~OxygenTankPressureLossData ()
        {
        }

        static
        const ::xsd::cxx::tree::type_factory_initializer< 0, char, OxygenTankPressureLossData >
        _xsd_OxygenTankPressureLossData_type_factory_init (
          "OxygenTankPressureLossData",
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
        operator<< (::std::ostream& o, const OxygenTankPressureLossData& i)
        {
          o << static_cast< const ::mil::tatrc::physiology::datamodel::AnesthesiaMachineActionData& > (i);

          o << ::std::endl << "State: " << i.State ();
          return o;
        }

        static
        const ::xsd::cxx::tree::std_ostream_initializer< 0, char, OxygenTankPressureLossData >
        _xsd_OxygenTankPressureLossData_std_ostream_init;
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
        operator<< (::xercesc::DOMElement& e, const OxygenTankPressureLossData& i)
        {
          e << static_cast< const ::mil::tatrc::physiology::datamodel::AnesthesiaMachineActionData& > (i);

          // State
          //
          {
            ::xercesc::DOMAttr& a (
              ::xsd::cxx::xml::dom::create_attribute (
                "State",
                e));

            a << i.State ();
          }
        }

        static
        const ::xsd::cxx::tree::type_serializer_initializer< 0, char, OxygenTankPressureLossData >
        _xsd_OxygenTankPressureLossData_type_serializer_init (
          "OxygenTankPressureLossData",
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

