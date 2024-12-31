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

#include "SubstanceConcentrationData.hxx"

#include "ScalarMassPerVolumeData.hxx"

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        // SubstanceConcentrationData
        // 

        const SubstanceConcentrationData::Concentration_type& SubstanceConcentrationData::
        Concentration () const
        {
          return this->Concentration_.get ();
        }

        SubstanceConcentrationData::Concentration_type& SubstanceConcentrationData::
        Concentration ()
        {
          return this->Concentration_.get ();
        }

        void SubstanceConcentrationData::
        Concentration (const Concentration_type& x)
        {
          this->Concentration_.set (x);
        }

        void SubstanceConcentrationData::
        Concentration (::std::unique_ptr< Concentration_type > x)
        {
          this->Concentration_.set (std::move (x));
        }

        const SubstanceConcentrationData::Name_type& SubstanceConcentrationData::
        Name () const
        {
          return this->Name_.get ();
        }

        SubstanceConcentrationData::Name_type& SubstanceConcentrationData::
        Name ()
        {
          return this->Name_.get ();
        }

        void SubstanceConcentrationData::
        Name (const Name_type& x)
        {
          this->Name_.set (x);
        }

        void SubstanceConcentrationData::
        Name (::std::unique_ptr< Name_type > x)
        {
          this->Name_.set (std::move (x));
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
        // SubstanceConcentrationData
        //

        SubstanceConcentrationData::
        SubstanceConcentrationData ()
        : ::xml_schema::type (),
          Concentration_ (this),
          Name_ (this)
        {
        }

        SubstanceConcentrationData::
        SubstanceConcentrationData (const Concentration_type& Concentration,
                                    const Name_type& Name)
        : ::xml_schema::type (),
          Concentration_ (Concentration, this),
          Name_ (Name, this)
        {
        }

        SubstanceConcentrationData::
        SubstanceConcentrationData (::std::unique_ptr< Concentration_type > Concentration,
                                    const Name_type& Name)
        : ::xml_schema::type (),
          Concentration_ (std::move (Concentration), this),
          Name_ (Name, this)
        {
        }

        SubstanceConcentrationData::
        SubstanceConcentrationData (const SubstanceConcentrationData& x,
                                    ::xml_schema::flags f,
                                    ::xml_schema::container* c)
        : ::xml_schema::type (x, f, c),
          Concentration_ (x.Concentration_, f, this),
          Name_ (x.Name_, f, this)
        {
        }

        SubstanceConcentrationData::
        SubstanceConcentrationData (const ::xercesc::DOMElement& e,
                                    ::xml_schema::flags f,
                                    ::xml_schema::container* c)
        : ::xml_schema::type (e, f | ::xml_schema::flags::base, c),
          Concentration_ (this),
          Name_ (this)
        {
          if ((f & ::xml_schema::flags::base) == 0)
          {
            ::xsd::cxx::xml::dom::parser< char > p (e, true, false, true);
            this->parse (p, f);
          }
        }

        void SubstanceConcentrationData::
        parse (::xsd::cxx::xml::dom::parser< char >& p,
               ::xml_schema::flags f)
        {
          for (; p.more_content (); p.next_content (false))
          {
            const ::xercesc::DOMElement& i (p.cur_element ());
            const ::xsd::cxx::xml::qualified_name< char > n (
              ::xsd::cxx::xml::dom::name< char > (i));

            // Concentration
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "Concentration",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< Concentration_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                if (!Concentration_.present ())
                {
                  ::std::unique_ptr< Concentration_type > r (
                    dynamic_cast< Concentration_type* > (tmp.get ()));

                  if (r.get ())
                    tmp.release ();
                  else
                    throw ::xsd::cxx::tree::not_derived< char > ();

                  this->Concentration_.set (::std::move (r));
                  continue;
                }
              }
            }

            break;
          }

          if (!Concentration_.present ())
          {
            throw ::xsd::cxx::tree::expected_element< char > (
              "Concentration",
              "uri:/mil/tatrc/physiology/datamodel");
          }

          while (p.more_attributes ())
          {
            const ::xercesc::DOMAttr& i (p.next_attribute ());
            const ::xsd::cxx::xml::qualified_name< char > n (
              ::xsd::cxx::xml::dom::name< char > (i));

            if (n.name () == "Name" && n.namespace_ ().empty ())
            {
              this->Name_.set (Name_traits::create (i, f, this));
              continue;
            }
          }

          if (!Name_.present ())
          {
            throw ::xsd::cxx::tree::expected_attribute< char > (
              "Name",
              "");
          }
        }

        SubstanceConcentrationData* SubstanceConcentrationData::
        _clone (::xml_schema::flags f,
                ::xml_schema::container* c) const
        {
          return new class SubstanceConcentrationData (*this, f, c);
        }

        SubstanceConcentrationData& SubstanceConcentrationData::
        operator= (const SubstanceConcentrationData& x)
        {
          if (this != &x)
          {
            static_cast< ::xml_schema::type& > (*this) = x;
            this->Concentration_ = x.Concentration_;
            this->Name_ = x.Name_;
          }

          return *this;
        }

        SubstanceConcentrationData::
        ~SubstanceConcentrationData ()
        {
        }

        static
        const ::xsd::cxx::tree::type_factory_initializer< 0, char, SubstanceConcentrationData >
        _xsd_SubstanceConcentrationData_type_factory_init (
          "SubstanceConcentrationData",
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
        operator<< (::std::ostream& o, const SubstanceConcentrationData& i)
        {
          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            o << ::std::endl << "Concentration: ";
            om.insert (o, i.Concentration ());
          }

          o << ::std::endl << "Name: " << i.Name ();
          return o;
        }

        static
        const ::xsd::cxx::tree::std_ostream_initializer< 0, char, SubstanceConcentrationData >
        _xsd_SubstanceConcentrationData_std_ostream_init;
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
        operator<< (::xercesc::DOMElement& e, const SubstanceConcentrationData& i)
        {
          e << static_cast< const ::xml_schema::type& > (i);

          // Concentration
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            const SubstanceConcentrationData::Concentration_type& x (i.Concentration ());
            if (typeid (SubstanceConcentrationData::Concentration_type) == typeid (x))
            {
              ::xercesc::DOMElement& s (
                ::xsd::cxx::xml::dom::create_element (
                  "Concentration",
                  "uri:/mil/tatrc/physiology/datamodel",
                  e));

              s << x;
            }
            else
              tsm.serialize (
                "Concentration",
                "uri:/mil/tatrc/physiology/datamodel",
                false, true, e, x);
          }

          // Name
          //
          {
            ::xercesc::DOMAttr& a (
              ::xsd::cxx::xml::dom::create_attribute (
                "Name",
                e));

            a << i.Name ();
          }
        }

        static
        const ::xsd::cxx::tree::type_serializer_initializer< 0, char, SubstanceConcentrationData >
        _xsd_SubstanceConcentrationData_type_serializer_init (
          "SubstanceConcentrationData",
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

