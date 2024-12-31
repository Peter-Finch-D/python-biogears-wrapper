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

#include "SubstanceCompoundData.hxx"

#include "SubstanceConcentrationData.hxx"

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        // SubstanceCompoundData
        // 

        const SubstanceCompoundData::Name_type& SubstanceCompoundData::
        Name () const
        {
          return this->Name_.get ();
        }

        SubstanceCompoundData::Name_type& SubstanceCompoundData::
        Name ()
        {
          return this->Name_.get ();
        }

        void SubstanceCompoundData::
        Name (const Name_type& x)
        {
          this->Name_.set (x);
        }

        void SubstanceCompoundData::
        Name (::std::unique_ptr< Name_type > x)
        {
          this->Name_.set (std::move (x));
        }

        const SubstanceCompoundData::Component_sequence& SubstanceCompoundData::
        Component () const
        {
          return this->Component_;
        }

        SubstanceCompoundData::Component_sequence& SubstanceCompoundData::
        Component ()
        {
          return this->Component_;
        }

        void SubstanceCompoundData::
        Component (const Component_sequence& s)
        {
          this->Component_ = s;
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
        // SubstanceCompoundData
        //

        SubstanceCompoundData::
        SubstanceCompoundData ()
        : ::mil::tatrc::physiology::datamodel::ObjectData (),
          Name_ (this),
          Component_ (this)
        {
        }

        SubstanceCompoundData::
        SubstanceCompoundData (const Name_type& Name)
        : ::mil::tatrc::physiology::datamodel::ObjectData (),
          Name_ (Name, this),
          Component_ (this)
        {
        }

        SubstanceCompoundData::
        SubstanceCompoundData (::std::unique_ptr< Name_type > Name)
        : ::mil::tatrc::physiology::datamodel::ObjectData (),
          Name_ (std::move (Name), this),
          Component_ (this)
        {
        }

        SubstanceCompoundData::
        SubstanceCompoundData (const SubstanceCompoundData& x,
                               ::xml_schema::flags f,
                               ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::ObjectData (x, f, c),
          Name_ (x.Name_, f, this),
          Component_ (x.Component_, f, this)
        {
        }

        SubstanceCompoundData::
        SubstanceCompoundData (const ::xercesc::DOMElement& e,
                               ::xml_schema::flags f,
                               ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::ObjectData (e, f | ::xml_schema::flags::base, c),
          Name_ (this),
          Component_ (this)
        {
          if ((f & ::xml_schema::flags::base) == 0)
          {
            ::xsd::cxx::xml::dom::parser< char > p (e, true, false, true);
            this->parse (p, f);
          }
        }

        void SubstanceCompoundData::
        parse (::xsd::cxx::xml::dom::parser< char >& p,
               ::xml_schema::flags f)
        {
          this->::mil::tatrc::physiology::datamodel::ObjectData::parse (p, f);

          for (; p.more_content (); p.next_content (false))
          {
            const ::xercesc::DOMElement& i (p.cur_element ());
            const ::xsd::cxx::xml::qualified_name< char > n (
              ::xsd::cxx::xml::dom::name< char > (i));

            // Name
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "Name",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< Name_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                if (!Name_.present ())
                {
                  ::std::unique_ptr< Name_type > r (
                    dynamic_cast< Name_type* > (tmp.get ()));

                  if (r.get ())
                    tmp.release ();
                  else
                    throw ::xsd::cxx::tree::not_derived< char > ();

                  this->Name_.set (::std::move (r));
                  continue;
                }
              }
            }

            // Component
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "Component",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< Component_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                ::std::unique_ptr< Component_type > r (
                  dynamic_cast< Component_type* > (tmp.get ()));

                if (r.get ())
                  tmp.release ();
                else
                  throw ::xsd::cxx::tree::not_derived< char > ();

                this->Component_.push_back (::std::move (r));
                continue;
              }
            }

            break;
          }

          if (!Name_.present ())
          {
            throw ::xsd::cxx::tree::expected_element< char > (
              "Name",
              "uri:/mil/tatrc/physiology/datamodel");
          }
        }

        SubstanceCompoundData* SubstanceCompoundData::
        _clone (::xml_schema::flags f,
                ::xml_schema::container* c) const
        {
          return new class SubstanceCompoundData (*this, f, c);
        }

        SubstanceCompoundData& SubstanceCompoundData::
        operator= (const SubstanceCompoundData& x)
        {
          if (this != &x)
          {
            static_cast< ::mil::tatrc::physiology::datamodel::ObjectData& > (*this) = x;
            this->Name_ = x.Name_;
            this->Component_ = x.Component_;
          }

          return *this;
        }

        SubstanceCompoundData::
        ~SubstanceCompoundData ()
        {
        }

        static
        const ::xsd::cxx::tree::type_factory_initializer< 0, char, SubstanceCompoundData >
        _xsd_SubstanceCompoundData_type_factory_init (
          "SubstanceCompoundData",
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
        operator<< (::std::ostream& o, const SubstanceCompoundData& i)
        {
          o << static_cast< const ::mil::tatrc::physiology::datamodel::ObjectData& > (i);

          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            o << ::std::endl << "Name: ";
            om.insert (o, i.Name ());
          }

          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            for (SubstanceCompoundData::Component_const_iterator
                 b (i.Component ().begin ()), e (i.Component ().end ());
                 b != e; ++b)
            {
              o << ::std::endl << "Component: ";
              om.insert (o, *b);
            }
          }

          return o;
        }

        static
        const ::xsd::cxx::tree::std_ostream_initializer< 0, char, SubstanceCompoundData >
        _xsd_SubstanceCompoundData_std_ostream_init;
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
        operator<< (::xercesc::DOMElement& e, const SubstanceCompoundData& i)
        {
          e << static_cast< const ::mil::tatrc::physiology::datamodel::ObjectData& > (i);

          // Name
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            const SubstanceCompoundData::Name_type& x (i.Name ());
            if (typeid (SubstanceCompoundData::Name_type) == typeid (x))
            {
              ::xercesc::DOMElement& s (
                ::xsd::cxx::xml::dom::create_element (
                  "Name",
                  "uri:/mil/tatrc/physiology/datamodel",
                  e));

              s << x;
            }
            else
              tsm.serialize (
                "Name",
                "uri:/mil/tatrc/physiology/datamodel",
                false, true, e, x);
          }

          // Component
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            for (SubstanceCompoundData::Component_const_iterator
                 b (i.Component ().begin ()), n (i.Component ().end ());
                 b != n; ++b)
            {
              if (typeid (SubstanceCompoundData::Component_type) == typeid (*b))
              {
                ::xercesc::DOMElement& s (
                  ::xsd::cxx::xml::dom::create_element (
                    "Component",
                    "uri:/mil/tatrc/physiology/datamodel",
                    e));

                s << *b;
              }
              else
                tsm.serialize (
                  "Component",
                  "uri:/mil/tatrc/physiology/datamodel",
                  false, true, e, *b);
            }
          }
        }

        static
        const ::xsd::cxx::tree::type_serializer_initializer< 0, char, SubstanceCompoundData >
        _xsd_SubstanceCompoundData_type_serializer_init (
          "SubstanceCompoundData",
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

