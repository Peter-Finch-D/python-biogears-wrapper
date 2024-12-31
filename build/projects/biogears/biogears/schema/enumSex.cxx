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

#include "enumSex.hxx"

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        // enumSex
        // 

        enumSex::
        enumSex ()
        : ::xml_schema::string ()
        {
        }

        enumSex::
        enumSex (value v)
        : ::xml_schema::string (_xsd_enumSex_literals_[v])
        {
        }

        enumSex::
        enumSex (const char* v)
        : ::xml_schema::string (v)
        {
        }

        enumSex::
        enumSex (const ::std::string& v)
        : ::xml_schema::string (v)
        {
        }

        enumSex::
        enumSex (const ::xml_schema::string& v)
        : ::xml_schema::string (v)
        {
        }

        enumSex::
        enumSex (const enumSex& v,
                 ::xml_schema::flags f,
                 ::xml_schema::container* c)
        : ::xml_schema::string (v, f, c)
        {
        }

        enumSex& enumSex::
        operator= (value v)
        {
          static_cast< ::xml_schema::string& > (*this) = 
          ::xml_schema::string (_xsd_enumSex_literals_[v]);

          return *this;
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
        // enumSex
        //

        enumSex::
        enumSex (const ::xercesc::DOMElement& e,
                 ::xml_schema::flags f,
                 ::xml_schema::container* c)
        : ::xml_schema::string (e, f, c)
        {
          _xsd_enumSex_convert ();
        }

        enumSex::
        enumSex (const ::xercesc::DOMAttr& a,
                 ::xml_schema::flags f,
                 ::xml_schema::container* c)
        : ::xml_schema::string (a, f, c)
        {
          _xsd_enumSex_convert ();
        }

        enumSex::
        enumSex (const ::std::string& s,
                 const ::xercesc::DOMElement* e,
                 ::xml_schema::flags f,
                 ::xml_schema::container* c)
        : ::xml_schema::string (s, e, f, c)
        {
          _xsd_enumSex_convert ();
        }

        enumSex* enumSex::
        _clone (::xml_schema::flags f,
                ::xml_schema::container* c) const
        {
          return new class enumSex (*this, f, c);
        }

        enumSex::value enumSex::
        _xsd_enumSex_convert () const
        {
          ::xsd::cxx::tree::enum_comparator< char > c (_xsd_enumSex_literals_);
          const value* i (::std::lower_bound (
                            _xsd_enumSex_indexes_,
                            _xsd_enumSex_indexes_ + 2,
                            *this,
                            c));

          if (i == _xsd_enumSex_indexes_ + 2 || _xsd_enumSex_literals_[*i] != *this)
          {
            throw ::xsd::cxx::tree::unexpected_enumerator < char > (*this);
          }

          return *i;
        }

        const char* const enumSex::
        _xsd_enumSex_literals_[2] =
        {
          "Male",
          "Female"
        };

        const enumSex::value enumSex::
        _xsd_enumSex_indexes_[2] =
        {
          ::mil::tatrc::physiology::datamodel::enumSex::Female,
          ::mil::tatrc::physiology::datamodel::enumSex::Male
        };

        static
        const ::xsd::cxx::tree::type_factory_initializer< 0, char, enumSex >
        _xsd_enumSex_type_factory_init (
          "enumSex",
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
        operator<< (::std::ostream& o, enumSex::value i)
        {
          return o << enumSex::_xsd_enumSex_literals_[i];
        }

        ::std::ostream&
        operator<< (::std::ostream& o, const enumSex& i)
        {
          return o << static_cast< const ::xml_schema::string& > (i);
        }

        static
        const ::xsd::cxx::tree::std_ostream_initializer< 0, char, enumSex >
        _xsd_enumSex_std_ostream_init;
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
        operator<< (::xercesc::DOMElement& e, const enumSex& i)
        {
          e << static_cast< const ::xml_schema::string& > (i);
        }

        void
        operator<< (::xercesc::DOMAttr& a, const enumSex& i)
        {
          a << static_cast< const ::xml_schema::string& > (i);
        }

        void
        operator<< (::xml_schema::list_stream& l,
                    const enumSex& i)
        {
          l << static_cast< const ::xml_schema::string& > (i);
        }

        static
        const ::xsd::cxx::tree::type_serializer_initializer< 0, char, enumSex >
        _xsd_enumSex_type_serializer_init (
          "enumSex",
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

