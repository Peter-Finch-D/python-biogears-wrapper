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

#include "enumPresenceIndicator.hxx"

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        // enumPresenceIndicator
        // 

        enumPresenceIndicator::
        enumPresenceIndicator ()
        : ::xml_schema::string ()
        {
        }

        enumPresenceIndicator::
        enumPresenceIndicator (value v)
        : ::xml_schema::string (_xsd_enumPresenceIndicator_literals_[v])
        {
        }

        enumPresenceIndicator::
        enumPresenceIndicator (const char* v)
        : ::xml_schema::string (v)
        {
        }

        enumPresenceIndicator::
        enumPresenceIndicator (const ::std::string& v)
        : ::xml_schema::string (v)
        {
        }

        enumPresenceIndicator::
        enumPresenceIndicator (const ::xml_schema::string& v)
        : ::xml_schema::string (v)
        {
        }

        enumPresenceIndicator::
        enumPresenceIndicator (const enumPresenceIndicator& v,
                               ::xml_schema::flags f,
                               ::xml_schema::container* c)
        : ::xml_schema::string (v, f, c)
        {
        }

        enumPresenceIndicator& enumPresenceIndicator::
        operator= (value v)
        {
          static_cast< ::xml_schema::string& > (*this) = 
          ::xml_schema::string (_xsd_enumPresenceIndicator_literals_[v]);

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
        // enumPresenceIndicator
        //

        enumPresenceIndicator::
        enumPresenceIndicator (const ::xercesc::DOMElement& e,
                               ::xml_schema::flags f,
                               ::xml_schema::container* c)
        : ::xml_schema::string (e, f, c)
        {
          _xsd_enumPresenceIndicator_convert ();
        }

        enumPresenceIndicator::
        enumPresenceIndicator (const ::xercesc::DOMAttr& a,
                               ::xml_schema::flags f,
                               ::xml_schema::container* c)
        : ::xml_schema::string (a, f, c)
        {
          _xsd_enumPresenceIndicator_convert ();
        }

        enumPresenceIndicator::
        enumPresenceIndicator (const ::std::string& s,
                               const ::xercesc::DOMElement* e,
                               ::xml_schema::flags f,
                               ::xml_schema::container* c)
        : ::xml_schema::string (s, e, f, c)
        {
          _xsd_enumPresenceIndicator_convert ();
        }

        enumPresenceIndicator* enumPresenceIndicator::
        _clone (::xml_schema::flags f,
                ::xml_schema::container* c) const
        {
          return new class enumPresenceIndicator (*this, f, c);
        }

        enumPresenceIndicator::value enumPresenceIndicator::
        _xsd_enumPresenceIndicator_convert () const
        {
          ::xsd::cxx::tree::enum_comparator< char > c (_xsd_enumPresenceIndicator_literals_);
          const value* i (::std::lower_bound (
                            _xsd_enumPresenceIndicator_indexes_,
                            _xsd_enumPresenceIndicator_indexes_ + 2,
                            *this,
                            c));

          if (i == _xsd_enumPresenceIndicator_indexes_ + 2 || _xsd_enumPresenceIndicator_literals_[*i] != *this)
          {
            throw ::xsd::cxx::tree::unexpected_enumerator < char > (*this);
          }

          return *i;
        }

        const char* const enumPresenceIndicator::
        _xsd_enumPresenceIndicator_literals_[2] =
        {
          "Positive",
          "Negative"
        };

        const enumPresenceIndicator::value enumPresenceIndicator::
        _xsd_enumPresenceIndicator_indexes_[2] =
        {
          ::mil::tatrc::physiology::datamodel::enumPresenceIndicator::Negative,
          ::mil::tatrc::physiology::datamodel::enumPresenceIndicator::Positive
        };

        static
        const ::xsd::cxx::tree::type_factory_initializer< 0, char, enumPresenceIndicator >
        _xsd_enumPresenceIndicator_type_factory_init (
          "enumPresenceIndicator",
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
        operator<< (::std::ostream& o, enumPresenceIndicator::value i)
        {
          return o << enumPresenceIndicator::_xsd_enumPresenceIndicator_literals_[i];
        }

        ::std::ostream&
        operator<< (::std::ostream& o, const enumPresenceIndicator& i)
        {
          return o << static_cast< const ::xml_schema::string& > (i);
        }

        static
        const ::xsd::cxx::tree::std_ostream_initializer< 0, char, enumPresenceIndicator >
        _xsd_enumPresenceIndicator_std_ostream_init;
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
        operator<< (::xercesc::DOMElement& e, const enumPresenceIndicator& i)
        {
          e << static_cast< const ::xml_schema::string& > (i);
        }

        void
        operator<< (::xercesc::DOMAttr& a, const enumPresenceIndicator& i)
        {
          a << static_cast< const ::xml_schema::string& > (i);
        }

        void
        operator<< (::xml_schema::list_stream& l,
                    const enumPresenceIndicator& i)
        {
          l << static_cast< const ::xml_schema::string& > (i);
        }

        static
        const ::xsd::cxx::tree::type_serializer_initializer< 0, char, enumPresenceIndicator >
        _xsd_enumPresenceIndicator_type_serializer_init (
          "enumPresenceIndicator",
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

