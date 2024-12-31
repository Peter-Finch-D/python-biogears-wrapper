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

#include "NervousConfigurationData.hxx"

#include "ScalarLengthData.hxx"

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        // NervousConfigurationData
        // 

        const NervousConfigurationData::PupilDiameterBaseline_optional& NervousConfigurationData::
        PupilDiameterBaseline () const
        {
          return this->PupilDiameterBaseline_;
        }

        NervousConfigurationData::PupilDiameterBaseline_optional& NervousConfigurationData::
        PupilDiameterBaseline ()
        {
          return this->PupilDiameterBaseline_;
        }

        void NervousConfigurationData::
        PupilDiameterBaseline (const PupilDiameterBaseline_type& x)
        {
          this->PupilDiameterBaseline_.set (x);
        }

        void NervousConfigurationData::
        PupilDiameterBaseline (const PupilDiameterBaseline_optional& x)
        {
          this->PupilDiameterBaseline_ = x;
        }

        void NervousConfigurationData::
        PupilDiameterBaseline (::std::unique_ptr< PupilDiameterBaseline_type > x)
        {
          this->PupilDiameterBaseline_.set (std::move (x));
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
        // NervousConfigurationData
        //

        NervousConfigurationData::
        NervousConfigurationData ()
        : ::mil::tatrc::physiology::datamodel::ObjectData (),
          PupilDiameterBaseline_ (this)
        {
        }

        NervousConfigurationData::
        NervousConfigurationData (const NervousConfigurationData& x,
                                  ::xml_schema::flags f,
                                  ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::ObjectData (x, f, c),
          PupilDiameterBaseline_ (x.PupilDiameterBaseline_, f, this)
        {
        }

        NervousConfigurationData::
        NervousConfigurationData (const ::xercesc::DOMElement& e,
                                  ::xml_schema::flags f,
                                  ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::ObjectData (e, f | ::xml_schema::flags::base, c),
          PupilDiameterBaseline_ (this)
        {
          if ((f & ::xml_schema::flags::base) == 0)
          {
            ::xsd::cxx::xml::dom::parser< char > p (e, true, false, true);
            this->parse (p, f);
          }
        }

        void NervousConfigurationData::
        parse (::xsd::cxx::xml::dom::parser< char >& p,
               ::xml_schema::flags f)
        {
          this->::mil::tatrc::physiology::datamodel::ObjectData::parse (p, f);

          for (; p.more_content (); p.next_content (false))
          {
            const ::xercesc::DOMElement& i (p.cur_element ());
            const ::xsd::cxx::xml::qualified_name< char > n (
              ::xsd::cxx::xml::dom::name< char > (i));

            // PupilDiameterBaseline
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "PupilDiameterBaseline",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< PupilDiameterBaseline_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                if (!this->PupilDiameterBaseline_)
                {
                  ::std::unique_ptr< PupilDiameterBaseline_type > r (
                    dynamic_cast< PupilDiameterBaseline_type* > (tmp.get ()));

                  if (r.get ())
                    tmp.release ();
                  else
                    throw ::xsd::cxx::tree::not_derived< char > ();

                  this->PupilDiameterBaseline_.set (::std::move (r));
                  continue;
                }
              }
            }

            break;
          }
        }

        NervousConfigurationData* NervousConfigurationData::
        _clone (::xml_schema::flags f,
                ::xml_schema::container* c) const
        {
          return new class NervousConfigurationData (*this, f, c);
        }

        NervousConfigurationData& NervousConfigurationData::
        operator= (const NervousConfigurationData& x)
        {
          if (this != &x)
          {
            static_cast< ::mil::tatrc::physiology::datamodel::ObjectData& > (*this) = x;
            this->PupilDiameterBaseline_ = x.PupilDiameterBaseline_;
          }

          return *this;
        }

        NervousConfigurationData::
        ~NervousConfigurationData ()
        {
        }

        static
        const ::xsd::cxx::tree::type_factory_initializer< 0, char, NervousConfigurationData >
        _xsd_NervousConfigurationData_type_factory_init (
          "NervousConfigurationData",
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
        operator<< (::std::ostream& o, const NervousConfigurationData& i)
        {
          o << static_cast< const ::mil::tatrc::physiology::datamodel::ObjectData& > (i);

          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            if (i.PupilDiameterBaseline ())
            {
              o << ::std::endl << "PupilDiameterBaseline: ";
              om.insert (o, *i.PupilDiameterBaseline ());
            }
          }

          return o;
        }

        static
        const ::xsd::cxx::tree::std_ostream_initializer< 0, char, NervousConfigurationData >
        _xsd_NervousConfigurationData_std_ostream_init;
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
        operator<< (::xercesc::DOMElement& e, const NervousConfigurationData& i)
        {
          e << static_cast< const ::mil::tatrc::physiology::datamodel::ObjectData& > (i);

          // PupilDiameterBaseline
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            if (i.PupilDiameterBaseline ())
            {
              const NervousConfigurationData::PupilDiameterBaseline_type& x (*i.PupilDiameterBaseline ());
              if (typeid (NervousConfigurationData::PupilDiameterBaseline_type) == typeid (x))
              {
                ::xercesc::DOMElement& s (
                  ::xsd::cxx::xml::dom::create_element (
                    "PupilDiameterBaseline",
                    "uri:/mil/tatrc/physiology/datamodel",
                    e));

                s << x;
              }
              else
                tsm.serialize (
                  "PupilDiameterBaseline",
                  "uri:/mil/tatrc/physiology/datamodel",
                  false, true, e, x);
            }
          }
        }

        static
        const ::xsd::cxx::tree::type_serializer_initializer< 0, char, NervousConfigurationData >
        _xsd_NervousConfigurationData_type_serializer_init (
          "NervousConfigurationData",
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

