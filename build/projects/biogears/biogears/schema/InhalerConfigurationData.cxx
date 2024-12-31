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

#include "InhalerConfigurationData.hxx"

#include "InhalerData.hxx"

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        // InhalerConfigurationData
        // 

        const InhalerConfigurationData::ConfigurationFile_optional& InhalerConfigurationData::
        ConfigurationFile () const
        {
          return this->ConfigurationFile_;
        }

        InhalerConfigurationData::ConfigurationFile_optional& InhalerConfigurationData::
        ConfigurationFile ()
        {
          return this->ConfigurationFile_;
        }

        void InhalerConfigurationData::
        ConfigurationFile (const ConfigurationFile_type& x)
        {
          this->ConfigurationFile_.set (x);
        }

        void InhalerConfigurationData::
        ConfigurationFile (const ConfigurationFile_optional& x)
        {
          this->ConfigurationFile_ = x;
        }

        void InhalerConfigurationData::
        ConfigurationFile (::std::unique_ptr< ConfigurationFile_type > x)
        {
          this->ConfigurationFile_.set (std::move (x));
        }

        const InhalerConfigurationData::Configuration_optional& InhalerConfigurationData::
        Configuration () const
        {
          return this->Configuration_;
        }

        InhalerConfigurationData::Configuration_optional& InhalerConfigurationData::
        Configuration ()
        {
          return this->Configuration_;
        }

        void InhalerConfigurationData::
        Configuration (const Configuration_type& x)
        {
          this->Configuration_.set (x);
        }

        void InhalerConfigurationData::
        Configuration (const Configuration_optional& x)
        {
          this->Configuration_ = x;
        }

        void InhalerConfigurationData::
        Configuration (::std::unique_ptr< Configuration_type > x)
        {
          this->Configuration_.set (std::move (x));
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
        // InhalerConfigurationData
        //

        InhalerConfigurationData::
        InhalerConfigurationData ()
        : ::mil::tatrc::physiology::datamodel::InhalerActionData (),
          ConfigurationFile_ (this),
          Configuration_ (this)
        {
        }

        InhalerConfigurationData::
        InhalerConfigurationData (const InhalerConfigurationData& x,
                                  ::xml_schema::flags f,
                                  ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::InhalerActionData (x, f, c),
          ConfigurationFile_ (x.ConfigurationFile_, f, this),
          Configuration_ (x.Configuration_, f, this)
        {
        }

        InhalerConfigurationData::
        InhalerConfigurationData (const ::xercesc::DOMElement& e,
                                  ::xml_schema::flags f,
                                  ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::InhalerActionData (e, f | ::xml_schema::flags::base, c),
          ConfigurationFile_ (this),
          Configuration_ (this)
        {
          if ((f & ::xml_schema::flags::base) == 0)
          {
            ::xsd::cxx::xml::dom::parser< char > p (e, true, false, false);
            this->parse (p, f);
          }
        }

        void InhalerConfigurationData::
        parse (::xsd::cxx::xml::dom::parser< char >& p,
               ::xml_schema::flags f)
        {
          this->::mil::tatrc::physiology::datamodel::InhalerActionData::parse (p, f);

          for (; p.more_content (); p.next_content (false))
          {
            const ::xercesc::DOMElement& i (p.cur_element ());
            const ::xsd::cxx::xml::qualified_name< char > n (
              ::xsd::cxx::xml::dom::name< char > (i));

            // ConfigurationFile
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "ConfigurationFile",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< ConfigurationFile_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                if (!this->ConfigurationFile_)
                {
                  ::std::unique_ptr< ConfigurationFile_type > r (
                    dynamic_cast< ConfigurationFile_type* > (tmp.get ()));

                  if (r.get ())
                    tmp.release ();
                  else
                    throw ::xsd::cxx::tree::not_derived< char > ();

                  this->ConfigurationFile_.set (::std::move (r));
                  continue;
                }
              }
            }

            // Configuration
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "Configuration",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< Configuration_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                if (!this->Configuration_)
                {
                  ::std::unique_ptr< Configuration_type > r (
                    dynamic_cast< Configuration_type* > (tmp.get ()));

                  if (r.get ())
                    tmp.release ();
                  else
                    throw ::xsd::cxx::tree::not_derived< char > ();

                  this->Configuration_.set (::std::move (r));
                  continue;
                }
              }
            }

            break;
          }
        }

        InhalerConfigurationData* InhalerConfigurationData::
        _clone (::xml_schema::flags f,
                ::xml_schema::container* c) const
        {
          return new class InhalerConfigurationData (*this, f, c);
        }

        InhalerConfigurationData& InhalerConfigurationData::
        operator= (const InhalerConfigurationData& x)
        {
          if (this != &x)
          {
            static_cast< ::mil::tatrc::physiology::datamodel::InhalerActionData& > (*this) = x;
            this->ConfigurationFile_ = x.ConfigurationFile_;
            this->Configuration_ = x.Configuration_;
          }

          return *this;
        }

        InhalerConfigurationData::
        ~InhalerConfigurationData ()
        {
        }

        static
        const ::xsd::cxx::tree::type_factory_initializer< 0, char, InhalerConfigurationData >
        _xsd_InhalerConfigurationData_type_factory_init (
          "InhalerConfigurationData",
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
        operator<< (::std::ostream& o, const InhalerConfigurationData& i)
        {
          o << static_cast< const ::mil::tatrc::physiology::datamodel::InhalerActionData& > (i);

          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            if (i.ConfigurationFile ())
            {
              o << ::std::endl << "ConfigurationFile: ";
              om.insert (o, *i.ConfigurationFile ());
            }
          }

          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            if (i.Configuration ())
            {
              o << ::std::endl << "Configuration: ";
              om.insert (o, *i.Configuration ());
            }
          }

          return o;
        }

        static
        const ::xsd::cxx::tree::std_ostream_initializer< 0, char, InhalerConfigurationData >
        _xsd_InhalerConfigurationData_std_ostream_init;
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
        operator<< (::xercesc::DOMElement& e, const InhalerConfigurationData& i)
        {
          e << static_cast< const ::mil::tatrc::physiology::datamodel::InhalerActionData& > (i);

          // ConfigurationFile
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            if (i.ConfigurationFile ())
            {
              const InhalerConfigurationData::ConfigurationFile_type& x (*i.ConfigurationFile ());
              if (typeid (InhalerConfigurationData::ConfigurationFile_type) == typeid (x))
              {
                ::xercesc::DOMElement& s (
                  ::xsd::cxx::xml::dom::create_element (
                    "ConfigurationFile",
                    "uri:/mil/tatrc/physiology/datamodel",
                    e));

                s << x;
              }
              else
                tsm.serialize (
                  "ConfigurationFile",
                  "uri:/mil/tatrc/physiology/datamodel",
                  false, true, e, x);
            }
          }

          // Configuration
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            if (i.Configuration ())
            {
              const InhalerConfigurationData::Configuration_type& x (*i.Configuration ());
              if (typeid (InhalerConfigurationData::Configuration_type) == typeid (x))
              {
                ::xercesc::DOMElement& s (
                  ::xsd::cxx::xml::dom::create_element (
                    "Configuration",
                    "uri:/mil/tatrc/physiology/datamodel",
                    e));

                s << x;
              }
              else
                tsm.serialize (
                  "Configuration",
                  "uri:/mil/tatrc/physiology/datamodel",
                  false, true, e, x);
            }
          }
        }

        static
        const ::xsd::cxx::tree::type_serializer_initializer< 0, char, InhalerConfigurationData >
        _xsd_InhalerConfigurationData_type_serializer_init (
          "InhalerConfigurationData",
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

