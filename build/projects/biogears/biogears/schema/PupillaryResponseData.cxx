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

#include "PupillaryResponseData.hxx"

#include "ScalarNeg1To1Data.hxx"

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        // PupillaryResponseData
        // 

        const PupillaryResponseData::ReactivityModifier_optional& PupillaryResponseData::
        ReactivityModifier () const
        {
          return this->ReactivityModifier_;
        }

        PupillaryResponseData::ReactivityModifier_optional& PupillaryResponseData::
        ReactivityModifier ()
        {
          return this->ReactivityModifier_;
        }

        void PupillaryResponseData::
        ReactivityModifier (const ReactivityModifier_type& x)
        {
          this->ReactivityModifier_.set (x);
        }

        void PupillaryResponseData::
        ReactivityModifier (const ReactivityModifier_optional& x)
        {
          this->ReactivityModifier_ = x;
        }

        void PupillaryResponseData::
        ReactivityModifier (::std::unique_ptr< ReactivityModifier_type > x)
        {
          this->ReactivityModifier_.set (std::move (x));
        }

        const PupillaryResponseData::ShapeModifier_optional& PupillaryResponseData::
        ShapeModifier () const
        {
          return this->ShapeModifier_;
        }

        PupillaryResponseData::ShapeModifier_optional& PupillaryResponseData::
        ShapeModifier ()
        {
          return this->ShapeModifier_;
        }

        void PupillaryResponseData::
        ShapeModifier (const ShapeModifier_type& x)
        {
          this->ShapeModifier_.set (x);
        }

        void PupillaryResponseData::
        ShapeModifier (const ShapeModifier_optional& x)
        {
          this->ShapeModifier_ = x;
        }

        void PupillaryResponseData::
        ShapeModifier (::std::unique_ptr< ShapeModifier_type > x)
        {
          this->ShapeModifier_.set (std::move (x));
        }

        const PupillaryResponseData::SizeModifier_optional& PupillaryResponseData::
        SizeModifier () const
        {
          return this->SizeModifier_;
        }

        PupillaryResponseData::SizeModifier_optional& PupillaryResponseData::
        SizeModifier ()
        {
          return this->SizeModifier_;
        }

        void PupillaryResponseData::
        SizeModifier (const SizeModifier_type& x)
        {
          this->SizeModifier_.set (x);
        }

        void PupillaryResponseData::
        SizeModifier (const SizeModifier_optional& x)
        {
          this->SizeModifier_ = x;
        }

        void PupillaryResponseData::
        SizeModifier (::std::unique_ptr< SizeModifier_type > x)
        {
          this->SizeModifier_.set (std::move (x));
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
        // PupillaryResponseData
        //

        PupillaryResponseData::
        PupillaryResponseData ()
        : ::mil::tatrc::physiology::datamodel::ObjectData (),
          ReactivityModifier_ (this),
          ShapeModifier_ (this),
          SizeModifier_ (this)
        {
        }

        PupillaryResponseData::
        PupillaryResponseData (const PupillaryResponseData& x,
                               ::xml_schema::flags f,
                               ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::ObjectData (x, f, c),
          ReactivityModifier_ (x.ReactivityModifier_, f, this),
          ShapeModifier_ (x.ShapeModifier_, f, this),
          SizeModifier_ (x.SizeModifier_, f, this)
        {
        }

        PupillaryResponseData::
        PupillaryResponseData (const ::xercesc::DOMElement& e,
                               ::xml_schema::flags f,
                               ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::ObjectData (e, f | ::xml_schema::flags::base, c),
          ReactivityModifier_ (this),
          ShapeModifier_ (this),
          SizeModifier_ (this)
        {
          if ((f & ::xml_schema::flags::base) == 0)
          {
            ::xsd::cxx::xml::dom::parser< char > p (e, true, false, true);
            this->parse (p, f);
          }
        }

        void PupillaryResponseData::
        parse (::xsd::cxx::xml::dom::parser< char >& p,
               ::xml_schema::flags f)
        {
          this->::mil::tatrc::physiology::datamodel::ObjectData::parse (p, f);

          for (; p.more_content (); p.next_content (false))
          {
            const ::xercesc::DOMElement& i (p.cur_element ());
            const ::xsd::cxx::xml::qualified_name< char > n (
              ::xsd::cxx::xml::dom::name< char > (i));

            // ReactivityModifier
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "ReactivityModifier",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< ReactivityModifier_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                if (!this->ReactivityModifier_)
                {
                  ::std::unique_ptr< ReactivityModifier_type > r (
                    dynamic_cast< ReactivityModifier_type* > (tmp.get ()));

                  if (r.get ())
                    tmp.release ();
                  else
                    throw ::xsd::cxx::tree::not_derived< char > ();

                  this->ReactivityModifier_.set (::std::move (r));
                  continue;
                }
              }
            }

            // ShapeModifier
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "ShapeModifier",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< ShapeModifier_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                if (!this->ShapeModifier_)
                {
                  ::std::unique_ptr< ShapeModifier_type > r (
                    dynamic_cast< ShapeModifier_type* > (tmp.get ()));

                  if (r.get ())
                    tmp.release ();
                  else
                    throw ::xsd::cxx::tree::not_derived< char > ();

                  this->ShapeModifier_.set (::std::move (r));
                  continue;
                }
              }
            }

            // SizeModifier
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "SizeModifier",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< SizeModifier_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                if (!this->SizeModifier_)
                {
                  ::std::unique_ptr< SizeModifier_type > r (
                    dynamic_cast< SizeModifier_type* > (tmp.get ()));

                  if (r.get ())
                    tmp.release ();
                  else
                    throw ::xsd::cxx::tree::not_derived< char > ();

                  this->SizeModifier_.set (::std::move (r));
                  continue;
                }
              }
            }

            break;
          }
        }

        PupillaryResponseData* PupillaryResponseData::
        _clone (::xml_schema::flags f,
                ::xml_schema::container* c) const
        {
          return new class PupillaryResponseData (*this, f, c);
        }

        PupillaryResponseData& PupillaryResponseData::
        operator= (const PupillaryResponseData& x)
        {
          if (this != &x)
          {
            static_cast< ::mil::tatrc::physiology::datamodel::ObjectData& > (*this) = x;
            this->ReactivityModifier_ = x.ReactivityModifier_;
            this->ShapeModifier_ = x.ShapeModifier_;
            this->SizeModifier_ = x.SizeModifier_;
          }

          return *this;
        }

        PupillaryResponseData::
        ~PupillaryResponseData ()
        {
        }

        static
        const ::xsd::cxx::tree::type_factory_initializer< 0, char, PupillaryResponseData >
        _xsd_PupillaryResponseData_type_factory_init (
          "PupillaryResponseData",
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
        operator<< (::std::ostream& o, const PupillaryResponseData& i)
        {
          o << static_cast< const ::mil::tatrc::physiology::datamodel::ObjectData& > (i);

          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            if (i.ReactivityModifier ())
            {
              o << ::std::endl << "ReactivityModifier: ";
              om.insert (o, *i.ReactivityModifier ());
            }
          }

          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            if (i.ShapeModifier ())
            {
              o << ::std::endl << "ShapeModifier: ";
              om.insert (o, *i.ShapeModifier ());
            }
          }

          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            if (i.SizeModifier ())
            {
              o << ::std::endl << "SizeModifier: ";
              om.insert (o, *i.SizeModifier ());
            }
          }

          return o;
        }

        static
        const ::xsd::cxx::tree::std_ostream_initializer< 0, char, PupillaryResponseData >
        _xsd_PupillaryResponseData_std_ostream_init;
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
        operator<< (::xercesc::DOMElement& e, const PupillaryResponseData& i)
        {
          e << static_cast< const ::mil::tatrc::physiology::datamodel::ObjectData& > (i);

          // ReactivityModifier
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            if (i.ReactivityModifier ())
            {
              const PupillaryResponseData::ReactivityModifier_type& x (*i.ReactivityModifier ());
              if (typeid (PupillaryResponseData::ReactivityModifier_type) == typeid (x))
              {
                ::xercesc::DOMElement& s (
                  ::xsd::cxx::xml::dom::create_element (
                    "ReactivityModifier",
                    "uri:/mil/tatrc/physiology/datamodel",
                    e));

                s << x;
              }
              else
                tsm.serialize (
                  "ReactivityModifier",
                  "uri:/mil/tatrc/physiology/datamodel",
                  false, true, e, x);
            }
          }

          // ShapeModifier
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            if (i.ShapeModifier ())
            {
              const PupillaryResponseData::ShapeModifier_type& x (*i.ShapeModifier ());
              if (typeid (PupillaryResponseData::ShapeModifier_type) == typeid (x))
              {
                ::xercesc::DOMElement& s (
                  ::xsd::cxx::xml::dom::create_element (
                    "ShapeModifier",
                    "uri:/mil/tatrc/physiology/datamodel",
                    e));

                s << x;
              }
              else
                tsm.serialize (
                  "ShapeModifier",
                  "uri:/mil/tatrc/physiology/datamodel",
                  false, true, e, x);
            }
          }

          // SizeModifier
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            if (i.SizeModifier ())
            {
              const PupillaryResponseData::SizeModifier_type& x (*i.SizeModifier ());
              if (typeid (PupillaryResponseData::SizeModifier_type) == typeid (x))
              {
                ::xercesc::DOMElement& s (
                  ::xsd::cxx::xml::dom::create_element (
                    "SizeModifier",
                    "uri:/mil/tatrc/physiology/datamodel",
                    e));

                s << x;
              }
              else
                tsm.serialize (
                  "SizeModifier",
                  "uri:/mil/tatrc/physiology/datamodel",
                  false, true, e, x);
            }
          }
        }

        static
        const ::xsd::cxx::tree::type_serializer_initializer< 0, char, PupillaryResponseData >
        _xsd_PupillaryResponseData_type_serializer_init (
          "PupillaryResponseData",
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

