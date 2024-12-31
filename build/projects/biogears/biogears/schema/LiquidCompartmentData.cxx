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

#include "LiquidCompartmentData.hxx"

#include "ScalarData.hxx"

#include "ScalarFractionData.hxx"

#include "LiquidSubstanceQuantityData.hxx"

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        // LiquidCompartmentData
        // 

        const LiquidCompartmentData::pH_optional& LiquidCompartmentData::
        pH () const
        {
          return this->pH_;
        }

        LiquidCompartmentData::pH_optional& LiquidCompartmentData::
        pH ()
        {
          return this->pH_;
        }

        void LiquidCompartmentData::
        pH (const pH_type& x)
        {
          this->pH_.set (x);
        }

        void LiquidCompartmentData::
        pH (const pH_optional& x)
        {
          this->pH_ = x;
        }

        void LiquidCompartmentData::
        pH (::std::unique_ptr< pH_type > x)
        {
          this->pH_.set (std::move (x));
        }

        const LiquidCompartmentData::WaterVolumeFraction_optional& LiquidCompartmentData::
        WaterVolumeFraction () const
        {
          return this->WaterVolumeFraction_;
        }

        LiquidCompartmentData::WaterVolumeFraction_optional& LiquidCompartmentData::
        WaterVolumeFraction ()
        {
          return this->WaterVolumeFraction_;
        }

        void LiquidCompartmentData::
        WaterVolumeFraction (const WaterVolumeFraction_type& x)
        {
          this->WaterVolumeFraction_.set (x);
        }

        void LiquidCompartmentData::
        WaterVolumeFraction (const WaterVolumeFraction_optional& x)
        {
          this->WaterVolumeFraction_ = x;
        }

        void LiquidCompartmentData::
        WaterVolumeFraction (::std::unique_ptr< WaterVolumeFraction_type > x)
        {
          this->WaterVolumeFraction_.set (std::move (x));
        }

        const LiquidCompartmentData::SubstanceQuantity_sequence& LiquidCompartmentData::
        SubstanceQuantity () const
        {
          return this->SubstanceQuantity_;
        }

        LiquidCompartmentData::SubstanceQuantity_sequence& LiquidCompartmentData::
        SubstanceQuantity ()
        {
          return this->SubstanceQuantity_;
        }

        void LiquidCompartmentData::
        SubstanceQuantity (const SubstanceQuantity_sequence& s)
        {
          this->SubstanceQuantity_ = s;
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
        // LiquidCompartmentData
        //

        LiquidCompartmentData::
        LiquidCompartmentData ()
        : ::mil::tatrc::physiology::datamodel::FluidCompartmentData (),
          pH_ (this),
          WaterVolumeFraction_ (this),
          SubstanceQuantity_ (this)
        {
        }

        LiquidCompartmentData::
        LiquidCompartmentData (const Name_type& Name)
        : ::mil::tatrc::physiology::datamodel::FluidCompartmentData (Name),
          pH_ (this),
          WaterVolumeFraction_ (this),
          SubstanceQuantity_ (this)
        {
        }

        LiquidCompartmentData::
        LiquidCompartmentData (::std::unique_ptr< Name_type > Name)
        : ::mil::tatrc::physiology::datamodel::FluidCompartmentData (std::move (Name)),
          pH_ (this),
          WaterVolumeFraction_ (this),
          SubstanceQuantity_ (this)
        {
        }

        LiquidCompartmentData::
        LiquidCompartmentData (const LiquidCompartmentData& x,
                               ::xml_schema::flags f,
                               ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::FluidCompartmentData (x, f, c),
          pH_ (x.pH_, f, this),
          WaterVolumeFraction_ (x.WaterVolumeFraction_, f, this),
          SubstanceQuantity_ (x.SubstanceQuantity_, f, this)
        {
        }

        LiquidCompartmentData::
        LiquidCompartmentData (const ::xercesc::DOMElement& e,
                               ::xml_schema::flags f,
                               ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::FluidCompartmentData (e, f | ::xml_schema::flags::base, c),
          pH_ (this),
          WaterVolumeFraction_ (this),
          SubstanceQuantity_ (this)
        {
          if ((f & ::xml_schema::flags::base) == 0)
          {
            ::xsd::cxx::xml::dom::parser< char > p (e, true, false, true);
            this->parse (p, f);
          }
        }

        void LiquidCompartmentData::
        parse (::xsd::cxx::xml::dom::parser< char >& p,
               ::xml_schema::flags f)
        {
          this->::mil::tatrc::physiology::datamodel::FluidCompartmentData::parse (p, f);

          for (; p.more_content (); p.next_content (false))
          {
            const ::xercesc::DOMElement& i (p.cur_element ());
            const ::xsd::cxx::xml::qualified_name< char > n (
              ::xsd::cxx::xml::dom::name< char > (i));

            // pH
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "pH",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< pH_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                if (!this->pH_)
                {
                  ::std::unique_ptr< pH_type > r (
                    dynamic_cast< pH_type* > (tmp.get ()));

                  if (r.get ())
                    tmp.release ();
                  else
                    throw ::xsd::cxx::tree::not_derived< char > ();

                  this->pH_.set (::std::move (r));
                  continue;
                }
              }
            }

            // WaterVolumeFraction
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "WaterVolumeFraction",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< WaterVolumeFraction_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                if (!this->WaterVolumeFraction_)
                {
                  ::std::unique_ptr< WaterVolumeFraction_type > r (
                    dynamic_cast< WaterVolumeFraction_type* > (tmp.get ()));

                  if (r.get ())
                    tmp.release ();
                  else
                    throw ::xsd::cxx::tree::not_derived< char > ();

                  this->WaterVolumeFraction_.set (::std::move (r));
                  continue;
                }
              }
            }

            // SubstanceQuantity
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "SubstanceQuantity",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< SubstanceQuantity_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                ::std::unique_ptr< SubstanceQuantity_type > r (
                  dynamic_cast< SubstanceQuantity_type* > (tmp.get ()));

                if (r.get ())
                  tmp.release ();
                else
                  throw ::xsd::cxx::tree::not_derived< char > ();

                this->SubstanceQuantity_.push_back (::std::move (r));
                continue;
              }
            }

            break;
          }
        }

        LiquidCompartmentData* LiquidCompartmentData::
        _clone (::xml_schema::flags f,
                ::xml_schema::container* c) const
        {
          return new class LiquidCompartmentData (*this, f, c);
        }

        LiquidCompartmentData& LiquidCompartmentData::
        operator= (const LiquidCompartmentData& x)
        {
          if (this != &x)
          {
            static_cast< ::mil::tatrc::physiology::datamodel::FluidCompartmentData& > (*this) = x;
            this->pH_ = x.pH_;
            this->WaterVolumeFraction_ = x.WaterVolumeFraction_;
            this->SubstanceQuantity_ = x.SubstanceQuantity_;
          }

          return *this;
        }

        LiquidCompartmentData::
        ~LiquidCompartmentData ()
        {
        }

        static
        const ::xsd::cxx::tree::type_factory_initializer< 0, char, LiquidCompartmentData >
        _xsd_LiquidCompartmentData_type_factory_init (
          "LiquidCompartmentData",
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
        operator<< (::std::ostream& o, const LiquidCompartmentData& i)
        {
          o << static_cast< const ::mil::tatrc::physiology::datamodel::FluidCompartmentData& > (i);

          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            if (i.pH ())
            {
              o << ::std::endl << "pH: ";
              om.insert (o, *i.pH ());
            }
          }

          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            if (i.WaterVolumeFraction ())
            {
              o << ::std::endl << "WaterVolumeFraction: ";
              om.insert (o, *i.WaterVolumeFraction ());
            }
          }

          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            for (LiquidCompartmentData::SubstanceQuantity_const_iterator
                 b (i.SubstanceQuantity ().begin ()), e (i.SubstanceQuantity ().end ());
                 b != e; ++b)
            {
              o << ::std::endl << "SubstanceQuantity: ";
              om.insert (o, *b);
            }
          }

          return o;
        }

        static
        const ::xsd::cxx::tree::std_ostream_initializer< 0, char, LiquidCompartmentData >
        _xsd_LiquidCompartmentData_std_ostream_init;
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
        operator<< (::xercesc::DOMElement& e, const LiquidCompartmentData& i)
        {
          e << static_cast< const ::mil::tatrc::physiology::datamodel::FluidCompartmentData& > (i);

          // pH
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            if (i.pH ())
            {
              const LiquidCompartmentData::pH_type& x (*i.pH ());
              if (typeid (LiquidCompartmentData::pH_type) == typeid (x))
              {
                ::xercesc::DOMElement& s (
                  ::xsd::cxx::xml::dom::create_element (
                    "pH",
                    "uri:/mil/tatrc/physiology/datamodel",
                    e));

                s << x;
              }
              else
                tsm.serialize (
                  "pH",
                  "uri:/mil/tatrc/physiology/datamodel",
                  false, true, e, x);
            }
          }

          // WaterVolumeFraction
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            if (i.WaterVolumeFraction ())
            {
              const LiquidCompartmentData::WaterVolumeFraction_type& x (*i.WaterVolumeFraction ());
              if (typeid (LiquidCompartmentData::WaterVolumeFraction_type) == typeid (x))
              {
                ::xercesc::DOMElement& s (
                  ::xsd::cxx::xml::dom::create_element (
                    "WaterVolumeFraction",
                    "uri:/mil/tatrc/physiology/datamodel",
                    e));

                s << x;
              }
              else
                tsm.serialize (
                  "WaterVolumeFraction",
                  "uri:/mil/tatrc/physiology/datamodel",
                  false, true, e, x);
            }
          }

          // SubstanceQuantity
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            for (LiquidCompartmentData::SubstanceQuantity_const_iterator
                 b (i.SubstanceQuantity ().begin ()), n (i.SubstanceQuantity ().end ());
                 b != n; ++b)
            {
              if (typeid (LiquidCompartmentData::SubstanceQuantity_type) == typeid (*b))
              {
                ::xercesc::DOMElement& s (
                  ::xsd::cxx::xml::dom::create_element (
                    "SubstanceQuantity",
                    "uri:/mil/tatrc/physiology/datamodel",
                    e));

                s << *b;
              }
              else
                tsm.serialize (
                  "SubstanceQuantity",
                  "uri:/mil/tatrc/physiology/datamodel",
                  false, true, e, *b);
            }
          }
        }

        static
        const ::xsd::cxx::tree::type_serializer_initializer< 0, char, LiquidCompartmentData >
        _xsd_LiquidCompartmentData_type_serializer_init (
          "LiquidCompartmentData",
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

