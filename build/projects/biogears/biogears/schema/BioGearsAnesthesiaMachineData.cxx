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

#include "BioGearsAnesthesiaMachineData.hxx"

#include "ScalarTimeData.hxx"

#include "ScalarData.hxx"

namespace mil
{
  namespace tatrc
  {
    namespace physiology
    {
      namespace datamodel
      {
        // BioGearsAnesthesiaMachineData
        // 

        const BioGearsAnesthesiaMachineData::Inhaling_type& BioGearsAnesthesiaMachineData::
        Inhaling () const
        {
          return this->Inhaling_.get ();
        }

        BioGearsAnesthesiaMachineData::Inhaling_type& BioGearsAnesthesiaMachineData::
        Inhaling ()
        {
          return this->Inhaling_.get ();
        }

        void BioGearsAnesthesiaMachineData::
        Inhaling (const Inhaling_type& x)
        {
          this->Inhaling_.set (x);
        }

        const BioGearsAnesthesiaMachineData::CurrentBreathingCycleTime_type& BioGearsAnesthesiaMachineData::
        CurrentBreathingCycleTime () const
        {
          return this->CurrentBreathingCycleTime_.get ();
        }

        BioGearsAnesthesiaMachineData::CurrentBreathingCycleTime_type& BioGearsAnesthesiaMachineData::
        CurrentBreathingCycleTime ()
        {
          return this->CurrentBreathingCycleTime_.get ();
        }

        void BioGearsAnesthesiaMachineData::
        CurrentBreathingCycleTime (const CurrentBreathingCycleTime_type& x)
        {
          this->CurrentBreathingCycleTime_.set (x);
        }

        void BioGearsAnesthesiaMachineData::
        CurrentBreathingCycleTime (::std::unique_ptr< CurrentBreathingCycleTime_type > x)
        {
          this->CurrentBreathingCycleTime_.set (std::move (x));
        }

        const BioGearsAnesthesiaMachineData::InspirationTime_type& BioGearsAnesthesiaMachineData::
        InspirationTime () const
        {
          return this->InspirationTime_.get ();
        }

        BioGearsAnesthesiaMachineData::InspirationTime_type& BioGearsAnesthesiaMachineData::
        InspirationTime ()
        {
          return this->InspirationTime_.get ();
        }

        void BioGearsAnesthesiaMachineData::
        InspirationTime (const InspirationTime_type& x)
        {
          this->InspirationTime_.set (x);
        }

        void BioGearsAnesthesiaMachineData::
        InspirationTime (::std::unique_ptr< InspirationTime_type > x)
        {
          this->InspirationTime_.set (std::move (x));
        }

        const BioGearsAnesthesiaMachineData::OxygenInletVolumeFraction_type& BioGearsAnesthesiaMachineData::
        OxygenInletVolumeFraction () const
        {
          return this->OxygenInletVolumeFraction_.get ();
        }

        BioGearsAnesthesiaMachineData::OxygenInletVolumeFraction_type& BioGearsAnesthesiaMachineData::
        OxygenInletVolumeFraction ()
        {
          return this->OxygenInletVolumeFraction_.get ();
        }

        void BioGearsAnesthesiaMachineData::
        OxygenInletVolumeFraction (const OxygenInletVolumeFraction_type& x)
        {
          this->OxygenInletVolumeFraction_.set (x);
        }

        void BioGearsAnesthesiaMachineData::
        OxygenInletVolumeFraction (::std::unique_ptr< OxygenInletVolumeFraction_type > x)
        {
          this->OxygenInletVolumeFraction_.set (std::move (x));
        }

        const BioGearsAnesthesiaMachineData::TotalBreathingCycleTime_type& BioGearsAnesthesiaMachineData::
        TotalBreathingCycleTime () const
        {
          return this->TotalBreathingCycleTime_.get ();
        }

        BioGearsAnesthesiaMachineData::TotalBreathingCycleTime_type& BioGearsAnesthesiaMachineData::
        TotalBreathingCycleTime ()
        {
          return this->TotalBreathingCycleTime_.get ();
        }

        void BioGearsAnesthesiaMachineData::
        TotalBreathingCycleTime (const TotalBreathingCycleTime_type& x)
        {
          this->TotalBreathingCycleTime_.set (x);
        }

        void BioGearsAnesthesiaMachineData::
        TotalBreathingCycleTime (::std::unique_ptr< TotalBreathingCycleTime_type > x)
        {
          this->TotalBreathingCycleTime_.set (std::move (x));
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
        // BioGearsAnesthesiaMachineData
        //

        BioGearsAnesthesiaMachineData::
        BioGearsAnesthesiaMachineData ()
        : ::mil::tatrc::physiology::datamodel::AnesthesiaMachineData (),
          Inhaling_ (this),
          CurrentBreathingCycleTime_ (this),
          InspirationTime_ (this),
          OxygenInletVolumeFraction_ (this),
          TotalBreathingCycleTime_ (this)
        {
        }

        BioGearsAnesthesiaMachineData::
        BioGearsAnesthesiaMachineData (const Inhaling_type& Inhaling,
                                       const CurrentBreathingCycleTime_type& CurrentBreathingCycleTime,
                                       const InspirationTime_type& InspirationTime,
                                       const OxygenInletVolumeFraction_type& OxygenInletVolumeFraction,
                                       const TotalBreathingCycleTime_type& TotalBreathingCycleTime)
        : ::mil::tatrc::physiology::datamodel::AnesthesiaMachineData (),
          Inhaling_ (Inhaling, this),
          CurrentBreathingCycleTime_ (CurrentBreathingCycleTime, this),
          InspirationTime_ (InspirationTime, this),
          OxygenInletVolumeFraction_ (OxygenInletVolumeFraction, this),
          TotalBreathingCycleTime_ (TotalBreathingCycleTime, this)
        {
        }

        BioGearsAnesthesiaMachineData::
        BioGearsAnesthesiaMachineData (const Inhaling_type& Inhaling,
                                       ::std::unique_ptr< CurrentBreathingCycleTime_type > CurrentBreathingCycleTime,
                                       ::std::unique_ptr< InspirationTime_type > InspirationTime,
                                       ::std::unique_ptr< OxygenInletVolumeFraction_type > OxygenInletVolumeFraction,
                                       ::std::unique_ptr< TotalBreathingCycleTime_type > TotalBreathingCycleTime)
        : ::mil::tatrc::physiology::datamodel::AnesthesiaMachineData (),
          Inhaling_ (Inhaling, this),
          CurrentBreathingCycleTime_ (std::move (CurrentBreathingCycleTime), this),
          InspirationTime_ (std::move (InspirationTime), this),
          OxygenInletVolumeFraction_ (std::move (OxygenInletVolumeFraction), this),
          TotalBreathingCycleTime_ (std::move (TotalBreathingCycleTime), this)
        {
        }

        BioGearsAnesthesiaMachineData::
        BioGearsAnesthesiaMachineData (const BioGearsAnesthesiaMachineData& x,
                                       ::xml_schema::flags f,
                                       ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::AnesthesiaMachineData (x, f, c),
          Inhaling_ (x.Inhaling_, f, this),
          CurrentBreathingCycleTime_ (x.CurrentBreathingCycleTime_, f, this),
          InspirationTime_ (x.InspirationTime_, f, this),
          OxygenInletVolumeFraction_ (x.OxygenInletVolumeFraction_, f, this),
          TotalBreathingCycleTime_ (x.TotalBreathingCycleTime_, f, this)
        {
        }

        BioGearsAnesthesiaMachineData::
        BioGearsAnesthesiaMachineData (const ::xercesc::DOMElement& e,
                                       ::xml_schema::flags f,
                                       ::xml_schema::container* c)
        : ::mil::tatrc::physiology::datamodel::AnesthesiaMachineData (e, f | ::xml_schema::flags::base, c),
          Inhaling_ (this),
          CurrentBreathingCycleTime_ (this),
          InspirationTime_ (this),
          OxygenInletVolumeFraction_ (this),
          TotalBreathingCycleTime_ (this)
        {
          if ((f & ::xml_schema::flags::base) == 0)
          {
            ::xsd::cxx::xml::dom::parser< char > p (e, true, false, true);
            this->parse (p, f);
          }
        }

        void BioGearsAnesthesiaMachineData::
        parse (::xsd::cxx::xml::dom::parser< char >& p,
               ::xml_schema::flags f)
        {
          this->::mil::tatrc::physiology::datamodel::AnesthesiaMachineData::parse (p, f);

          for (; p.more_content (); p.next_content (false))
          {
            const ::xercesc::DOMElement& i (p.cur_element ());
            const ::xsd::cxx::xml::qualified_name< char > n (
              ::xsd::cxx::xml::dom::name< char > (i));

            // Inhaling
            //
            if (n.name () == "Inhaling" && n.namespace_ () == "uri:/mil/tatrc/physiology/datamodel")
            {
              if (!Inhaling_.present ())
              {
                this->Inhaling_.set (Inhaling_traits::create (i, f, this));
                continue;
              }
            }

            // CurrentBreathingCycleTime
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "CurrentBreathingCycleTime",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< CurrentBreathingCycleTime_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                if (!CurrentBreathingCycleTime_.present ())
                {
                  ::std::unique_ptr< CurrentBreathingCycleTime_type > r (
                    dynamic_cast< CurrentBreathingCycleTime_type* > (tmp.get ()));

                  if (r.get ())
                    tmp.release ();
                  else
                    throw ::xsd::cxx::tree::not_derived< char > ();

                  this->CurrentBreathingCycleTime_.set (::std::move (r));
                  continue;
                }
              }
            }

            // InspirationTime
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "InspirationTime",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< InspirationTime_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                if (!InspirationTime_.present ())
                {
                  ::std::unique_ptr< InspirationTime_type > r (
                    dynamic_cast< InspirationTime_type* > (tmp.get ()));

                  if (r.get ())
                    tmp.release ();
                  else
                    throw ::xsd::cxx::tree::not_derived< char > ();

                  this->InspirationTime_.set (::std::move (r));
                  continue;
                }
              }
            }

            // OxygenInletVolumeFraction
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "OxygenInletVolumeFraction",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< OxygenInletVolumeFraction_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                if (!OxygenInletVolumeFraction_.present ())
                {
                  ::std::unique_ptr< OxygenInletVolumeFraction_type > r (
                    dynamic_cast< OxygenInletVolumeFraction_type* > (tmp.get ()));

                  if (r.get ())
                    tmp.release ();
                  else
                    throw ::xsd::cxx::tree::not_derived< char > ();

                  this->OxygenInletVolumeFraction_.set (::std::move (r));
                  continue;
                }
              }
            }

            // TotalBreathingCycleTime
            //
            {
              ::std::unique_ptr< ::xsd::cxx::tree::type > tmp (
                ::xsd::cxx::tree::type_factory_map_instance< 0, char > ().create (
                  "TotalBreathingCycleTime",
                  "uri:/mil/tatrc/physiology/datamodel",
                  &::xsd::cxx::tree::factory_impl< TotalBreathingCycleTime_type >,
                  false, true, i, n, f, this));

              if (tmp.get () != 0)
              {
                if (!TotalBreathingCycleTime_.present ())
                {
                  ::std::unique_ptr< TotalBreathingCycleTime_type > r (
                    dynamic_cast< TotalBreathingCycleTime_type* > (tmp.get ()));

                  if (r.get ())
                    tmp.release ();
                  else
                    throw ::xsd::cxx::tree::not_derived< char > ();

                  this->TotalBreathingCycleTime_.set (::std::move (r));
                  continue;
                }
              }
            }

            break;
          }

          if (!Inhaling_.present ())
          {
            throw ::xsd::cxx::tree::expected_element< char > (
              "Inhaling",
              "uri:/mil/tatrc/physiology/datamodel");
          }

          if (!CurrentBreathingCycleTime_.present ())
          {
            throw ::xsd::cxx::tree::expected_element< char > (
              "CurrentBreathingCycleTime",
              "uri:/mil/tatrc/physiology/datamodel");
          }

          if (!InspirationTime_.present ())
          {
            throw ::xsd::cxx::tree::expected_element< char > (
              "InspirationTime",
              "uri:/mil/tatrc/physiology/datamodel");
          }

          if (!OxygenInletVolumeFraction_.present ())
          {
            throw ::xsd::cxx::tree::expected_element< char > (
              "OxygenInletVolumeFraction",
              "uri:/mil/tatrc/physiology/datamodel");
          }

          if (!TotalBreathingCycleTime_.present ())
          {
            throw ::xsd::cxx::tree::expected_element< char > (
              "TotalBreathingCycleTime",
              "uri:/mil/tatrc/physiology/datamodel");
          }
        }

        BioGearsAnesthesiaMachineData* BioGearsAnesthesiaMachineData::
        _clone (::xml_schema::flags f,
                ::xml_schema::container* c) const
        {
          return new class BioGearsAnesthesiaMachineData (*this, f, c);
        }

        BioGearsAnesthesiaMachineData& BioGearsAnesthesiaMachineData::
        operator= (const BioGearsAnesthesiaMachineData& x)
        {
          if (this != &x)
          {
            static_cast< ::mil::tatrc::physiology::datamodel::AnesthesiaMachineData& > (*this) = x;
            this->Inhaling_ = x.Inhaling_;
            this->CurrentBreathingCycleTime_ = x.CurrentBreathingCycleTime_;
            this->InspirationTime_ = x.InspirationTime_;
            this->OxygenInletVolumeFraction_ = x.OxygenInletVolumeFraction_;
            this->TotalBreathingCycleTime_ = x.TotalBreathingCycleTime_;
          }

          return *this;
        }

        BioGearsAnesthesiaMachineData::
        ~BioGearsAnesthesiaMachineData ()
        {
        }

        static
        const ::xsd::cxx::tree::type_factory_initializer< 0, char, BioGearsAnesthesiaMachineData >
        _xsd_BioGearsAnesthesiaMachineData_type_factory_init (
          "BioGearsAnesthesiaMachineData",
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
        operator<< (::std::ostream& o, const BioGearsAnesthesiaMachineData& i)
        {
          o << static_cast< const ::mil::tatrc::physiology::datamodel::AnesthesiaMachineData& > (i);

          o << ::std::endl << "Inhaling: " << i.Inhaling ();
          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            o << ::std::endl << "CurrentBreathingCycleTime: ";
            om.insert (o, i.CurrentBreathingCycleTime ());
          }

          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            o << ::std::endl << "InspirationTime: ";
            om.insert (o, i.InspirationTime ());
          }

          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            o << ::std::endl << "OxygenInletVolumeFraction: ";
            om.insert (o, i.OxygenInletVolumeFraction ());
          }

          {
            ::xsd::cxx::tree::std_ostream_map< char >& om (
              ::xsd::cxx::tree::std_ostream_map_instance< 0, char > ());

            o << ::std::endl << "TotalBreathingCycleTime: ";
            om.insert (o, i.TotalBreathingCycleTime ());
          }

          return o;
        }

        static
        const ::xsd::cxx::tree::std_ostream_initializer< 0, char, BioGearsAnesthesiaMachineData >
        _xsd_BioGearsAnesthesiaMachineData_std_ostream_init;
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
        operator<< (::xercesc::DOMElement& e, const BioGearsAnesthesiaMachineData& i)
        {
          e << static_cast< const ::mil::tatrc::physiology::datamodel::AnesthesiaMachineData& > (i);

          // Inhaling
          //
          {
            ::xercesc::DOMElement& s (
              ::xsd::cxx::xml::dom::create_element (
                "Inhaling",
                "uri:/mil/tatrc/physiology/datamodel",
                e));

            s << i.Inhaling ();
          }

          // CurrentBreathingCycleTime
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            const BioGearsAnesthesiaMachineData::CurrentBreathingCycleTime_type& x (i.CurrentBreathingCycleTime ());
            if (typeid (BioGearsAnesthesiaMachineData::CurrentBreathingCycleTime_type) == typeid (x))
            {
              ::xercesc::DOMElement& s (
                ::xsd::cxx::xml::dom::create_element (
                  "CurrentBreathingCycleTime",
                  "uri:/mil/tatrc/physiology/datamodel",
                  e));

              s << x;
            }
            else
              tsm.serialize (
                "CurrentBreathingCycleTime",
                "uri:/mil/tatrc/physiology/datamodel",
                false, true, e, x);
          }

          // InspirationTime
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            const BioGearsAnesthesiaMachineData::InspirationTime_type& x (i.InspirationTime ());
            if (typeid (BioGearsAnesthesiaMachineData::InspirationTime_type) == typeid (x))
            {
              ::xercesc::DOMElement& s (
                ::xsd::cxx::xml::dom::create_element (
                  "InspirationTime",
                  "uri:/mil/tatrc/physiology/datamodel",
                  e));

              s << x;
            }
            else
              tsm.serialize (
                "InspirationTime",
                "uri:/mil/tatrc/physiology/datamodel",
                false, true, e, x);
          }

          // OxygenInletVolumeFraction
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            const BioGearsAnesthesiaMachineData::OxygenInletVolumeFraction_type& x (i.OxygenInletVolumeFraction ());
            if (typeid (BioGearsAnesthesiaMachineData::OxygenInletVolumeFraction_type) == typeid (x))
            {
              ::xercesc::DOMElement& s (
                ::xsd::cxx::xml::dom::create_element (
                  "OxygenInletVolumeFraction",
                  "uri:/mil/tatrc/physiology/datamodel",
                  e));

              s << x;
            }
            else
              tsm.serialize (
                "OxygenInletVolumeFraction",
                "uri:/mil/tatrc/physiology/datamodel",
                false, true, e, x);
          }

          // TotalBreathingCycleTime
          //
          {
            ::xsd::cxx::tree::type_serializer_map< char >& tsm (
              ::xsd::cxx::tree::type_serializer_map_instance< 0, char > ());

            const BioGearsAnesthesiaMachineData::TotalBreathingCycleTime_type& x (i.TotalBreathingCycleTime ());
            if (typeid (BioGearsAnesthesiaMachineData::TotalBreathingCycleTime_type) == typeid (x))
            {
              ::xercesc::DOMElement& s (
                ::xsd::cxx::xml::dom::create_element (
                  "TotalBreathingCycleTime",
                  "uri:/mil/tatrc/physiology/datamodel",
                  e));

              s << x;
            }
            else
              tsm.serialize (
                "TotalBreathingCycleTime",
                "uri:/mil/tatrc/physiology/datamodel",
                false, true, e, x);
          }
        }

        static
        const ::xsd::cxx::tree::type_serializer_initializer< 0, char, BioGearsAnesthesiaMachineData >
        _xsd_BioGearsAnesthesiaMachineData_type_serializer_init (
          "BioGearsAnesthesiaMachineData",
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

