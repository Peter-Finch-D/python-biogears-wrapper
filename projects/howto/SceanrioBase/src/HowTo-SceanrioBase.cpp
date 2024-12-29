/**************************************************************************************
Copyright 2015 Applied Research Associates, Inc.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the License
at:
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under
the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
**************************************************************************************/

#include "HowToTracker.h"

// Include the various types you will be using in your code
#include <biogears/cdm/engine/PhysiologyEngineTrack.h>
#include <biogears/cdm/scenario/SEScenario.h>
#include <biogears/cdm/scenario/SEScenarioInitialParameters.h>
#include <biogears/cdm/scenario/SEAdvanceTime.h>
#include <biogears/cdm/properties/SEScalarTime.h>
#include <biogears/cdm/Serializer.h>
#include <biogears/schema/DataRequestsData.hxx>

//--------------------------------------------------------------------------------------------------
/// \brief
/// Creating an engine based on a scenario file
///
/// \details
/// Read a scenario file and pull things out for your use case
//--------------------------------------------------------------------------------------------------
void HowToScenarioBase()
{
  // Create our engine
  std::unique_ptr<PhysiologyEngine> bg = CreateBioGearsEngine("HowToScenarioBase.log");
  bg->GetLogger()->Info("HowToScenarioBase");

  // Let's read the scenario we want to base this engine on
  SEScenario sce(bg->GetSubstanceManager());
  sce.LoadFile("Scenarios/Dynamic_Scenario.xml");

  if (sce.HasEngineStateFile())
  {
    if (!bg->LoadState(sce.GetEngineStateFile()))
    {
      bg->GetLogger()->Error("Could not load state, check the error");
      return;
    }
  }
  else if (sce.HasInitialParameters())
  {
    SEScenarioInitialParameters& sip = sce.GetInitialParameters();
    if (sip.HasPatientFile())
    {
      std::vector<const SECondition*> conditions;
      for (SECondition* c : sip.GetConditions())
        conditions.push_back(c); // Copy to const
      if (!bg->InitializeEngine(sip.GetPatientFile(), &conditions, &sip.GetConfiguration()))
      {
        bg->GetLogger()->Error("Could not load state, check the error");
        return;
      }
    }
    else if (sip.HasPatient())
    {
      std::vector<const SECondition*> conditions;
      for (SECondition* c : sip.GetConditions())
        conditions.push_back(c); // Copy to const
      if (!bg->InitializeEngine(sip.GetPatient(), &conditions, &sip.GetConfiguration()))
      {
        bg->GetLogger()->Error("Could not load state, check the error");
        return;
      }
    }
  }

  // This will clear out the data requests if any exist in the DataRequestManager
  CDM::DataRequestsData* drData = sce.GetDataRequestManager().Unload();
  bg->GetEngineTrack()->GetDataRequestManager().Load(*drData, bg->GetSubstanceManager());
  delete drData;

  // Set the results filename to test_results.csv
  bg->GetEngineTrack()->GetDataRequestManager().SetResultsFilename("./test_results.csv");
  bg->GetLogger()->Info("Results filename set to ./test_results.csv");

  // Set the sampling interval to 1 second (or any desired interval)
  bg->GetEngineTrack()->GetDataRequestManager().SetSamplesPerSecond(10);
  bg->GetLogger()->Info("Sampling interval set to 1 second.");

  // Log the data requests to ensure they are present
  if (bg->GetEngineTrack()->GetDataRequestManager().GetDataRequests().empty())
  {
    bg->GetLogger()->Error("No data requests found in the scenario.");
  }
  else
  {
    bg->GetLogger()->Info("Data requests found in the scenario.");
  }

  // Let's request data to be tracked that is in the scenario
  HowToTracker tracker(*bg);
  SEAdvanceTime* adv;
  // Now run the scenario actions
  for (SEAction* a : sce.GetActions())
  {
    // We want the tracker to process an advance time action so it will write each time step of data to our track file
    adv = dynamic_cast<SEAdvanceTime*>(a);
    if (adv != nullptr)
      tracker.AdvanceModelTime(adv->GetTime(TimeUnit::s)); // you could just do bg->AdvanceModelTime without tracking timesteps
    else
      bg->ProcessAction(*a);
  }

  // Finalize the engine to ensure all data is written
  bg->GetLogger()->Info("Finalizing the engine.");
  bg->GetEngineTrack()->GetDataRequestManager().SetResultsFilename("./test_results.csv");

  // At this point your engine is where you want it to be
  // You could read in a new scenario and run its actions
}
