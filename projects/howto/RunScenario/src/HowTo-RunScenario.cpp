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
#include <biogears/cdm/scenario/SEScenario.h>
#include <biogears/cdm/scenario/SEScenarioInitialParameters.h>
#include <biogears/cdm/scenario/SEScenarioExec.h>
#include <biogears/cdm/compartment/SECompartmentManager.h>
#include <biogears/cdm/scenario/SEAdvanceTime.h>

#include <biogears/cdm/properties/SEScalarFrequency.h>
#include <biogears/cdm/properties/SEScalarTime.h>
#include <biogears/cdm/properties/SEScalarVolume.h>

class MyCustomExec : public SEScenarioCustomExec
{
public:
  void CustomExec(double time_s, PhysiologyEngine* engine) override
  {
    // Called at the end of EACH time step
    // Add any custom logic you want here...
  }
};

void HowToRunScenario(const std::string& scenarioXML)
{
  // 1) Create a BioGears engine
  std::unique_ptr<PhysiologyEngine> bg = CreateBioGearsEngine("HowToRunScenario.log");
  bg->GetLogger()->Info("HowToRunScenario");
  bg->GetLogger()->SetLogLevel(log4cpp::Priority::INFO);

  // 2) Create our scenario
  SEScenario sce(bg->GetSubstanceManager());

  // 3) Load from the XML string
  if (!sce.LoadString(scenarioXML)) {
    bg->GetLogger()->Error("Could not load scenario from XML string!");
    return; // Nothing else to do
  }

  // 4) You can set requests, sampling, etc.
  sce.GetDataRequestManager().SetSamplesPerSecond(1);

  // 5) Execute the scenario
  SEScenarioExec executor(*bg);
  executor.Execute(sce, "./HowTo-RunScenarioResults.csv", new MyCustomExec());
}
