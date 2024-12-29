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


//--------------------------------------------------------------------------------------------------
/// \brief
/// A class used to handle any specific logic you may want to do each time step
///
/// \details
/// This method will be called at the end of EACH time step of the engine
/// The SEScenarioExecutor will process the advance time actions in a scenario and 
/// step the engine, calling this method each time step
//--------------------------------------------------------------------------------------------------
class MyCustomExec : public SEScenarioCustomExec
{
public:
	void CustomExec(double time_s, PhysiologyEngine* engine)
	{
		// you are given the current scenairo time and the engine, so you can do what ever you want
	}
};

//--------------------------------------------------------------------------------------------------
/// \brief
/// Usage of creating and running a scenario
///
/// \details
//--------------------------------------------------------------------------------------------------
void HowToRunScenario()
{
	std::unique_ptr<PhysiologyEngine> bg = CreateBioGearsEngine("HowToRunScenario.log");

  	bg->GetLogger()->Info("HowToRunScenario");
	bg->GetLogger()->SetLogLevel(log4cpp::Priority::INFO);

	SEScenarioExec executor(*bg);
	SEScenario sce(bg->GetSubstanceManager());

	sce.LoadFile("Scenarios/Dynamic_Scenario.xml");
	//sce.LoadStream(std::cin);
	sce.GetDataRequestManager().SetSamplesPerSecond(1); // Set sampling interval to 1 second

	executor.Execute(sce, "./HowTo-RunScenarioResults.csv", new MyCustomExec());
}
