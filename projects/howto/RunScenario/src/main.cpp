#include <iostream>

#include "HowTo-RunScenario.h"
#include "HowToTracker.h"

#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
  // We expect the full XML to be passed as a single argument
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " \"<FullScenarioXML>\"\n";
    return 1;
  }

  // Concatenate argv[1] through argv[argc-1] if needed
  // (If your XML might have spaces, newlines, etc. you might need to combine them.)
  // If your XML is guaranteed to be in argv[1] only, just do:
  std::string scenarioXML = argv[1];

  // Call the function that executes the scenario
  HowToRunScenario(scenarioXML);

  return 0;
}
