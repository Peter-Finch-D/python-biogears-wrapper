
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was BiogearsConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

####################################################################################

include(CMakeFindDependencyMacro)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

find_dependency(Eigen3)
find_dependency(Log4cpp)
find_dependency(CodeSynthesis)

include ( "${CMAKE_CURRENT_LIST_DIR}/Biogears.cmake" )
