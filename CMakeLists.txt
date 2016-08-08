# ---------------------------------------- #
#                                          #
#                   GGM                    #
#                                          #
# ---------------------------------------- #
cmake_minimum_required(VERSION 3.1.1)
include(Scripts/common.cmake)
project(ggm)

# Sources
# -------------------------------------------------------------------------------------------------
set(GGM_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
globrecurse_c_cpp_sources(GGM_SOURCES ${GGM_ROOT_DIR}/Source)
globrecurse_c_cpp_sources(GGM_INCLUDES ${GGM_ROOT_DIR}/Include)
group_sources(${GGM_ROOT_DIR}/Source PREFIX "ggm" ${GGM_SOURCES})
group_sources(${GGM_ROOT_DIR}/Include PREFIX "include" ${GGM_INCLUDES})

# Some CMake options
# -------------------------------------------------------------------------------------------------
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")
set (CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Client target
# -------------------------------------------------------------------------------------------------
add_executable(GGM ${GGM_SOURCES} ${GGM_INCLUDES})

target_link_libraries(GGM)

set_property(TARGET GGM PROPERTY FOLDER "ggm")
setup_target_paths(GGM ${GGM_ROOT_DIR}/Bin)

# Build config
# -------------------------------------------------------------------------------------------------
# Include our own source folder + shared includes
target_include_directories(GGM
	PRIVATE ${GGM_ROOT_DIR}/Source
	PRIVATE ${GGM_ROOT_DIR}/Include)

# Require C++14
set_target_properties(GGM PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON)

if(WIN32)
	target_compile_options(GGM
		PRIVATE "/MP"
		PRIVATE "/W4"
		PRIVATE "/arch:AVX")
endif()

if(MINGW OR CMAKE_COMPILER_IS_GNUCXX OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
	target_compile_options(GGM
		PRIVATE "-Wall"
		PRIVATE "-Wextra"
		PRIVATE "-mavx"
		PRIVATE "-Wno-format-zero-length")

	# Debug infos in debug
	target_compile_options(GGM PRIVATE $<$<CONFIG:Debug>:-g>)

	target_link_libraries(GGM PUBLIC m stdc++)
endif()
