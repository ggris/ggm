# ---------------------------------------- #
#                                          #
#             Helper functions             #
#                                          #
# ---------------------------------------- #

# Source files helpers
# -------------------------------------------------------------------------------------------------
function(globrecurse_c_cpp_sources result folder)
	file(GLOB_RECURSE res
		"${folder}/*.c"
		"${folder}/*.cpp"
		"${folder}/*.cxx"
		"${folder}/*.h"
		"${folder}/*.hpp"
		"${folder}/*.hxx"
		"${folder}/*.inl")
	set(${result} ${res} PARENT_SCOPE)
endfunction()

function(group_sources RelDir)
	get_filename_component(RelDir ${RelDir}/void REALPATH)
	get_filename_component(RelDir ${RelDir} PATH)
	set(prefix "")
	set(tokenIsPrefix 0)
	foreach(src ${ARGN})
		if(${tokenIsPrefix})
			set(prefix "${src}\\")
			set(tokenIsPrefix 0)
		else()
			if(${src} STREQUAL "PREFIX")
				set(tokenIsPrefix 1)
			else()
				set(grname ${src})
				get_filename_component(grname ${grname} REALPATH)
				get_filename_component(grname ${grname} PATH)
				string(REPLACE "${RelDir}/" "" grname "${grname}")
				string(REPLACE "${RelDir}" "" grname "${grname}")
				string(REPLACE "/" "\\\\" grname "${grname}")
				source_group("${prefix}${grname}" FILES ${src})
			endif()
		endif()
	endforeach()
endfunction()

function(setup_target_paths Target LibDir)
	set_target_properties(${Target} PROPERTIES
		OUTPUT_NAME ${Target}
		DEBUG_POSTFIX _d
		RELWITHDEBINFO_POSTFIX _rd
		MINSIZEREL_POSTFIX _rm
		RUNTIME_OUTPUT_DIRECTORY ${LibDir}
		LIBRARY_OUTPUT_DIRECTORY ${LibDir}
		ARCHIVE_OUTPUT_DIRECTORY ${LibDir}
		PDB_OUTPUT_DIRECTORY ${LibDir}
		RUNTIME_OUTPUT_DIRECTORY_DEBUG ${LibDir}
		LIBRARY_OUTPUT_DIRECTORY_DEBUG ${LibDir}
		ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${LibDir}
		PDB_OUTPUT_DIRECTORY_DEBUG ${LibDir}
		RUNTIME_OUTPUT_DIRECTORY_RELEASE ${LibDir}
		LIBRARY_OUTPUT_DIRECTORY_RELEASE ${LibDir}
		ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${LibDir}
		PDB_OUTPUT_DIRECTORY_RELEASE ${LibDir}
		RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${LibDir}
		LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${LibDir}
		ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${LibDir}
		PDB_OUTPUT_DIRECTORY_RELWITHDEBINFO ${LibDir}
		RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${LibDir}
		LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${LibDir}
		ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${LibDir}
		PDB_OUTPUT_DIRECTORY_MINSIZEREL ${LibDir}
	)
endfunction()