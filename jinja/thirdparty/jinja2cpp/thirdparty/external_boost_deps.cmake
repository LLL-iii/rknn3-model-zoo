if (JINJA2CPP_VERBOSE)
	set (FIND_BOOST_PACKAGE_QUIET)
else ()
	set (FIND_BOOST_PACKAGE_QUIET QUIET)
endif ()

if (MSVC)
	if (NOT DEFINED Boost_USE_STATIC_LIBS)
		if (THIRDPARTY_RUNTIME_TYPE STREQUAL "/MD" OR THIRDPARTY_RUNTIME_TYPE STREQUAL "/MDd")
			set (Boost_USE_STATIC_LIBS OFF)
			set (Boost_USE_STATIC_RUNTIME OFF)
		else ()
			set (Boost_USE_STATIC_LIBS ON)
			set (Boost_USE_STATIC_RUNTIME ON)
		endif ()
		if (JINJA2CPP_VERBOSE)
			message (STATUS ">>>DEBUG<<< Boost_USE_STATIC_RUNTIME = ${Boost_USE_STATIC_RUNTIME}")
		endif ()
	endif ()
endif ()

find_package(boost_algorithm          ${FIND_BOOST_PACKAGE_QUIET})
find_package(boost_any                ${FIND_BOOST_PACKAGE_QUIET})
find_package(boost_assert             ${FIND_BOOST_PACKAGE_QUIET})
find_package(boost_atomic             ${FIND_BOOST_PACKAGE_QUIET})
find_package(boost_filesystem         ${FIND_BOOST_PACKAGE_QUIET})
find_package(boost_numeric_conversion ${FIND_BOOST_PACKAGE_QUIET})
find_package(boost_json               ${FIND_BOOST_PACKAGE_QUIET})
find_package(boost_optional           ${FIND_BOOST_PACKAGE_QUIET})
find_package(boost_variant            ${FIND_BOOST_PACKAGE_QUIET})
find_package(boost_regex              ${FIND_BOOST_PACKAGE_QUIET})
find_package(boost_lexical_cast       ${FIND_BOOST_PACKAGE_QUIET})

if (boost_algorithm_FOUND AND
   boost_any_FOUND AND
   boost_filesystem_FOUND AND
   boost_numeric_conversion_FOUND AND
   boost_json_FOUND AND
   boost_optional_FOUND AND
   boost_variant_FOUND AND boost_regex_FOUND)
   imported_target_alias(boost_algorithm          ALIAS boost_algorithm::boost_algorithm)
   imported_target_alias(boost_any                ALIAS boost_any::boost_any)
   imported_target_alias(boost_filesystem         ALIAS boost_filesystem::boost_filesystem)
   imported_target_alias(boost_numeric_conversion ALIAS numeric_conversion::numeric_conversion)
   imported_target_alias(boost_json               ALIAS boost_json::boost_json)
   imported_target_alias(boost_optional           ALIAS boost_optional::boost_optional)
   imported_target_alias(boost_variant            ALIAS boost_variant::boost_variant)
   imported_target_alias(boost_regex              ALIAS boost_regex::boost_regex)
   imported_target_alias(boost_lexical_cast       ALIAS boost_regex::lexical_cast)

else ()
    # Local patch (RKNN3): boost < 1.70 has no per-library cmake packages and
    # CMake >= 4.x FindBoost mis-handles header-only components. Fall back to
    # a plain version check and create the Boost component targets manually.
    find_package(Boost 1.65 ${FIND_BOOST_PACKAGE_QUIET})
    if (NOT Boost_FOUND)
        message(FATAL_ERROR "Boost >= 1.65 not found (Boost_INCLUDE_DIR=${Boost_INCLUDE_DIR})")
    endif()

    foreach(_boost_hdr_lib algorithm any lexical_cast numeric_conversion optional variant)
        if(NOT TARGET Boost::${_boost_hdr_lib})
            add_library(Boost::${_boost_hdr_lib} INTERFACE IMPORTED)
            set_target_properties(Boost::${_boost_hdr_lib} PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIR}")
        endif()
    endforeach()

    find_library(BOOST_FILESYSTEM_LIB NAMES boost_filesystem HINTS ${Boost_LIBRARY_DIR} PATHS /lib /usr/lib)
    find_library(BOOST_REGEX_LIB        NAMES boost_regex        HINTS ${Boost_LIBRARY_DIR} PATHS /lib /usr/lib)
    foreach(_boost_comp filesystem regex)
        if(NOT TARGET Boost::${_boost_comp})
            add_library(Boost::${_boost_comp} INTERFACE IMPORTED)
            set_target_properties(Boost::${_boost_comp} PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIR}"
                IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                INTERFACE_LINK_LIBRARIES "${BOOST_FILESYSTEM_LIB}" )
        endif()
    endforeach()
    if(NOT TARGET Boost::regex)
        set_target_properties(Boost::regex PROPERTIES
            INTERFACE_LINK_LIBRARIES "${BOOST_REGEX_LIB}")
    endif()

    imported_target_alias(boost_algorithm          ALIAS Boost::algorithm)
    imported_target_alias(boost_any                ALIAS Boost::any)
    imported_target_alias(boost_filesystem         ALIAS Boost::filesystem)
    imported_target_alias(boost_numeric_conversion ALIAS Boost::numeric_conversion)
    imported_target_alias(boost_optional           ALIAS Boost::optional)
    imported_target_alias(boost_variant            ALIAS Boost::variant)
    imported_target_alias(boost_regex              ALIAS Boost::regex)
    imported_target_alias(boost_lexical_cast       ALIAS Boost::lexical_cast)
endif ()

set(_additional_boost_install_targets)
if ("${JINJA2CPP_USE_REGEX}" STREQUAL "boost")
    set(_additional_boost_install_targets "boost_regex")
endif()

if(JINJA2CPP_INSTALL)
    install(TARGETS boost_algorithm boost_any boost_filesystem boost_numeric_conversion boost_json boost_optional boost_variant ${_additional_boost_install_targets}
            EXPORT InstallTargets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/static
            PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/boost
            )
endif()
