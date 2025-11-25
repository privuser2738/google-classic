cmake_minimum_required(VERSION 3.16)
project(Holo VERSION 0.1.0 LANGUAGES C CXX)

# C/C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Platform-specific settings
if(WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    add_definitions(-DNOMINMAX)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Holo core library
add_library(holo_core STATIC
    src/holo.c
    src/commands.c
)

# Holo executable
add_executable(holo
    src/main.c
)

target_link_libraries(holo PRIVATE holo_core)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux: link readline if available
    find_library(READLINE_LIBRARY readline)
    if(READLINE_LIBRARY)
        target_link_libraries(holo PRIVATE ${READLINE_LIBRARY})
        target_compile_definitions(holo_core PRIVATE HAVE_READLINE)
    endif()
endif()

# Installation
install(TARGETS holo RUNTIME DESTINATION bin)
install(FILES include/holo.h DESTINATION include)

# Summary
message(STATUS "")
message(STATUS "Holo v${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
