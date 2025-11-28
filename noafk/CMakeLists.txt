cmake_minimum_required(VERSION 3.16)
project(NoAFK VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Windows-specific settings
if(WIN32)
    # Use Unicode
    add_definitions(-DUNICODE -D_UNICODE)

    # Hide console window for GUI app
    set(CMAKE_WIN32_EXECUTABLE ON)
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/settings.cpp
    src/antiafk.cpp
)

set(HEADERS
    src/settings.h
    src/antiafk.h
    src/resource.h
)

# Resource file (Windows)
set(RESOURCES
    src/resource.rc
)

# Create executable
add_executable(${PROJECT_NAME} WIN32
    ${SOURCES}
    ${HEADERS}
    ${RESOURCES}
)

# Link Windows libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    comctl32
    shell32
    user32
    advapi32
)

# Set output directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Install target
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

# CPack configuration for creating installer
set(CPACK_PACKAGE_NAME "NoAFK")
set(CPACK_PACKAGE_VENDOR "NoAFK")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "World of Warcraft Anti-AFK Tool")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

include(CPack)
