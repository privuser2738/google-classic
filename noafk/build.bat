@echo off
REM NoAFK Build Script for Windows
REM Requires Visual Studio or MinGW with CMake

setlocal enabledelayedexpansion

echo ========================================
echo NoAFK Build Script
echo ========================================
echo.

REM Check for CMake
where cmake >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CMake not found in PATH
    echo Please install CMake from https://cmake.org/
    exit /b 1
)

REM Check for compiler
set GENERATOR=

REM Try to find Visual Studio
where cl >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Found Visual Studio compiler
    set GENERATOR="Visual Studio 17 2022"
    goto :build
)

REM Try MinGW
where g++ >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Found MinGW compiler
    set GENERATOR="MinGW Makefiles"
    goto :build
)

REM Try to use MSVC from Developer Command Prompt
if defined VSINSTALLDIR (
    echo Found Visual Studio environment
    set GENERATOR="Visual Studio 17 2022"
    goto :build
)

echo ERROR: No suitable compiler found
echo Please run from Visual Studio Developer Command Prompt
echo or install MinGW-w64
exit /b 1

:build
echo.
echo Using generator: %GENERATOR%
echo.

REM Create build directory
if not exist build mkdir build
cd build

REM Configure
echo Configuring project...
cmake -G %GENERATOR% -DCMAKE_BUILD_TYPE=Release ..
if %ERRORLEVEL% NEQ 0 (
    echo Configuration failed!
    exit /b 1
)

REM Build
echo.
echo Building...
cmake --build . --config Release
if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    exit /b 1
)

echo.
echo ========================================
echo Build successful!
echo Executable: build\bin\NoAFK.exe
echo ========================================

cd ..
