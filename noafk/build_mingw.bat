@echo off
REM NoAFK Build Script using MinGW from MSYS2
REM Run this from a standard command prompt

setlocal enabledelayedexpansion

REM Try to find MinGW
set MINGW_PATH=
if exist "C:\msys64\mingw64\bin\g++.exe" (
    set MINGW_PATH=C:\msys64\mingw64\bin
) else if exist "C:\mingw64\bin\g++.exe" (
    set MINGW_PATH=C:\mingw64\bin
) else if exist "C:\MinGW\bin\g++.exe" (
    set MINGW_PATH=C:\MinGW\bin
)

if "%MINGW_PATH%"=="" (
    echo ERROR: MinGW not found!
    echo Please install MSYS2 with MinGW-w64 from https://www.msys2.org/
    exit /b 1
)

set PATH=%MINGW_PATH%;%PATH%

echo ========================================
echo NoAFK Build Script [MinGW]
echo Using: %MINGW_PATH%
echo ========================================
echo.

REM Create output directories
if not exist build mkdir build
if not exist build\bin mkdir build\bin

REM Compile resources
echo [1/5] Compiling resources...
pushd src
"%MINGW_PATH%\windres.exe" -I. resource.rc -o ..\build\resource.o
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Resource compilation failed!
    popd
    exit /b 1
)
popd

REM Compile source files
echo [2/5] Compiling main.cpp...
"%MINGW_PATH%\g++.exe" -std=c++17 -Wall -O2 -DUNICODE -D_UNICODE -DNDEBUG -Isrc -c src\main.cpp -o build\main.o
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: main.cpp compilation failed!
    exit /b 1
)

echo [3/5] Compiling settings.cpp...
"%MINGW_PATH%\g++.exe" -std=c++17 -Wall -O2 -DUNICODE -D_UNICODE -DNDEBUG -Isrc -c src\settings.cpp -o build\settings.o
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: settings.cpp compilation failed!
    exit /b 1
)

echo [4/5] Compiling antiafk.cpp...
"%MINGW_PATH%\g++.exe" -std=c++17 -Wall -O2 -DUNICODE -D_UNICODE -DNDEBUG -Isrc -c src\antiafk.cpp -o build\antiafk.o
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: antiafk.cpp compilation failed!
    exit /b 1
)

REM Link executable
echo [5/5] Linking NoAFK.exe...
"%MINGW_PATH%\g++.exe" -mwindows -static -o build\bin\NoAFK.exe build\main.o build\settings.o build\antiafk.o build\resource.o -lcomctl32 -lshell32 -luser32 -ladvapi32 -lpthread
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Linking failed!
    exit /b 1
)

echo.
echo ========================================
echo BUILD SUCCESSFUL
echo Executable: build\bin\NoAFK.exe
echo ========================================
echo.
echo Run: build\bin\NoAFK.exe

endlocal
