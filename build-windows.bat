@echo off
setlocal

:: Holo Build Script for Windows
:: Usage: build-windows.bat [clean] [debug]

set "VERSION=0.2.0"
set "CLEAN=0"
set "DEBUG=0"

:parse
if "%~1"=="" goto :main
if /i "%~1"=="clean" set "CLEAN=1"
if /i "%~1"=="debug" set "DEBUG=1"
if /i "%~1"=="help" goto :help
if /i "%~1"=="-h" goto :help
shift
goto :parse

:main
echo.
echo  ========================================
echo   Holo Build System v%VERSION%
echo  ========================================
echo.

:: Check for MinGW GCC
if not exist "C:\msys64\mingw64\bin\gcc.exe" (
    echo [ERROR] MinGW-w64 GCC not found
    echo         Please install MSYS2 from https://www.msys2.org/
    echo         Then run: pacman -S mingw-w64-x86_64-gcc
    exit /b 1
)

:: Clean
if "%CLEAN%"=="1" (
    echo [CLEAN] Removing old build...
    if exist "%~dp0dist" rd /s /q "%~dp0dist"
    if exist "%~dp0build" rd /s /q "%~dp0build"
    del /f /q "%~dp0*.exe" 2>nul
    echo [CLEAN] Done.
    echo.
)

:: Create dist directory
if not exist "%~dp0dist\bin" mkdir "%~dp0dist\bin"

:: Set build flags
if "%DEBUG%"=="1" (
    echo [BUILD] Mode: Debug
    set "CFLAGS=-Wall -Wextra -std=c11 -g -O0 -DDEBUG"
) else (
    echo [BUILD] Mode: Release + AVX2 + OpenMP
    set "CFLAGS=-Wall -Wextra -std=c11 -O3 -ffast-math -march=native -mavx2 -mfma -fopenmp -s -DNDEBUG"
)

echo [BUILD] Compiling with MinGW-w64...

:: Set PATH to include MinGW
set "PATH=C:\msys64\mingw64\bin;%PATH%"

:: Compile directly with gcc (including AI modules)
gcc %CFLAGS% -I"%~dp0include" -I"%~dp0src" ^
    "%~dp0src\main.c" ^
    "%~dp0src\holo.c" ^
    "%~dp0src\commands.c" ^
    "%~dp0src\ai\gguf.c" ^
    "%~dp0src\ai\tensor.c" ^
    "%~dp0src\ai\quant.c" ^
    "%~dp0src\ai\llm.c" ^
    -o "%~dp0dist\bin\holo.exe" -lm 2>&1

if errorlevel 1 (
    echo.
    echo [ERROR] Compilation failed!
    exit /b 1
)

if not exist "%~dp0dist\bin\holo.exe" (
    echo.
    echo [ERROR] holo.exe was not created!
    exit /b 1
)

:: Copy headers
echo [DIST] Copying headers...
if not exist "%~dp0dist\include\holo" mkdir "%~dp0dist\include\holo"
copy /y "%~dp0include\holo.h" "%~dp0dist\include\" >nul 2>&1
copy /y "%~dp0include\holo\*.h" "%~dp0dist\include\holo\" >nul 2>&1

:: Version file
> "%~dp0dist\VERSION.txt" (
    echo Holo v%VERSION%
    echo Build: %DATE% %TIME%
    echo Platform: Windows x64
)

:: Get file size
for %%A in ("%~dp0dist\bin\holo.exe") do set "FSIZE=%%~zA"
set /a "KB=%FSIZE%/1024"

echo.
echo  ========================================
echo   Build Successful!
echo  ========================================
echo.
echo   Output:  dist\bin\holo.exe
echo   Size:    %KB% KB
echo.
echo   Run:     dist\bin\holo.exe
echo.

exit /b 0

:help
echo.
echo  Holo Build Script for Windows
echo.
echo  Usage: build-windows.bat [clean] [debug]
echo.
echo  Options:
echo    clean   Remove old builds first
echo    debug   Build with debug symbols
echo.
echo  Examples:
echo    build-windows.bat           Release build
echo    build-windows.bat clean     Clean + release build
echo    build-windows.bat debug     Debug build
echo.
exit /b 0
