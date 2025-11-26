@echo off
setlocal

:: Holo CUDA Build Script for Windows
:: Usage: build-cuda.bat [clean] [debug]

set "VERSION=0.3.0"
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
echo   Holo CUDA Build System v%VERSION%
echo  ========================================
echo.

:: Check for CUDA
set "CUDA_PATH="
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\nvcc.exe" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\nvcc.exe" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3"
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\nvcc.exe" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
)

if "%CUDA_PATH%"=="" (
    echo [ERROR] CUDA Toolkit not found!
    echo         Please install CUDA from https://developer.nvidia.com/cuda-downloads
    echo.
    echo         Checked locations:
    echo           - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x
    echo           - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
    exit /b 1
)

echo [CUDA] Found: %CUDA_PATH%

:: Check for Visual Studio
set "VS_PATH="
for /f "tokens=*" %%i in ('"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath 2^>nul') do set "VS_PATH=%%i"

if "%VS_PATH%"=="" (
    echo [ERROR] Visual Studio not found!
    echo         Please install Visual Studio with C++ development tools
    exit /b 1
)
echo [VS] Found: %VS_PATH%

:: Set up Visual Studio environment
call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to set up VS environment
    exit /b 1
)
echo [VS] Environment configured

:: MinGW not needed - using MSVC for all C code

:: Clean
if "%CLEAN%"=="1" (
    echo [CLEAN] Removing old build...
    if exist "%~dp0dist" rd /s /q "%~dp0dist"
    if exist "%~dp0build" rd /s /q "%~dp0build"
    del /f /q "%~dp0*.exe" 2>nul
    del /f /q "%~dp0*.obj" 2>nul
    echo [CLEAN] Done.
    echo.
)

:: Create directories
if not exist "%~dp0dist\bin" mkdir "%~dp0dist\bin"
if not exist "%~dp0build" mkdir "%~dp0build"

:: Set paths
set "PATH=%CUDA_PATH%\bin;C:\msys64\mingw64\bin;%PATH%"

:: Set build flags
if "%DEBUG%"=="1" (
    echo [BUILD] Mode: Debug
    set "NVCC_FLAGS=-g -G -O0 -DDEBUG -allow-unsupported-compiler"
    set "CL_FLAGS=/nologo /W3 /Od /Zi /DDEBUG /DHOLO_USE_CUDA /D_CRT_SECURE_NO_WARNINGS"
) else (
    echo [BUILD] Mode: Release
    set "NVCC_FLAGS=-O3 -DNDEBUG -allow-unsupported-compiler"
    set "CL_FLAGS=/nologo /W3 /O2 /DNDEBUG /DHOLO_USE_CUDA /D_CRT_SECURE_NO_WARNINGS"
)

:: Detect GPU compute capability (default to common ones)
:: RTX 4090/4080/4070 = sm_89, RTX 3090/3080/3070 = sm_86
:: RTX 2080/2070 = sm_75, GTX 1080/1070 = sm_61
set "GPU_ARCH=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89"

echo [BUILD] Compiling CUDA kernels...

:: Compile CUDA code to object file
"%CUDA_PATH%\bin\nvcc.exe" %NVCC_FLAGS% %GPU_ARCH% ^
    -I"%~dp0include" ^
    -c "%~dp0src\ai\cuda_ops.cu" ^
    -o "%~dp0build\cuda_ops.obj" 2>&1

if errorlevel 1 (
    echo.
    echo [ERROR] CUDA compilation failed!
    exit /b 1
)

echo [BUILD] Compiling C code with MSVC...

:: Compile C files with cl.exe
cl %CL_FLAGS% /I"%~dp0include" /I"%~dp0src" /I"%CUDA_PATH%\include" /c "%~dp0src\main.c" /Fo"%~dp0build\main.obj"
if errorlevel 1 goto :compile_error

cl %CL_FLAGS% /I"%~dp0include" /I"%~dp0src" /c "%~dp0src\holo.c" /Fo"%~dp0build\holo.obj"
if errorlevel 1 goto :compile_error

cl %CL_FLAGS% /I"%~dp0include" /I"%~dp0src" /c "%~dp0src\commands.c" /Fo"%~dp0build\commands.obj"
if errorlevel 1 goto :compile_error

cl %CL_FLAGS% /I"%~dp0include" /I"%~dp0src" /c "%~dp0src\ai\gguf.c" /Fo"%~dp0build\gguf.obj"
if errorlevel 1 goto :compile_error

cl %CL_FLAGS% /I"%~dp0include" /I"%~dp0src" /c "%~dp0src\ai\tensor.c" /Fo"%~dp0build\tensor.obj"
if errorlevel 1 goto :compile_error

cl %CL_FLAGS% /I"%~dp0include" /I"%~dp0src" /c "%~dp0src\ai\quant.c" /Fo"%~dp0build\quant.obj"
if errorlevel 1 goto :compile_error

cl %CL_FLAGS% /I"%~dp0include" /I"%~dp0src" /I"%CUDA_PATH%\include" /c "%~dp0src\ai\llm.c" /Fo"%~dp0build\llm.obj"
if errorlevel 1 goto :compile_error

cl %CL_FLAGS% /I"%~dp0include" /I"%~dp0src" /I"%CUDA_PATH%\include" /c "%~dp0src\ai\llm_cuda.c" /Fo"%~dp0build\llm_cuda.obj"
if errorlevel 1 goto :compile_error

echo [BUILD] Linking...

:: Link everything together
:: Use nvcc for final link to handle CUDA runtime
:: Note: -lm not needed on Windows, math is in MSVCRT
"%CUDA_PATH%\bin\nvcc.exe" -allow-unsupported-compiler ^
    "%~dp0build\main.obj" ^
    "%~dp0build\holo.obj" ^
    "%~dp0build\commands.obj" ^
    "%~dp0build\gguf.obj" ^
    "%~dp0build\tensor.obj" ^
    "%~dp0build\quant.obj" ^
    "%~dp0build\llm.obj" ^
    "%~dp0build\llm_cuda.obj" ^
    "%~dp0build\cuda_ops.obj" ^
    -o "%~dp0dist\bin\holo.exe" ^
    -lcudart 2>&1

if errorlevel 1 (
    echo.
    echo [ERROR] Linking failed!
    exit /b 1
)

if not exist "%~dp0dist\bin\holo.exe" (
    echo.
    echo [ERROR] holo.exe was not created!
    exit /b 1
)

:: Copy CUDA runtime DLLs
echo [DIST] Copying CUDA runtime...
copy /y "%CUDA_PATH%\bin\cudart64_*.dll" "%~dp0dist\bin\" >nul 2>&1

:: Copy headers
echo [DIST] Copying headers...
if not exist "%~dp0dist\include\holo" mkdir "%~dp0dist\include\holo"
copy /y "%~dp0include\holo.h" "%~dp0dist\include\" >nul 2>&1
copy /y "%~dp0include\holo\*.h" "%~dp0dist\include\holo\" >nul 2>&1
copy /y "%~dp0include\cuda_ops.h" "%~dp0dist\include\" >nul 2>&1

:: Version file
> "%~dp0dist\VERSION.txt" (
    echo Holo v%VERSION% (CUDA)
    echo Build: %DATE% %TIME%
    echo Platform: Windows x64
    echo CUDA: %CUDA_PATH%
)

:: Get file size
for %%A in ("%~dp0dist\bin\holo.exe") do set "FSIZE=%%~zA"
set /a "KB=%FSIZE%/1024"

echo.
echo  ========================================
echo   CUDA Build Successful!
echo  ========================================
echo.
echo   Output:  dist\bin\holo.exe
echo   Size:    %KB% KB
echo   GPU:     CUDA enabled
echo.
echo   Run:     dist\bin\holo.exe
echo.

exit /b 0

:compile_error
echo.
echo [ERROR] C compilation failed!
exit /b 1

:help
echo.
echo  Holo CUDA Build Script for Windows
echo.
echo  Usage: build-cuda.bat [clean] [debug]
echo.
echo  Options:
echo    clean   Remove old builds first
echo    debug   Build with debug symbols
echo.
echo  Requirements:
echo    - CUDA Toolkit 11.8+ (https://developer.nvidia.com/cuda-downloads)
echo    - MinGW-w64 GCC (via MSYS2)
echo    - NVIDIA GPU with compute capability 7.5+
echo.
echo  Examples:
echo    build-cuda.bat           Release build with CUDA
echo    build-cuda.bat clean     Clean + release build
echo    build-cuda.bat debug     Debug build with CUDA
echo.
exit /b 0
