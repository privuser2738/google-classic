#!/bin/bash
# NoAFK Build Script for MinGW/MSYS2
# Usage: ./build.sh [debug|release|clean]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directories
SRC_DIR="src"
BUILD_DIR="build"
BIN_DIR="$BUILD_DIR/bin"

# Build type
BUILD_TYPE="${1:-release}"

echo "========================================"
echo "NoAFK Build Script"
echo "Build type: $BUILD_TYPE"
echo "========================================"
echo

# Clean if requested
if [ "$BUILD_TYPE" = "clean" ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"
    echo -e "${GREEN}Clean complete!${NC}"
    exit 0
fi

# Create directories
mkdir -p "$BUILD_DIR"
mkdir -p "$BIN_DIR"

# Compiler flags
CXXFLAGS="-std=c++17 -Wall -Wextra -DUNICODE -D_UNICODE -I$SRC_DIR"

if [ "$BUILD_TYPE" = "debug" ]; then
    CXXFLAGS="$CXXFLAGS -g -O0 -DDEBUG"
else
    CXXFLAGS="$CXXFLAGS -O2 -DNDEBUG"
fi

LDFLAGS="-mwindows -static"
LIBS="-lcomctl32 -lshell32 -luser32 -ladvapi32 -lpthread"

# Compile resources
echo -e "${YELLOW}Compiling resources...${NC}"
cd "$SRC_DIR"
windres -I. resource.rc -o "../$BUILD_DIR/resource.o"
cd ..

# Compile source files
echo -e "${YELLOW}Compiling source files...${NC}"

echo "  main.cpp"
g++ $CXXFLAGS -c "$SRC_DIR/main.cpp" -o "$BUILD_DIR/main.o"

echo "  settings.cpp"
g++ $CXXFLAGS -c "$SRC_DIR/settings.cpp" -o "$BUILD_DIR/settings.o"

echo "  antiafk.cpp"
g++ $CXXFLAGS -c "$SRC_DIR/antiafk.cpp" -o "$BUILD_DIR/antiafk.o"

# Link
echo -e "${YELLOW}Linking executable...${NC}"
g++ $LDFLAGS -o "$BIN_DIR/NoAFK.exe" \
    "$BUILD_DIR/main.o" \
    "$BUILD_DIR/settings.o" \
    "$BUILD_DIR/antiafk.o" \
    "$BUILD_DIR/resource.o" \
    $LIBS

echo
echo "========================================"
echo -e "${GREEN}Build successful!${NC}"
echo "Executable: $BIN_DIR/NoAFK.exe"
echo "========================================"
