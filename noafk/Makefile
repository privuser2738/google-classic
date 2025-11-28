# NoAFK Makefile for MinGW/MSYS2
# Usage: make [debug|release|clean]

CXX = g++
WINDRES = windres

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = $(BUILD_DIR)/bin

# Source files
SOURCES = $(SRC_DIR)/main.cpp \
          $(SRC_DIR)/settings.cpp \
          $(SRC_DIR)/antiafk.cpp

HEADERS = $(SRC_DIR)/settings.h \
          $(SRC_DIR)/antiafk.h \
          $(SRC_DIR)/resource.h

RESOURCES = $(SRC_DIR)/resource.rc

# Object files
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
RES_OBJ = $(BUILD_DIR)/resource.o

# Output
TARGET = $(BIN_DIR)/NoAFK.exe

# Compiler flags
CXXFLAGS = -std=c++17 -Wall -Wextra -DUNICODE -D_UNICODE -I$(SRC_DIR)
LDFLAGS = -mwindows -static
LIBS = -lcomctl32 -lshell32 -luser32 -ladvapi32 -lpthread

# Debug/Release flags
ifeq ($(DEBUG),1)
    CXXFLAGS += -g -O0 -DDEBUG
else
    CXXFLAGS += -O2 -DNDEBUG
endif

# Default target
.PHONY: all
all: release

# Release build
.PHONY: release
release: CXXFLAGS += -O2 -DNDEBUG
release: $(TARGET)
	@echo "Release build complete: $(TARGET)"

# Debug build
.PHONY: debug
debug: CXXFLAGS += -g -O0 -DDEBUG
debug: $(TARGET)
	@echo "Debug build complete: $(TARGET)"

# Create directories
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(BUILD_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile resources
$(RES_OBJ): $(RESOURCES) $(SRC_DIR)/resource.h | $(BUILD_DIR)
	@echo "Compiling resources..."
	cd $(SRC_DIR) && $(WINDRES) -I. resource.rc -o ../$(RES_OBJ)

# Link executable
$(TARGET): $(OBJECTS) $(RES_OBJ) | $(BIN_DIR)
	@echo "Linking $(TARGET)..."
	$(CXX) $(LDFLAGS) -o $@ $(OBJECTS) $(RES_OBJ) $(LIBS)

# Clean build files
.PHONY: clean
clean:
	@echo "Cleaning build files..."
	rm -rf $(BUILD_DIR)

# Install (copy to user's local bin)
.PHONY: install
install: release
	@echo "Installing to ~/bin..."
	@mkdir -p ~/bin
	cp $(TARGET) ~/bin/

# Run the application
.PHONY: run
run: release
	@echo "Starting NoAFK..."
	$(TARGET) &

# Help
.PHONY: help
help:
	@echo "NoAFK Build System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build release version (default)"
	@echo "  release   - Build optimized release version"
	@echo "  debug     - Build debug version with symbols"
	@echo "  clean     - Remove all build files"
	@echo "  install   - Install to ~/bin"
	@echo "  run       - Build and run the application"
	@echo "  help      - Show this help message"
