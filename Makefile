# Holo - Assistive AI with REPL Interface
# Simple Makefile for quick builds (use CMake for full builds)

CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -Iinclude
LDFLAGS =

# Platform detection
ifeq ($(OS),Windows_NT)
    TARGET = holo.exe
    RM = del /Q
    MKDIR = mkdir
else
    TARGET = holo
    RM = rm -f
    MKDIR = mkdir -p
    LDFLAGS += -lreadline
endif

# Source files
SRCS = src/main.c src/holo.c src/commands.c
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJS) $(TARGET)

# For development
debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

release: CFLAGS += -O2 -DNDEBUG
release: clean $(TARGET)

.PHONY: all clean debug release
