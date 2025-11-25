# Holo Architecture

## Enterprise-Grade, Cross-Platform AI Assistant
**Pure C/C++ | Your Own API | No External AI Dependencies**

---

## Target Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| Windows (x64) | Primary | Win10/11, MSVC + MinGW |
| Linux (x64/ARM) | Primary | glibc 2.17+, musl |
| macOS (x64/ARM) | Primary | 10.15+, Universal Binary |
| iOS | Secondary | 14.0+, Framework |
| Android | Secondary | API 24+, NDK, JNI wrapper |

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HOLO CORE                                       │
│                         (Pure C, Platform Agnostic)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   REPL      │  │   API       │  │   AI        │  │   Tools     │        │
│  │   Engine    │  │   Server    │  │   Engine    │  │   Runtime   │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                │
│         └────────────────┴────────────────┴────────────────┘                │
│                                   │                                         │
│                     ┌─────────────┴─────────────┐                           │
│                     │    Platform Abstraction   │                           │
│                     │         Layer (PAL)       │                           │
│                     └─────────────┬─────────────┘                           │
│                                   │                                         │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                    │
        ┌───────────────┬───────────┼───────────┬───────────────┐
        │               │           │           │               │
   ┌────┴────┐    ┌────┴────┐ ┌────┴────┐ ┌────┴────┐    ┌────┴────┐
   │ Windows │    │  Linux  │ │  macOS  │ │   iOS   │    │ Android │
   │  PAL    │    │   PAL   │ │   PAL   │ │   PAL   │    │   PAL   │
   └─────────┘    └─────────┘ └─────────┘ └─────────┘    └─────────┘
```

---

## Directory Structure

```
holo/
├── src/
│   ├── core/                   # Platform-agnostic core
│   │   ├── holo.c              # Main initialization
│   │   ├── repl.c              # REPL engine
│   │   ├── commands.c          # Command system
│   │   ├── config.c            # Configuration management
│   │   └── memory.c            # Memory management
│   │
│   ├── ai/                     # AI/Inference engine
│   │   ├── engine.c            # AI engine abstraction
│   │   ├── tokenizer.c         # Text tokenization
│   │   ├── inference.c         # Model inference
│   │   ├── context.c           # Conversation context
│   │   └── models/             # Model loading/management
│   │
│   ├── api/                    # Holo API server
│   │   ├── server.c            # HTTP/WebSocket server
│   │   ├── router.c            # Request routing
│   │   ├── handlers.c          # API endpoint handlers
│   │   ├── auth.c              # Authentication
│   │   └── protocol.c          # Wire protocol
│   │
│   ├── net/                    # Networking (pure C)
│   │   ├── tcp.c               # TCP sockets
│   │   ├── http.c              # HTTP 1.1 parser/builder
│   │   ├── websocket.c         # WebSocket protocol
│   │   ├── tls.c               # TLS (via platform or mbedtls)
│   │   └── dns.c               # DNS resolution
│   │
│   ├── output/                 # Output formatting
│   │   ├── format.c            # Text formatting
│   │   ├── json.c              # JSON encode/decode
│   │   └── markdown.c          # Markdown rendering
│   │
│   ├── tools/                  # Built-in tools
│   │   ├── shell.c             # Shell execution
│   │   ├── files.c             # File operations
│   │   ├── search.c            # Code/text search
│   │   └── git.c               # Git operations
│   │
│   ├── pal/                    # Platform Abstraction Layer
│   │   ├── pal.h               # Common interface
│   │   ├── pal_windows.c       # Windows implementation
│   │   ├── pal_linux.c         # Linux implementation
│   │   ├── pal_macos.c         # macOS implementation
│   │   ├── pal_ios.m           # iOS implementation (Obj-C)
│   │   └── pal_android.c       # Android implementation
│   │
│   └── main.c                  # Entry point (CLI)
│
├── include/
│   └── holo/
│       ├── holo.h              # Main public header
│       ├── ai.h                # AI engine interface
│       ├── api.h               # API server interface
│       ├── net.h               # Networking interface
│       ├── pal.h               # Platform abstraction
│       ├── format.h            # Output formatting
│       └── tools.h             # Tools interface
│
├── platform/                   # Platform-specific projects
│   ├── windows/                # Windows build files
│   ├── linux/                  # Linux build files
│   ├── macos/                  # Xcode project
│   ├── ios/                    # iOS framework
│   └── android/                # Android NDK/Gradle
│
├── models/                     # AI model storage
│   └── README.md               # Model documentation
│
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
│
├── CMakeLists.txt              # Cross-platform build
├── Makefile                    # Quick build
└── ARCHITECTURE.md             # This file
```

---

## Platform Abstraction Layer (PAL)

The PAL provides a unified C interface for platform-specific operations:

```c
// pal.h - Platform Abstraction Layer Interface

#ifndef HOLO_PAL_H
#define HOLO_PAL_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// Platform identification
typedef enum {
    HOLO_PLATFORM_WINDOWS,
    HOLO_PLATFORM_LINUX,
    HOLO_PLATFORM_MACOS,
    HOLO_PLATFORM_IOS,
    HOLO_PLATFORM_ANDROID
} holo_platform_t;

holo_platform_t holo_get_platform(void);
const char *holo_get_platform_name(void);

// Memory
void *holo_alloc(size_t size);
void *holo_realloc(void *ptr, size_t size);
void holo_free(void *ptr);
void *holo_alloc_aligned(size_t size, size_t alignment);

// Threading
typedef struct holo_thread holo_thread_t;
typedef struct holo_mutex holo_mutex_t;
typedef struct holo_cond holo_cond_t;
typedef void (*holo_thread_fn)(void *arg);

holo_thread_t *holo_thread_create(holo_thread_fn fn, void *arg);
void holo_thread_join(holo_thread_t *thread);
holo_mutex_t *holo_mutex_create(void);
void holo_mutex_lock(holo_mutex_t *mutex);
void holo_mutex_unlock(holo_mutex_t *mutex);
void holo_mutex_destroy(holo_mutex_t *mutex);

// File I/O
typedef struct holo_file holo_file_t;

holo_file_t *holo_file_open(const char *path, const char *mode);
size_t holo_file_read(holo_file_t *file, void *buf, size_t size);
size_t holo_file_write(holo_file_t *file, const void *buf, size_t size);
void holo_file_close(holo_file_t *file);
bool holo_file_exists(const char *path);
bool holo_dir_create(const char *path);
char *holo_get_home_dir(void);
char *holo_get_config_dir(void);

// Networking
typedef struct holo_socket holo_socket_t;

holo_socket_t *holo_socket_create(int type);  // HOLO_SOCK_TCP, HOLO_SOCK_UDP
int holo_socket_connect(holo_socket_t *sock, const char *host, uint16_t port);
int holo_socket_bind(holo_socket_t *sock, const char *host, uint16_t port);
int holo_socket_listen(holo_socket_t *sock, int backlog);
holo_socket_t *holo_socket_accept(holo_socket_t *sock);
ssize_t holo_socket_send(holo_socket_t *sock, const void *buf, size_t len);
ssize_t holo_socket_recv(holo_socket_t *sock, void *buf, size_t len);
void holo_socket_close(holo_socket_t *sock);

// Time
uint64_t holo_time_ms(void);      // Milliseconds since epoch
uint64_t holo_time_us(void);      // Microseconds (monotonic)
void holo_sleep_ms(uint32_t ms);

// Console/Terminal
int holo_term_width(void);
int holo_term_height(void);
bool holo_term_is_tty(void);
void holo_term_set_raw(bool raw);
int holo_term_read_char(void);    // Non-blocking

// Dynamic loading
typedef struct holo_lib holo_lib_t;

holo_lib_t *holo_lib_load(const char *path);
void *holo_lib_symbol(holo_lib_t *lib, const char *name);
void holo_lib_unload(holo_lib_t *lib);

// Process
int holo_exec(const char *cmd, char **output, size_t *output_len);
int holo_spawn(const char *cmd, int *pid);

#endif // HOLO_PAL_H
```

---

## Holo API Specification

### Protocol Overview

- **Transport**: HTTP/1.1 + WebSocket
- **Format**: JSON
- **Auth**: API Key / JWT tokens
- **Port**: 8420 (default)

### REST Endpoints

```
POST /v1/chat/completions     # Chat completion (streaming supported)
POST /v1/completions          # Text completion
GET  /v1/models               # List available models
GET  /v1/models/{id}          # Model details
POST /v1/embeddings           # Text embeddings
GET  /v1/health               # Health check
```

### WebSocket Endpoints

```
WS /v1/chat/stream            # Real-time chat streaming
WS /v1/tools/execute          # Tool execution with progress
```

### Request Format

```json
{
  "model": "holo-1",
  "messages": [
    {"role": "system", "content": "You are Holo, an AI assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 2048,
  "stream": true
}
```

### Response Format (Streaming)

```json
{"id":"msg_001","object":"chat.chunk","delta":{"content":"Hello"}}
{"id":"msg_001","object":"chat.chunk","delta":{"content":"!"}}
{"id":"msg_001","object":"chat.chunk","delta":{},"finish_reason":"stop"}
```

---

## AI Engine Architecture

### Design Goals
1. **No external dependencies** - Pure C/C++ inference
2. **Model agnostic** - Support multiple architectures (Transformer, etc.)
3. **Quantization** - INT4/INT8 for efficiency
4. **Hardware acceleration** - CPU SIMD, GPU optional

### Components

```c
// ai.h - AI Engine Interface

typedef struct holo_ai_engine holo_ai_engine_t;
typedef struct holo_ai_model holo_ai_model_t;
typedef struct holo_ai_context holo_ai_context_t;

// Engine lifecycle
holo_ai_engine_t *holo_ai_init(void);
void holo_ai_shutdown(holo_ai_engine_t *engine);

// Model management
holo_ai_model_t *holo_ai_load_model(holo_ai_engine_t *engine, const char *path);
void holo_ai_unload_model(holo_ai_model_t *model);

// Inference
holo_ai_context_t *holo_ai_context_create(holo_ai_model_t *model, int context_size);
int holo_ai_generate(holo_ai_context_t *ctx, const char *prompt,
                     void (*callback)(const char *token, void *user), void *user);
void holo_ai_context_destroy(holo_ai_context_t *ctx);

// Tokenization
int holo_ai_tokenize(holo_ai_model_t *model, const char *text, int *tokens, int max_tokens);
char *holo_ai_detokenize(holo_ai_model_t *model, const int *tokens, int count);
```

### Supported Model Formats
- **GGUF** - llama.cpp compatible (primary)
- **HOLO** - Custom optimized format (future)

---

## Build System

### CMake (Primary)

```cmake
cmake_minimum_required(VERSION 3.16)
project(holo C CXX)

# Options
option(HOLO_BUILD_TESTS "Build tests" ON)
option(HOLO_BUILD_API_SERVER "Build API server" ON)
option(HOLO_USE_SIMD "Enable SIMD optimizations" ON)

# Platform detection
if(WIN32)
    set(HOLO_PLATFORM "windows")
elseif(APPLE)
    if(IOS)
        set(HOLO_PLATFORM "ios")
    else()
        set(HOLO_PLATFORM "macos")
    endif()
elseif(ANDROID)
    set(HOLO_PLATFORM "android")
else()
    set(HOLO_PLATFORM "linux")
endif()
```

### Build Commands

```bash
# Windows (MSVC)
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release

# Windows (MinGW)
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Linux/macOS
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Android (NDK)
cmake -B build -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a -DCMAKE_BUILD_TYPE=Release
cmake --build build

# iOS
cmake -B build -G Xcode -DCMAKE_SYSTEM_NAME=iOS
cmake --build build --config Release
```

---

## Enterprise Features

### Security
- [ ] TLS 1.3 support (via mbedTLS or platform)
- [ ] API key authentication
- [ ] JWT token support
- [ ] Rate limiting
- [ ] Request signing

### Scalability
- [ ] Connection pooling
- [ ] Request queuing
- [ ] Load balancing (external)
- [ ] Horizontal scaling

### Monitoring
- [ ] Prometheus metrics endpoint
- [ ] Request logging
- [ ] Performance profiling
- [ ] Health checks

### Deployment
- [ ] Static binary builds
- [ ] Docker support
- [ ] Kubernetes manifests
- [ ] systemd service files

---

## Development Phases

### Phase 1: Foundation (Current)
- [x] REPL core
- [x] Output formatting
- [ ] Platform abstraction layer
- [ ] Cross-platform build system

### Phase 2: Networking
- [ ] TCP/HTTP server
- [ ] WebSocket support
- [ ] TLS integration
- [ ] API endpoints

### Phase 3: AI Engine
- [ ] Model loading (GGUF)
- [ ] Tokenization
- [ ] Inference engine
- [ ] Context management

### Phase 4: Enterprise
- [ ] Authentication
- [ ] Rate limiting
- [ ] Monitoring
- [ ] Documentation

### Phase 5: Mobile
- [ ] iOS framework
- [ ] Android library
- [ ] Mobile-optimized inference
