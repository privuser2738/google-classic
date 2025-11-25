# Holo Roadmap

## Vision
Holo is an assistive AI REPL that provides **better-than-human** responses through:
- Structured, scannable output formatting
- Context-aware intelligence
- Multi-modal capabilities (text, code, files, web)
- Local-first with optional cloud AI backends

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      HOLO REPL                               │
├─────────────────────────────────────────────────────────────┤
│  Input Layer        │  Processing Layer  │  Output Layer    │
│  ─────────────────  │  ────────────────  │  ──────────────  │
│  • Line editor      │  • Command router  │  • Formatter     │
│  • Tab completion   │  • AI dispatcher   │  • Syntax HL     │
│  • History/search   │  • Context mgr     │  • Pagination    │
│  • Multi-line       │  • Tool executor   │  • Streaming     │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ Local AI │   │ Cloud AI │   │  Tools   │
        │ (llama)  │   │ (Claude) │   │ (shell)  │
        └──────────┘   └──────────┘   └──────────┘
```

---

## Phase 1: Enhanced REPL Foundation (v0.2)

### 1.1 Rich Output Formatting
- [ ] ANSI color support (already started)
- [ ] Box drawing for structured output
- [ ] Markdown rendering in terminal
- [ ] Syntax highlighting for code blocks
- [ ] Progress indicators and spinners

### 1.2 Better Input Handling
- [ ] Arrow key navigation (up/down history, left/right cursor)
- [ ] Tab completion for commands and paths
- [ ] Multi-line input mode (for code/prompts)
- [ ] Ctrl+R reverse history search
- [ ] Line editing (backspace, delete, home, end)

### 1.3 Output Formatting System
```c
// Example API
holo_print_header("Results");
holo_print_table(headers, rows, cols);
holo_print_code("python", code_string);
holo_print_list(items, count);
holo_print_box("Important note here");
```

---

## Phase 2: AI Integration (v0.3)

### 2.1 AI Backend Abstraction
```c
typedef struct {
    const char *name;           // "claude", "ollama", "openai"
    holo_result_t (*init)(void *config);
    holo_result_t (*chat)(const char *prompt, char **response);
    holo_result_t (*stream)(const char *prompt, stream_callback cb);
    void (*cleanup)(void);
} holo_ai_backend_t;
```

### 2.2 Supported Backends
- [ ] **Claude API** - Primary cloud backend
- [ ] **Ollama** - Local LLM (llama, mistral, etc.)
- [ ] **OpenAI-compatible** - Any OpenAI API endpoint
- [ ] **Local inference** - llama.cpp integration

### 2.3 Conversation Management
- [ ] Chat history persistence
- [ ] Context window management
- [ ] System prompts / personas
- [ ] Multi-turn conversations

---

## Phase 3: Human-Readable Responses (v0.4)

### 3.1 Response Formatting Rules
1. **Headers first** - Start with a clear summary
2. **Structured data** - Tables, lists, trees for data
3. **Code in blocks** - Syntax highlighted, copy-ready
4. **Progressive disclosure** - Summary → Details → Deep dive
5. **Actionable items** - Clear next steps when relevant

### 3.2 Smart Output Modes
```
holo> explain malloc

┌─ Summary ────────────────────────────────────────────┐
│ malloc() allocates memory on the heap and returns   │
│ a pointer to it. Returns NULL on failure.           │
└──────────────────────────────────────────────────────┘

┌─ Signature ──────────────────────────────────────────┐
│ void *malloc(size_t size);                          │
└──────────────────────────────────────────────────────┘

┌─ Example ────────────────────────────────────────────┐
│ int *arr = malloc(10 * sizeof(int));                │
│ if (arr == NULL) {                                  │
│     // handle error                                 │
│ }                                                   │
│ // use arr...                                       │
│ free(arr);                                          │
└──────────────────────────────────────────────────────┘

💡 Remember: Always check for NULL and call free()
```

### 3.3 Context-Aware Formatting
- Detect terminal width and adapt
- Collapse long outputs with "show more"
- Offer to save large outputs to file
- Stream long responses progressively

---

## Phase 4: Tool Integration (v0.5)

### 4.1 Built-in Tools
- [ ] **File operations** - read, write, search, diff
- [ ] **Shell execution** - run commands, capture output
- [ ] **Web fetch** - HTTP requests, scraping
- [ ] **Code analysis** - syntax check, format, lint

### 4.2 Subproject Integration
- [ ] List and manage Holo subprojects
- [ ] Run builds for subprojects
- [ ] Search across all codebases
- [ ] Git operations across repos

### 4.3 Tool Definition Format
```c
typedef struct {
    const char *name;
    const char *description;
    const char *parameters_json;  // JSON schema
    holo_result_t (*execute)(const char *params_json, char **result);
} holo_tool_t;
```

---

## Phase 5: Advanced Features (v0.6+)

### 5.1 Agents & Automation
- [ ] Multi-step task execution
- [ ] Background task queue
- [ ] Scheduled tasks
- [ ] Workflow definitions

### 5.2 Memory & Learning
- [ ] Persistent knowledge base
- [ ] User preference learning
- [ ] Project-specific context
- [ ] Semantic search over history

### 5.3 Multi-Modal
- [ ] Image input (screenshots, diagrams)
- [ ] File attachments
- [ ] Audio transcription (whisper)

---

## File Structure (Target)

```
holo/
├── src/
│   ├── main.c              # Entry point
│   ├── holo.c              # Core REPL
│   ├── commands.c          # Built-in commands
│   ├── input/
│   │   ├── readline.c      # Line editing
│   │   ├── completion.c    # Tab completion
│   │   └── history.c       # History management
│   ├── output/
│   │   ├── format.c        # Output formatting
│   │   ├── color.c         # ANSI colors
│   │   ├── markdown.c      # MD rendering
│   │   └── syntax.c        # Syntax highlighting
│   ├── ai/
│   │   ├── backend.c       # Backend abstraction
│   │   ├── claude.c        # Claude API
│   │   ├── ollama.c        # Ollama integration
│   │   └── context.c       # Conversation context
│   ├── tools/
│   │   ├── shell.c         # Shell execution
│   │   ├── files.c         # File operations
│   │   └── web.c           # HTTP/web tools
│   └── util/
│       ├── json.c          # JSON parsing
│       ├── http.c          # HTTP client
│       └── string.c        # String utilities
├── include/
│   └── holo/
│       ├── holo.h          # Main header
│       ├── ai.h            # AI interfaces
│       ├── format.h        # Formatting
│       └── tools.h         # Tool definitions
├── lib/                    # Third-party libs
├── config/                 # Default configs
└── [subprojects]/          # Your existing projects
```

---

## Next Immediate Steps

1. **Output formatting module** - Create `src/output/format.c` with box drawing, tables, colors
2. **Better line editing** - Implement arrow keys and basic editing on Windows
3. **Config system** - Load settings from `~/.holo/config`
4. **First AI integration** - Add Claude API backend (HTTP client needed)

---

## Dependencies to Add

| Library | Purpose | Notes |
|---------|---------|-------|
| cJSON | JSON parsing | Lightweight, single file |
| libcurl | HTTP client | For API calls |
| libuv | Async I/O | Optional, for streaming |

Or keep it minimal and implement HTTP with raw sockets + WinHTTP/libcurl.
