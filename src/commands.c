/*
 * Holo - Assistive AI with REPL Interface
 * Built-in commands
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "holo.h"
#include "ai/llm.h"

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define getcwd _getcwd
#define chdir _chdir
#else
#include <unistd.h>
#endif

/* Command: help */
static holo_result_t cmd_help(holo_context_t *ctx, int argc, char **argv) {
    (void)argc;
    (void)argv;

    holo_print("\nAvailable commands:\n");
    holo_print("-------------------\n");

    for (size_t i = 0; i < ctx->command_count; i++) {
        holo_print("  %-12s %s\n",
                   ctx->commands[i].name,
                   ctx->commands[i].description ? ctx->commands[i].description : "");
    }

    holo_print("\nType '<command> --help' for more information on a specific command.\n\n");
    return HOLO_OK;
}

/* Command: exit/quit */
static holo_result_t cmd_exit(holo_context_t *ctx, int argc, char **argv) {
    (void)argc;
    (void)argv;

    ctx->running = false;
    holo_print("Goodbye!\n");
    return HOLO_EXIT;
}

/* Command: version */
static holo_result_t cmd_version(holo_context_t *ctx, int argc, char **argv) {
    (void)ctx;
    (void)argc;
    (void)argv;

    holo_print("Holo v%s\n", holo_version());
    return HOLO_OK;
}

/* Command: clear */
static holo_result_t cmd_clear(holo_context_t *ctx, int argc, char **argv) {
    (void)ctx;
    (void)argc;
    (void)argv;

#ifdef _WIN32
    system("cls");
#else
    printf("\033[2J\033[H");
#endif
    return HOLO_OK;
}

/* Command: history */
static holo_result_t cmd_history(holo_context_t *ctx, int argc, char **argv) {
    (void)argv;

    if (ctx->history_count == 0) {
        holo_print("History is empty.\n");
        return HOLO_OK;
    }

    size_t start = 0;
    size_t count = ctx->history_count;

    /* Optional: show last N entries */
    if (argc > 1) {
        int n = atoi(argv[1]);
        if (n > 0 && (size_t)n < count) {
            start = count - n;
        }
    }

    for (size_t i = start; i < ctx->history_count; i++) {
        holo_print("  %4zu  %s\n", i + 1, ctx->history[i]);
    }

    return HOLO_OK;
}

/* Command: pwd */
static holo_result_t cmd_pwd(holo_context_t *ctx, int argc, char **argv) {
    (void)ctx;
    (void)argc;
    (void)argv;

    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd))) {
        holo_print("%s\n", cwd);
    } else {
        holo_print_error("Could not get current directory\n");
        return HOLO_ERROR_IO;
    }
    return HOLO_OK;
}

/* Command: cd */
static holo_result_t cmd_cd(holo_context_t *ctx, int argc, char **argv) {
    (void)ctx;

    if (argc < 2) {
        /* Go to home directory */
        const char *home = getenv("HOME");
        if (!home) home = getenv("USERPROFILE");
        if (home) {
            if (chdir(home) != 0) {
                holo_print_error("Could not change to home directory\n");
                return HOLO_ERROR_IO;
            }
        }
    } else {
        if (chdir(argv[1]) != 0) {
            holo_print_error("Could not change to directory: %s\n", argv[1]);
            return HOLO_ERROR_IO;
        }
    }

    /* Print new directory */
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd))) {
        holo_print("%s\n", cwd);
    }

    return HOLO_OK;
}

/* Command: ls (basic listing) */
static holo_result_t cmd_ls(holo_context_t *ctx, int argc, char **argv) {
    (void)ctx;

    const char *path = (argc > 1) ? argv[1] : ".";

#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    char search_path[1024];
    snprintf(search_path, sizeof(search_path), "%s\\*", path);

    HANDLE hFind = FindFirstFileA(search_path, &fd);
    if (hFind == INVALID_HANDLE_VALUE) {
        holo_print_error("Could not list directory: %s\n", path);
        return HOLO_ERROR_IO;
    }

    do {
        if (strcmp(fd.cFileName, ".") == 0 || strcmp(fd.cFileName, "..") == 0) {
            continue;
        }

        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            holo_print("  \033[34m%s/\033[0m\n", fd.cFileName);
        } else {
            holo_print("  %s\n", fd.cFileName);
        }
    } while (FindNextFileA(hFind, &fd));

    FindClose(hFind);
#else
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "ls -la '%s'", path);
    system(cmd);
#endif

    return HOLO_OK;
}

/* Command: echo */
static holo_result_t cmd_echo(holo_context_t *ctx, int argc, char **argv) {
    (void)ctx;

    for (int i = 1; i < argc; i++) {
        holo_print("%s", argv[i]);
        if (i < argc - 1) holo_print(" ");
    }
    holo_print("\n");

    return HOLO_OK;
}

/* Global LLM context */
static llm_ctx_t *g_llm = NULL;

/* Token callback for streaming output */
static int g_token_count = 0;

static bool token_callback(int token, const char *text, void *user_data) {
    (void)token;
    (void)user_data;

    if (text && *text) {
        /* Print the token text */
        printf("%s", text);
        fflush(stdout);
        g_token_count++;
    }
    return true;  /* Continue generation */
}

/* Command: chat (AI chat interface) */
static holo_result_t cmd_chat(holo_context_t *ctx, int argc, char **argv) {
    (void)ctx;

    if (argc < 2) {
        holo_print("Usage: chat <message>\n");
        holo_print("Start a conversation with Holo AI.\n");
        holo_print("\nOptions:\n");
        holo_print("  chat <message>        Send a message to the AI\n");
        holo_print("  chat --model <path>   Load a GGUF model\n");
        holo_print("  chat --unload         Unload current model\n");
        holo_print("  chat --info           Show current model info\n");
        return HOLO_OK;
    }

    /* Handle --model flag */
    if (strcmp(argv[1], "--model") == 0 || strcmp(argv[1], "-m") == 0) {
        if (argc < 3) {
            holo_print("Usage: chat --model <path/to/model.gguf>\n");
            return HOLO_OK;
        }

        /* Unload existing model */
        if (g_llm) {
            llm_free(g_llm);
            g_llm = NULL;
        }

        /* Load new model */
        holo_print("\nLoading model...\n");
        g_llm = llm_load(argv[2]);

        if (g_llm) {
            /* Try to initialize CUDA for GPU acceleration */
#ifdef HOLO_USE_CUDA
            if (llm_cuda_init(g_llm) == 0) {
                holo_print("  \033[32m[CUDA]\033[0m GPU acceleration enabled\n");
            }
#endif
            /* Extract filename */
            const char *name = strrchr(argv[2], '\\');
            if (!name) name = strrchr(argv[2], '/');
            if (name) name++; else name = argv[2];

            holo_print("\033[32m[OK]\033[0m Model loaded: %s\n", name);
            holo_print("     Ready to chat! Type: chat <your message>\n\n");
        } else {
            holo_print_error("Failed to load model: %s\n", argv[2]);
            return HOLO_ERROR_IO;
        }
        return HOLO_OK;
    }

    /* Handle --unload flag */
    if (strcmp(argv[1], "--unload") == 0) {
        if (g_llm) {
            llm_free(g_llm);
            g_llm = NULL;
            holo_print("Model unloaded.\n");
        } else {
            holo_print("No model is currently loaded.\n");
        }
        return HOLO_OK;
    }

    /* Handle --info flag */
    if (strcmp(argv[1], "--info") == 0) {
        if (g_llm) {
            llm_print_info(g_llm);
        } else {
            holo_print("\nNo model loaded.\n");
            holo_print("Use: chat --model <path/to/model.gguf>\n\n");
        }
        return HOLO_OK;
    }

    /* Build message from arguments */
    char message[HOLO_MAX_INPUT_SIZE] = {0};
    for (int i = 1; i < argc; i++) {
        if (i > 1) strcat(message, " ");
        strncat(message, argv[i], sizeof(message) - strlen(message) - 1);
    }

    /* Display user message */
    holo_print("\n\033[1;34m[You]:\033[0m %s\n\n", message);

    /* Check if model is loaded */
    if (!g_llm) {
        holo_print("\033[1;33m[Holo]:\033[0m No model loaded.\n");
        holo_print("        Load a model first: chat --model <path/to/model.gguf>\n\n");
        return HOLO_OK;
    }

    /* Generate response using chat template */
    holo_print("\033[1;32m[Holo]:\033[0m ");
    fflush(stdout);

    g_token_count = 0;
    llm_sampler_t sampler = LLM_SAMPLER_DEFAULT;
    sampler.temperature = 0.0f;  /* Greedy decoding for now */
    int generated = llm_chat(g_llm, message, 512, &sampler, token_callback, NULL);

    if (generated < 0) {
        holo_print("(generation error)\n");
    } else if (generated == 0) {
        holo_print("(no response generated)\n");
    }
    holo_print("\n\n");

    return HOLO_OK;
}

/* Command: model (AI model management) */
static holo_result_t cmd_model(holo_context_t *ctx, int argc, char **argv) {
    (void)ctx;

    if (argc < 2) {
        holo_print("\nAI Model Management\n");
        holo_print("===================\n\n");
        holo_print("Usage:\n");
        holo_print("  model load <path>     Load a GGUF model file\n");
        holo_print("  model unload          Unload current model\n");
        holo_print("  model info            Show loaded model info\n");
        holo_print("  model list            List available models\n");
        holo_print("  model backends        Show available compute backends\n");
        holo_print("\nSupported Formats:\n");
        holo_print("  GGUF (.gguf)          Recommended format\n");
        holo_print("\nSupported Architectures:\n");
        holo_print("  Llama, Llama2, Llama3, Mistral, Mixtral\n");
        holo_print("  Qwen, Qwen2, Phi, Phi2, Phi3\n");
        holo_print("  Gemma, Gemma2, StarCoder, StarCoder2\n");
        holo_print("  DeepSeek, Command-R, Falcon, MPT\n");
        holo_print("\n");
        return HOLO_OK;
    }

    const char *subcmd = argv[1];

    if (strcmp(subcmd, "backends") == 0) {
        holo_print("\nAvailable Compute Backends:\n");
        holo_print("---------------------------\n");
        holo_print("  \033[32m[x]\033[0m CPU       Always available\n");
#ifdef _WIN32
        holo_print("  [ ] CUDA      NVIDIA GPU (requires nvcuda.dll)\n");
        holo_print("  [ ] Vulkan    Cross-platform GPU (requires vulkan-1.dll)\n");
#elif defined(__APPLE__)
        holo_print("  \033[32m[x]\033[0m Metal     Apple GPU (native)\n");
#else
        holo_print("  [ ] CUDA      NVIDIA GPU (requires libcuda.so)\n");
        holo_print("  [ ] ROCm      AMD GPU (requires libamdhip64.so)\n");
        holo_print("  [ ] Vulkan    Cross-platform GPU (requires libvulkan.so)\n");
#endif
        holo_print("\n");
        return HOLO_OK;
    }

    if (strcmp(subcmd, "info") == 0) {
        if (g_llm) {
            llm_print_info(g_llm);
        } else {
            holo_print("\nModel Status: No model loaded\n");
            holo_print("Use 'model load <path>' to load a GGUF model.\n\n");
        }
        return HOLO_OK;
    }

    if (strcmp(subcmd, "unload") == 0) {
        if (g_llm) {
            llm_free(g_llm);
            g_llm = NULL;
            holo_print("Model unloaded.\n");
        } else {
            holo_print("No model is currently loaded.\n");
        }
        return HOLO_OK;
    }

    if (strcmp(subcmd, "load") == 0) {
        if (argc < 3) {
            holo_print("Usage: model load <path/to/model.gguf>\n");
            return HOLO_OK;
        }

        /* Unload existing model */
        if (g_llm) {
            llm_free(g_llm);
            g_llm = NULL;
        }

        holo_print("\nLoading model: %s\n", argv[2]);
        g_llm = llm_load(argv[2]);

        if (g_llm) {
            /* Try to initialize CUDA for GPU acceleration */
#ifdef HOLO_USE_CUDA
            if (llm_cuda_init(g_llm) == 0) {
                holo_print("  \033[32m[CUDA]\033[0m GPU acceleration enabled\n");
            }
#endif
            /* Extract filename */
            const char *name = strrchr(argv[2], '\\');
            if (!name) name = strrchr(argv[2], '/');
            if (name) name++; else name = argv[2];

            holo_print("\033[32m[OK]\033[0m Model loaded: %s\n", name);
            holo_print("     Ready to chat! Type: chat <your message>\n\n");
        } else {
            holo_print_error("Failed to load model: %s\n", argv[2]);
            return HOLO_ERROR_IO;
        }
        return HOLO_OK;
    }

    if (strcmp(subcmd, "list") == 0) {
        holo_print("\nLocal Models (models/):\n");
        holo_print("-----------------------\n");

        /* Try to list models directory */
#ifdef _WIN32
        WIN32_FIND_DATAA fd;
        HANDLE hFind = FindFirstFileA("models\\*.gguf", &fd);
        if (hFind != INVALID_HANDLE_VALUE) {
            int count = 0;
            do {
                /* Get file size */
                LARGE_INTEGER size;
                size.LowPart = fd.nFileSizeLow;
                size.HighPart = fd.nFileSizeHigh;
                double gb = size.QuadPart / (1024.0 * 1024.0 * 1024.0);

                holo_print("  %-40s  %.2f GB\n", fd.cFileName, gb);
                count++;
            } while (FindNextFileA(hFind, &fd));
            FindClose(hFind);

            if (count == 0) {
                holo_print("  (no models found)\n");
            }
            holo_print("\n");
        } else {
            holo_print("  (models/ directory not found or empty)\n\n");
        }
#else
        holo_print("  (use 'ls models/*.gguf' to list)\n\n");
#endif

        holo_print("Download models with: model download <name>\n");
        holo_print("Available: ministral-3b, qwen2-7b, llama3-8b, phi3-mini\n\n");
        return HOLO_OK;
    }

    if (strcmp(subcmd, "download") == 0) {
        if (argc < 3) {
            holo_print("\nUsage: model download <name>\n\n");
            holo_print("Available models:\n");
            holo_print("  ministral-3b   Ministral 3B Instruct Q8 (3.5 GB)\n");
            holo_print("  qwen2-7b       Qwen2 7B Instruct Q4_K_M (4.4 GB)\n");
            holo_print("  llama3-8b      Llama 3 8B Instruct Q4_K_M (4.7 GB)\n");
            holo_print("  phi3-mini      Phi-3 Mini 4K Q4_K_M (2.4 GB)\n");
            holo_print("  gemma2-2b      Gemma 2 2B Instruct Q8 (2.8 GB)\n");
            holo_print("\nModels are downloaded to: models/\n\n");
            return HOLO_OK;
        }

        const char *name = argv[2];
        const char *url = NULL;
        const char *filename = NULL;

        if (strcmp(name, "ministral-3b") == 0) {
            url = "https://huggingface.co/bartowski/Ministral-3b-instruct-GGUF/resolve/main/Ministral-3b-instruct-Q8_0.gguf";
            filename = "Ministral-3b-instruct.Q8_0.gguf";
        } else if (strcmp(name, "qwen2-7b") == 0) {
            url = "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q4_k_m.gguf";
            filename = "qwen2-7b-instruct.Q4_K_M.gguf";
        } else if (strcmp(name, "llama3-8b") == 0) {
            url = "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf";
            filename = "llama3-8b-instruct.Q4_K_M.gguf";
        } else if (strcmp(name, "phi3-mini") == 0) {
            url = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf";
            filename = "phi3-mini-4k.Q4_K_M.gguf";
        } else if (strcmp(name, "gemma2-2b") == 0) {
            url = "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q8_0.gguf";
            filename = "gemma2-2b-it.Q8_0.gguf";
        } else {
            holo_print("Unknown model: %s\n", name);
            holo_print("Use 'model download' to see available models.\n");
            return HOLO_OK;
        }

        holo_print("\nDownloading %s...\n", name);
        holo_print("URL: %s\n\n", url);

        /* Create models directory if needed */
#ifdef _WIN32
        CreateDirectoryA("models", NULL);

        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
            "powershell -Command \"& { "
            "$ProgressPreference = 'SilentlyContinue'; "
            "Invoke-WebRequest -Uri '%s' -OutFile 'models/%s' "
            "}\"", url, filename);

        holo_print("Running: powershell download...\n");
        holo_print("This may take a few minutes depending on your connection.\n\n");

        int ret = system(cmd);
        if (ret == 0) {
            holo_print("\n\033[32m[OK]\033[0m Downloaded: models/%s\n", filename);
            holo_print("     Load with: model load models/%s\n\n", filename);
        } else {
            holo_print("\n\033[31m[ERROR]\033[0m Download failed.\n");
            holo_print("Try manually: curl -L -o models/%s \"%s\"\n\n", filename, url);
        }
#else
        char cmd[2048];
        snprintf(cmd, sizeof(cmd), "mkdir -p models && curl -L -o models/%s '%s'", filename, url);
        system(cmd);
#endif
        return HOLO_OK;
    }

    holo_print("Unknown model subcommand: %s\n", subcmd);
    return HOLO_OK;
}

/* Command: projects (list subprojects) */
static holo_result_t cmd_projects(holo_context_t *ctx, int argc, char **argv) {
    (void)ctx;
    (void)argc;
    (void)argv;

    holo_print("\nHolo Subprojects:\n");
    holo_print("-----------------\n");
    holo_print("Use 'project <name>' to switch to a subproject.\n\n");

    /* This will be populated dynamically later */
    holo_print("  (Project listing will be implemented)\n\n");

    return HOLO_OK;
}

/* Command: serve (start API server) */
static holo_result_t cmd_serve(holo_context_t *ctx, int argc, char **argv) {
    (void)ctx;

    uint16_t port = 8420;
    if (argc > 1) {
        port = (uint16_t)atoi(argv[1]);
    }

    holo_print("\n");
    holo_print("Holo API Server\n");
    holo_print("===============\n");
    holo_print("  Port: %d\n", port);
    holo_print("  Endpoints:\n");
    holo_print("    POST /v1/chat/completions\n");
    holo_print("    GET  /v1/models\n");
    holo_print("    GET  /health\n");
    holo_print("\n");
    holo_print("Server implementation ready - integrate with 'holo_http_server_*' API\n");
    holo_print("See: include/holo/net.h and src/net/http.c\n");
    holo_print("\n");

    return HOLO_OK;
}

/* Register all built-in commands */
holo_result_t holo_register_builtin_commands(holo_context_t *ctx) {
    static const holo_command_t builtins[] = {
        {"help",     "Show available commands",    "help [command]",   cmd_help},
        {"exit",     "Exit Holo",                  "exit",             cmd_exit},
        {"quit",     "Exit Holo",                  "quit",             cmd_exit},
        {"version",  "Show version information",   "version",          cmd_version},
        {"clear",    "Clear the screen",           "clear",            cmd_clear},
        {"history",  "Show command history",       "history [n]",      cmd_history},
        {"pwd",      "Print working directory",    "pwd",              cmd_pwd},
        {"cd",       "Change directory",           "cd <path>",        cmd_cd},
        {"ls",       "List directory contents",    "ls [path]",        cmd_ls},
        {"echo",     "Print arguments",            "echo <text>",      cmd_echo},
        {"chat",     "Chat with Holo AI",          "chat <message>",   cmd_chat},
        {"model",    "Manage AI models",           "model <cmd>",      cmd_model},
        {"projects", "List Holo subprojects",      "projects",         cmd_projects},
        {"serve",    "Start API server",           "serve [port]",     cmd_serve},
    };

    size_t count = sizeof(builtins) / sizeof(builtins[0]);
    for (size_t i = 0; i < count; i++) {
        holo_result_t result = holo_register_command(ctx, &builtins[i]);
        if (result != HOLO_OK) {
            return result;
        }
    }

    return HOLO_OK;
}
