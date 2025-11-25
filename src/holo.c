/*
 * Holo - Assistive AI with REPL Interface
 * Core implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>
#include "holo.h"

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#include <readline/readline.h>
#include <readline/history.h>
#endif

/* Initialize Holo context */
holo_context_t *holo_init(void) {
    holo_context_t *ctx = (holo_context_t *)calloc(1, sizeof(holo_context_t));
    if (!ctx) {
        return NULL;
    }

    ctx->running = true;
    ctx->prompt = strdup(HOLO_PROMPT_DEFAULT);
    ctx->history_capacity = 64;
    ctx->history = (char **)calloc(ctx->history_capacity, sizeof(char *));
    ctx->commands = NULL;
    ctx->command_count = 0;
    ctx->user_data = NULL;

    if (!ctx->prompt || !ctx->history) {
        holo_destroy(ctx);
        return NULL;
    }

#ifdef _WIN32
    /* Enable ANSI escape sequences on Windows */
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    if (GetConsoleMode(hOut, &dwMode)) {
        dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        SetConsoleMode(hOut, dwMode);
    }
#endif

    return ctx;
}

/* Destroy Holo context */
void holo_destroy(holo_context_t *ctx) {
    if (!ctx) return;

    free(ctx->prompt);
    holo_history_clear(ctx);
    free(ctx->history);
    free(ctx->commands);
    free(ctx);
}

/* Read a line of input */
char *holo_readline(holo_context_t *ctx) {
    static char buffer[HOLO_MAX_INPUT_SIZE];

#ifdef _WIN32
    /* Windows: simple line input */
    printf("%s", ctx->prompt);
    fflush(stdout);

    if (!fgets(buffer, sizeof(buffer), stdin)) {
        return NULL;
    }

    /* Remove trailing newline */
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
        buffer[len - 1] = '\0';
        len--;
    }
    if (len > 0 && buffer[len - 1] == '\r') {
        buffer[len - 1] = '\0';
    }

    return buffer[0] ? strdup(buffer) : strdup("");
#else
    /* Unix: use readline library if available */
    char *line = readline(ctx->prompt);
    if (line && *line) {
        add_history(line);
    }
    return line;
#endif
}

/* Print formatted output */
void holo_print(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

/* Print error message */
void holo_print_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "\033[31mError: \033[0m");
    vfprintf(stderr, fmt, args);
    va_end(args);
}

/* Add line to history */
holo_result_t holo_history_add(holo_context_t *ctx, const char *line) {
    if (!ctx || !line || !*line) {
        return HOLO_ERROR;
    }

    /* Expand history if needed */
    if (ctx->history_count >= ctx->history_capacity) {
        size_t new_capacity = ctx->history_capacity * 2;
        char **new_history = (char **)realloc(ctx->history,
                                               new_capacity * sizeof(char *));
        if (!new_history) {
            return HOLO_ERROR_MEMORY;
        }
        ctx->history = new_history;
        ctx->history_capacity = new_capacity;
    }

    ctx->history[ctx->history_count] = strdup(line);
    if (!ctx->history[ctx->history_count]) {
        return HOLO_ERROR_MEMORY;
    }
    ctx->history_count++;

    return HOLO_OK;
}

/* Clear history */
void holo_history_clear(holo_context_t *ctx) {
    if (!ctx || !ctx->history) return;

    for (size_t i = 0; i < ctx->history_count; i++) {
        free(ctx->history[i]);
        ctx->history[i] = NULL;
    }
    ctx->history_count = 0;
}

/* Parse command line into arguments */
char **holo_parse_args(const char *line, int *argc) {
    if (!line || !argc) {
        return NULL;
    }

    *argc = 0;

    /* Count arguments first */
    const char *p = line;
    bool in_quote = false;
    bool in_arg = false;
    int count = 0;

    while (*p) {
        if (*p == '"') {
            in_quote = !in_quote;
            if (!in_arg) {
                count++;
                in_arg = true;
            }
        } else if (isspace((unsigned char)*p) && !in_quote) {
            in_arg = false;
        } else if (!in_arg) {
            count++;
            in_arg = true;
        }
        p++;
    }

    if (count == 0) {
        return NULL;
    }

    /* Allocate argument array */
    char **argv = (char **)calloc(count + 1, sizeof(char *));
    if (!argv) {
        return NULL;
    }

    /* Parse arguments */
    p = line;
    int idx = 0;
    char arg_buffer[HOLO_MAX_INPUT_SIZE];
    int buf_pos = 0;
    in_quote = false;
    in_arg = false;

    while (*p) {
        if (*p == '"') {
            in_quote = !in_quote;
            if (!in_arg) {
                in_arg = true;
                buf_pos = 0;
            }
        } else if (isspace((unsigned char)*p) && !in_quote) {
            if (in_arg) {
                arg_buffer[buf_pos] = '\0';
                argv[idx++] = strdup(arg_buffer);
                in_arg = false;
                buf_pos = 0;
            }
        } else {
            if (!in_arg) {
                in_arg = true;
                buf_pos = 0;
            }
            arg_buffer[buf_pos++] = *p;
        }
        p++;
    }

    /* Handle last argument */
    if (in_arg && buf_pos > 0) {
        arg_buffer[buf_pos] = '\0';
        argv[idx++] = strdup(arg_buffer);
    }

    *argc = idx;
    argv[idx] = NULL;

    return argv;
}

/* Free parsed arguments */
void holo_free_args(char **argv, int argc) {
    if (!argv) return;

    for (int i = 0; i < argc; i++) {
        free(argv[i]);
    }
    free(argv);
}

/* Get version string */
const char *holo_version(void) {
    return HOLO_VERSION_STRING;
}

/* Register a command */
holo_result_t holo_register_command(holo_context_t *ctx, const holo_command_t *cmd) {
    if (!ctx || !cmd || !cmd->name || !cmd->handler) {
        return HOLO_ERROR;
    }

    /* Reallocate command array */
    holo_command_t *new_commands = (holo_command_t *)realloc(
        ctx->commands,
        (ctx->command_count + 1) * sizeof(holo_command_t)
    );
    if (!new_commands) {
        return HOLO_ERROR_MEMORY;
    }

    ctx->commands = new_commands;
    ctx->commands[ctx->command_count] = *cmd;
    ctx->command_count++;

    return HOLO_OK;
}

/* Main REPL loop */
holo_result_t holo_run(holo_context_t *ctx) {
    if (!ctx) {
        return HOLO_ERROR;
    }

    while (ctx->running) {
        char *line = holo_readline(ctx);
        if (!line) {
            /* EOF */
            printf("\n");
            break;
        }

        /* Skip empty lines */
        if (!*line) {
            free(line);
            continue;
        }

        /* Add to history */
        holo_history_add(ctx, line);

        /* Parse and execute */
        int argc;
        char **argv = holo_parse_args(line, &argc);
        free(line);

        if (!argv || argc == 0) {
            continue;
        }

        /* Find and execute command */
        bool found = false;
        for (size_t i = 0; i < ctx->command_count; i++) {
            if (strcmp(ctx->commands[i].name, argv[0]) == 0) {
                holo_result_t result = ctx->commands[i].handler(ctx, argc, argv);
                if (result == HOLO_EXIT) {
                    holo_free_args(argv, argc);
                    return HOLO_EXIT;
                }
                found = true;
                break;
            }
        }

        if (!found) {
            holo_print_error("Unknown command: %s\n", argv[0]);
            holo_print("Type 'help' for available commands.\n");
        }

        holo_free_args(argv, argc);
    }

    return HOLO_OK;
}
