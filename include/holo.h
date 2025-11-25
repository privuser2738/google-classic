/*
 * Holo - Assistive AI with REPL Interface
 * Main header file
 */

#ifndef HOLO_H
#define HOLO_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdbool.h>

/* Version info */
#define HOLO_VERSION_MAJOR 0
#define HOLO_VERSION_MINOR 1
#define HOLO_VERSION_PATCH 0
#define HOLO_VERSION_STRING "0.1.0"

/* Configuration */
#define HOLO_MAX_INPUT_SIZE 4096
#define HOLO_MAX_HISTORY 1000
#define HOLO_PROMPT_DEFAULT "holo> "

/* Result codes */
typedef enum {
    HOLO_OK = 0,
    HOLO_ERROR = -1,
    HOLO_ERROR_MEMORY = -2,
    HOLO_ERROR_IO = -3,
    HOLO_ERROR_PARSE = -4,
    HOLO_EXIT = 1
} holo_result_t;

/* Forward declarations */
typedef struct holo_context holo_context_t;
typedef struct holo_command holo_command_t;

/* Command handler function type */
typedef holo_result_t (*holo_command_fn)(holo_context_t *ctx, int argc, char **argv);

/* Command structure */
struct holo_command {
    const char *name;
    const char *description;
    const char *usage;
    holo_command_fn handler;
};

/* Context structure - main state */
struct holo_context {
    bool running;
    char *prompt;
    char **history;
    size_t history_count;
    size_t history_capacity;
    holo_command_t *commands;
    size_t command_count;
    void *user_data;
};

/* Core functions */
holo_context_t *holo_init(void);
void holo_destroy(holo_context_t *ctx);
holo_result_t holo_run(holo_context_t *ctx);

/* Command registration */
holo_result_t holo_register_command(holo_context_t *ctx, const holo_command_t *cmd);
holo_result_t holo_register_builtin_commands(holo_context_t *ctx);

/* Input/Output */
char *holo_readline(holo_context_t *ctx);
void holo_print(const char *fmt, ...);
void holo_print_error(const char *fmt, ...);

/* History */
holo_result_t holo_history_add(holo_context_t *ctx, const char *line);
void holo_history_clear(holo_context_t *ctx);

/* Utilities */
char **holo_parse_args(const char *line, int *argc);
void holo_free_args(char **argv, int argc);
const char *holo_version(void);

#ifdef __cplusplus
}
#endif

#endif /* HOLO_H */
