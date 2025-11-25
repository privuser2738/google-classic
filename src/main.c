/*
 * Holo - Assistive AI with REPL Interface
 * Main entry point
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "holo.h"

static void print_banner(void) {
    printf("\n");
    printf("  _    _       _       \n");
    printf(" | |  | |     | |      \n");
    printf(" | |__| | ___ | | ___  \n");
    printf(" |  __  |/ _ \\| |/ _ \\ \n");
    printf(" | |  | | (_) | | (_) |\n");
    printf(" |_|  |_|\\___/|_|\\___/ \n");
    printf("\n");
    printf(" Assistive AI v%s\n", HOLO_VERSION_STRING);
    printf(" Type 'help' for available commands, 'exit' to quit.\n");
    printf("\n");
}

static void print_usage(const char *program) {
    printf("Usage: %s [options]\n", program);
    printf("\n");
    printf("Options:\n");
    printf("  -h, --help     Show this help message\n");
    printf("  -v, --version  Show version information\n");
    printf("  -c <command>   Execute a single command and exit\n");
    printf("\n");
}

int main(int argc, char *argv[]) {
    holo_context_t *ctx = NULL;
    holo_result_t result;
    bool interactive = true;
    const char *single_command = NULL;

    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--version") == 0) {
            printf("Holo v%s\n", HOLO_VERSION_STRING);
            return 0;
        }
        else if (strcmp(argv[i], "-c") == 0) {
            if (i + 1 < argc) {
                single_command = argv[++i];
                interactive = false;
            } else {
                fprintf(stderr, "Error: -c requires a command argument\n");
                return 1;
            }
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Initialize Holo context */
    ctx = holo_init();
    if (!ctx) {
        fprintf(stderr, "Failed to initialize Holo\n");
        return 1;
    }

    /* Register built-in commands */
    result = holo_register_builtin_commands(ctx);
    if (result != HOLO_OK) {
        fprintf(stderr, "Failed to register built-in commands\n");
        holo_destroy(ctx);
        return 1;
    }

    if (interactive) {
        /* Interactive REPL mode */
        print_banner();
        result = holo_run(ctx);
    } else {
        /* Single command mode */
        int cmd_argc;
        char **cmd_argv = holo_parse_args(single_command, &cmd_argc);
        if (cmd_argv && cmd_argc > 0) {
            /* Find and execute the command */
            bool found = false;
            for (size_t i = 0; i < ctx->command_count; i++) {
                if (strcmp(ctx->commands[i].name, cmd_argv[0]) == 0) {
                    result = ctx->commands[i].handler(ctx, cmd_argc, cmd_argv);
                    found = true;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "Unknown command: %s\n", cmd_argv[0]);
                result = HOLO_ERROR;
            }
            holo_free_args(cmd_argv, cmd_argc);
        } else {
            result = HOLO_ERROR_PARSE;
        }
    }

    holo_destroy(ctx);

    return (result == HOLO_OK || result == HOLO_EXIT) ? 0 : 1;
}
