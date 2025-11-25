/*
 * Holo - Assistive AI with REPL Interface
 * Built-in commands
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "holo.h"

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

/* Command: chat (placeholder for AI chat) */
static holo_result_t cmd_chat(holo_context_t *ctx, int argc, char **argv) {
    (void)ctx;

    if (argc < 2) {
        holo_print("Usage: chat <message>\n");
        holo_print("Start a conversation with Holo AI.\n");
        return HOLO_OK;
    }

    /* Build message from arguments */
    char message[HOLO_MAX_INPUT_SIZE] = {0};
    for (int i = 1; i < argc; i++) {
        if (i > 1) strcat(message, " ");
        strncat(message, argv[i], sizeof(message) - strlen(message) - 1);
    }

    holo_print("\n[Holo]: AI chat integration coming soon...\n");
    holo_print("        Your message: \"%s\"\n\n", message);

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
        {"projects", "List Holo subprojects",      "projects",         cmd_projects},
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
