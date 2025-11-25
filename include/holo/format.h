/*
 * Holo - Output Formatting
 * Human-readable, structured terminal output
 */

#ifndef HOLO_FORMAT_H
#define HOLO_FORMAT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdbool.h>

/* ============================================================================
 * ANSI Color Codes
 * ============================================================================ */

/* Reset */
#define HOLO_RESET       "\033[0m"

/* Regular colors */
#define HOLO_BLACK       "\033[30m"
#define HOLO_RED         "\033[31m"
#define HOLO_GREEN       "\033[32m"
#define HOLO_YELLOW      "\033[33m"
#define HOLO_BLUE        "\033[34m"
#define HOLO_MAGENTA     "\033[35m"
#define HOLO_CYAN        "\033[36m"
#define HOLO_WHITE       "\033[37m"

/* Bright colors */
#define HOLO_BRIGHT_BLACK   "\033[90m"
#define HOLO_BRIGHT_RED     "\033[91m"
#define HOLO_BRIGHT_GREEN   "\033[92m"
#define HOLO_BRIGHT_YELLOW  "\033[93m"
#define HOLO_BRIGHT_BLUE    "\033[94m"
#define HOLO_BRIGHT_MAGENTA "\033[95m"
#define HOLO_BRIGHT_CYAN    "\033[96m"
#define HOLO_BRIGHT_WHITE   "\033[97m"

/* Background colors */
#define HOLO_BG_BLACK    "\033[40m"
#define HOLO_BG_RED      "\033[41m"
#define HOLO_BG_GREEN    "\033[42m"
#define HOLO_BG_YELLOW   "\033[43m"
#define HOLO_BG_BLUE     "\033[44m"
#define HOLO_BG_MAGENTA  "\033[45m"
#define HOLO_BG_CYAN     "\033[46m"
#define HOLO_BG_WHITE    "\033[47m"

/* Styles */
#define HOLO_BOLD        "\033[1m"
#define HOLO_DIM         "\033[2m"
#define HOLO_ITALIC      "\033[3m"
#define HOLO_UNDERLINE   "\033[4m"

/* ============================================================================
 * Box Drawing Characters (UTF-8)
 * ============================================================================ */

/* Light box */
#define BOX_H       "\xe2\x94\x80"  /* ─ horizontal */
#define BOX_V       "\xe2\x94\x82"  /* │ vertical */
#define BOX_TL      "\xe2\x94\x8c"  /* ┌ top-left */
#define BOX_TR      "\xe2\x94\x90"  /* ┐ top-right */
#define BOX_BL      "\xe2\x94\x94"  /* └ bottom-left */
#define BOX_BR      "\xe2\x94\x98"  /* ┘ bottom-right */
#define BOX_LT      "\xe2\x94\x9c"  /* ├ left-tee */
#define BOX_RT      "\xe2\x94\xa4"  /* ┤ right-tee */
#define BOX_TT      "\xe2\x94\xac"  /* ┬ top-tee */
#define BOX_BT      "\xe2\x94\xb4"  /* ┴ bottom-tee */
#define BOX_X       "\xe2\x94\xbc"  /* ┼ cross */

/* Double box */
#define BOX2_H      "\xe2\x95\x90"  /* ═ */
#define BOX2_V      "\xe2\x95\x91"  /* ║ */
#define BOX2_TL     "\xe2\x95\x94"  /* ╔ */
#define BOX2_TR     "\xe2\x95\x97"  /* ╗ */
#define BOX2_BL     "\xe2\x95\x9a"  /* ╚ */
#define BOX2_BR     "\xe2\x95\x9d"  /* ╝ */

/* Rounded corners */
#define BOX_R_TL    "\xe2\x95\xad"  /* ╭ */
#define BOX_R_TR    "\xe2\x95\xae"  /* ╮ */
#define BOX_R_BL    "\xe2\x95\xb0"  /* ╰ */
#define BOX_R_BR    "\xe2\x95\xaf"  /* ╯ */

/* ============================================================================
 * Symbols
 * ============================================================================ */

#define SYM_CHECK    "\xe2\x9c\x93"  /* ✓ */
#define SYM_CROSS    "\xe2\x9c\x97"  /* ✗ */
#define SYM_BULLET   "\xe2\x80\xa2"  /* • */
#define SYM_ARROW    "\xe2\x86\x92"  /* → */
#define SYM_INFO     "\xe2\x84\xb9"  /* ℹ */
#define SYM_WARN     "\xe2\x9a\xa0"  /* ⚠ */
#define SYM_STAR     "\xe2\x98\x85"  /* ★ */
#define SYM_LIGHT    "\xe2\x98\x80"  /* ☀ - light bulb alternative */

/* ============================================================================
 * Types
 * ============================================================================ */

typedef enum {
    HOLO_BOX_SINGLE,    /* ┌─┐ */
    HOLO_BOX_DOUBLE,    /* ╔═╗ */
    HOLO_BOX_ROUNDED,   /* ╭─╮ */
    HOLO_BOX_ASCII      /* +-+ fallback */
} holo_box_style_t;

typedef enum {
    HOLO_ALIGN_LEFT,
    HOLO_ALIGN_CENTER,
    HOLO_ALIGN_RIGHT
} holo_align_t;

typedef enum {
    HOLO_MSG_INFO,
    HOLO_MSG_SUCCESS,
    HOLO_MSG_WARNING,
    HOLO_MSG_ERROR,
    HOLO_MSG_TIP
} holo_msg_type_t;

/* ============================================================================
 * Configuration
 * ============================================================================ */

typedef struct {
    int terminal_width;         /* Auto-detect or manual */
    bool use_color;             /* Enable ANSI colors */
    bool use_unicode;           /* Use Unicode box drawing */
    holo_box_style_t box_style; /* Default box style */
} holo_format_config_t;

/* Initialize formatting system */
void holo_format_init(void);

/* Get/set configuration */
holo_format_config_t *holo_format_config(void);
int holo_get_terminal_width(void);

/* ============================================================================
 * Output Functions
 * ============================================================================ */

/* Headers and sections */
void holo_print_header(const char *title);
void holo_print_subheader(const char *title);
void holo_print_divider(void);

/* Boxes */
void holo_print_box(const char *content);
void holo_print_box_titled(const char *title, const char *content);

/* Messages with icons */
void holo_print_msg(holo_msg_type_t type, const char *fmt, ...);
void holo_print_info(const char *fmt, ...);
void holo_print_success(const char *fmt, ...);
void holo_print_warning(const char *fmt, ...);
void holo_print_tip(const char *fmt, ...);

/* Lists */
void holo_print_list(const char **items, size_t count);
void holo_print_numbered_list(const char **items, size_t count);

/* Tables */
void holo_print_table(const char **headers, const char **rows,
                      size_t cols, size_t row_count);
void holo_print_key_value(const char *key, const char *value);

/* Code blocks */
void holo_print_code(const char *language, const char *code);
void holo_print_code_line(int line_num, const char *code);

/* Progress */
void holo_print_progress(int current, int total, const char *label);
void holo_print_spinner_frame(int frame, const char *label);

/* Utility */
void holo_print_wrapped(const char *text, int indent);
void holo_print_padded(const char *text, int width, holo_align_t align);

#ifdef __cplusplus
}
#endif

#endif /* HOLO_FORMAT_H */
