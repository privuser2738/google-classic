/*
 * Holo - Output Formatting Implementation
 * Human-readable, structured terminal output
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "holo/format.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

/* Global configuration */
static holo_format_config_t g_config = {
    .terminal_width = 80,
    .use_color = true,
    .use_unicode = true,
    .box_style = HOLO_BOX_ROUNDED
};

/* Spinner frames */
static const char *spinner_frames[] = {
    "\xe2\xa0\x8b", "\xe2\xa0\x99", "\xe2\xa0\xb9", "\xe2\xa0\xb8",
    "\xe2\xa0\xbc", "\xe2\xa0\xb4", "\xe2\xa0\xa6", "\xe2\xa0\xa7",
    "\xe2\xa0\x87", "\xe2\xa0\x8f"  /* ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ */
};
#define SPINNER_FRAME_COUNT 10

/* ============================================================================
 * Initialization & Configuration
 * ============================================================================ */

void holo_format_init(void) {
    g_config.terminal_width = holo_get_terminal_width();

#ifdef _WIN32
    /* Enable UTF-8 output on Windows */
    SetConsoleOutputCP(CP_UTF8);

    /* Enable ANSI escape sequences */
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    if (GetConsoleMode(hOut, &dwMode)) {
        dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        SetConsoleMode(hOut, dwMode);
    }
#endif
}

holo_format_config_t *holo_format_config(void) {
    return &g_config;
}

int holo_get_terminal_width(void) {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        return csbi.srWindow.Right - csbi.srWindow.Left + 1;
    }
    return 80;
#else
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) {
        return w.ws_col;
    }
    return 80;
#endif
}

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static void print_horizontal_line(int width, const char *left,
                                  const char *mid, const char *right) {
    printf("%s", left);
    for (int i = 0; i < width - 2; i++) {
        printf("%s", mid);
    }
    printf("%s\n", right);
}

static size_t utf8_strlen(const char *s) {
    size_t len = 0;
    while (*s) {
        if ((*s & 0xC0) != 0x80) len++;
        s++;
    }
    return len;
}

static void print_repeated(const char *str, int count) {
    for (int i = 0; i < count; i++) {
        printf("%s", str);
    }
}

/* ============================================================================
 * Headers and Sections
 * ============================================================================ */

void holo_print_header(const char *title) {
    int width = g_config.terminal_width;
    size_t title_len = utf8_strlen(title);

    printf("\n");
    if (g_config.use_color) printf(HOLO_BOLD HOLO_CYAN);

    if (g_config.use_unicode) {
        printf("%s %s %s", BOX2_H BOX2_H, title, "");
        print_repeated(BOX2_H, (int)(width - title_len - 5));
    } else {
        printf("== %s ", title);
        print_repeated("=", (int)(width - title_len - 5));
    }

    if (g_config.use_color) printf(HOLO_RESET);
    printf("\n\n");
}

void holo_print_subheader(const char *title) {
    if (g_config.use_color) printf(HOLO_BOLD);
    printf("%s", title);
    if (g_config.use_color) printf(HOLO_RESET);
    printf("\n");

    size_t len = utf8_strlen(title);
    if (g_config.use_unicode) {
        print_repeated(BOX_H, (int)len);
    } else {
        print_repeated("-", (int)len);
    }
    printf("\n");
}

void holo_print_divider(void) {
    int width = g_config.terminal_width - 4;
    if (g_config.use_color) printf(HOLO_DIM);
    if (g_config.use_unicode) {
        print_repeated(BOX_H, width);
    } else {
        print_repeated("-", width);
    }
    if (g_config.use_color) printf(HOLO_RESET);
    printf("\n");
}

/* ============================================================================
 * Boxes
 * ============================================================================ */

void holo_print_box(const char *content) {
    holo_print_box_titled(NULL, content);
}

void holo_print_box_titled(const char *title, const char *content) {
    int width = g_config.terminal_width - 4;
    if (width > 76) width = 76;  /* Max width for readability */

    const char *tl, *tr, *bl, *br, *h, *v;

    if (g_config.use_unicode) {
        switch (g_config.box_style) {
            case HOLO_BOX_DOUBLE:
                tl = BOX2_TL; tr = BOX2_TR; bl = BOX2_BL; br = BOX2_BR;
                h = BOX2_H; v = BOX2_V;
                break;
            case HOLO_BOX_ROUNDED:
                tl = BOX_R_TL; tr = BOX_R_TR; bl = BOX_R_BL; br = BOX_R_BR;
                h = BOX_H; v = BOX_V;
                break;
            default:
                tl = BOX_TL; tr = BOX_TR; bl = BOX_BL; br = BOX_BR;
                h = BOX_H; v = BOX_V;
        }
    } else {
        tl = "+"; tr = "+"; bl = "+"; br = "+";
        h = "-"; v = "|";
    }

    if (g_config.use_color) printf(HOLO_CYAN);

    /* Top border with optional title */
    printf("%s", tl);
    if (title) {
        printf("%s ", h);
        if (g_config.use_color) printf(HOLO_BOLD HOLO_WHITE);
        printf("%s", title);
        if (g_config.use_color) printf(HOLO_RESET HOLO_CYAN);
        printf(" ");
        size_t title_len = utf8_strlen(title);
        print_repeated(h, width - (int)title_len - 4);
    } else {
        print_repeated(h, width - 2);
    }
    printf("%s\n", tr);

    /* Content lines */
    const char *line_start = content;
    const char *p = content;

    while (*p) {
        if (*p == '\n' || (p - line_start) >= (width - 4)) {
            printf("%s ", v);
            if (g_config.use_color) printf(HOLO_RESET);

            int line_len = (int)(p - line_start);
            if (*p == '\n') {
                fwrite(line_start, 1, line_len, stdout);
            } else {
                /* Word wrap: find last space */
                const char *wrap = p;
                while (wrap > line_start && *wrap != ' ') wrap--;
                if (wrap == line_start) wrap = p;

                line_len = (int)(wrap - line_start);
                fwrite(line_start, 1, line_len, stdout);
                p = wrap;
                if (*p == ' ') p++;
            }

            /* Padding */
            for (int i = line_len; i < width - 4; i++) printf(" ");

            if (g_config.use_color) printf(HOLO_CYAN);
            printf(" %s\n", v);

            line_start = (*p == '\n') ? p + 1 : p;
            if (*p == '\n') p++;
        } else {
            p++;
        }
    }

    /* Last line if any */
    if (line_start < p) {
        printf("%s ", v);
        if (g_config.use_color) printf(HOLO_RESET);
        int line_len = (int)(p - line_start);
        fwrite(line_start, 1, line_len, stdout);
        for (int i = line_len; i < width - 4; i++) printf(" ");
        if (g_config.use_color) printf(HOLO_CYAN);
        printf(" %s\n", v);
    }

    /* Bottom border */
    printf("%s", bl);
    print_repeated(h, width - 2);
    printf("%s", br);

    if (g_config.use_color) printf(HOLO_RESET);
    printf("\n");
}

/* ============================================================================
 * Messages with Icons
 * ============================================================================ */

void holo_print_msg(holo_msg_type_t type, const char *fmt, ...) {
    const char *icon;
    const char *color;

    switch (type) {
        case HOLO_MSG_SUCCESS:
            icon = g_config.use_unicode ? SYM_CHECK : "[OK]";
            color = HOLO_GREEN;
            break;
        case HOLO_MSG_WARNING:
            icon = g_config.use_unicode ? SYM_WARN : "[!]";
            color = HOLO_YELLOW;
            break;
        case HOLO_MSG_ERROR:
            icon = g_config.use_unicode ? SYM_CROSS : "[X]";
            color = HOLO_RED;
            break;
        case HOLO_MSG_TIP:
            icon = g_config.use_unicode ? SYM_LIGHT : "[*]";
            color = HOLO_MAGENTA;
            break;
        case HOLO_MSG_INFO:
        default:
            icon = g_config.use_unicode ? SYM_INFO : "[i]";
            color = HOLO_BLUE;
            break;
    }

    if (g_config.use_color) printf("%s", color);
    printf("%s ", icon);
    if (g_config.use_color) printf(HOLO_RESET);

    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);

    printf("\n");
}

void holo_print_info(const char *fmt, ...) {
    if (g_config.use_color) printf(HOLO_BLUE);
    printf("%s ", g_config.use_unicode ? SYM_INFO : "[i]");
    if (g_config.use_color) printf(HOLO_RESET);

    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
}

void holo_print_success(const char *fmt, ...) {
    if (g_config.use_color) printf(HOLO_GREEN);
    printf("%s ", g_config.use_unicode ? SYM_CHECK : "[OK]");
    if (g_config.use_color) printf(HOLO_RESET);

    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
}

void holo_print_warning(const char *fmt, ...) {
    if (g_config.use_color) printf(HOLO_YELLOW);
    printf("%s ", g_config.use_unicode ? SYM_WARN : "[!]");
    if (g_config.use_color) printf(HOLO_RESET);

    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
}

void holo_print_tip(const char *fmt, ...) {
    if (g_config.use_color) printf(HOLO_MAGENTA);
    printf("%s ", g_config.use_unicode ? SYM_LIGHT : "[*]");
    if (g_config.use_color) printf(HOLO_RESET);

    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
}

/* ============================================================================
 * Lists
 * ============================================================================ */

void holo_print_list(const char **items, size_t count) {
    const char *bullet = g_config.use_unicode ? SYM_BULLET : "*";

    for (size_t i = 0; i < count; i++) {
        if (g_config.use_color) printf(HOLO_CYAN);
        printf("  %s ", bullet);
        if (g_config.use_color) printf(HOLO_RESET);
        printf("%s\n", items[i]);
    }
}

void holo_print_numbered_list(const char **items, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (g_config.use_color) printf(HOLO_CYAN);
        printf("  %zu. ", i + 1);
        if (g_config.use_color) printf(HOLO_RESET);
        printf("%s\n", items[i]);
    }
}

/* ============================================================================
 * Tables
 * ============================================================================ */

void holo_print_table(const char **headers, const char **rows,
                      size_t cols, size_t row_count) {
    if (cols == 0 || !headers) return;

    /* Calculate column widths */
    size_t *widths = calloc(cols, sizeof(size_t));
    if (!widths) return;

    for (size_t c = 0; c < cols; c++) {
        widths[c] = utf8_strlen(headers[c]);
    }

    for (size_t r = 0; r < row_count; r++) {
        for (size_t c = 0; c < cols; c++) {
            size_t len = utf8_strlen(rows[r * cols + c]);
            if (len > widths[c]) widths[c] = len;
        }
    }

    /* Print header */
    if (g_config.use_color) printf(HOLO_BOLD);
    for (size_t c = 0; c < cols; c++) {
        printf("  %-*s", (int)widths[c] + 2, headers[c]);
    }
    printf("\n");
    if (g_config.use_color) printf(HOLO_RESET);

    /* Header separator */
    if (g_config.use_color) printf(HOLO_DIM);
    for (size_t c = 0; c < cols; c++) {
        printf("  ");
        print_repeated(g_config.use_unicode ? BOX_H : "-", (int)widths[c]);
        printf("  ");
    }
    printf("\n");
    if (g_config.use_color) printf(HOLO_RESET);

    /* Print rows */
    for (size_t r = 0; r < row_count; r++) {
        for (size_t c = 0; c < cols; c++) {
            printf("  %-*s", (int)widths[c] + 2, rows[r * cols + c]);
        }
        printf("\n");
    }

    free(widths);
}

void holo_print_key_value(const char *key, const char *value) {
    if (g_config.use_color) printf(HOLO_CYAN);
    printf("  %s: ", key);
    if (g_config.use_color) printf(HOLO_RESET);
    printf("%s\n", value);
}

/* ============================================================================
 * Code Blocks
 * ============================================================================ */

void holo_print_code(const char *language, const char *code) {
    int width = g_config.terminal_width - 4;
    if (width > 76) width = 76;

    /* Header */
    if (g_config.use_color) printf(HOLO_DIM);
    if (g_config.use_unicode) {
        printf("%s", BOX_R_TL);
        print_repeated(BOX_H, 2);
    } else {
        printf("+--");
    }

    if (language && *language) {
        if (g_config.use_color) printf(HOLO_RESET HOLO_YELLOW);
        printf(" %s ", language);
        if (g_config.use_color) printf(HOLO_DIM);
    }

    int remaining = width - (language ? (int)utf8_strlen(language) + 6 : 4);
    if (g_config.use_unicode) {
        print_repeated(BOX_H, remaining);
        printf("%s\n", BOX_R_TR);
    } else {
        print_repeated("-", remaining);
        printf("+\n");
    }

    /* Code content */
    const char *p = code;
    int line_num = 1;

    while (*p) {
        const char *line_end = strchr(p, '\n');
        if (!line_end) line_end = p + strlen(p);

        if (g_config.use_color) printf(HOLO_DIM);
        printf("%s ", g_config.use_unicode ? BOX_V : "|");
        if (g_config.use_color) printf(HOLO_BRIGHT_BLACK);
        printf("%3d ", line_num++);
        if (g_config.use_color) printf(HOLO_DIM);
        printf("%s ", g_config.use_unicode ? BOX_V : "|");
        if (g_config.use_color) printf(HOLO_RESET);

        fwrite(p, 1, line_end - p, stdout);
        printf("\n");

        p = (*line_end) ? line_end + 1 : line_end;
    }

    /* Footer */
    if (g_config.use_color) printf(HOLO_DIM);
    if (g_config.use_unicode) {
        printf("%s", BOX_R_BL);
        print_repeated(BOX_H, width - 2);
        printf("%s", BOX_R_BR);
    } else {
        printf("+");
        print_repeated("-", width - 2);
        printf("+");
    }
    if (g_config.use_color) printf(HOLO_RESET);
    printf("\n");
}

void holo_print_code_line(int line_num, const char *code) {
    if (g_config.use_color) printf(HOLO_BRIGHT_BLACK);
    printf("%4d ", line_num);
    if (g_config.use_color) printf(HOLO_DIM);
    printf("%s ", g_config.use_unicode ? BOX_V : "|");
    if (g_config.use_color) printf(HOLO_RESET);
    printf("%s\n", code);
}

/* ============================================================================
 * Progress
 * ============================================================================ */

void holo_print_progress(int current, int total, const char *label) {
    int bar_width = 30;
    int filled = (current * bar_width) / total;
    int percent = (current * 100) / total;

    printf("\r");
    if (label) printf("%s ", label);

    if (g_config.use_color) printf(HOLO_CYAN);
    printf("[");

    if (g_config.use_color) printf(HOLO_GREEN);
    for (int i = 0; i < filled; i++) {
        printf("%s", g_config.use_unicode ? "\xe2\x96\x88" : "#");
    }

    if (g_config.use_color) printf(HOLO_DIM);
    for (int i = filled; i < bar_width; i++) {
        printf("%s", g_config.use_unicode ? "\xe2\x96\x91" : "-");
    }

    if (g_config.use_color) printf(HOLO_RESET HOLO_CYAN);
    printf("]");
    if (g_config.use_color) printf(HOLO_RESET);

    printf(" %3d%%", percent);
    fflush(stdout);

    if (current >= total) printf("\n");
}

void holo_print_spinner_frame(int frame, const char *label) {
    frame = frame % SPINNER_FRAME_COUNT;

    printf("\r");
    if (g_config.use_color) printf(HOLO_CYAN);

    if (g_config.use_unicode) {
        printf("%s ", spinner_frames[frame]);
    } else {
        const char *ascii_frames = "|/-\\";
        printf("%c ", ascii_frames[frame % 4]);
    }

    if (g_config.use_color) printf(HOLO_RESET);
    if (label) printf("%s", label);
    fflush(stdout);
}

/* ============================================================================
 * Utility
 * ============================================================================ */

void holo_print_wrapped(const char *text, int indent) {
    int width = g_config.terminal_width - indent - 2;
    const char *p = text;

    while (*p) {
        /* Print indent */
        for (int i = 0; i < indent; i++) printf(" ");

        /* Find line end */
        const char *line_end = p;
        const char *last_space = NULL;
        int col = 0;

        while (*line_end && *line_end != '\n' && col < width) {
            if (*line_end == ' ') last_space = line_end;
            line_end++;
            col++;
        }

        /* Word wrap at space if possible */
        if (*line_end && *line_end != '\n' && last_space) {
            line_end = last_space;
        }

        fwrite(p, 1, line_end - p, stdout);
        printf("\n");

        p = line_end;
        while (*p == ' ' || *p == '\n') p++;
    }
}

void holo_print_padded(const char *text, int width, holo_align_t align) {
    size_t len = utf8_strlen(text);
    int padding = (int)width - (int)len;

    if (padding <= 0) {
        printf("%s", text);
        return;
    }

    switch (align) {
        case HOLO_ALIGN_CENTER:
            for (int i = 0; i < padding / 2; i++) printf(" ");
            printf("%s", text);
            for (int i = 0; i < (padding + 1) / 2; i++) printf(" ");
            break;
        case HOLO_ALIGN_RIGHT:
            for (int i = 0; i < padding; i++) printf(" ");
            printf("%s", text);
            break;
        case HOLO_ALIGN_LEFT:
        default:
            printf("%s", text);
            for (int i = 0; i < padding; i++) printf(" ");
            break;
    }
}
