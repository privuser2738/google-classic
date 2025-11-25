/*
 * Holo - Platform Abstraction Layer
 * Common/shared implementation
 */

#include "holo/pal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * Platform Identification
 * ============================================================================ */

holo_platform_id_t holo_platform_id(void) {
#if defined(HOLO_PLATFORM_WINDOWS)
    return HOLO_PLAT_WINDOWS;
#elif defined(HOLO_PLATFORM_LINUX)
    return HOLO_PLAT_LINUX;
#elif defined(HOLO_PLATFORM_MACOS)
    return HOLO_PLAT_MACOS;
#elif defined(HOLO_PLATFORM_IOS)
    return HOLO_PLAT_IOS;
#elif defined(HOLO_PLATFORM_ANDROID)
    return HOLO_PLAT_ANDROID;
#else
    return 0;
#endif
}

const char *holo_platform_name(void) {
    return HOLO_PLATFORM_NAME;
}

/* ============================================================================
 * Basic Memory (wraps malloc - platforms can override)
 * ============================================================================ */

void *holo_alloc(size_t size) {
    return malloc(size);
}

void *holo_calloc(size_t count, size_t size) {
    return calloc(count, size);
}

void *holo_realloc(void *ptr, size_t size) {
    return realloc(ptr, size);
}

void holo_free(void *ptr) {
    free(ptr);
}

/* ============================================================================
 * Path Utilities (cross-platform)
 * ============================================================================ */

char *holo_path_join(const char *base, const char *path) {
    if (!base || !path) return NULL;

    size_t base_len = strlen(base);
    size_t path_len = strlen(path);

    /* Remove trailing slash from base */
    while (base_len > 0 && (base[base_len - 1] == '/' || base[base_len - 1] == '\\')) {
        base_len--;
    }

    /* Skip leading slash from path */
    while (*path == '/' || *path == '\\') {
        path++;
        path_len--;
    }

    char *result = (char *)holo_alloc(base_len + 1 + path_len + 1);
    if (!result) return NULL;

    memcpy(result, base, base_len);
#ifdef HOLO_PLATFORM_WINDOWS
    result[base_len] = '\\';
#else
    result[base_len] = '/';
#endif
    memcpy(result + base_len + 1, path, path_len);
    result[base_len + 1 + path_len] = '\0';

    return result;
}

char *holo_path_dirname(const char *path) {
    if (!path) return NULL;

    const char *last_sep = NULL;
    const char *p = path;

    while (*p) {
        if (*p == '/' || *p == '\\') {
            last_sep = p;
        }
        p++;
    }

    if (!last_sep) {
        return strdup(".");
    }

    size_t len = last_sep - path;
    if (len == 0) len = 1;  /* Root directory */

    char *result = (char *)holo_alloc(len + 1);
    if (!result) return NULL;

    memcpy(result, path, len);
    result[len] = '\0';

    return result;
}

char *holo_path_basename(const char *path) {
    if (!path) return NULL;

    const char *last_sep = NULL;
    const char *p = path;

    while (*p) {
        if (*p == '/' || *p == '\\') {
            last_sep = p;
        }
        p++;
    }

    const char *base = last_sep ? last_sep + 1 : path;
    return strdup(base);
}

char *holo_path_extension(const char *path) {
    if (!path) return NULL;

    const char *dot = NULL;
    const char *p = path;

    /* Find last dot after last separator */
    while (*p) {
        if (*p == '/' || *p == '\\') {
            dot = NULL;  /* Reset on directory separator */
        } else if (*p == '.') {
            dot = p;
        }
        p++;
    }

    if (!dot || dot == path) {
        return strdup("");
    }

    return strdup(dot);
}

/* ============================================================================
 * File Utilities
 * ============================================================================ */

char *holo_file_read_all(const char *path, size_t *out_size) {
    holo_file_t *file = holo_file_open(path, HOLO_FILE_READ | HOLO_FILE_BINARY);
    if (!file) return NULL;

    int64_t size = holo_file_size(file);
    if (size < 0) {
        holo_file_close(file);
        return NULL;
    }

    char *buf = (char *)holo_alloc((size_t)size + 1);
    if (!buf) {
        holo_file_close(file);
        return NULL;
    }

    size_t read = holo_file_read(file, buf, (size_t)size);
    holo_file_close(file);

    if (read != (size_t)size) {
        holo_free(buf);
        return NULL;
    }

    buf[size] = '\0';
    if (out_size) *out_size = (size_t)size;

    return buf;
}

/* ============================================================================
 * Time Formatting
 * ============================================================================ */

char *holo_time_format(const holo_datetime_t *dt, const char *fmt) {
    if (!dt || !fmt) return NULL;

    char buf[256];
    char *out = buf;
    char *end = buf + sizeof(buf) - 1;

    while (*fmt && out < end) {
        if (*fmt != '%') {
            *out++ = *fmt++;
            continue;
        }

        fmt++;
        int n = 0;

        switch (*fmt) {
            case 'Y':  /* Year (4 digit) */
                n = snprintf(out, end - out, "%04d", dt->year);
                break;
            case 'm':  /* Month (01-12) */
                n = snprintf(out, end - out, "%02d", dt->month);
                break;
            case 'd':  /* Day (01-31) */
                n = snprintf(out, end - out, "%02d", dt->day);
                break;
            case 'H':  /* Hour (00-23) */
                n = snprintf(out, end - out, "%02d", dt->hour);
                break;
            case 'M':  /* Minute (00-59) */
                n = snprintf(out, end - out, "%02d", dt->minute);
                break;
            case 'S':  /* Second (00-59) */
                n = snprintf(out, end - out, "%02d", dt->second);
                break;
            case 's':  /* Milliseconds (000-999) */
                n = snprintf(out, end - out, "%03d", dt->ms);
                break;
            case '%':
                *out++ = '%';
                fmt++;
                continue;
            default:
                *out++ = '%';
                *out++ = *fmt;
                fmt++;
                continue;
        }

        out += n;
        fmt++;
    }

    *out = '\0';
    return strdup(buf);
}

/* ============================================================================
 * Error Handling
 * ============================================================================ */

static __thread int g_last_error = 0;

int holo_get_last_error(void) {
    return g_last_error;
}

const char *holo_get_error_string(int error) {
    switch (error) {
        case HOLO_OK:           return "Success";
        case HOLO_ERR_NOMEM:    return "Out of memory";
        case HOLO_ERR_IO:       return "I/O error";
        case HOLO_ERR_NOTFOUND: return "Not found";
        case HOLO_ERR_PERM:     return "Permission denied";
        case HOLO_ERR_BUSY:     return "Resource busy";
        case HOLO_ERR_TIMEOUT:  return "Timeout";
        case HOLO_ERR_INVAL:    return "Invalid argument";
        case HOLO_ERR_NET:      return "Network error";
        case HOLO_ERR_CONNRESET: return "Connection reset";
        case HOLO_ERR_CONNREFUSED: return "Connection refused";
        default: return "Unknown error";
    }
}
