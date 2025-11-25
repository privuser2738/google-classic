/*
 * Holo - Platform Abstraction Layer (PAL)
 * Cross-platform interface for Windows, Linux, macOS, iOS, Android
 */

#ifndef HOLO_PAL_H
#define HOLO_PAL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* ============================================================================
 * Platform Detection
 * ============================================================================ */

#if defined(_WIN32) || defined(_WIN64)
    #define HOLO_PLATFORM_WINDOWS 1
    #define HOLO_PLATFORM_NAME "windows"
#elif defined(__APPLE__)
    #include <TargetConditionals.h>
    #if TARGET_OS_IPHONE || TARGET_OS_SIMULATOR
        #define HOLO_PLATFORM_IOS 1
        #define HOLO_PLATFORM_NAME "ios"
    #else
        #define HOLO_PLATFORM_MACOS 1
        #define HOLO_PLATFORM_NAME "macos"
    #endif
    #define HOLO_PLATFORM_APPLE 1
#elif defined(__ANDROID__)
    #define HOLO_PLATFORM_ANDROID 1
    #define HOLO_PLATFORM_NAME "android"
#elif defined(__linux__)
    #define HOLO_PLATFORM_LINUX 1
    #define HOLO_PLATFORM_NAME "linux"
#else
    #error "Unsupported platform"
#endif

/* Architecture detection */
#if defined(__x86_64__) || defined(_M_X64)
    #define HOLO_ARCH_X64 1
    #define HOLO_ARCH_NAME "x64"
#elif defined(__i386__) || defined(_M_IX86)
    #define HOLO_ARCH_X86 1
    #define HOLO_ARCH_NAME "x86"
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define HOLO_ARCH_ARM64 1
    #define HOLO_ARCH_NAME "arm64"
#elif defined(__arm__) || defined(_M_ARM)
    #define HOLO_ARCH_ARM 1
    #define HOLO_ARCH_NAME "arm"
#else
    #define HOLO_ARCH_NAME "unknown"
#endif

/* Export/Import macros */
#ifdef HOLO_PLATFORM_WINDOWS
    #ifdef HOLO_BUILD_DLL
        #define HOLO_API __declspec(dllexport)
    #elif defined(HOLO_USE_DLL)
        #define HOLO_API __declspec(dllimport)
    #else
        #define HOLO_API
    #endif
#else
    #define HOLO_API __attribute__((visibility("default")))
#endif

/* ============================================================================
 * Platform Identification
 * ============================================================================ */

typedef enum {
    HOLO_PLAT_WINDOWS = 1,
    HOLO_PLAT_LINUX   = 2,
    HOLO_PLAT_MACOS   = 3,
    HOLO_PLAT_IOS     = 4,
    HOLO_PLAT_ANDROID = 5
} holo_platform_id_t;

typedef struct {
    holo_platform_id_t id;
    const char *name;
    const char *arch;
    const char *version;
    int version_major;
    int version_minor;
    int cpu_cores;
    uint64_t total_memory;
} holo_platform_info_t;

HOLO_API holo_platform_id_t holo_platform_id(void);
HOLO_API const char *holo_platform_name(void);
HOLO_API int holo_platform_info(holo_platform_info_t *info);

/* ============================================================================
 * Memory Management
 * ============================================================================ */

HOLO_API void *holo_alloc(size_t size);
HOLO_API void *holo_calloc(size_t count, size_t size);
HOLO_API void *holo_realloc(void *ptr, size_t size);
HOLO_API void holo_free(void *ptr);

/* Aligned allocation (for SIMD) */
HOLO_API void *holo_alloc_aligned(size_t size, size_t alignment);
HOLO_API void holo_free_aligned(void *ptr);

/* Memory info */
HOLO_API uint64_t holo_mem_available(void);
HOLO_API uint64_t holo_mem_used(void);

/* ============================================================================
 * Threading
 * ============================================================================ */

typedef struct holo_thread holo_thread_t;
typedef struct holo_mutex holo_mutex_t;
typedef struct holo_cond holo_cond_t;
typedef struct holo_rwlock holo_rwlock_t;
typedef struct holo_semaphore holo_semaphore_t;

typedef void (*holo_thread_func_t)(void *arg);

/* Thread management */
HOLO_API holo_thread_t *holo_thread_create(holo_thread_func_t func, void *arg);
HOLO_API int holo_thread_join(holo_thread_t *thread);
HOLO_API void holo_thread_detach(holo_thread_t *thread);
HOLO_API void holo_thread_yield(void);
HOLO_API uint64_t holo_thread_id(void);
HOLO_API int holo_thread_set_name(const char *name);

/* Mutex */
HOLO_API holo_mutex_t *holo_mutex_create(void);
HOLO_API void holo_mutex_destroy(holo_mutex_t *mutex);
HOLO_API void holo_mutex_lock(holo_mutex_t *mutex);
HOLO_API bool holo_mutex_trylock(holo_mutex_t *mutex);
HOLO_API void holo_mutex_unlock(holo_mutex_t *mutex);

/* Condition variable */
HOLO_API holo_cond_t *holo_cond_create(void);
HOLO_API void holo_cond_destroy(holo_cond_t *cond);
HOLO_API void holo_cond_wait(holo_cond_t *cond, holo_mutex_t *mutex);
HOLO_API bool holo_cond_timedwait(holo_cond_t *cond, holo_mutex_t *mutex, uint32_t timeout_ms);
HOLO_API void holo_cond_signal(holo_cond_t *cond);
HOLO_API void holo_cond_broadcast(holo_cond_t *cond);

/* Read-write lock */
HOLO_API holo_rwlock_t *holo_rwlock_create(void);
HOLO_API void holo_rwlock_destroy(holo_rwlock_t *rwlock);
HOLO_API void holo_rwlock_rdlock(holo_rwlock_t *rwlock);
HOLO_API void holo_rwlock_wrlock(holo_rwlock_t *rwlock);
HOLO_API void holo_rwlock_unlock(holo_rwlock_t *rwlock);

/* Atomics */
HOLO_API int32_t holo_atomic_inc(volatile int32_t *val);
HOLO_API int32_t holo_atomic_dec(volatile int32_t *val);
HOLO_API int32_t holo_atomic_add(volatile int32_t *val, int32_t add);
HOLO_API bool holo_atomic_cas(volatile int32_t *val, int32_t expected, int32_t desired);

/* ============================================================================
 * File System
 * ============================================================================ */

typedef struct holo_file holo_file_t;
typedef struct holo_dir holo_dir_t;

/* File open modes */
#define HOLO_FILE_READ      0x01
#define HOLO_FILE_WRITE     0x02
#define HOLO_FILE_APPEND    0x04
#define HOLO_FILE_CREATE    0x08
#define HOLO_FILE_TRUNCATE  0x10
#define HOLO_FILE_BINARY    0x20

/* File operations */
HOLO_API holo_file_t *holo_file_open(const char *path, uint32_t mode);
HOLO_API void holo_file_close(holo_file_t *file);
HOLO_API size_t holo_file_read(holo_file_t *file, void *buf, size_t size);
HOLO_API size_t holo_file_write(holo_file_t *file, const void *buf, size_t size);
HOLO_API int64_t holo_file_seek(holo_file_t *file, int64_t offset, int whence);
HOLO_API int64_t holo_file_tell(holo_file_t *file);
HOLO_API int holo_file_flush(holo_file_t *file);
HOLO_API int64_t holo_file_size(holo_file_t *file);

/* File utilities */
HOLO_API bool holo_path_exists(const char *path);
HOLO_API bool holo_path_is_file(const char *path);
HOLO_API bool holo_path_is_dir(const char *path);
HOLO_API int holo_file_delete(const char *path);
HOLO_API int holo_file_rename(const char *old_path, const char *new_path);
HOLO_API int holo_file_copy(const char *src, const char *dst);
HOLO_API char *holo_file_read_all(const char *path, size_t *size);

/* Directory operations */
HOLO_API int holo_dir_create(const char *path);
HOLO_API int holo_dir_create_recursive(const char *path);
HOLO_API int holo_dir_delete(const char *path);
HOLO_API holo_dir_t *holo_dir_open(const char *path);
HOLO_API const char *holo_dir_next(holo_dir_t *dir);
HOLO_API void holo_dir_close(holo_dir_t *dir);

/* Path utilities */
HOLO_API char *holo_path_join(const char *base, const char *path);
HOLO_API char *holo_path_dirname(const char *path);
HOLO_API char *holo_path_basename(const char *path);
HOLO_API char *holo_path_extension(const char *path);
HOLO_API char *holo_path_absolute(const char *path);
HOLO_API char *holo_path_normalize(const char *path);

/* Standard directories */
HOLO_API char *holo_dir_home(void);
HOLO_API char *holo_dir_config(void);      /* ~/.config/holo or %APPDATA%\Holo */
HOLO_API char *holo_dir_data(void);        /* ~/.local/share/holo */
HOLO_API char *holo_dir_cache(void);       /* ~/.cache/holo */
HOLO_API char *holo_dir_temp(void);
HOLO_API char *holo_dir_current(void);
HOLO_API int holo_dir_set_current(const char *path);

/* ============================================================================
 * Networking
 * ============================================================================ */

typedef struct holo_socket holo_socket_t;

/* Socket types */
#define HOLO_SOCK_TCP       1
#define HOLO_SOCK_UDP       2

/* Socket options */
#define HOLO_SOCKOPT_NONBLOCK   0x01
#define HOLO_SOCKOPT_REUSEADDR  0x02
#define HOLO_SOCKOPT_KEEPALIVE  0x04
#define HOLO_SOCKOPT_NODELAY    0x08

/* Socket operations */
HOLO_API int holo_net_init(void);
HOLO_API void holo_net_shutdown(void);

HOLO_API holo_socket_t *holo_socket_create(int type);
HOLO_API void holo_socket_close(holo_socket_t *sock);
HOLO_API int holo_socket_set_option(holo_socket_t *sock, uint32_t option, int value);

HOLO_API int holo_socket_connect(holo_socket_t *sock, const char *host, uint16_t port);
HOLO_API int holo_socket_bind(holo_socket_t *sock, const char *host, uint16_t port);
HOLO_API int holo_socket_listen(holo_socket_t *sock, int backlog);
HOLO_API holo_socket_t *holo_socket_accept(holo_socket_t *sock, char *client_addr, size_t addr_len);

HOLO_API ssize_t holo_socket_send(holo_socket_t *sock, const void *buf, size_t len);
HOLO_API ssize_t holo_socket_recv(holo_socket_t *sock, void *buf, size_t len);
HOLO_API ssize_t holo_socket_sendto(holo_socket_t *sock, const void *buf, size_t len,
                                     const char *host, uint16_t port);
HOLO_API ssize_t holo_socket_recvfrom(holo_socket_t *sock, void *buf, size_t len,
                                       char *host, size_t host_len, uint16_t *port);

/* Polling */
typedef struct {
    holo_socket_t *sock;
    uint32_t events;      /* HOLO_POLL_READ, HOLO_POLL_WRITE */
    uint32_t revents;     /* Output: events that occurred */
} holo_pollfd_t;

#define HOLO_POLL_READ   0x01
#define HOLO_POLL_WRITE  0x02
#define HOLO_POLL_ERROR  0x04

HOLO_API int holo_socket_poll(holo_pollfd_t *fds, size_t nfds, int timeout_ms);

/* DNS */
HOLO_API int holo_dns_resolve(const char *hostname, char *ip_out, size_t ip_len);

/* ============================================================================
 * Time
 * ============================================================================ */

HOLO_API uint64_t holo_time_ms(void);       /* Milliseconds since epoch */
HOLO_API uint64_t holo_time_us(void);       /* Microseconds (monotonic) */
HOLO_API uint64_t holo_time_ns(void);       /* Nanoseconds (monotonic) */
HOLO_API void holo_sleep_ms(uint32_t ms);
HOLO_API void holo_sleep_us(uint32_t us);

typedef struct {
    int year;
    int month;      /* 1-12 */
    int day;        /* 1-31 */
    int hour;       /* 0-23 */
    int minute;     /* 0-59 */
    int second;     /* 0-59 */
    int ms;         /* 0-999 */
    int weekday;    /* 0=Sunday, 6=Saturday */
    int yearday;    /* 1-366 */
    int tz_offset;  /* UTC offset in minutes */
} holo_datetime_t;

HOLO_API void holo_time_now(holo_datetime_t *dt);
HOLO_API void holo_time_utc(holo_datetime_t *dt);
HOLO_API char *holo_time_format(const holo_datetime_t *dt, const char *fmt);
HOLO_API uint64_t holo_time_to_epoch(const holo_datetime_t *dt);
HOLO_API void holo_time_from_epoch(uint64_t epoch_ms, holo_datetime_t *dt);

/* ============================================================================
 * Terminal / Console
 * ============================================================================ */

HOLO_API int holo_term_width(void);
HOLO_API int holo_term_height(void);
HOLO_API bool holo_term_is_tty(int fd);     /* 0=stdin, 1=stdout, 2=stderr */
HOLO_API bool holo_term_supports_color(void);
HOLO_API bool holo_term_supports_unicode(void);

/* Raw mode for interactive input */
HOLO_API int holo_term_set_raw(bool enable);
HOLO_API int holo_term_read_char(void);     /* Returns -1 if no input */
HOLO_API int holo_term_read_key(void);      /* Returns key code (handles escape sequences) */

/* Key codes */
#define HOLO_KEY_UP       1000
#define HOLO_KEY_DOWN     1001
#define HOLO_KEY_LEFT     1002
#define HOLO_KEY_RIGHT    1003
#define HOLO_KEY_HOME     1004
#define HOLO_KEY_END      1005
#define HOLO_KEY_PGUP     1006
#define HOLO_KEY_PGDN     1007
#define HOLO_KEY_INSERT   1008
#define HOLO_KEY_DELETE   1009
#define HOLO_KEY_BACKSPACE 127
#define HOLO_KEY_TAB      9
#define HOLO_KEY_ENTER    10
#define HOLO_KEY_ESCAPE   27

/* ============================================================================
 * Process / Execution
 * ============================================================================ */

typedef struct holo_process holo_process_t;

HOLO_API holo_process_t *holo_process_spawn(const char *cmd, const char **args);
HOLO_API int holo_process_wait(holo_process_t *proc);
HOLO_API int holo_process_kill(holo_process_t *proc);
HOLO_API bool holo_process_running(holo_process_t *proc);
HOLO_API ssize_t holo_process_read(holo_process_t *proc, char *buf, size_t len);
HOLO_API ssize_t holo_process_write(holo_process_t *proc, const char *buf, size_t len);
HOLO_API void holo_process_close(holo_process_t *proc);

/* Simple command execution */
HOLO_API int holo_exec(const char *cmd, char **output, size_t *output_len);

/* Environment variables */
HOLO_API char *holo_getenv(const char *name);
HOLO_API int holo_setenv(const char *name, const char *value);
HOLO_API int holo_unsetenv(const char *name);

/* ============================================================================
 * Dynamic Libraries
 * ============================================================================ */

typedef struct holo_lib holo_lib_t;

HOLO_API holo_lib_t *holo_lib_load(const char *path);
HOLO_API void *holo_lib_symbol(holo_lib_t *lib, const char *name);
HOLO_API void holo_lib_unload(holo_lib_t *lib);
HOLO_API const char *holo_lib_error(void);

/* ============================================================================
 * Error Handling
 * ============================================================================ */

HOLO_API int holo_get_last_error(void);
HOLO_API const char *holo_get_error_string(int error);

/* Common error codes */
#define HOLO_OK             0
#define HOLO_ERR_NOMEM      -1
#define HOLO_ERR_IO         -2
#define HOLO_ERR_NOTFOUND   -3
#define HOLO_ERR_PERM       -4
#define HOLO_ERR_BUSY       -5
#define HOLO_ERR_TIMEOUT    -6
#define HOLO_ERR_INVAL      -7
#define HOLO_ERR_NET        -8
#define HOLO_ERR_CONNRESET  -9
#define HOLO_ERR_CONNREFUSED -10

/* ============================================================================
 * Initialization
 * ============================================================================ */

HOLO_API int holo_pal_init(void);
HOLO_API void holo_pal_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* HOLO_PAL_H */
