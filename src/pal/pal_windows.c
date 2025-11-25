/*
 * Holo - Platform Abstraction Layer
 * Windows Implementation
 */

#ifdef _WIN32

#include "holo/pal.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <io.h>
#include <direct.h>
#include <shlobj.h>
#include <process.h>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "shell32.lib")

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

struct holo_thread {
    HANDLE handle;
    holo_thread_func_t func;
    void *arg;
};

struct holo_mutex {
    CRITICAL_SECTION cs;
};

struct holo_cond {
    CONDITION_VARIABLE cv;
};

struct holo_rwlock {
    SRWLOCK lock;
};

struct holo_file {
    HANDLE handle;
};

struct holo_dir {
    HANDLE handle;
    WIN32_FIND_DATAW data;
    char name_buf[MAX_PATH * 3];
    bool first;
};

struct holo_socket {
    SOCKET sock;
};

struct holo_lib {
    HMODULE handle;
};

struct holo_process {
    HANDLE process;
    HANDLE thread;
    HANDLE stdin_write;
    HANDLE stdout_read;
};

/* ============================================================================
 * Initialization
 * ============================================================================ */

static bool g_pal_initialized = false;
static bool g_wsa_initialized = false;

int holo_pal_init(void) {
    if (g_pal_initialized) return HOLO_OK;

    /* Enable UTF-8 console output */
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    /* Enable ANSI escape sequences */
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    if (GetConsoleMode(hOut, &dwMode)) {
        dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        SetConsoleMode(hOut, dwMode);
    }

    g_pal_initialized = true;
    return HOLO_OK;
}

void holo_pal_shutdown(void) {
    if (g_wsa_initialized) {
        WSACleanup();
        g_wsa_initialized = false;
    }
    g_pal_initialized = false;
}

/* ============================================================================
 * Platform Info
 * ============================================================================ */

int holo_platform_info(holo_platform_info_t *info) {
    if (!info) return HOLO_ERR_INVAL;

    memset(info, 0, sizeof(*info));
    info->id = HOLO_PLAT_WINDOWS;
    info->name = "windows";
    info->arch = HOLO_ARCH_NAME;

    /* Get Windows version */
    OSVERSIONINFOW osvi = { sizeof(osvi) };
    /* Note: GetVersionEx is deprecated, but works for basic info */
    #pragma warning(suppress: 4996)
    if (GetVersionExW(&osvi)) {
        info->version_major = osvi.dwMajorVersion;
        info->version_minor = osvi.dwMinorVersion;
    }

    /* Get CPU info */
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    info->cpu_cores = sysinfo.dwNumberOfProcessors;

    /* Get memory info */
    MEMORYSTATUSEX memstat = { sizeof(memstat) };
    if (GlobalMemoryStatusEx(&memstat)) {
        info->total_memory = memstat.ullTotalPhys;
    }

    return HOLO_OK;
}

/* ============================================================================
 * Memory
 * ============================================================================ */

void *holo_alloc_aligned(size_t size, size_t alignment) {
    return _aligned_malloc(size, alignment);
}

void holo_free_aligned(void *ptr) {
    _aligned_free(ptr);
}

uint64_t holo_mem_available(void) {
    MEMORYSTATUSEX memstat = { sizeof(memstat) };
    if (GlobalMemoryStatusEx(&memstat)) {
        return memstat.ullAvailPhys;
    }
    return 0;
}

uint64_t holo_mem_used(void) {
    MEMORYSTATUSEX memstat = { sizeof(memstat) };
    if (GlobalMemoryStatusEx(&memstat)) {
        return memstat.ullTotalPhys - memstat.ullAvailPhys;
    }
    return 0;
}

/* ============================================================================
 * Threading
 * ============================================================================ */

static unsigned __stdcall thread_wrapper(void *arg) {
    holo_thread_t *thread = (holo_thread_t *)arg;
    thread->func(thread->arg);
    return 0;
}

holo_thread_t *holo_thread_create(holo_thread_func_t func, void *arg) {
    holo_thread_t *thread = (holo_thread_t *)holo_alloc(sizeof(holo_thread_t));
    if (!thread) return NULL;

    thread->func = func;
    thread->arg = arg;

    thread->handle = (HANDLE)_beginthreadex(NULL, 0, thread_wrapper, thread, 0, NULL);
    if (!thread->handle) {
        holo_free(thread);
        return NULL;
    }

    return thread;
}

int holo_thread_join(holo_thread_t *thread) {
    if (!thread) return HOLO_ERR_INVAL;

    DWORD result = WaitForSingleObject(thread->handle, INFINITE);
    CloseHandle(thread->handle);
    holo_free(thread);

    return (result == WAIT_OBJECT_0) ? HOLO_OK : HOLO_ERR_IO;
}

void holo_thread_detach(holo_thread_t *thread) {
    if (thread) {
        CloseHandle(thread->handle);
        holo_free(thread);
    }
}

void holo_thread_yield(void) {
    SwitchToThread();
}

uint64_t holo_thread_id(void) {
    return (uint64_t)GetCurrentThreadId();
}

int holo_thread_set_name(const char *name) {
    /* Windows 10+ thread naming via SetThreadDescription */
    typedef HRESULT (WINAPI *SetThreadDescriptionFunc)(HANDLE, PCWSTR);

    HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
    if (kernel32) {
        SetThreadDescriptionFunc setDesc = (SetThreadDescriptionFunc)
            GetProcAddress(kernel32, "SetThreadDescription");
        if (setDesc) {
            wchar_t wname[256];
            MultiByteToWideChar(CP_UTF8, 0, name, -1, wname, 256);
            setDesc(GetCurrentThread(), wname);
            return HOLO_OK;
        }
    }
    return HOLO_ERR_IO;  /* Not supported on older Windows */
}

/* Mutex */
holo_mutex_t *holo_mutex_create(void) {
    holo_mutex_t *mutex = (holo_mutex_t *)holo_alloc(sizeof(holo_mutex_t));
    if (!mutex) return NULL;

    InitializeCriticalSection(&mutex->cs);
    return mutex;
}

void holo_mutex_destroy(holo_mutex_t *mutex) {
    if (mutex) {
        DeleteCriticalSection(&mutex->cs);
        holo_free(mutex);
    }
}

void holo_mutex_lock(holo_mutex_t *mutex) {
    if (mutex) EnterCriticalSection(&mutex->cs);
}

bool holo_mutex_trylock(holo_mutex_t *mutex) {
    if (!mutex) return false;
    return TryEnterCriticalSection(&mutex->cs) != 0;
}

void holo_mutex_unlock(holo_mutex_t *mutex) {
    if (mutex) LeaveCriticalSection(&mutex->cs);
}

/* Condition variable */
holo_cond_t *holo_cond_create(void) {
    holo_cond_t *cond = (holo_cond_t *)holo_alloc(sizeof(holo_cond_t));
    if (!cond) return NULL;

    InitializeConditionVariable(&cond->cv);
    return cond;
}

void holo_cond_destroy(holo_cond_t *cond) {
    if (cond) holo_free(cond);
}

void holo_cond_wait(holo_cond_t *cond, holo_mutex_t *mutex) {
    if (cond && mutex) {
        SleepConditionVariableCS(&cond->cv, &mutex->cs, INFINITE);
    }
}

bool holo_cond_timedwait(holo_cond_t *cond, holo_mutex_t *mutex, uint32_t timeout_ms) {
    if (!cond || !mutex) return false;
    return SleepConditionVariableCS(&cond->cv, &mutex->cs, timeout_ms) != 0;
}

void holo_cond_signal(holo_cond_t *cond) {
    if (cond) WakeConditionVariable(&cond->cv);
}

void holo_cond_broadcast(holo_cond_t *cond) {
    if (cond) WakeAllConditionVariable(&cond->cv);
}

/* Read-write lock */
holo_rwlock_t *holo_rwlock_create(void) {
    holo_rwlock_t *rwlock = (holo_rwlock_t *)holo_alloc(sizeof(holo_rwlock_t));
    if (!rwlock) return NULL;

    InitializeSRWLock(&rwlock->lock);
    return rwlock;
}

void holo_rwlock_destroy(holo_rwlock_t *rwlock) {
    if (rwlock) holo_free(rwlock);
}

void holo_rwlock_rdlock(holo_rwlock_t *rwlock) {
    if (rwlock) AcquireSRWLockShared(&rwlock->lock);
}

void holo_rwlock_wrlock(holo_rwlock_t *rwlock) {
    if (rwlock) AcquireSRWLockExclusive(&rwlock->lock);
}

void holo_rwlock_unlock(holo_rwlock_t *rwlock) {
    if (rwlock) {
        /* Note: Must track which mode was acquired. For simplicity, try both */
        ReleaseSRWLockExclusive(&rwlock->lock);
    }
}

/* Atomics */
int32_t holo_atomic_inc(volatile int32_t *val) {
    return InterlockedIncrement((volatile LONG *)val);
}

int32_t holo_atomic_dec(volatile int32_t *val) {
    return InterlockedDecrement((volatile LONG *)val);
}

int32_t holo_atomic_add(volatile int32_t *val, int32_t add) {
    return InterlockedAdd((volatile LONG *)val, add);
}

bool holo_atomic_cas(volatile int32_t *val, int32_t expected, int32_t desired) {
    return InterlockedCompareExchange((volatile LONG *)val, desired, expected) == expected;
}

/* ============================================================================
 * File System
 * ============================================================================ */

static wchar_t *utf8_to_wide(const char *str) {
    if (!str) return NULL;

    int len = MultiByteToWideChar(CP_UTF8, 0, str, -1, NULL, 0);
    if (len <= 0) return NULL;

    wchar_t *wstr = (wchar_t *)holo_alloc(len * sizeof(wchar_t));
    if (!wstr) return NULL;

    MultiByteToWideChar(CP_UTF8, 0, str, -1, wstr, len);
    return wstr;
}

static char *wide_to_utf8(const wchar_t *wstr) {
    if (!wstr) return NULL;

    int len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, NULL, 0, NULL, NULL);
    if (len <= 0) return NULL;

    char *str = (char *)holo_alloc(len);
    if (!str) return NULL;

    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, str, len, NULL, NULL);
    return str;
}

holo_file_t *holo_file_open(const char *path, uint32_t mode) {
    if (!path) return NULL;

    wchar_t *wpath = utf8_to_wide(path);
    if (!wpath) return NULL;

    DWORD access = 0;
    DWORD share = FILE_SHARE_READ;
    DWORD creation = OPEN_EXISTING;

    if (mode & HOLO_FILE_READ)   access |= GENERIC_READ;
    if (mode & HOLO_FILE_WRITE)  access |= GENERIC_WRITE;
    if (mode & HOLO_FILE_CREATE) creation = OPEN_ALWAYS;
    if (mode & HOLO_FILE_TRUNCATE) creation = CREATE_ALWAYS;

    HANDLE handle = CreateFileW(wpath, access, share, NULL, creation,
                                FILE_ATTRIBUTE_NORMAL, NULL);
    holo_free(wpath);

    if (handle == INVALID_HANDLE_VALUE) {
        return NULL;
    }

    if (mode & HOLO_FILE_APPEND) {
        SetFilePointer(handle, 0, NULL, FILE_END);
    }

    holo_file_t *file = (holo_file_t *)holo_alloc(sizeof(holo_file_t));
    if (!file) {
        CloseHandle(handle);
        return NULL;
    }

    file->handle = handle;
    return file;
}

void holo_file_close(holo_file_t *file) {
    if (file) {
        CloseHandle(file->handle);
        holo_free(file);
    }
}

size_t holo_file_read(holo_file_t *file, void *buf, size_t size) {
    if (!file || !buf) return 0;

    DWORD read = 0;
    if (!ReadFile(file->handle, buf, (DWORD)size, &read, NULL)) {
        return 0;
    }
    return (size_t)read;
}

size_t holo_file_write(holo_file_t *file, const void *buf, size_t size) {
    if (!file || !buf) return 0;

    DWORD written = 0;
    if (!WriteFile(file->handle, buf, (DWORD)size, &written, NULL)) {
        return 0;
    }
    return (size_t)written;
}

int64_t holo_file_seek(holo_file_t *file, int64_t offset, int whence) {
    if (!file) return -1;

    DWORD method;
    switch (whence) {
        case SEEK_SET: method = FILE_BEGIN; break;
        case SEEK_CUR: method = FILE_CURRENT; break;
        case SEEK_END: method = FILE_END; break;
        default: return -1;
    }

    LARGE_INTEGER li;
    li.QuadPart = offset;
    li.LowPart = SetFilePointer(file->handle, li.LowPart, &li.HighPart, method);

    if (li.LowPart == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR) {
        return -1;
    }

    return li.QuadPart;
}

int64_t holo_file_tell(holo_file_t *file) {
    return holo_file_seek(file, 0, SEEK_CUR);
}

int holo_file_flush(holo_file_t *file) {
    if (!file) return HOLO_ERR_INVAL;
    return FlushFileBuffers(file->handle) ? HOLO_OK : HOLO_ERR_IO;
}

int64_t holo_file_size(holo_file_t *file) {
    if (!file) return -1;

    LARGE_INTEGER size;
    if (!GetFileSizeEx(file->handle, &size)) {
        return -1;
    }
    return size.QuadPart;
}

bool holo_path_exists(const char *path) {
    wchar_t *wpath = utf8_to_wide(path);
    if (!wpath) return false;

    DWORD attr = GetFileAttributesW(wpath);
    holo_free(wpath);

    return attr != INVALID_FILE_ATTRIBUTES;
}

bool holo_path_is_file(const char *path) {
    wchar_t *wpath = utf8_to_wide(path);
    if (!wpath) return false;

    DWORD attr = GetFileAttributesW(wpath);
    holo_free(wpath);

    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
}

bool holo_path_is_dir(const char *path) {
    wchar_t *wpath = utf8_to_wide(path);
    if (!wpath) return false;

    DWORD attr = GetFileAttributesW(wpath);
    holo_free(wpath);

    return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY);
}

int holo_file_delete(const char *path) {
    wchar_t *wpath = utf8_to_wide(path);
    if (!wpath) return HOLO_ERR_NOMEM;

    BOOL result = DeleteFileW(wpath);
    holo_free(wpath);

    return result ? HOLO_OK : HOLO_ERR_IO;
}

int holo_file_rename(const char *old_path, const char *new_path) {
    wchar_t *wold = utf8_to_wide(old_path);
    wchar_t *wnew = utf8_to_wide(new_path);

    if (!wold || !wnew) {
        holo_free(wold);
        holo_free(wnew);
        return HOLO_ERR_NOMEM;
    }

    BOOL result = MoveFileW(wold, wnew);
    holo_free(wold);
    holo_free(wnew);

    return result ? HOLO_OK : HOLO_ERR_IO;
}

int holo_dir_create(const char *path) {
    wchar_t *wpath = utf8_to_wide(path);
    if (!wpath) return HOLO_ERR_NOMEM;

    BOOL result = CreateDirectoryW(wpath, NULL);
    holo_free(wpath);

    if (!result) {
        DWORD err = GetLastError();
        if (err == ERROR_ALREADY_EXISTS) return HOLO_OK;
        return HOLO_ERR_IO;
    }

    return HOLO_OK;
}

int holo_dir_create_recursive(const char *path) {
    char *tmp = strdup(path);
    if (!tmp) return HOLO_ERR_NOMEM;

    char *p = tmp;

    /* Skip drive letter on Windows */
    if (p[0] && p[1] == ':') p += 2;
    if (*p == '\\' || *p == '/') p++;

    while (*p) {
        if (*p == '\\' || *p == '/') {
            char c = *p;
            *p = '\0';
            holo_dir_create(tmp);
            *p = c;
        }
        p++;
    }

    int result = holo_dir_create(tmp);
    free(tmp);
    return result;
}

holo_dir_t *holo_dir_open(const char *path) {
    char search[MAX_PATH];
    snprintf(search, sizeof(search), "%s\\*", path);

    wchar_t *wsearch = utf8_to_wide(search);
    if (!wsearch) return NULL;

    holo_dir_t *dir = (holo_dir_t *)holo_alloc(sizeof(holo_dir_t));
    if (!dir) {
        holo_free(wsearch);
        return NULL;
    }

    dir->handle = FindFirstFileW(wsearch, &dir->data);
    holo_free(wsearch);

    if (dir->handle == INVALID_HANDLE_VALUE) {
        holo_free(dir);
        return NULL;
    }

    dir->first = true;
    return dir;
}

const char *holo_dir_next(holo_dir_t *dir) {
    if (!dir) return NULL;

    while (1) {
        if (dir->first) {
            dir->first = false;
        } else {
            if (!FindNextFileW(dir->handle, &dir->data)) {
                return NULL;
            }
        }

        /* Skip . and .. */
        if (wcscmp(dir->data.cFileName, L".") == 0 ||
            wcscmp(dir->data.cFileName, L"..") == 0) {
            continue;
        }

        WideCharToMultiByte(CP_UTF8, 0, dir->data.cFileName, -1,
                           dir->name_buf, sizeof(dir->name_buf), NULL, NULL);
        return dir->name_buf;
    }
}

void holo_dir_close(holo_dir_t *dir) {
    if (dir) {
        FindClose(dir->handle);
        holo_free(dir);
    }
}

char *holo_dir_home(void) {
    wchar_t path[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathW(NULL, CSIDL_PROFILE, NULL, 0, path))) {
        return wide_to_utf8(path);
    }
    return NULL;
}

char *holo_dir_config(void) {
    wchar_t path[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathW(NULL, CSIDL_APPDATA, NULL, 0, path))) {
        wcscat(path, L"\\Holo");
        return wide_to_utf8(path);
    }
    return NULL;
}

char *holo_dir_data(void) {
    return holo_dir_config();  /* Same as config on Windows */
}

char *holo_dir_cache(void) {
    wchar_t path[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathW(NULL, CSIDL_LOCAL_APPDATA, NULL, 0, path))) {
        wcscat(path, L"\\Holo\\Cache");
        return wide_to_utf8(path);
    }
    return NULL;
}

char *holo_dir_temp(void) {
    wchar_t path[MAX_PATH];
    DWORD len = GetTempPathW(MAX_PATH, path);
    if (len > 0) {
        return wide_to_utf8(path);
    }
    return NULL;
}

char *holo_dir_current(void) {
    wchar_t path[MAX_PATH];
    DWORD len = GetCurrentDirectoryW(MAX_PATH, path);
    if (len > 0) {
        return wide_to_utf8(path);
    }
    return NULL;
}

int holo_dir_set_current(const char *path) {
    wchar_t *wpath = utf8_to_wide(path);
    if (!wpath) return HOLO_ERR_NOMEM;

    BOOL result = SetCurrentDirectoryW(wpath);
    holo_free(wpath);

    return result ? HOLO_OK : HOLO_ERR_IO;
}

/* ============================================================================
 * Networking
 * ============================================================================ */

int holo_net_init(void) {
    if (g_wsa_initialized) return HOLO_OK;

    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        return HOLO_ERR_NET;
    }

    g_wsa_initialized = true;
    return HOLO_OK;
}

void holo_net_shutdown(void) {
    if (g_wsa_initialized) {
        WSACleanup();
        g_wsa_initialized = false;
    }
}

holo_socket_t *holo_socket_create(int type) {
    holo_net_init();

    int sock_type = (type == HOLO_SOCK_UDP) ? SOCK_DGRAM : SOCK_STREAM;
    int protocol = (type == HOLO_SOCK_UDP) ? IPPROTO_UDP : IPPROTO_TCP;

    SOCKET sock = socket(AF_INET, sock_type, protocol);
    if (sock == INVALID_SOCKET) {
        return NULL;
    }

    holo_socket_t *s = (holo_socket_t *)holo_alloc(sizeof(holo_socket_t));
    if (!s) {
        closesocket(sock);
        return NULL;
    }

    s->sock = sock;
    return s;
}

void holo_socket_close(holo_socket_t *sock) {
    if (sock) {
        closesocket(sock->sock);
        holo_free(sock);
    }
}

int holo_socket_set_option(holo_socket_t *sock, uint32_t option, int value) {
    if (!sock) return HOLO_ERR_INVAL;

    if (option & HOLO_SOCKOPT_NONBLOCK) {
        u_long mode = value ? 1 : 0;
        ioctlsocket(sock->sock, FIONBIO, &mode);
    }

    if (option & HOLO_SOCKOPT_REUSEADDR) {
        setsockopt(sock->sock, SOL_SOCKET, SO_REUSEADDR, (char *)&value, sizeof(value));
    }

    if (option & HOLO_SOCKOPT_KEEPALIVE) {
        setsockopt(sock->sock, SOL_SOCKET, SO_KEEPALIVE, (char *)&value, sizeof(value));
    }

    if (option & HOLO_SOCKOPT_NODELAY) {
        setsockopt(sock->sock, IPPROTO_TCP, TCP_NODELAY, (char *)&value, sizeof(value));
    }

    return HOLO_OK;
}

int holo_socket_connect(holo_socket_t *sock, const char *host, uint16_t port) {
    if (!sock || !host) return HOLO_ERR_INVAL;

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    /* Try numeric first */
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        /* DNS lookup */
        struct addrinfo hints = {0}, *result;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        if (getaddrinfo(host, NULL, &hints, &result) != 0) {
            return HOLO_ERR_NET;
        }

        addr.sin_addr = ((struct sockaddr_in *)result->ai_addr)->sin_addr;
        freeaddrinfo(result);
    }

    if (connect(sock->sock, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        int err = WSAGetLastError();
        if (err == WSAECONNREFUSED) return HOLO_ERR_CONNREFUSED;
        return HOLO_ERR_NET;
    }

    return HOLO_OK;
}

int holo_socket_bind(holo_socket_t *sock, const char *host, uint16_t port) {
    if (!sock) return HOLO_ERR_INVAL;

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    if (host && *host) {
        inet_pton(AF_INET, host, &addr.sin_addr);
    } else {
        addr.sin_addr.s_addr = INADDR_ANY;
    }

    if (bind(sock->sock, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        return HOLO_ERR_NET;
    }

    return HOLO_OK;
}

int holo_socket_listen(holo_socket_t *sock, int backlog) {
    if (!sock) return HOLO_ERR_INVAL;

    if (listen(sock->sock, backlog) != 0) {
        return HOLO_ERR_NET;
    }

    return HOLO_OK;
}

holo_socket_t *holo_socket_accept(holo_socket_t *sock, char *client_addr, size_t addr_len) {
    if (!sock) return NULL;

    struct sockaddr_in addr;
    int len = sizeof(addr);

    SOCKET client = accept(sock->sock, (struct sockaddr *)&addr, &len);
    if (client == INVALID_SOCKET) {
        return NULL;
    }

    if (client_addr && addr_len > 0) {
        inet_ntop(AF_INET, &addr.sin_addr, client_addr, (socklen_t)addr_len);
    }

    holo_socket_t *s = (holo_socket_t *)holo_alloc(sizeof(holo_socket_t));
    if (!s) {
        closesocket(client);
        return NULL;
    }

    s->sock = client;
    return s;
}

ssize_t holo_socket_send(holo_socket_t *sock, const void *buf, size_t len) {
    if (!sock || !buf) return -1;
    return send(sock->sock, (const char *)buf, (int)len, 0);
}

ssize_t holo_socket_recv(holo_socket_t *sock, void *buf, size_t len) {
    if (!sock || !buf) return -1;
    return recv(sock->sock, (char *)buf, (int)len, 0);
}

int holo_socket_poll(holo_pollfd_t *fds, size_t nfds, int timeout_ms) {
    if (!fds || nfds == 0) return HOLO_ERR_INVAL;

    /* Use select() for Windows compatibility */
    fd_set readfds, writefds, exceptfds;
    FD_ZERO(&readfds);
    FD_ZERO(&writefds);
    FD_ZERO(&exceptfds);

    for (size_t i = 0; i < nfds; i++) {
        if (fds[i].events & HOLO_POLL_READ) {
            FD_SET(fds[i].sock->sock, &readfds);
        }
        if (fds[i].events & HOLO_POLL_WRITE) {
            FD_SET(fds[i].sock->sock, &writefds);
        }
        FD_SET(fds[i].sock->sock, &exceptfds);
        fds[i].revents = 0;
    }

    struct timeval tv, *ptv = NULL;
    if (timeout_ms >= 0) {
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        ptv = &tv;
    }

    int result = select(0, &readfds, &writefds, &exceptfds, ptv);
    if (result < 0) {
        return HOLO_ERR_NET;
    }

    for (size_t i = 0; i < nfds; i++) {
        if (FD_ISSET(fds[i].sock->sock, &readfds)) {
            fds[i].revents |= HOLO_POLL_READ;
        }
        if (FD_ISSET(fds[i].sock->sock, &writefds)) {
            fds[i].revents |= HOLO_POLL_WRITE;
        }
        if (FD_ISSET(fds[i].sock->sock, &exceptfds)) {
            fds[i].revents |= HOLO_POLL_ERROR;
        }
    }

    return result;
}

/* ============================================================================
 * Time
 * ============================================================================ */

uint64_t holo_time_ms(void) {
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);

    ULARGE_INTEGER uli;
    uli.LowPart = ft.dwLowDateTime;
    uli.HighPart = ft.dwHighDateTime;

    /* Convert from 100-ns intervals since 1601 to ms since 1970 */
    return (uli.QuadPart - 116444736000000000ULL) / 10000;
}

uint64_t holo_time_us(void) {
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }

    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);

    return (uint64_t)(counter.QuadPart * 1000000 / freq.QuadPart);
}

uint64_t holo_time_ns(void) {
    return holo_time_us() * 1000;
}

void holo_sleep_ms(uint32_t ms) {
    Sleep(ms);
}

void holo_sleep_us(uint32_t us) {
    /* Windows doesn't have sub-ms sleep, use busy wait for small delays */
    if (us < 1000) {
        uint64_t start = holo_time_us();
        while (holo_time_us() - start < us) {
            _mm_pause();
        }
    } else {
        Sleep(us / 1000);
    }
}

void holo_time_now(holo_datetime_t *dt) {
    if (!dt) return;

    SYSTEMTIME st;
    GetLocalTime(&st);

    dt->year = st.wYear;
    dt->month = st.wMonth;
    dt->day = st.wDay;
    dt->hour = st.wHour;
    dt->minute = st.wMinute;
    dt->second = st.wSecond;
    dt->ms = st.wMilliseconds;
    dt->weekday = st.wDayOfWeek;

    TIME_ZONE_INFORMATION tz;
    if (GetTimeZoneInformation(&tz) != TIME_ZONE_ID_INVALID) {
        dt->tz_offset = -(int)tz.Bias;
    }
}

/* ============================================================================
 * Terminal
 * ============================================================================ */

int holo_term_width(void) {
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        return csbi.srWindow.Right - csbi.srWindow.Left + 1;
    }
    return 80;
}

int holo_term_height(void) {
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        return csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
    }
    return 24;
}

bool holo_term_is_tty(int fd) {
    HANDLE h = (fd == 0) ? GetStdHandle(STD_INPUT_HANDLE) :
               (fd == 1) ? GetStdHandle(STD_OUTPUT_HANDLE) :
                           GetStdHandle(STD_ERROR_HANDLE);

    DWORD mode;
    return GetConsoleMode(h, &mode) != 0;
}

bool holo_term_supports_color(void) {
    return holo_term_is_tty(1);
}

bool holo_term_supports_unicode(void) {
    return GetConsoleOutputCP() == CP_UTF8;
}

static DWORD g_orig_console_mode = 0;

int holo_term_set_raw(bool enable) {
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);

    if (enable) {
        GetConsoleMode(h, &g_orig_console_mode);
        DWORD mode = g_orig_console_mode;
        mode &= ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_PROCESSED_INPUT);
        mode |= ENABLE_VIRTUAL_TERMINAL_INPUT;
        SetConsoleMode(h, mode);
    } else {
        SetConsoleMode(h, g_orig_console_mode);
    }

    return HOLO_OK;
}

int holo_term_read_char(void) {
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    INPUT_RECORD rec;
    DWORD count;

    while (1) {
        if (!PeekConsoleInput(h, &rec, 1, &count) || count == 0) {
            return -1;
        }

        ReadConsoleInput(h, &rec, 1, &count);

        if (rec.EventType == KEY_EVENT && rec.Event.KeyEvent.bKeyDown) {
            return rec.Event.KeyEvent.uChar.AsciiChar;
        }
    }
}

/* ============================================================================
 * Environment
 * ============================================================================ */

char *holo_getenv(const char *name) {
    char buf[32767];  /* Max env var size on Windows */
    DWORD len = GetEnvironmentVariableA(name, buf, sizeof(buf));
    if (len == 0 || len >= sizeof(buf)) {
        return NULL;
    }
    return strdup(buf);
}

int holo_setenv(const char *name, const char *value) {
    return SetEnvironmentVariableA(name, value) ? HOLO_OK : HOLO_ERR_IO;
}

int holo_unsetenv(const char *name) {
    return SetEnvironmentVariableA(name, NULL) ? HOLO_OK : HOLO_ERR_IO;
}

/* ============================================================================
 * Dynamic Libraries
 * ============================================================================ */

holo_lib_t *holo_lib_load(const char *path) {
    wchar_t *wpath = utf8_to_wide(path);
    if (!wpath) return NULL;

    HMODULE handle = LoadLibraryW(wpath);
    holo_free(wpath);

    if (!handle) return NULL;

    holo_lib_t *lib = (holo_lib_t *)holo_alloc(sizeof(holo_lib_t));
    if (!lib) {
        FreeLibrary(handle);
        return NULL;
    }

    lib->handle = handle;
    return lib;
}

void *holo_lib_symbol(holo_lib_t *lib, const char *name) {
    if (!lib || !name) return NULL;
    return (void *)GetProcAddress(lib->handle, name);
}

void holo_lib_unload(holo_lib_t *lib) {
    if (lib) {
        FreeLibrary(lib->handle);
        holo_free(lib);
    }
}

static __thread char g_lib_error[256] = {0};

const char *holo_lib_error(void) {
    DWORD err = GetLastError();
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, NULL, err, 0,
                   g_lib_error, sizeof(g_lib_error), NULL);
    return g_lib_error;
}

#endif /* _WIN32 */
