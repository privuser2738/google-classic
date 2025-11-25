/*
 * Holo - Platform Abstraction Layer
 * Unix/POSIX Implementation (Linux, macOS)
 */

#if defined(__linux__) || defined(__APPLE__)

#include "holo/pal.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <dirent.h>
#include <dlfcn.h>
#include <pthread.h>
#include <termios.h>
#include <poll.h>
#include <time.h>
#include <pwd.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#else
#include <sys/sysinfo.h>
#endif

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

struct holo_thread {
    pthread_t handle;
    holo_thread_func_t func;
    void *arg;
};

struct holo_mutex {
    pthread_mutex_t mutex;
};

struct holo_cond {
    pthread_cond_t cond;
};

struct holo_rwlock {
    pthread_rwlock_t rwlock;
};

struct holo_file {
    int fd;
};

struct holo_dir {
    DIR *dir;
    struct dirent *entry;
};

struct holo_socket {
    int fd;
};

struct holo_lib {
    void *handle;
};

struct holo_process {
    pid_t pid;
    int stdin_fd;
    int stdout_fd;
};

/* ============================================================================
 * Initialization
 * ============================================================================ */

static bool g_pal_initialized = false;

int holo_pal_init(void) {
    if (g_pal_initialized) return HOLO_OK;

    /* Ignore SIGPIPE to handle broken pipes gracefully */
    signal(SIGPIPE, SIG_IGN);

    g_pal_initialized = true;
    return HOLO_OK;
}

void holo_pal_shutdown(void) {
    g_pal_initialized = false;
}

/* ============================================================================
 * Platform Info
 * ============================================================================ */

int holo_platform_info(holo_platform_info_t *info) {
    if (!info) return HOLO_ERR_INVAL;

    memset(info, 0, sizeof(*info));

#ifdef __APPLE__
    info->id = HOLO_PLAT_MACOS;
    info->name = "macos";

    /* Get macOS version */
    char version[64];
    size_t len = sizeof(version);
    if (sysctlbyname("kern.osproductversion", version, &len, NULL, 0) == 0) {
        sscanf(version, "%d.%d", &info->version_major, &info->version_minor);
    }

    /* CPU cores */
    int cores;
    len = sizeof(cores);
    if (sysctlbyname("hw.ncpu", &cores, &len, NULL, 0) == 0) {
        info->cpu_cores = cores;
    }

    /* Memory */
    int64_t mem;
    len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, NULL, 0) == 0) {
        info->total_memory = (uint64_t)mem;
    }
#else
    info->id = HOLO_PLAT_LINUX;
    info->name = "linux";

    /* Get kernel version */
    struct utsname uts;
    if (uname(&uts) == 0) {
        sscanf(uts.release, "%d.%d", &info->version_major, &info->version_minor);
    }

    /* CPU cores */
    info->cpu_cores = sysconf(_SC_NPROCESSORS_ONLN);

    /* Memory */
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        info->total_memory = si.totalram * si.mem_unit;
    }
#endif

    info->arch = HOLO_ARCH_NAME;
    return HOLO_OK;
}

/* ============================================================================
 * Memory
 * ============================================================================ */

void *holo_alloc_aligned(size_t size, size_t alignment) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

void holo_free_aligned(void *ptr) {
    free(ptr);
}

uint64_t holo_mem_available(void) {
#ifdef __APPLE__
    vm_size_t page_size;
    mach_port_t mach_port = mach_host_self();
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = sizeof(vm_stats) / sizeof(natural_t);

    if (host_page_size(mach_port, &page_size) == KERN_SUCCESS &&
        host_statistics64(mach_port, HOST_VM_INFO64, (host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
        return (uint64_t)vm_stats.free_count * page_size;
    }
    return 0;
#else
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        return si.freeram * si.mem_unit;
    }
    return 0;
#endif
}

uint64_t holo_mem_used(void) {
#ifdef __APPLE__
    vm_size_t page_size;
    mach_port_t mach_port = mach_host_self();
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = sizeof(vm_stats) / sizeof(natural_t);

    if (host_page_size(mach_port, &page_size) == KERN_SUCCESS &&
        host_statistics64(mach_port, HOST_VM_INFO64, (host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
        return (uint64_t)(vm_stats.active_count + vm_stats.wire_count) * page_size;
    }
    return 0;
#else
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        return (si.totalram - si.freeram) * si.mem_unit;
    }
    return 0;
#endif
}

/* ============================================================================
 * Threading
 * ============================================================================ */

static void *thread_wrapper(void *arg) {
    holo_thread_t *thread = (holo_thread_t *)arg;
    thread->func(thread->arg);
    return NULL;
}

holo_thread_t *holo_thread_create(holo_thread_func_t func, void *arg) {
    holo_thread_t *thread = (holo_thread_t *)holo_alloc(sizeof(holo_thread_t));
    if (!thread) return NULL;

    thread->func = func;
    thread->arg = arg;

    if (pthread_create(&thread->handle, NULL, thread_wrapper, thread) != 0) {
        holo_free(thread);
        return NULL;
    }

    return thread;
}

int holo_thread_join(holo_thread_t *thread) {
    if (!thread) return HOLO_ERR_INVAL;

    int result = pthread_join(thread->handle, NULL);
    holo_free(thread);

    return (result == 0) ? HOLO_OK : HOLO_ERR_IO;
}

void holo_thread_detach(holo_thread_t *thread) {
    if (thread) {
        pthread_detach(thread->handle);
        holo_free(thread);
    }
}

void holo_thread_yield(void) {
    sched_yield();
}

uint64_t holo_thread_id(void) {
#ifdef __APPLE__
    uint64_t tid;
    pthread_threadid_np(NULL, &tid);
    return tid;
#else
    return (uint64_t)pthread_self();
#endif
}

int holo_thread_set_name(const char *name) {
#ifdef __APPLE__
    return pthread_setname_np(name) == 0 ? HOLO_OK : HOLO_ERR_IO;
#else
    return pthread_setname_np(pthread_self(), name) == 0 ? HOLO_OK : HOLO_ERR_IO;
#endif
}

/* Mutex */
holo_mutex_t *holo_mutex_create(void) {
    holo_mutex_t *mutex = (holo_mutex_t *)holo_alloc(sizeof(holo_mutex_t));
    if (!mutex) return NULL;

    if (pthread_mutex_init(&mutex->mutex, NULL) != 0) {
        holo_free(mutex);
        return NULL;
    }

    return mutex;
}

void holo_mutex_destroy(holo_mutex_t *mutex) {
    if (mutex) {
        pthread_mutex_destroy(&mutex->mutex);
        holo_free(mutex);
    }
}

void holo_mutex_lock(holo_mutex_t *mutex) {
    if (mutex) pthread_mutex_lock(&mutex->mutex);
}

bool holo_mutex_trylock(holo_mutex_t *mutex) {
    if (!mutex) return false;
    return pthread_mutex_trylock(&mutex->mutex) == 0;
}

void holo_mutex_unlock(holo_mutex_t *mutex) {
    if (mutex) pthread_mutex_unlock(&mutex->mutex);
}

/* Condition variable */
holo_cond_t *holo_cond_create(void) {
    holo_cond_t *cond = (holo_cond_t *)holo_alloc(sizeof(holo_cond_t));
    if (!cond) return NULL;

    if (pthread_cond_init(&cond->cond, NULL) != 0) {
        holo_free(cond);
        return NULL;
    }

    return cond;
}

void holo_cond_destroy(holo_cond_t *cond) {
    if (cond) {
        pthread_cond_destroy(&cond->cond);
        holo_free(cond);
    }
}

void holo_cond_wait(holo_cond_t *cond, holo_mutex_t *mutex) {
    if (cond && mutex) {
        pthread_cond_wait(&cond->cond, &mutex->mutex);
    }
}

bool holo_cond_timedwait(holo_cond_t *cond, holo_mutex_t *mutex, uint32_t timeout_ms) {
    if (!cond || !mutex) return false;

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_ms / 1000;
    ts.tv_nsec += (timeout_ms % 1000) * 1000000;
    if (ts.tv_nsec >= 1000000000) {
        ts.tv_sec++;
        ts.tv_nsec -= 1000000000;
    }

    return pthread_cond_timedwait(&cond->cond, &mutex->mutex, &ts) == 0;
}

void holo_cond_signal(holo_cond_t *cond) {
    if (cond) pthread_cond_signal(&cond->cond);
}

void holo_cond_broadcast(holo_cond_t *cond) {
    if (cond) pthread_cond_broadcast(&cond->cond);
}

/* Read-write lock */
holo_rwlock_t *holo_rwlock_create(void) {
    holo_rwlock_t *rwlock = (holo_rwlock_t *)holo_alloc(sizeof(holo_rwlock_t));
    if (!rwlock) return NULL;

    if (pthread_rwlock_init(&rwlock->rwlock, NULL) != 0) {
        holo_free(rwlock);
        return NULL;
    }

    return rwlock;
}

void holo_rwlock_destroy(holo_rwlock_t *rwlock) {
    if (rwlock) {
        pthread_rwlock_destroy(&rwlock->rwlock);
        holo_free(rwlock);
    }
}

void holo_rwlock_rdlock(holo_rwlock_t *rwlock) {
    if (rwlock) pthread_rwlock_rdlock(&rwlock->rwlock);
}

void holo_rwlock_wrlock(holo_rwlock_t *rwlock) {
    if (rwlock) pthread_rwlock_wrlock(&rwlock->rwlock);
}

void holo_rwlock_unlock(holo_rwlock_t *rwlock) {
    if (rwlock) pthread_rwlock_unlock(&rwlock->rwlock);
}

/* Atomics - use GCC/Clang builtins */
int32_t holo_atomic_inc(volatile int32_t *val) {
    return __sync_add_and_fetch(val, 1);
}

int32_t holo_atomic_dec(volatile int32_t *val) {
    return __sync_sub_and_fetch(val, 1);
}

int32_t holo_atomic_add(volatile int32_t *val, int32_t add) {
    return __sync_add_and_fetch(val, add);
}

bool holo_atomic_cas(volatile int32_t *val, int32_t expected, int32_t desired) {
    return __sync_bool_compare_and_swap(val, expected, desired);
}

/* ============================================================================
 * File System
 * ============================================================================ */

holo_file_t *holo_file_open(const char *path, uint32_t mode) {
    if (!path) return NULL;

    int flags = 0;

    if ((mode & HOLO_FILE_READ) && (mode & HOLO_FILE_WRITE)) {
        flags = O_RDWR;
    } else if (mode & HOLO_FILE_WRITE) {
        flags = O_WRONLY;
    } else {
        flags = O_RDONLY;
    }

    if (mode & HOLO_FILE_CREATE) flags |= O_CREAT;
    if (mode & HOLO_FILE_TRUNCATE) flags |= O_TRUNC;
    if (mode & HOLO_FILE_APPEND) flags |= O_APPEND;

    int fd = open(path, flags, 0644);
    if (fd < 0) {
        return NULL;
    }

    holo_file_t *file = (holo_file_t *)holo_alloc(sizeof(holo_file_t));
    if (!file) {
        close(fd);
        return NULL;
    }

    file->fd = fd;
    return file;
}

void holo_file_close(holo_file_t *file) {
    if (file) {
        close(file->fd);
        holo_free(file);
    }
}

size_t holo_file_read(holo_file_t *file, void *buf, size_t size) {
    if (!file || !buf) return 0;

    ssize_t n = read(file->fd, buf, size);
    return (n > 0) ? (size_t)n : 0;
}

size_t holo_file_write(holo_file_t *file, const void *buf, size_t size) {
    if (!file || !buf) return 0;

    ssize_t n = write(file->fd, buf, size);
    return (n > 0) ? (size_t)n : 0;
}

int64_t holo_file_seek(holo_file_t *file, int64_t offset, int whence) {
    if (!file) return -1;
    return lseek(file->fd, offset, whence);
}

int64_t holo_file_tell(holo_file_t *file) {
    return holo_file_seek(file, 0, SEEK_CUR);
}

int holo_file_flush(holo_file_t *file) {
    if (!file) return HOLO_ERR_INVAL;
    return fsync(file->fd) == 0 ? HOLO_OK : HOLO_ERR_IO;
}

int64_t holo_file_size(holo_file_t *file) {
    if (!file) return -1;

    struct stat st;
    if (fstat(file->fd, &st) != 0) {
        return -1;
    }
    return st.st_size;
}

bool holo_path_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

bool holo_path_is_file(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

bool holo_path_is_dir(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

int holo_file_delete(const char *path) {
    return unlink(path) == 0 ? HOLO_OK : HOLO_ERR_IO;
}

int holo_file_rename(const char *old_path, const char *new_path) {
    return rename(old_path, new_path) == 0 ? HOLO_OK : HOLO_ERR_IO;
}

int holo_dir_create(const char *path) {
    if (mkdir(path, 0755) != 0) {
        if (errno == EEXIST) return HOLO_OK;
        return HOLO_ERR_IO;
    }
    return HOLO_OK;
}

int holo_dir_create_recursive(const char *path) {
    char *tmp = strdup(path);
    if (!tmp) return HOLO_ERR_NOMEM;

    char *p = tmp;
    if (*p == '/') p++;

    while (*p) {
        if (*p == '/') {
            *p = '\0';
            holo_dir_create(tmp);
            *p = '/';
        }
        p++;
    }

    int result = holo_dir_create(tmp);
    free(tmp);
    return result;
}

holo_dir_t *holo_dir_open(const char *path) {
    DIR *dir = opendir(path);
    if (!dir) return NULL;

    holo_dir_t *d = (holo_dir_t *)holo_alloc(sizeof(holo_dir_t));
    if (!d) {
        closedir(dir);
        return NULL;
    }

    d->dir = dir;
    d->entry = NULL;
    return d;
}

const char *holo_dir_next(holo_dir_t *dir) {
    if (!dir) return NULL;

    while ((dir->entry = readdir(dir->dir)) != NULL) {
        if (strcmp(dir->entry->d_name, ".") == 0 ||
            strcmp(dir->entry->d_name, "..") == 0) {
            continue;
        }
        return dir->entry->d_name;
    }
    return NULL;
}

void holo_dir_close(holo_dir_t *dir) {
    if (dir) {
        closedir(dir->dir);
        holo_free(dir);
    }
}

char *holo_dir_home(void) {
    const char *home = getenv("HOME");
    if (home) return strdup(home);

    struct passwd *pw = getpwuid(getuid());
    if (pw) return strdup(pw->pw_dir);

    return NULL;
}

char *holo_dir_config(void) {
    const char *xdg = getenv("XDG_CONFIG_HOME");
    if (xdg) {
        return holo_path_join(xdg, "holo");
    }

    char *home = holo_dir_home();
    if (!home) return NULL;

#ifdef __APPLE__
    char *config = holo_path_join(home, "Library/Application Support/Holo");
#else
    char *config = holo_path_join(home, ".config/holo");
#endif

    holo_free(home);
    return config;
}

char *holo_dir_data(void) {
    const char *xdg = getenv("XDG_DATA_HOME");
    if (xdg) {
        return holo_path_join(xdg, "holo");
    }

    char *home = holo_dir_home();
    if (!home) return NULL;

#ifdef __APPLE__
    char *data = holo_path_join(home, "Library/Application Support/Holo");
#else
    char *data = holo_path_join(home, ".local/share/holo");
#endif

    holo_free(home);
    return data;
}

char *holo_dir_cache(void) {
    const char *xdg = getenv("XDG_CACHE_HOME");
    if (xdg) {
        return holo_path_join(xdg, "holo");
    }

    char *home = holo_dir_home();
    if (!home) return NULL;

#ifdef __APPLE__
    char *cache = holo_path_join(home, "Library/Caches/Holo");
#else
    char *cache = holo_path_join(home, ".cache/holo");
#endif

    holo_free(home);
    return cache;
}

char *holo_dir_temp(void) {
    const char *tmp = getenv("TMPDIR");
    if (tmp) return strdup(tmp);
    return strdup("/tmp");
}

char *holo_dir_current(void) {
    char buf[PATH_MAX];
    if (getcwd(buf, sizeof(buf))) {
        return strdup(buf);
    }
    return NULL;
}

int holo_dir_set_current(const char *path) {
    return chdir(path) == 0 ? HOLO_OK : HOLO_ERR_IO;
}

/* ============================================================================
 * Networking
 * ============================================================================ */

int holo_net_init(void) {
    return HOLO_OK;  /* No init needed on Unix */
}

void holo_net_shutdown(void) {
    /* Nothing to do */
}

holo_socket_t *holo_socket_create(int type) {
    int sock_type = (type == HOLO_SOCK_UDP) ? SOCK_DGRAM : SOCK_STREAM;
    int protocol = (type == HOLO_SOCK_UDP) ? IPPROTO_UDP : IPPROTO_TCP;

    int fd = socket(AF_INET, sock_type, protocol);
    if (fd < 0) {
        return NULL;
    }

    holo_socket_t *sock = (holo_socket_t *)holo_alloc(sizeof(holo_socket_t));
    if (!sock) {
        close(fd);
        return NULL;
    }

    sock->fd = fd;
    return sock;
}

void holo_socket_close(holo_socket_t *sock) {
    if (sock) {
        close(sock->fd);
        holo_free(sock);
    }
}

int holo_socket_set_option(holo_socket_t *sock, uint32_t option, int value) {
    if (!sock) return HOLO_ERR_INVAL;

    if (option & HOLO_SOCKOPT_NONBLOCK) {
        int flags = fcntl(sock->fd, F_GETFL, 0);
        if (value) {
            fcntl(sock->fd, F_SETFL, flags | O_NONBLOCK);
        } else {
            fcntl(sock->fd, F_SETFL, flags & ~O_NONBLOCK);
        }
    }

    if (option & HOLO_SOCKOPT_REUSEADDR) {
        setsockopt(sock->fd, SOL_SOCKET, SO_REUSEADDR, &value, sizeof(value));
    }

    if (option & HOLO_SOCKOPT_KEEPALIVE) {
        setsockopt(sock->fd, SOL_SOCKET, SO_KEEPALIVE, &value, sizeof(value));
    }

    if (option & HOLO_SOCKOPT_NODELAY) {
        setsockopt(sock->fd, IPPROTO_TCP, TCP_NODELAY, &value, sizeof(value));
    }

    return HOLO_OK;
}

int holo_socket_connect(holo_socket_t *sock, const char *host, uint16_t port) {
    if (!sock || !host) return HOLO_ERR_INVAL;

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        struct addrinfo hints = {0}, *result;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        if (getaddrinfo(host, NULL, &hints, &result) != 0) {
            return HOLO_ERR_NET;
        }

        addr.sin_addr = ((struct sockaddr_in *)result->ai_addr)->sin_addr;
        freeaddrinfo(result);
    }

    if (connect(sock->fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        if (errno == ECONNREFUSED) return HOLO_ERR_CONNREFUSED;
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

    if (bind(sock->fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        return HOLO_ERR_NET;
    }

    return HOLO_OK;
}

int holo_socket_listen(holo_socket_t *sock, int backlog) {
    if (!sock) return HOLO_ERR_INVAL;
    return listen(sock->fd, backlog) == 0 ? HOLO_OK : HOLO_ERR_NET;
}

holo_socket_t *holo_socket_accept(holo_socket_t *sock, char *client_addr, size_t addr_len) {
    if (!sock) return NULL;

    struct sockaddr_in addr;
    socklen_t len = sizeof(addr);

    int client = accept(sock->fd, (struct sockaddr *)&addr, &len);
    if (client < 0) {
        return NULL;
    }

    if (client_addr && addr_len > 0) {
        inet_ntop(AF_INET, &addr.sin_addr, client_addr, addr_len);
    }

    holo_socket_t *s = (holo_socket_t *)holo_alloc(sizeof(holo_socket_t));
    if (!s) {
        close(client);
        return NULL;
    }

    s->fd = client;
    return s;
}

ssize_t holo_socket_send(holo_socket_t *sock, const void *buf, size_t len) {
    if (!sock || !buf) return -1;
    return send(sock->fd, buf, len, 0);
}

ssize_t holo_socket_recv(holo_socket_t *sock, void *buf, size_t len) {
    if (!sock || !buf) return -1;
    return recv(sock->fd, buf, len, 0);
}

int holo_socket_poll(holo_pollfd_t *fds, size_t nfds, int timeout_ms) {
    if (!fds || nfds == 0) return HOLO_ERR_INVAL;

    struct pollfd *pfds = (struct pollfd *)holo_alloc(nfds * sizeof(struct pollfd));
    if (!pfds) return HOLO_ERR_NOMEM;

    for (size_t i = 0; i < nfds; i++) {
        pfds[i].fd = fds[i].sock->fd;
        pfds[i].events = 0;
        if (fds[i].events & HOLO_POLL_READ) pfds[i].events |= POLLIN;
        if (fds[i].events & HOLO_POLL_WRITE) pfds[i].events |= POLLOUT;
        fds[i].revents = 0;
    }

    int result = poll(pfds, nfds, timeout_ms);

    if (result > 0) {
        for (size_t i = 0; i < nfds; i++) {
            if (pfds[i].revents & POLLIN) fds[i].revents |= HOLO_POLL_READ;
            if (pfds[i].revents & POLLOUT) fds[i].revents |= HOLO_POLL_WRITE;
            if (pfds[i].revents & (POLLERR | POLLHUP)) fds[i].revents |= HOLO_POLL_ERROR;
        }
    }

    holo_free(pfds);
    return result < 0 ? HOLO_ERR_NET : result;
}

/* ============================================================================
 * Time
 * ============================================================================ */

uint64_t holo_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

uint64_t holo_time_us(void) {
#ifdef __APPLE__
    static mach_timebase_info_data_t timebase = {0};
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    return mach_absolute_time() * timebase.numer / timebase.denom / 1000;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
#endif
}

uint64_t holo_time_ns(void) {
#ifdef __APPLE__
    static mach_timebase_info_data_t timebase = {0};
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    return mach_absolute_time() * timebase.numer / timebase.denom;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
#endif
}

void holo_sleep_ms(uint32_t ms) {
    usleep(ms * 1000);
}

void holo_sleep_us(uint32_t us) {
    usleep(us);
}

void holo_time_now(holo_datetime_t *dt) {
    if (!dt) return;

    time_t t = time(NULL);
    struct tm *tm = localtime(&t);

    dt->year = tm->tm_year + 1900;
    dt->month = tm->tm_mon + 1;
    dt->day = tm->tm_mday;
    dt->hour = tm->tm_hour;
    dt->minute = tm->tm_min;
    dt->second = tm->tm_sec;
    dt->ms = 0;
    dt->weekday = tm->tm_wday;
    dt->yearday = tm->tm_yday + 1;

#ifdef __APPLE__
    dt->tz_offset = (int)(-tm->tm_gmtoff / 60);
#else
    dt->tz_offset = (int)(-timezone / 60);
#endif
}

/* ============================================================================
 * Terminal
 * ============================================================================ */

int holo_term_width(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        return ws.ws_col;
    }
    return 80;
}

int holo_term_height(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        return ws.ws_row;
    }
    return 24;
}

bool holo_term_is_tty(int fd) {
    return isatty(fd) != 0;
}

bool holo_term_supports_color(void) {
    const char *term = getenv("TERM");
    if (!term) return false;
    return strstr(term, "color") != NULL ||
           strstr(term, "256") != NULL ||
           strcmp(term, "xterm") == 0;
}

bool holo_term_supports_unicode(void) {
    const char *lang = getenv("LANG");
    if (lang && (strstr(lang, "UTF-8") || strstr(lang, "utf8"))) {
        return true;
    }
    return false;
}

static struct termios g_orig_termios;
static bool g_raw_mode = false;

int holo_term_set_raw(bool enable) {
    if (enable && !g_raw_mode) {
        if (tcgetattr(STDIN_FILENO, &g_orig_termios) != 0) {
            return HOLO_ERR_IO;
        }

        struct termios raw = g_orig_termios;
        raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
        raw.c_oflag &= ~(OPOST);
        raw.c_cflag |= (CS8);
        raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 1;

        if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) != 0) {
            return HOLO_ERR_IO;
        }

        g_raw_mode = true;
    } else if (!enable && g_raw_mode) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_termios);
        g_raw_mode = false;
    }

    return HOLO_OK;
}

int holo_term_read_char(void) {
    char c;
    if (read(STDIN_FILENO, &c, 1) == 1) {
        return (unsigned char)c;
    }
    return -1;
}

/* ============================================================================
 * Environment
 * ============================================================================ */

char *holo_getenv(const char *name) {
    const char *val = getenv(name);
    return val ? strdup(val) : NULL;
}

int holo_setenv(const char *name, const char *value) {
    return setenv(name, value, 1) == 0 ? HOLO_OK : HOLO_ERR_IO;
}

int holo_unsetenv(const char *name) {
    return unsetenv(name) == 0 ? HOLO_OK : HOLO_ERR_IO;
}

/* ============================================================================
 * Dynamic Libraries
 * ============================================================================ */

holo_lib_t *holo_lib_load(const char *path) {
    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) return NULL;

    holo_lib_t *lib = (holo_lib_t *)holo_alloc(sizeof(holo_lib_t));
    if (!lib) {
        dlclose(handle);
        return NULL;
    }

    lib->handle = handle;
    return lib;
}

void *holo_lib_symbol(holo_lib_t *lib, const char *name) {
    if (!lib || !name) return NULL;
    return dlsym(lib->handle, name);
}

void holo_lib_unload(holo_lib_t *lib) {
    if (lib) {
        dlclose(lib->handle);
        holo_free(lib);
    }
}

const char *holo_lib_error(void) {
    return dlerror();
}

#endif /* __linux__ || __APPLE__ */
