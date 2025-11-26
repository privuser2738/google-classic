/*
 * Holo - HTTP Server Implementation
 * Pure C, cross-platform, no external dependencies
 */

#include "holo/net.h"
#include "holo/pal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

typedef struct {
    holo_http_method_t method;
    char *pattern;
    holo_http_handler_t handler;
    void *user_data;
} http_route_t;

struct holo_http_server {
    holo_http_config_t config;
    holo_socket_t *listen_socket;

    http_route_t *routes;
    size_t route_count;
    size_t route_capacity;

    holo_http_conn_t **connections;
    size_t conn_count;
    size_t conn_capacity;

    bool running;
    holo_mutex_t *mutex;
};

/* ============================================================================
 * String Utilities
 * ============================================================================ */

static char *str_dup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    char *d = (char *)holo_alloc(len + 1);
    if (d) memcpy(d, s, len + 1);
    return d;
}

static char *str_ndup(const char *s, size_t n) {
    if (!s) return NULL;
    char *d = (char *)holo_alloc(n + 1);
    if (d) {
        memcpy(d, s, n);
        d[n] = '\0';
    }
    return d;
}

static int str_casecmp(const char *a, const char *b) {
    while (*a && *b) {
        int diff = tolower((unsigned char)*a) - tolower((unsigned char)*b);
        if (diff != 0) return diff;
        a++;
        b++;
    }
    return tolower((unsigned char)*a) - tolower((unsigned char)*b);
}

/* ============================================================================
 * HTTP Method Parsing
 * ============================================================================ */

holo_http_method_t holo_http_parse_method(const char *method) {
    if (!method) return HOLO_HTTP_UNKNOWN;

    if (strcmp(method, "GET") == 0) return HOLO_HTTP_GET;
    if (strcmp(method, "POST") == 0) return HOLO_HTTP_POST;
    if (strcmp(method, "PUT") == 0) return HOLO_HTTP_PUT;
    if (strcmp(method, "DELETE") == 0) return HOLO_HTTP_DELETE;
    if (strcmp(method, "PATCH") == 0) return HOLO_HTTP_PATCH;
    if (strcmp(method, "HEAD") == 0) return HOLO_HTTP_HEAD;
    if (strcmp(method, "OPTIONS") == 0) return HOLO_HTTP_OPTIONS;

    return HOLO_HTTP_UNKNOWN;
}

const char *holo_http_method_string(holo_http_method_t method) {
    switch (method) {
        case HOLO_HTTP_GET:     return "GET";
        case HOLO_HTTP_POST:    return "POST";
        case HOLO_HTTP_PUT:     return "PUT";
        case HOLO_HTTP_DELETE:  return "DELETE";
        case HOLO_HTTP_PATCH:   return "PATCH";
        case HOLO_HTTP_HEAD:    return "HEAD";
        case HOLO_HTTP_OPTIONS: return "OPTIONS";
        default: return "UNKNOWN";
    }
}

const char *holo_http_status_text(holo_http_status_t status) {
    switch (status) {
        case HOLO_HTTP_200_OK:                  return "OK";
        case HOLO_HTTP_201_CREATED:             return "Created";
        case HOLO_HTTP_204_NO_CONTENT:          return "No Content";
        case HOLO_HTTP_400_BAD_REQUEST:         return "Bad Request";
        case HOLO_HTTP_401_UNAUTHORIZED:        return "Unauthorized";
        case HOLO_HTTP_403_FORBIDDEN:           return "Forbidden";
        case HOLO_HTTP_404_NOT_FOUND:           return "Not Found";
        case HOLO_HTTP_405_METHOD_NOT_ALLOWED:  return "Method Not Allowed";
        case HOLO_HTTP_413_PAYLOAD_TOO_LARGE:   return "Payload Too Large";
        case HOLO_HTTP_429_TOO_MANY_REQUESTS:   return "Too Many Requests";
        case HOLO_HTTP_500_INTERNAL_ERROR:      return "Internal Server Error";
        case HOLO_HTTP_503_SERVICE_UNAVAILABLE: return "Service Unavailable";
        default: return "Unknown";
    }
}

/* ============================================================================
 * URL Encoding/Decoding
 * ============================================================================ */

static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

char *holo_url_decode(const char *str) {
    if (!str) return NULL;

    size_t len = strlen(str);
    char *out = (char *)holo_alloc(len + 1);
    if (!out) return NULL;

    char *p = out;
    while (*str) {
        if (*str == '%' && str[1] && str[2]) {
            int h1 = hex_digit(str[1]);
            int h2 = hex_digit(str[2]);
            if (h1 >= 0 && h2 >= 0) {
                *p++ = (char)((h1 << 4) | h2);
                str += 3;
                continue;
            }
        } else if (*str == '+') {
            *p++ = ' ';
            str++;
            continue;
        }
        *p++ = *str++;
    }
    *p = '\0';

    return out;
}

char *holo_url_encode(const char *str) {
    if (!str) return NULL;

    size_t len = strlen(str);
    char *out = (char *)holo_alloc(len * 3 + 1);
    if (!out) return NULL;

    char *p = out;
    while (*str) {
        char c = *str++;
        if (isalnum((unsigned char)c) || c == '-' || c == '_' || c == '.' || c == '~') {
            *p++ = c;
        } else if (c == ' ') {
            *p++ = '+';
        } else {
            sprintf(p, "%%%02X", (unsigned char)c);
            p += 3;
        }
    }
    *p = '\0';

    return out;
}

/* ============================================================================
 * Request Parsing
 * ============================================================================ */

static void request_init(holo_http_request_t *req) {
    memset(req, 0, sizeof(*req));
    req->method = HOLO_HTTP_UNKNOWN;
    req->keep_alive = true;  /* HTTP/1.1 default */
}

static void request_cleanup(holo_http_request_t *req) {
    holo_free(req->path);
    holo_free(req->query_string);
    holo_free(req->http_version);
    holo_free(req->body);

    for (size_t i = 0; i < req->header_count; i++) {
        holo_free(req->headers[i].name);
        holo_free(req->headers[i].value);
    }

    memset(req, 0, sizeof(*req));
}

static int parse_request_line(holo_http_request_t *req, const char *line) {
    /* Parse: METHOD PATH HTTP/VERSION */
    const char *p = line;

    /* Method */
    const char *method_end = strchr(p, ' ');
    if (!method_end) return -1;

    char method[16];
    size_t method_len = method_end - p;
    if (method_len >= sizeof(method)) return -1;
    memcpy(method, p, method_len);
    method[method_len] = '\0';
    req->method = holo_http_parse_method(method);

    p = method_end + 1;
    while (*p == ' ') p++;

    /* Path (may include query string) */
    const char *path_end = strchr(p, ' ');
    if (!path_end) return -1;

    size_t path_len = path_end - p;
    char *full_path = str_ndup(p, path_len);
    if (!full_path) return -1;

    /* Split path and query string */
    char *query = strchr(full_path, '?');
    if (query) {
        *query = '\0';
        req->query_string = str_dup(query + 1);
    }
    req->path = full_path;

    p = path_end + 1;
    while (*p == ' ') p++;

    /* HTTP version */
    req->http_version = str_dup(p);

    /* Determine keep-alive default based on version */
    if (strstr(req->http_version, "1.0")) {
        req->keep_alive = false;
    }

    return 0;
}

static int parse_header(holo_http_request_t *req, const char *line) {
    if (req->header_count >= HOLO_HTTP_MAX_HEADERS) {
        return -1;
    }

    const char *colon = strchr(line, ':');
    if (!colon) return -1;

    size_t name_len = colon - line;
    const char *value = colon + 1;
    while (*value == ' ' || *value == '\t') value++;

    req->headers[req->header_count].name = str_ndup(line, name_len);
    req->headers[req->header_count].value = str_dup(value);
    req->header_count++;

    /* Parse special headers */
    if (str_casecmp(line, "Content-Length") == 0 ||
        strncasecmp(line, "Content-Length", 14) == 0) {
        req->content_length = (size_t)atoll(value);
    } else if (str_casecmp(line, "Content-Type") == 0 ||
               strncasecmp(line, "Content-Type", 12) == 0) {
        req->content_type = req->headers[req->header_count - 1].value;
    } else if (str_casecmp(line, "Authorization") == 0 ||
               strncasecmp(line, "Authorization", 13) == 0) {
        req->authorization = req->headers[req->header_count - 1].value;
    } else if (str_casecmp(line, "Host") == 0 ||
               strncasecmp(line, "Host", 4) == 0) {
        req->host = req->headers[req->header_count - 1].value;
    } else if (str_casecmp(line, "Connection") == 0 ||
               strncasecmp(line, "Connection", 10) == 0) {
        if (strstr(value, "close")) {
            req->keep_alive = false;
        } else if (strstr(value, "keep-alive")) {
            req->keep_alive = true;
        }
    } else if (str_casecmp(line, "Transfer-Encoding") == 0 ||
               strncasecmp(line, "Transfer-Encoding", 17) == 0) {
        if (strstr(value, "chunked")) {
            req->chunked = true;
        }
    }

    return 0;
}

static int parse_request(holo_http_conn_t *conn) {
    holo_http_request_t *req = &conn->request;
    char *buf = conn->recv_buffer;
    size_t len = conn->recv_size;

    /* Find end of headers */
    char *headers_end = strstr(buf, "\r\n\r\n");
    if (!headers_end) {
        return 0;  /* Need more data */
    }

    size_t headers_len = headers_end - buf;
    size_t body_start = headers_len + 4;

    /* Parse request line */
    char *line = buf;
    char *line_end = strstr(line, "\r\n");
    if (!line_end) return -1;

    *line_end = '\0';
    if (parse_request_line(req, line) < 0) {
        return -1;
    }

    /* Parse headers */
    line = line_end + 2;
    while (line < headers_end) {
        line_end = strstr(line, "\r\n");
        if (!line_end) break;
        if (line == line_end) break;  /* Empty line */

        *line_end = '\0';
        if (parse_header(req, line) < 0) {
            return -1;
        }
        line = line_end + 2;
    }

    /* Read body if present */
    if (req->content_length > 0) {
        size_t available = len - body_start;
        if (available < req->content_length) {
            return 0;  /* Need more data */
        }

        req->body = (char *)holo_alloc(req->content_length + 1);
        if (!req->body) return -1;

        memcpy(req->body, buf + body_start, req->content_length);
        req->body[req->content_length] = '\0';
        req->body_length = req->content_length;
    }

    return 1;  /* Complete request */
}

/* ============================================================================
 * Response Building
 * ============================================================================ */

static void response_init(holo_http_response_t *res) {
    memset(res, 0, sizeof(*res));
    res->status = HOLO_HTTP_200_OK;
}

static void response_cleanup(holo_http_response_t *res) {
    holo_free(res->status_text);
    holo_free(res->body);

    for (size_t i = 0; i < res->header_count; i++) {
        holo_free(res->headers[i].name);
        holo_free(res->headers[i].value);
    }

    memset(res, 0, sizeof(*res));
}

void holo_http_response_status(holo_http_conn_t *conn, holo_http_status_t status) {
    conn->response.status = status;
}

void holo_http_response_header(holo_http_conn_t *conn, const char *name, const char *value) {
    holo_http_response_t *res = &conn->response;

    if (res->header_count >= HOLO_HTTP_MAX_HEADERS) return;

    res->headers[res->header_count].name = str_dup(name);
    res->headers[res->header_count].value = str_dup(value);
    res->header_count++;
}

void holo_http_response_json(holo_http_conn_t *conn) {
    holo_http_response_header(conn, "Content-Type", "application/json; charset=utf-8");
}

void holo_http_response_text(holo_http_conn_t *conn) {
    holo_http_response_header(conn, "Content-Type", "text/plain; charset=utf-8");
}

void holo_http_response_html(holo_http_conn_t *conn) {
    holo_http_response_header(conn, "Content-Type", "text/html; charset=utf-8");
}

int holo_http_response_write(holo_http_conn_t *conn, const void *data, size_t length) {
    holo_http_response_t *res = &conn->response;

    if (res->body_length + length > res->body_capacity) {
        size_t new_cap = res->body_capacity ? res->body_capacity * 2 : 4096;
        while (new_cap < res->body_length + length) new_cap *= 2;

        char *new_body = (char *)holo_realloc(res->body, new_cap);
        if (!new_body) return -1;

        res->body = new_body;
        res->body_capacity = new_cap;
    }

    memcpy(res->body + res->body_length, data, length);
    res->body_length += length;

    return 0;
}

int holo_http_response_printf(holo_http_conn_t *conn, const char *fmt, ...) {
    char buf[4096];
    va_list args;

    va_start(args, fmt);
    int len = vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (len < 0) return -1;
    if ((size_t)len >= sizeof(buf)) {
        /* Buffer too small, allocate */
        char *big_buf = (char *)holo_alloc(len + 1);
        if (!big_buf) return -1;

        va_start(args, fmt);
        vsnprintf(big_buf, len + 1, fmt, args);
        va_end(args);

        int result = holo_http_response_write(conn, big_buf, len);
        holo_free(big_buf);
        return result;
    }

    return holo_http_response_write(conn, buf, len);
}

static int send_response_headers(holo_http_conn_t *conn) {
    holo_http_response_t *res = &conn->response;
    char buf[HOLO_HTTP_MAX_HEADER_SIZE];
    int pos = 0;

    /* Status line */
    pos += snprintf(buf + pos, sizeof(buf) - pos, "HTTP/1.1 %d %s\r\n",
                    res->status, holo_http_status_text(res->status));

    /* Headers */
    for (size_t i = 0; i < res->header_count; i++) {
        pos += snprintf(buf + pos, sizeof(buf) - pos, "%s: %s\r\n",
                        res->headers[i].name, res->headers[i].value);
    }

    /* Content-Length if not chunked */
    if (!res->chunked && res->body_length > 0) {
        pos += snprintf(buf + pos, sizeof(buf) - pos, "Content-Length: %zu\r\n",
                        res->body_length);
    }

    /* Connection header */
    if (!conn->keep_alive) {
        pos += snprintf(buf + pos, sizeof(buf) - pos, "Connection: close\r\n");
    }

    /* Server header */
    pos += snprintf(buf + pos, sizeof(buf) - pos, "Server: Holo/0.2.0\r\n");

    /* End of headers */
    pos += snprintf(buf + pos, sizeof(buf) - pos, "\r\n");

    /* Send */
    ssize_t sent = holo_socket_send(conn->socket, buf, pos);
    if (sent != pos) return -1;

    res->headers_sent = true;
    return 0;
}

int holo_http_response_send(holo_http_conn_t *conn) {
    holo_http_response_t *res = &conn->response;

    /* Send headers if not already sent */
    if (!res->headers_sent) {
        if (send_response_headers(conn) < 0) {
            return -1;
        }
    }

    /* Send body */
    if (res->body && res->body_length > 0) {
        ssize_t sent = holo_socket_send(conn->socket, res->body, res->body_length);
        if (sent != (ssize_t)res->body_length) {
            return -1;
        }
    }

    return 0;
}

int holo_http_response_send_json(holo_http_conn_t *conn, holo_http_status_t status,
                                  const char *json) {
    holo_http_response_status(conn, status);
    holo_http_response_json(conn);
    holo_http_response_write(conn, json, strlen(json));
    return holo_http_response_send(conn);
}

int holo_http_response_error(holo_http_conn_t *conn, holo_http_status_t status,
                              const char *message) {
    char json[512];
    snprintf(json, sizeof(json),
             "{\"error\":{\"code\":%d,\"message\":\"%s\"}}",
             status, message ? message : holo_http_status_text(status));

    return holo_http_response_send_json(conn, status, json);
}

/* ============================================================================
 * Streaming Response
 * ============================================================================ */

int holo_http_response_start_chunked(holo_http_conn_t *conn) {
    conn->response.chunked = true;
    holo_http_response_header(conn, "Transfer-Encoding", "chunked");
    return send_response_headers(conn);
}

int holo_http_response_write_chunk(holo_http_conn_t *conn, const void *data, size_t length) {
    if (length == 0) return 0;

    char header[32];
    int hlen = snprintf(header, sizeof(header), "%zx\r\n", length);

    if (holo_socket_send(conn->socket, header, hlen) != hlen) return -1;
    if (holo_socket_send(conn->socket, data, length) != (ssize_t)length) return -1;
    if (holo_socket_send(conn->socket, "\r\n", 2) != 2) return -1;

    return 0;
}

int holo_http_response_end_chunked(holo_http_conn_t *conn) {
    return holo_socket_send(conn->socket, "0\r\n\r\n", 5) == 5 ? 0 : -1;
}

int holo_http_response_sse_start(holo_http_conn_t *conn) {
    holo_http_response_header(conn, "Content-Type", "text/event-stream");
    holo_http_response_header(conn, "Cache-Control", "no-cache");
    holo_http_response_header(conn, "Connection", "keep-alive");
    conn->response.chunked = true;
    return send_response_headers(conn);
}

int holo_http_response_sse_event(holo_http_conn_t *conn, const char *event, const char *data) {
    char buf[4096];
    int len = 0;

    if (event) {
        len += snprintf(buf + len, sizeof(buf) - len, "event: %s\n", event);
    }

    /* Split data by newlines */
    const char *p = data;
    while (*p) {
        const char *nl = strchr(p, '\n');
        if (nl) {
            len += snprintf(buf + len, sizeof(buf) - len, "data: %.*s\n", (int)(nl - p), p);
            p = nl + 1;
        } else {
            len += snprintf(buf + len, sizeof(buf) - len, "data: %s\n", p);
            break;
        }
    }

    len += snprintf(buf + len, sizeof(buf) - len, "\n");

    return holo_socket_send(conn->socket, buf, len) == len ? 0 : -1;
}

/* ============================================================================
 * Request Accessors
 * ============================================================================ */

holo_http_method_t holo_http_request_method(holo_http_conn_t *conn) {
    return conn->request.method;
}

const char *holo_http_request_path(holo_http_conn_t *conn) {
    return conn->request.path;
}

const char *holo_http_request_query(holo_http_conn_t *conn) {
    return conn->request.query_string;
}

const char *holo_http_request_body(holo_http_conn_t *conn) {
    return conn->request.body;
}

size_t holo_http_request_body_length(holo_http_conn_t *conn) {
    return conn->request.body_length;
}

const char *holo_http_request_header(holo_http_conn_t *conn, const char *name) {
    for (size_t i = 0; i < conn->request.header_count; i++) {
        if (str_casecmp(conn->request.headers[i].name, name) == 0) {
            return conn->request.headers[i].value;
        }
    }
    return NULL;
}

/* ============================================================================
 * Connection Management
 * ============================================================================ */

static holo_http_conn_t *conn_create(holo_socket_t *socket, const char *addr) {
    holo_http_conn_t *conn = (holo_http_conn_t *)holo_calloc(1, sizeof(holo_http_conn_t));
    if (!conn) return NULL;

    conn->socket = socket;
    if (addr) {
        strncpy(conn->client_addr, addr, sizeof(conn->client_addr) - 1);
    }

    conn->recv_capacity = HOLO_HTTP_BUFFER_SIZE;
    conn->recv_buffer = (char *)holo_alloc(conn->recv_capacity);
    if (!conn->recv_buffer) {
        holo_free(conn);
        return NULL;
    }

    request_init(&conn->request);
    response_init(&conn->response);
    conn->keep_alive = true;
    conn->last_activity = holo_time_ms();

    return conn;
}

static void conn_destroy(holo_http_conn_t *conn) {
    if (!conn) return;

    if (conn->socket) {
        holo_socket_close(conn->socket);
    }

    holo_free(conn->recv_buffer);
    request_cleanup(&conn->request);
    response_cleanup(&conn->response);
    holo_free(conn);
}

static void conn_reset(holo_http_conn_t *conn) {
    request_cleanup(&conn->request);
    response_cleanup(&conn->response);
    request_init(&conn->request);
    response_init(&conn->response);
    conn->recv_size = 0;
}

/* ============================================================================
 * Server Implementation
 * ============================================================================ */

holo_http_server_t *holo_http_server_create(const holo_http_config_t *config) {
    holo_http_server_t *server = (holo_http_server_t *)holo_calloc(1, sizeof(holo_http_server_t));
    if (!server) return NULL;

    if (config) {
        server->config = *config;
    } else {
        holo_http_config_t def = HOLO_HTTP_CONFIG_DEFAULT;
        server->config = def;
    }

    server->route_capacity = 32;
    server->routes = (http_route_t *)holo_calloc(server->route_capacity, sizeof(http_route_t));

    server->conn_capacity = server->config.max_connections;
    server->connections = (holo_http_conn_t **)holo_calloc(server->conn_capacity,
                                                           sizeof(holo_http_conn_t *));

    server->mutex = holo_mutex_create();

    if (!server->routes || !server->connections || !server->mutex) {
        holo_http_server_destroy(server);
        return NULL;
    }

    return server;
}

void holo_http_server_destroy(holo_http_server_t *server) {
    if (!server) return;

    holo_http_server_stop(server);

    for (size_t i = 0; i < server->route_count; i++) {
        holo_free(server->routes[i].pattern);
    }
    holo_free(server->routes);

    for (size_t i = 0; i < server->conn_count; i++) {
        conn_destroy(server->connections[i]);
    }
    holo_free(server->connections);

    if (server->mutex) {
        holo_mutex_destroy(server->mutex);
    }

    holo_free(server);
}

int holo_http_server_route(holo_http_server_t *server, holo_http_method_t method,
                           const char *pattern, holo_http_handler_t handler, void *user_data) {
    if (!server || !pattern || !handler) return -1;

    if (server->route_count >= server->route_capacity) {
        size_t new_cap = server->route_capacity * 2;
        http_route_t *new_routes = (http_route_t *)holo_realloc(
            server->routes, new_cap * sizeof(http_route_t));
        if (!new_routes) return -1;
        server->routes = new_routes;
        server->route_capacity = new_cap;
    }

    server->routes[server->route_count].method = method;
    server->routes[server->route_count].pattern = str_dup(pattern);
    server->routes[server->route_count].handler = handler;
    server->routes[server->route_count].user_data = user_data;
    server->route_count++;

    return 0;
}

int holo_http_server_start(holo_http_server_t *server) {
    if (!server) return -1;

    holo_net_init();

    server->listen_socket = holo_socket_create(HOLO_SOCK_TCP);
    if (!server->listen_socket) {
        return -1;
    }

    holo_socket_set_option(server->listen_socket, HOLO_SOCKOPT_REUSEADDR, 1);
    holo_socket_set_option(server->listen_socket, HOLO_SOCKOPT_NONBLOCK, 1);

    if (holo_socket_bind(server->listen_socket, server->config.bind_address,
                         server->config.port) != HOLO_OK) {
        holo_socket_close(server->listen_socket);
        server->listen_socket = NULL;
        return -1;
    }

    if (holo_socket_listen(server->listen_socket, server->config.backlog) != HOLO_OK) {
        holo_socket_close(server->listen_socket);
        server->listen_socket = NULL;
        return -1;
    }

    server->running = true;
    return 0;
}

int holo_http_server_stop(holo_http_server_t *server) {
    if (!server) return -1;

    server->running = false;

    if (server->listen_socket) {
        holo_socket_close(server->listen_socket);
        server->listen_socket = NULL;
    }

    return 0;
}

bool holo_http_server_running(holo_http_server_t *server) {
    return server && server->running;
}

/* Route matching (simple prefix match for now) */
static http_route_t *find_route(holo_http_server_t *server, holo_http_conn_t *conn) {
    for (size_t i = 0; i < server->route_count; i++) {
        http_route_t *route = &server->routes[i];

        /* Check method */
        if (route->method != conn->request.method) continue;

        /* Simple pattern matching */
        const char *pattern = route->pattern;
        const char *path = conn->request.path;

        if (strcmp(pattern, path) == 0) {
            return route;
        }

        /* Wildcard support: /api/* matches /api/anything */
        size_t plen = strlen(pattern);
        if (plen > 1 && pattern[plen - 1] == '*') {
            if (strncmp(pattern, path, plen - 1) == 0) {
                return route;
            }
        }
    }

    return NULL;
}

static void handle_connection(holo_http_server_t *server, holo_http_conn_t *conn) {
    /* Receive data */
    if (conn->recv_size >= conn->recv_capacity - 1) {
        /* Buffer full, expand or reject */
        if (conn->recv_capacity >= HOLO_HTTP_MAX_BODY_SIZE) {
            holo_http_response_error(conn, HOLO_HTTP_413_PAYLOAD_TOO_LARGE, NULL);
            conn->keep_alive = false;
            return;
        }
        size_t new_cap = conn->recv_capacity * 2;
        char *new_buf = (char *)holo_realloc(conn->recv_buffer, new_cap);
        if (!new_buf) {
            holo_http_response_error(conn, HOLO_HTTP_500_INTERNAL_ERROR, NULL);
            conn->keep_alive = false;
            return;
        }
        conn->recv_buffer = new_buf;
        conn->recv_capacity = new_cap;
    }

    ssize_t n = holo_socket_recv(conn->socket,
                                  conn->recv_buffer + conn->recv_size,
                                  conn->recv_capacity - conn->recv_size - 1);

    if (n > 0) {
        conn->recv_size += n;
        conn->recv_buffer[conn->recv_size] = '\0';
        conn->last_activity = holo_time_ms();
    } else if (n == 0) {
        /* Connection closed */
        conn->keep_alive = false;
        return;
    }

    /* Try to parse request */
    int result = parse_request(conn);
    if (result < 0) {
        holo_http_response_error(conn, HOLO_HTTP_400_BAD_REQUEST, "Invalid request");
        conn->keep_alive = false;
        return;
    }
    if (result == 0) {
        /* Need more data */
        return;
    }

    /* Find and call handler */
    http_route_t *route = find_route(server, conn);
    if (route) {
        route->handler(conn, route->user_data);
    } else {
        holo_http_response_error(conn, HOLO_HTTP_404_NOT_FOUND, "Not found");
    }

    /* Ensure response is sent */
    if (!conn->response.headers_sent) {
        holo_http_response_send(conn);
    }

    /* Keep-alive handling */
    conn->keep_alive = conn->keep_alive && conn->request.keep_alive;

    if (conn->keep_alive) {
        conn_reset(conn);
    }
}

int holo_http_server_poll(holo_http_server_t *server, int timeout_ms) {
    if (!server || !server->running) return -1;

    /* Accept new connections */
    char client_addr[64];
    holo_socket_t *client = holo_socket_accept(server->listen_socket, client_addr, sizeof(client_addr));

    if (client) {
        holo_socket_set_option(client, HOLO_SOCKOPT_NONBLOCK, 1);
        holo_socket_set_option(client, HOLO_SOCKOPT_NODELAY, 1);

        holo_http_conn_t *conn = conn_create(client, client_addr);
        if (conn && server->conn_count < server->conn_capacity) {
            server->connections[server->conn_count++] = conn;
        } else {
            conn_destroy(conn);
        }
    }

    /* Process existing connections */
    uint64_t now = holo_time_ms();
    size_t i = 0;

    while (i < server->conn_count) {
        holo_http_conn_t *conn = server->connections[i];

        /* Check timeout */
        if (now - conn->last_activity > server->config.timeout_ms) {
            conn->keep_alive = false;
        }

        if (conn->keep_alive) {
            handle_connection(server, conn);
        }

        /* Remove closed connections */
        if (!conn->keep_alive) {
            conn_destroy(conn);
            server->connections[i] = server->connections[--server->conn_count];
        } else {
            i++;
        }
    }

    /* Small sleep to prevent busy loop */
    if (server->conn_count == 0) {
        holo_sleep_ms(timeout_ms > 0 ? (timeout_ms < 10 ? timeout_ms : 10) : 1);
    }

    return 0;
}
