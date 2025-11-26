/*
 * Holo - Networking and HTTP Server
 * Pure C implementation, no external dependencies
 */

#ifndef HOLO_NET_H
#define HOLO_NET_H

#ifdef __cplusplus
extern "C" {
#endif

#include "holo/pal.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* ============================================================================
 * HTTP Constants
 * ============================================================================ */

#define HOLO_HTTP_MAX_HEADERS     64
#define HOLO_HTTP_MAX_HEADER_SIZE 8192
#define HOLO_HTTP_MAX_BODY_SIZE   (16 * 1024 * 1024)  /* 16MB */
#define HOLO_HTTP_BUFFER_SIZE     65536

/* HTTP Methods */
typedef enum {
    HOLO_HTTP_GET = 0,
    HOLO_HTTP_POST,
    HOLO_HTTP_PUT,
    HOLO_HTTP_DELETE,
    HOLO_HTTP_PATCH,
    HOLO_HTTP_HEAD,
    HOLO_HTTP_OPTIONS,
    HOLO_HTTP_UNKNOWN
} holo_http_method_t;

/* HTTP Status Codes */
typedef enum {
    HOLO_HTTP_200_OK = 200,
    HOLO_HTTP_201_CREATED = 201,
    HOLO_HTTP_204_NO_CONTENT = 204,
    HOLO_HTTP_400_BAD_REQUEST = 400,
    HOLO_HTTP_401_UNAUTHORIZED = 401,
    HOLO_HTTP_403_FORBIDDEN = 403,
    HOLO_HTTP_404_NOT_FOUND = 404,
    HOLO_HTTP_405_METHOD_NOT_ALLOWED = 405,
    HOLO_HTTP_413_PAYLOAD_TOO_LARGE = 413,
    HOLO_HTTP_429_TOO_MANY_REQUESTS = 429,
    HOLO_HTTP_500_INTERNAL_ERROR = 500,
    HOLO_HTTP_503_SERVICE_UNAVAILABLE = 503
} holo_http_status_t;

/* ============================================================================
 * HTTP Header
 * ============================================================================ */

typedef struct {
    char *name;
    char *value;
} holo_http_header_t;

/* ============================================================================
 * HTTP Request
 * ============================================================================ */

typedef struct {
    holo_http_method_t method;
    char *path;
    char *query_string;
    char *http_version;

    holo_http_header_t headers[HOLO_HTTP_MAX_HEADERS];
    size_t header_count;

    char *body;
    size_t body_length;
    size_t content_length;

    /* Parsed from headers */
    const char *content_type;
    const char *authorization;
    const char *host;
    bool keep_alive;
    bool chunked;
} holo_http_request_t;

/* ============================================================================
 * HTTP Response
 * ============================================================================ */

typedef struct {
    holo_http_status_t status;
    char *status_text;

    holo_http_header_t headers[HOLO_HTTP_MAX_HEADERS];
    size_t header_count;

    char *body;
    size_t body_length;
    size_t body_capacity;

    bool headers_sent;
    bool chunked;
} holo_http_response_t;

/* ============================================================================
 * HTTP Connection
 * ============================================================================ */

typedef struct holo_http_conn {
    holo_socket_t *socket;
    char client_addr[64];

    char *recv_buffer;
    size_t recv_size;
    size_t recv_capacity;

    holo_http_request_t request;
    holo_http_response_t response;

    bool keep_alive;
    uint64_t last_activity;
    void *user_data;
} holo_http_conn_t;

/* ============================================================================
 * HTTP Server
 * ============================================================================ */

typedef struct holo_http_server holo_http_server_t;

/* Route handler callback */
typedef void (*holo_http_handler_t)(holo_http_conn_t *conn, void *user_data);

/* Route definition */
typedef struct {
    holo_http_method_t method;
    const char *pattern;
    holo_http_handler_t handler;
    void *user_data;
} holo_http_route_t;

/* Server configuration */
typedef struct {
    const char *bind_address;
    uint16_t port;
    int backlog;
    int max_connections;
    uint32_t timeout_ms;
    size_t max_body_size;
} holo_http_config_t;

/* Default configuration */
#define HOLO_HTTP_CONFIG_DEFAULT { \
    .bind_address = "0.0.0.0",     \
    .port = 8420,                   \
    .backlog = 128,                 \
    .max_connections = 1024,        \
    .timeout_ms = 30000,            \
    .max_body_size = HOLO_HTTP_MAX_BODY_SIZE \
}

/* ============================================================================
 * Server Functions
 * ============================================================================ */

/* Create and destroy server */
HOLO_API holo_http_server_t *holo_http_server_create(const holo_http_config_t *config);
HOLO_API void holo_http_server_destroy(holo_http_server_t *server);

/* Route registration */
HOLO_API int holo_http_server_route(holo_http_server_t *server,
                                     holo_http_method_t method,
                                     const char *pattern,
                                     holo_http_handler_t handler,
                                     void *user_data);

/* Convenience route macros */
#define holo_http_get(s, p, h, u)    holo_http_server_route(s, HOLO_HTTP_GET, p, h, u)
#define holo_http_post(s, p, h, u)   holo_http_server_route(s, HOLO_HTTP_POST, p, h, u)
#define holo_http_put(s, p, h, u)    holo_http_server_route(s, HOLO_HTTP_PUT, p, h, u)
#define holo_http_delete(s, p, h, u) holo_http_server_route(s, HOLO_HTTP_DELETE, p, h, u)

/* Server control */
HOLO_API int holo_http_server_start(holo_http_server_t *server);
HOLO_API int holo_http_server_stop(holo_http_server_t *server);
HOLO_API bool holo_http_server_running(holo_http_server_t *server);

/* Process connections (call in a loop or use threaded mode) */
HOLO_API int holo_http_server_poll(holo_http_server_t *server, int timeout_ms);

/* ============================================================================
 * Request Functions
 * ============================================================================ */

/* Get request info */
HOLO_API holo_http_method_t holo_http_request_method(holo_http_conn_t *conn);
HOLO_API const char *holo_http_request_path(holo_http_conn_t *conn);
HOLO_API const char *holo_http_request_query(holo_http_conn_t *conn);
HOLO_API const char *holo_http_request_body(holo_http_conn_t *conn);
HOLO_API size_t holo_http_request_body_length(holo_http_conn_t *conn);

/* Get header value */
HOLO_API const char *holo_http_request_header(holo_http_conn_t *conn, const char *name);

/* Parse query string parameter */
HOLO_API const char *holo_http_request_param(holo_http_conn_t *conn, const char *name);

/* ============================================================================
 * Response Functions
 * ============================================================================ */

/* Set response status */
HOLO_API void holo_http_response_status(holo_http_conn_t *conn, holo_http_status_t status);

/* Set response header */
HOLO_API void holo_http_response_header(holo_http_conn_t *conn,
                                         const char *name, const char *value);

/* Set content type shortcuts */
HOLO_API void holo_http_response_json(holo_http_conn_t *conn);
HOLO_API void holo_http_response_text(holo_http_conn_t *conn);
HOLO_API void holo_http_response_html(holo_http_conn_t *conn);

/* Write response body */
HOLO_API int holo_http_response_write(holo_http_conn_t *conn,
                                       const void *data, size_t length);
HOLO_API int holo_http_response_printf(holo_http_conn_t *conn,
                                        const char *fmt, ...);

/* Send complete response */
HOLO_API int holo_http_response_send(holo_http_conn_t *conn);

/* Send JSON response */
HOLO_API int holo_http_response_send_json(holo_http_conn_t *conn,
                                           holo_http_status_t status,
                                           const char *json);

/* Send error response */
HOLO_API int holo_http_response_error(holo_http_conn_t *conn,
                                       holo_http_status_t status,
                                       const char *message);

/* ============================================================================
 * Streaming Response (for SSE / chunked transfer)
 * ============================================================================ */

/* Start chunked response */
HOLO_API int holo_http_response_start_chunked(holo_http_conn_t *conn);

/* Write chunk */
HOLO_API int holo_http_response_write_chunk(holo_http_conn_t *conn,
                                             const void *data, size_t length);

/* End chunked response */
HOLO_API int holo_http_response_end_chunked(holo_http_conn_t *conn);

/* Server-Sent Events */
HOLO_API int holo_http_response_sse_start(holo_http_conn_t *conn);
HOLO_API int holo_http_response_sse_event(holo_http_conn_t *conn,
                                           const char *event,
                                           const char *data);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/* Parse HTTP method string */
HOLO_API holo_http_method_t holo_http_parse_method(const char *method);
HOLO_API const char *holo_http_method_string(holo_http_method_t method);

/* Get status text */
HOLO_API const char *holo_http_status_text(holo_http_status_t status);

/* URL encode/decode */
HOLO_API char *holo_url_encode(const char *str);
HOLO_API char *holo_url_decode(const char *str);

#ifdef __cplusplus
}
#endif

#endif /* HOLO_NET_H */
