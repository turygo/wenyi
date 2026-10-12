#ifndef WENYI_BRIDGE_H
#define WENYI_BRIDGE_H
#include <stddef.h>
#include <stdint.h>
#define WENYI_MAX_BYTES (512 * 1024 * 1024)
#define WENYI_MAX_DEPTH 256
/* 回调中的字符串只在该次调用内有效，Rust 必须立即复制。 */
typedef int (*wenyi_event)(void *, int, const char *, size_t, const char *, size_t,
                           const char *, size_t, const char *, size_t);
typedef struct {
    void *user;
    wenyi_event event;
    int failed;
    int safety;
    int encoding;
    size_t nodes;
    size_t bytes;
    int depth;
} wenyi_sink;
int wenyi_emit(wenyi_sink *, int, const char *, const char *, const char *, const char *);
int wenyi_xml(const char *, int, int, void *, wenyi_event);
int wenyi_expat(const char *, int, void *, wenyi_event);
int wenyi_convert(const char *, int, const char *, const char *, int, char **, int *);
void wenyi_free(void *);
const char *wenyi_xml_version(void);
const char *wenyi_expat_version(void);
void wenyi_icu_version(char output[32]);
int wenyi_encoding_init(void);
#endif
