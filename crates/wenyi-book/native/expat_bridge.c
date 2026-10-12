#include "bridge.h"
#include <expat.h>
#include <string.h>
#include <stdio.h>

typedef struct { wenyi_sink sink; XML_Parser parser; int cdata; } package_parser;

static void emit(package_parser *state, int kind, const char *a, const char *b, const char *c, const char *d) {
    if (!wenyi_emit(&state->sink, kind, a, b, c, d)) XML_StopParser(state->parser, XML_FALSE);
}
static void start(void *user, const XML_Char *name, const XML_Char **attrs) {
    package_parser *state = (package_parser *)user;
    if (state->sink.nodes == SIZE_MAX) {
        state->sink.failed = state->sink.safety = 1;
        XML_StopParser(state->parser, XML_FALSE); return;
    }
    ++state->sink.nodes;
    emit(state, 1, name, NULL, NULL, NULL);
    for (size_t i = 0; attrs[i] && !state->sink.failed; i += 2) emit(state, 3, attrs[i], NULL, NULL, attrs[i+1]);
}
static void end(void *user, const XML_Char *name) {
    package_parser *state = (package_parser *)user;
    (void)name;
    emit(state, 2, NULL, NULL, NULL, NULL);
}
static void text(void *user, const XML_Char *data, int length) {
    package_parser *state = (package_parser *)user;
    if (state->sink.failed || length < 0) return;
    size_t size = (size_t)length;
    if (state->sink.bytes > SIZE_MAX - size) {
        state->sink.failed = state->sink.safety = 1;
        XML_StopParser(state->parser, XML_FALSE); return;
    }
    state->sink.bytes += size;
    if (!state->sink.event(state->sink.user, state->cdata ? 6 : 5, data, size, NULL, 0, NULL, 0, NULL, 0)) {
        state->sink.failed = state->sink.safety = 1;
        XML_StopParser(state->parser, XML_FALSE);
    }
}
static void namespace_start(void *user, const XML_Char *prefix, const XML_Char *uri) {
    emit((package_parser *)user, 13, prefix, uri, NULL, NULL);
}
static void comment(void *user, const XML_Char *value) { emit((package_parser *)user, 7, value, NULL, NULL, NULL); }
static void instruction(void *user, const XML_Char *target, const XML_Char *value) {
    emit((package_parser *)user, 8, target, value, NULL, NULL);
}
static void cdata_start(void *user) { ((package_parser *)user)->cdata = 1; }
static void cdata_end(void *user) { ((package_parser *)user)->cdata = 0; }
static int external(XML_Parser parser, const XML_Char *context, const XML_Char *base,
                    const XML_Char *system_id, const XML_Char *public_id) {
    package_parser *state = (package_parser *)XML_GetUserData(parser);
    (void)context; (void)base; (void)system_id; (void)public_id;
    state->sink.failed = state->sink.safety = 1;
    return XML_STATUS_ERROR;
}
static void skipped(void *user, const XML_Char *name, int parameter) {
    package_parser *state = (package_parser *)user;
    (void)name; (void)parameter;
    state->sink.failed = state->sink.safety = 1;
    XML_StopParser(state->parser, XML_FALSE);
}
int wenyi_expat(const char *bytes, int length, void *user, wenyi_event event) {
    if (length < 0) return 2;
    package_parser state = {0};
    state.sink.user = user; state.sink.event = event;
    state.parser = XML_ParserCreateNS(NULL, '\x1f');
    if (!state.parser) return 2;
    XML_SetUserData(state.parser, &state);
    XML_SetReturnNSTriplet(state.parser, 1);
    XML_SetElementHandler(state.parser, start, end);
    XML_SetCharacterDataHandler(state.parser, text);
    XML_SetNamespaceDeclHandler(state.parser, namespace_start, NULL);
    XML_SetCommentHandler(state.parser, comment);
    XML_SetProcessingInstructionHandler(state.parser, instruction);
    XML_SetCdataSectionHandler(state.parser, cdata_start, cdata_end);
    XML_SetExternalEntityRefHandler(state.parser, external);
    XML_SetSkippedEntityHandler(state.parser, skipped);
    XML_SetParamEntityParsing(state.parser, XML_PARAM_ENTITY_PARSING_NEVER);
    XML_SetBillionLaughsAttackProtectionMaximumAmplification(state.parser, 16.0f);
    XML_SetBillionLaughsAttackProtectionActivationThreshold(state.parser, 1024 * 1024);
    enum XML_Status status = XML_Parse(state.parser, bytes, length, XML_TRUE);
    enum XML_Error error = XML_GetErrorCode(state.parser);
    int result = state.sink.safety || error == XML_ERROR_AMPLIFICATION_LIMIT_BREACH || error == XML_ERROR_NO_MEMORY ? 2
        : error == XML_ERROR_UNKNOWN_ENCODING || error == XML_ERROR_INCORRECT_ENCODING ? 3
        : status == XML_STATUS_OK ? 0 : 1;
    if (result) {
        char line[32], column[32];
        snprintf(line, sizeof(line), "%llu", (unsigned long long)XML_GetCurrentLineNumber64(state.parser));
        snprintf(column, sizeof(column), "%llu", (unsigned long long)XML_GetCurrentColumnNumber64(state.parser));
        const char *reason = XML_ErrorString(error);
        state.sink.event(user, 12, reason, reason ? strlen(reason) : 0, line, strlen(line), column, strlen(column), NULL, 0);
    }
    XML_ParserFree(state.parser);
    return result;
}
const char *wenyi_expat_version(void) { return XML_ExpatVersion(); }
