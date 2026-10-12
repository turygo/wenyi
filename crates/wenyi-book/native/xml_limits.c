#include "xml_limits.h"

static wenyi_xml_guard *guard(void *user) {
    return (wenyi_xml_guard *)xmlCtxtGetPrivate((xmlParserCtxt *)user);
}
static int admit(void *user, int delta, size_t bytes, int count) {
    wenyi_xml_guard *state = guard(user);
    state->depth += delta;
    if (count && state->nodes != SIZE_MAX) ++state->nodes;
    if (state->depth > WENYI_MAX_DEPTH || state->depth < 0 || state->nodes == SIZE_MAX ||
        state->text_bytes > SIZE_MAX - bytes) {
        state->sink->safety = state->sink->failed = 1;
        xmlStopParser((xmlParserCtxt *)user);
        return 0;
    }
    state->text_bytes += bytes;
    return !state->sink->failed;
}
static void start(void *user, const xmlChar *name, const xmlChar **attributes) {
    wenyi_xml_guard *state = guard(user);
    if (admit(user, 1, 0, 1) && state->original.startElement) state->original.startElement(user, name, attributes);
}
static void end(void *user, const xmlChar *name) {
    wenyi_xml_guard *state = guard(user);
    if (admit(user, -1, 0, 0) && state->original.endElement) state->original.endElement(user, name);
}
static void start_ns(void *user, const xmlChar *local, const xmlChar *prefix, const xmlChar *uri,
                     int ns_count, const xmlChar **namespaces, int attr_count, int defaults, const xmlChar **attributes) {
    wenyi_xml_guard *state = guard(user);
    if (admit(user, 1, 0, 1) && state->original.startElementNs) {
        state->original.startElementNs(user, local, prefix, uri, ns_count, namespaces, attr_count, defaults, attributes);
    }
}
static void end_ns(void *user, const xmlChar *local, const xmlChar *prefix, const xmlChar *uri) {
    wenyi_xml_guard *state = guard(user);
    if (admit(user, -1, 0, 0) && state->original.endElementNs) state->original.endElementNs(user, local, prefix, uri);
}
static void characters(void *user, const xmlChar *text, int size) {
    wenyi_xml_guard *state = guard(user);
    if (size < 0) { admit(user, 0, WENYI_MAX_BYTES + 1, 1); return; }
    if (admit(user, 0, (size_t)size, 1) && state->original.characters) state->original.characters(user, text, size);
}
static void whitespace(void *user, const xmlChar *text, int size) {
    wenyi_xml_guard *state = guard(user);
    if (size < 0) { admit(user, 0, WENYI_MAX_BYTES + 1, 1); return; }
    if (admit(user, 0, (size_t)size, 1) && state->original.ignorableWhitespace) state->original.ignorableWhitespace(user, text, size);
}
static void cdata(void *user, const xmlChar *text, int size) {
    wenyi_xml_guard *state = guard(user);
    if (size < 0) { admit(user, 0, WENYI_MAX_BYTES + 1, 1); return; }
    if (admit(user, 0, (size_t)size, 1) && state->original.cdataBlock) state->original.cdataBlock(user, text, size);
}
static void comment(void *user, const xmlChar *text) {
    wenyi_xml_guard *state = guard(user);
    if (admit(user, 0, 0, 1) && state->original.comment) state->original.comment(user, text);
}
static void instruction(void *user, const xmlChar *target, const xmlChar *text) {
    wenyi_xml_guard *state = guard(user);
    if (admit(user, 0, 0, 1) && state->original.processingInstruction) state->original.processingInstruction(user, target, text);
}
static void reference(void *user, const xmlChar *name) {
    wenyi_xml_guard *state = guard(user);
    if (admit(user, 0, 0, 1) && state->original.reference) state->original.reference(user, name);
}
int wenyi_xml_install_limits(xmlParserCtxt *context, wenyi_xml_guard *state, wenyi_sink *sink) {
    xmlSAXHandler *original = xmlCtxtGetSaxHandler(context);
    if (!original) return 0;
    state->sink = sink; state->original = *original;
    state->nodes = state->text_bytes = 0; state->depth = 0;
    xmlSAXHandler limited = *original;
    limited.startElement = start; limited.endElement = end;
    limited.startElementNs = start_ns; limited.endElementNs = end_ns;
    limited.characters = characters; limited.ignorableWhitespace = whitespace;
    limited.cdataBlock = cdata; limited.comment = comment;
    limited.processingInstruction = instruction; limited.reference = reference;
    /* 原解析器仍拥有 DOM 构建；限制回调在分配每个新节点前截断输入。 */
    xmlCtxtSetPrivate(context, state);
    return xmlCtxtSetSaxHandler(context, &limited) == 0;
}
