#include "bridge.h"
#include "xml_limits.h"
#include <string.h>
#include <libxml/parser.h>
#include <stdio.h>
#include <libxml/HTMLparser.h>
#include <libxml/xmlerror.h>
#include <libxml/entities.h>

int wenyi_emit(wenyi_sink *sink, int kind, const char *a, const char *b,
               const char *c, const char *d) {
    size_t al = a ? strlen(a) : 0, bl = b ? strlen(b) : 0;
    size_t cl = c ? strlen(c) : 0, dl = d ? strlen(d) : 0;
    if (sink->failed) return 0;
    if (al > SIZE_MAX - bl || al + bl > SIZE_MAX - cl || al + bl + cl > SIZE_MAX - dl ||
        sink->bytes > SIZE_MAX - (al + bl + cl + dl)) {
        sink->failed = sink->safety = 1;
        return 0;
    }
    sink->bytes += al + bl + cl + dl;
    if (!sink->event(sink->user, kind, a, al, b, bl, c, cl, d, dl)) {
        sink->failed = sink->safety = 1;
        return 0;
    }
    return 1;
}

static xmlParserErrors deny_resource(void *user, const char *url, const char *public_id,
                                    xmlResourceType type, xmlParserInputFlags flags, xmlParserInput **out) {
    wenyi_sink *sink = (wenyi_sink *)user;
    (void)url; (void)public_id; (void)type; (void)flags;
    *out = NULL;
    sink->safety = sink->failed = 1;
    return XML_IO_EACCES;
}

static void diagnostic(void *user, const xmlError *error) {
    wenyi_sink *sink = (wenyi_sink *)user;
    if (!error) return;
    if ((error->domain == XML_FROM_IO && error->code != XML_ERR_INVALID_ENCODING &&
         error->code != XML_ERR_UNSUPPORTED_ENCODING) || error->code == XML_ERR_RESOURCE_LIMIT ||
        error->code == XML_ERR_NAME_TOO_LONG || error->code == XML_ERR_ENTITY_LOOP ||
        error->code == XML_ERR_NO_MEMORY || error->code == XML_ERR_INTERNAL_ERROR) sink->safety = 1;
    char message[1025], line[32], column[32];
    size_t size = error->message ? strlen(error->message) : 0;
    if (size > 1024) size = 1024;
    if (size) memcpy(message, error->message, size);
    message[size] = 0;
    snprintf(line, sizeof(line), "%d", error->line);
    snprintf(column, sizeof(column), "%d", error->int2);
    wenyi_emit(sink, 12, message, line, column, NULL);
}

static int walk(wenyi_sink *sink, xmlDoc *doc, xmlNode *first, int depth) {
    if (depth > WENYI_MAX_DEPTH) { sink->failed = sink->safety = 1; return 0; }
    for (xmlNode *node = first; node && !sink->failed; node = node->next) {
        if (sink->nodes == SIZE_MAX) { sink->failed = sink->safety = 1; break; }
        ++sink->nodes;
        if (node->type == XML_ELEMENT_NODE) {
            const char *uri = node->ns ? (const char *)node->ns->href : NULL;
            const char *prefix = node->ns ? (const char *)node->ns->prefix : NULL;
            wenyi_emit(sink, 1, (const char *)node->name, uri, prefix, NULL);
            for (xmlNs *ns = node->nsDef; ns && !sink->failed; ns = ns->next) {
                wenyi_emit(sink, 4, (const char *)ns->prefix, (const char *)ns->href, NULL, NULL);
            }
            for (xmlAttr *attr = node->properties; attr && !sink->failed; attr = attr->next) {
                xmlChar *value = attr->children ? xmlNodeListGetString(doc, attr->children, 1) : NULL;
                if (attr->children && !value) { sink->failed = sink->safety = 1; break; }
                wenyi_emit(sink, 3, (const char *)attr->name, attr->ns ? (const char *)attr->ns->href : NULL,
                    attr->ns ? (const char *)attr->ns->prefix : NULL, (const char *)value);
                xmlFree(value);
            }
            if (!walk(sink, doc, node->children, depth + 1)) break;
            wenyi_emit(sink, 2, NULL, NULL, NULL, NULL);
        } else if (node->type == XML_TEXT_NODE || node->type == XML_CDATA_SECTION_NODE) {
            wenyi_emit(sink, node->type == XML_TEXT_NODE ? 5 : 6, (const char *)node->content, NULL, NULL, NULL);
        } else if (node->type == XML_COMMENT_NODE) {
            wenyi_emit(sink, 7, (const char *)node->content, NULL, NULL, NULL);
        } else if (node->type == XML_PI_NODE) {
            wenyi_emit(sink, 8, (const char *)node->name, (const char *)node->content, NULL, NULL);
        } else if (node->type == XML_ENTITY_REF_NODE) {
            xmlEntity *entity = xmlGetDocEntity(doc, node->name);
            if (entity && (entity->etype == XML_EXTERNAL_GENERAL_PARSED_ENTITY ||
                entity->etype == XML_EXTERNAL_GENERAL_UNPARSED_ENTITY || entity->etype == XML_EXTERNAL_PARAMETER_ENTITY)) {
                sink->failed = sink->safety = 1;
                break;
            }
            wenyi_emit(sink, 9, (const char *)node->name, NULL, NULL, NULL);
        } else if (node->type == XML_DTD_NODE) {
            xmlBuffer *buffer = xmlBufferCreate();
            if (!buffer) { sink->failed = sink->safety = 1; break; }
            if (xmlNodeDump(buffer, doc, node, 0, 0) < 0) sink->failed = sink->safety = 1;
            else wenyi_emit(sink, 10, (const char *)xmlBufferContent(buffer), NULL, NULL, NULL);
            xmlBufferFree(buffer);
        }
    }
    return !sink->failed;
}

int wenyi_xml(const char *bytes, int length, int html, void *user, wenyi_event event) {
    wenyi_sink sink = {0};
    sink.user = user; sink.event = event;
    if (length < 0 || length > WENYI_MAX_BYTES) return 2;
    xmlParserCtxt *context = html ? (xmlParserCtxt *)htmlNewParserCtxt() : xmlNewParserCtxt();
    if (!context) return 2;
    wenyi_xml_guard limits;
    if (!wenyi_xml_install_limits(context, &limits, &sink)) { xmlFreeParserCtxt(context); return 2; }
    xmlCtxtSetResourceLoader(context, deny_resource, &sink);
    xmlCtxtSetErrorHandler(context, diagnostic, &sink);
    int options = html ? HTML_PARSE_NONET | HTML_PARSE_NODEFDTD : XML_PARSE_NONET | XML_PARSE_NO_XXE;
    xmlDoc *doc = html ? htmlCtxtReadMemory((htmlParserCtxt *)context, bytes, length, NULL, NULL, options)
                       : xmlCtxtReadMemory(context, bytes, length, NULL, NULL, options);
    int result = sink.safety ? 2 : doc ? 0 : 1;
    if (doc && result == 0) {
        wenyi_emit(&sink, 11, doc->encoding ? (const char *)doc->encoding : "UTF-8", NULL, NULL, NULL);
        walk(&sink, doc, doc->children, 0);
        if (sink.failed) result = 2;
    }
    if (doc) xmlFreeDoc(doc);
    xmlFreeParserCtxt(context);
    return result;
}

const char *wenyi_xml_version(void) { return xmlParserVersion; }
