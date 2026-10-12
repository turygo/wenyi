#include "bridge.h"
#include <libxml/tree.h>
#include <libxml/xmlsave.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>

void *wenyi_document_new(void) { return xmlNewDoc(BAD_CAST "1.0"); }
void wenyi_document_free(void *document) { xmlFreeDoc((xmlDoc *)document); }
void *wenyi_node_new(void *document, void *parent, int kind, const char *name, const char *value) {
    xmlDoc *doc = (xmlDoc *)document;
    xmlNode *node = NULL;
    if (kind == 1 && (!name || xmlValidateNCName(BAD_CAST name, 0) != 0)) return NULL;
    if ((kind == 8 || kind == 9) && (!name || xmlValidateName(BAD_CAST name, 0) != 0)) return NULL;
    if (kind == 1) node = xmlNewDocNode(doc, NULL, BAD_CAST name, NULL);
    else if (kind == 7) node = xmlNewDocComment(doc, BAD_CAST value);
    else if (kind == 8) node = xmlNewDocPI(doc, BAD_CAST name, BAD_CAST value);
    else if (kind == 9) node = xmlNewReference(doc, BAD_CAST name);
    if (!node) return NULL;
    xmlNode *owner = parent ? (xmlNode *)parent : (xmlNode *)doc;
    if (!xmlAddChild(owner, node)) { xmlFreeNode(node); return NULL; }
    return node;
}
int wenyi_namespace(void *node, const char *prefix, const char *uri) {
    if (prefix && *prefix && xmlValidateNCName(BAD_CAST prefix, 0) != 0) return 2;
    return xmlNewNs((xmlNode *)node, BAD_CAST uri, prefix && *prefix ? BAD_CAST prefix : NULL) ? 0 : 2;
}
static xmlNs *namespace_for(xmlNode *node, const char *prefix, const char *uri) {
    if (!uri || !*uri) return NULL;
    xmlNs *ns = xmlSearchNs(node->doc, node, prefix && *prefix ? BAD_CAST prefix : NULL);
    return ns && xmlStrEqual(ns->href, BAD_CAST uri) ? ns : NULL;
}
int wenyi_element_namespace(void *pointer, const char *prefix, const char *uri) {
    xmlNode *node = (xmlNode *)pointer;
    xmlNs *ns = namespace_for(node, prefix, uri);
    if (uri && *uri && !ns) return 2;
    xmlSetNs(node, ns);
    return 0;
}
int wenyi_attribute(void *pointer, const char *name, const char *prefix, const char *uri, const char *value) {
    if (!name || (uri && *uri ? xmlValidateNCName(BAD_CAST name, 0) : xmlValidateName(BAD_CAST name, 0)) != 0) return 2;
    xmlNode *node = (xmlNode *)pointer;
    xmlNs *ns = namespace_for(node, prefix, uri);
    if (uri && *uri && !ns) return 2;
    return xmlNewNsProp(node, ns, BAD_CAST name, BAD_CAST value) ? 0 : 2;
}
int wenyi_text(void *document, void *parent, const char *value, int length, int cdata) {
    if (length < 0) return 2;
    xmlNode *node = cdata ? xmlNewCDataBlock((xmlDoc *)document, BAD_CAST value, length)
                          : xmlNewDocTextLen((xmlDoc *)document, BAD_CAST value, length);
    if (!node) return 2;
    if (!xmlAddChild((xmlNode *)parent, node)) { xmlFreeNode(node); return 2; }
    return 0;
}
int wenyi_document_dump(void *document, const char *encoding, char **output, int *length) {
    *output = NULL; *length = 0;
    xmlBuffer *buffer = xmlBufferCreate();
    if (!buffer) return 2;
    xmlSaveCtxt *save = xmlSaveToBuffer(buffer, encoding, XML_SAVE_NO_DECL | XML_SAVE_AS_XML);
    if (!save) { xmlBufferFree(buffer); return 3; }
    int failed = 0;
    for (xmlNode *node = ((xmlDoc *)document)->children; node && !failed; node = node->next) {
        if (xmlSaveTree(save, node) < 0) failed = 1;
    }
    if (xmlSaveClose(save) < 0) failed = 1;
    size_t size = xmlBufferLength(buffer);
    if (failed || size > INT_MAX) { xmlBufferFree(buffer); return 2; }
    char *owned = (char *)malloc(size ? size : 1);
    if (!owned) { xmlBufferFree(buffer); return 2; }
    if (size) memcpy(owned, xmlBufferContent(buffer), size);
    xmlBufferFree(buffer);
    *output = owned; *length = (int)size;
    return 0;
}
