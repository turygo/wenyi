#ifndef WENYI_XML_LIMITS_H
#define WENYI_XML_LIMITS_H
#include "bridge.h"
#include <libxml/parser.h>

typedef struct {
    wenyi_sink *sink;
    xmlSAXHandler original;
    size_t nodes;
    size_t text_bytes;
    int depth;
} wenyi_xml_guard;
int wenyi_xml_install_limits(xmlParserCtxt *, wenyi_xml_guard *, wenyi_sink *);
#endif
