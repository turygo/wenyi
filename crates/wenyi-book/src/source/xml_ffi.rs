#![allow(unsafe_code)]

use super::tree::{safety, source};
use super::{
    Attribute, ConversionPolicy, DocumentEnvelope, ElementData, MAX_BYTES, MAX_DEPTH,
    NamespaceDecl, NativeVersions, NodeId, NodeKind, ParseDiagnostic, ParsedDocument,
    QualifiedName, SourceNode, SourceTree,
};
use crate::error::{BookError, BookErrorKind};
use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::LazyLock,
};

type Event = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    *const c_char,
    usize,
    *const c_char,
    usize,
    *const c_char,
    usize,
    *const c_char,
    usize,
) -> c_int;
unsafe extern "C" {
    fn wenyi_xml(
        bytes: *const c_char,
        length: c_int,
        html: c_int,
        user: *mut c_void,
        event: Event,
    ) -> c_int;
    fn wenyi_expat(bytes: *const c_char, length: c_int, user: *mut c_void, event: Event) -> c_int;
    fn wenyi_convert(
        bytes: *const c_char,
        length: c_int,
        from: *const c_char,
        to: *const c_char,
        replace: c_int,
        output: *mut *mut c_char,
        output_length: *mut c_int,
    ) -> c_int;
    fn wenyi_free(pointer: *mut c_void);
    fn wenyi_xml_version() -> *const c_char;
    fn wenyi_expat_version() -> *const c_char;
    fn wenyi_icu_version(output: *mut c_char);
    fn wenyi_encoding_init() -> c_int;
    fn wenyi_document_new() -> *mut c_void;
    fn wenyi_document_free(document: *mut c_void);
    fn wenyi_node_new(
        document: *mut c_void,
        parent: *mut c_void,
        kind: c_int,
        name: *const c_char,
        value: *const c_char,
    ) -> *mut c_void;
    fn wenyi_namespace(node: *mut c_void, prefix: *const c_char, uri: *const c_char) -> c_int;
    fn wenyi_element_namespace(
        node: *mut c_void,
        prefix: *const c_char,
        uri: *const c_char,
    ) -> c_int;
    fn wenyi_attribute(
        node: *mut c_void,
        name: *const c_char,
        prefix: *const c_char,
        uri: *const c_char,
        value: *const c_char,
    ) -> c_int;
    fn wenyi_text(
        document: *mut c_void,
        parent: *mut c_void,
        value: *const c_char,
        length: c_int,
        cdata: c_int,
    ) -> c_int;
    fn wenyi_document_dump(
        document: *mut c_void,
        encoding: *const c_char,
        output: *mut *mut c_char,
        length: *mut c_int,
    ) -> c_int;
}
// ICU 文件访问策略只初始化一次；转换数据来自静态库，不读运行时环境路径。
static INIT: LazyLock<c_int> = LazyLock::new(|| unsafe { wenyi_encoding_init() });
fn initialize() -> Result<(), BookError> {
    let code = *INIT;
    if code != 0 {
        return Err(native_error(code, "ICU static data initialization failed"));
    }
    Ok(())
}
fn length(bytes: &[u8]) -> Result<c_int, BookError> {
    if bytes.len() > MAX_BYTES {
        return Err(safety("native input length limit exceeded"));
    }
    c_int::try_from(bytes.len()).map_err(|_| safety("native input length overflow"))
}
fn native_error(status: c_int, detail: &str) -> BookError {
    match status {
        1 => BookError::new(BookErrorKind::XmlSyntax, "xml-syntax", detail),
        3 => BookError::new(BookErrorKind::Encoding, "native-encoding", detail),
        _ => safety(detail),
    }
}
struct Builder {
    nodes: Vec<SourceNode>,
    stack: Vec<NodeId>,
    root: Option<NodeId>,
    before: Vec<NodeId>,
    after: Vec<NodeId>,
    namespaces: Vec<NamespaceDecl>,
    diagnostics: Vec<ParseDiagnostic>,
    envelope: DocumentEnvelope,
    error: Option<BookError>,
    bytes: usize,
    bounded_depth: bool,
}
impl Builder {
    fn add(&mut self, kind: NodeKind) -> Result<NodeId, BookError> {
        let id = NodeId::new(self.nodes.len())?;
        let parent = self.stack.last().copied();
        self.nodes.push(SourceNode {
            kind,
            parent,
            children: Vec::new(),
            text: None,
            tail: None,
            text_cdata: Vec::new(),
            tail_cdata: Vec::new(),
        });
        if let Some(parent) = parent {
            self.nodes[parent.index()].children.push(id);
        } else if matches!(self.nodes[id.index()].kind, NodeKind::Element(_)) {
            if self.root.replace(id).is_some() {
                return Err(source("multiple document roots"));
            }
        } else if self.root.is_some() {
            self.after.push(id);
        } else {
            self.before.push(id);
        }
        Ok(id)
    }
    fn event(&mut self, kind: c_int, fields: [String; 4]) -> Result<(), BookError> {
        let [a, b, c, d] = fields;
        match kind {
            1 => {
                if self.bounded_depth && self.stack.len() >= MAX_DEPTH {
                    return Err(safety("source depth limit exceeded"));
                }
                let namespaces = std::mem::take(&mut self.namespaces);
                let id = self.add(NodeKind::Element(ElementData {
                    name: name(a, b, c),
                    attributes: Vec::new(),
                    namespaces,
                }))?;
                self.stack.push(id);
            }
            2 => {
                self.stack
                    .pop()
                    .ok_or_else(|| source("unbalanced native element events"))?;
            }
            3 | 4 => {
                let id = self
                    .stack
                    .last()
                    .ok_or_else(|| source("native attribute without element"))?;
                let NodeKind::Element(element) = &mut self.nodes[id.index()].kind else {
                    return Err(source("invalid native element event"));
                };
                if kind == 3 {
                    element.attributes.push(Attribute {
                        name: name(a, b, c),
                        value: d,
                    });
                } else {
                    element.namespaces.push(NamespaceDecl {
                        prefix: optional(a),
                        uri: b,
                    });
                }
            }
            5 | 6 => self.text(a, kind == 6)?,
            7 => {
                self.add(NodeKind::Comment(a))?;
            }
            8 => {
                self.add(NodeKind::ProcessingInstruction { target: a, data: b })?;
            }
            9 => {
                self.add(NodeKind::EntityReference(a))?;
            }
            10 => {
                self.envelope.doctype = Some(a);
            }
            11 => {
                if !a.is_empty() {
                    self.envelope.encoding = a;
                }
            }
            12 => {
                if self.diagnostics.len() < 20 {
                    let mut reason = a;
                    if reason.len() > 1024 {
                        let mut end = 1024;
                        while !reason.is_char_boundary(end) {
                            end -= 1;
                        }
                        reason.truncate(end);
                    }
                    self.diagnostics.push(ParseDiagnostic {
                        line: b.parse().ok(),
                        column: c.parse().ok(),
                        reason,
                    });
                }
            }
            13 => self.namespaces.push(NamespaceDecl {
                prefix: optional(a),
                uri: b,
            }),
            _ => return Err(source("unknown native source event")),
        }
        Ok(())
    }
    fn text(&mut self, value: String, cdata: bool) -> Result<(), BookError> {
        let Some(parent) = self.stack.last().copied() else {
            return Ok(());
        };
        let last = self.nodes[parent.index()].children.last().copied();
        let (text, ranges) = if let Some(last) = last {
            let node = &mut self.nodes[last.index()];
            (&mut node.tail, &mut node.tail_cdata)
        } else {
            let node = &mut self.nodes[parent.index()];
            (&mut node.text, &mut node.text_cdata)
        };
        let text = text.get_or_insert_with(String::new);
        let start = text.len();
        if start.checked_add(value.len()).is_none() {
            return Err(safety("source text length overflow"));
        }
        text.push_str(&value);
        if cdata {
            ranges.push(start..text.len());
        }
        Ok(())
    }
}
fn optional(value: String) -> Option<String> {
    if value.is_empty() { None } else { Some(value) }
}
fn name(local: String, namespace: String, prefix: String) -> QualifiedName {
    if local.contains('\x1f') {
        let mut parts = local.split('\x1f');
        let uri = parts.next().unwrap_or_default();
        let local_name = parts.next().unwrap_or_default();
        let prefix = parts.next().map(str::to_owned);
        QualifiedName {
            namespace: Some(uri.to_owned()),
            prefix,
            local: local_name.to_owned(),
        }
    } else {
        QualifiedName {
            namespace: optional(namespace),
            prefix: optional(prefix),
            local,
        }
    }
}
unsafe fn copy(pointer: *const c_char, size: usize, diagnostic: bool) -> Result<String, BookError> {
    if size > isize::MAX as usize || (size != 0 && pointer.is_null()) {
        return Err(safety("invalid native string length/pointer"));
    }
    if size == 0 {
        return Ok(String::new());
    }
    // 原生回调明确传入长度，内存归原生调用栈所有；此处只复制，不保存指针。
    let bytes = unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), size) };
    if diagnostic {
        return Ok(String::from_utf8_lossy(bytes).into_owned());
    }
    String::from_utf8(bytes.to_vec()).map_err(|_| {
        BookError::new(
            BookErrorKind::Encoding,
            "native-utf8",
            "native output is not UTF-8",
        )
    })
}
unsafe extern "C" fn callback(
    user: *mut c_void,
    kind: c_int,
    a: *const c_char,
    al: usize,
    b: *const c_char,
    bl: usize,
    c: *const c_char,
    cl: usize,
    d: *const c_char,
    dl: usize,
) -> c_int {
    if user.is_null() {
        return 0;
    }
    // userdata 指向同步原生调用期间独占借用的 Builder，所有异常在 ABI 边界截获。
    let builder = unsafe { &mut *user.cast::<Builder>() };
    if builder.error.is_some() {
        return 0;
    }
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), BookError> {
        let size = [al, bl, cl, dl]
            .into_iter()
            .try_fold(0_usize, usize::checked_add)
            .ok_or_else(|| safety("native event size overflow"))?;
        builder.bytes = builder
            .bytes
            .checked_add(size)
            .ok_or_else(|| safety("native event total overflow"))?;
        // 指针只在本回调内读取；复制完成后只把拥有所有权的字符串交给安全模型。
        let fields = unsafe {
            [
                copy(a, al, kind == 12)?,
                copy(b, bl, false)?,
                copy(c, cl, false)?,
                copy(d, dl, false)?,
            ]
        };
        builder.event(kind, fields)
    }));
    match result {
        Ok(Ok(())) => 1,
        Ok(Err(error)) => {
            builder.error = Some(error);
            0
        }
        Err(_) => {
            builder.error = Some(safety("panic contained at native callback boundary"));
            0
        }
    }
}
pub(super) fn parse(
    bytes: &[u8],
    expat: bool,
    html: bool,
    envelope: DocumentEnvelope,
) -> Result<ParsedDocument, BookError> {
    initialize()?;
    let length = if expat {
        c_int::try_from(bytes.len()).map_err(|_| safety("native decoded input length overflow"))?
    } else {
        length(bytes)?
    };
    let html_mode = if !html {
        0
    } else if envelope.mode == super::ParseMode::Xml {
        2
    } else {
        1
    };
    let mut builder = Builder {
        nodes: Vec::new(),
        stack: Vec::new(),
        root: None,
        before: Vec::new(),
        after: Vec::new(),
        namespaces: Vec::new(),
        diagnostics: Vec::new(),
        envelope,
        error: None,
        bytes: 0,
        bounded_depth: !expat,
    };
    // 原生解析同步结束并释放文档后才返回；传入字节与 Builder 在整个调用内存活。
    let status = unsafe {
        if expat {
            wenyi_expat(
                bytes.as_ptr().cast(),
                length,
                (&mut builder as *mut Builder).cast(),
                callback,
            )
        } else {
            wenyi_xml(
                bytes.as_ptr().cast(),
                length,
                html_mode,
                (&mut builder as *mut Builder).cast(),
                callback,
            )
        }
    };
    if let Some(error) = builder.error {
        return Err(error);
    }
    if status != 0 {
        let detail = builder
            .diagnostics
            .first()
            .map_or("native parser rejected source", |entry| {
                entry.reason.as_str()
            });
        return Err(native_error(status, detail));
    }
    if !builder.stack.is_empty() {
        return Err(safety("unbalanced native tree"));
    }
    let root = builder
        .root
        .ok_or_else(|| native_error(1, "native parser returned no root element"))?;
    Ok(ParsedDocument {
        tree: SourceTree {
            nodes: builder.nodes,
            root,
            before_root: builder.before,
            after_root: builder.after,
        },
        envelope: builder.envelope,
        diagnostics: builder.diagnostics,
    })
}
pub(super) fn convert(
    bytes: &[u8],
    from: &str,
    to: &str,
    policy: ConversionPolicy,
) -> Result<Vec<u8>, BookError> {
    initialize()?;
    let length =
        c_int::try_from(bytes.len()).map_err(|_| safety("conversion input length overflow"))?;
    if from.is_empty() || to.is_empty() || from.len() > 100 || to.len() > 100 {
        return Err(native_error(
            3,
            "encoding name length is outside the admitted bound",
        ));
    }
    let from = CString::new(from).map_err(|_| native_error(3, "encoding name contains NUL"))?;
    let to = CString::new(to).map_err(|_| native_error(3, "encoding name contains NUL"))?;
    let mut output = std::ptr::null_mut();
    let mut size = 0;
    // 转换器只借用输入与名称；输出由配对的 wenyi_free 释放。
    let status = unsafe {
        wenyi_convert(
            bytes.as_ptr().cast(),
            length,
            from.as_ptr(),
            to.as_ptr(),
            c_int::from(policy == ConversionPolicy::Replace),
            &mut output,
            &mut size,
        )
    };
    if status != 0 {
        return Err(native_error(
            status,
            "ICU conversion rejected input or encoding",
        ));
    }
    if output.is_null() || size < 0 {
        if !output.is_null() {
            unsafe { wenyi_free(output.cast()) };
        }
        return Err(safety("invalid native conversion result"));
    }
    // 长度经检查，复制后立即释放，外部永远拿不到原生指针。
    let result = unsafe { std::slice::from_raw_parts(output.cast::<u8>(), size as usize) }.to_vec();
    unsafe { wenyi_free(output.cast()) };
    Ok(result)
}
pub(super) fn versions() -> NativeVersions {
    // 库版本由已链接的原生实现提供；ICU 的有界栈缓冲区在复制前始终存活。
    let mut icu = [0 as c_char; 32];
    unsafe {
        wenyi_icu_version(icu.as_mut_ptr());
        NativeVersions {
            libxml2: CStr::from_ptr(wenyi_xml_version())
                .to_string_lossy()
                .into_owned(),
            expat: CStr::from_ptr(wenyi_expat_version())
                .to_string_lossy()
                .into_owned(),
            icu: CStr::from_ptr(icu.as_ptr()).to_string_lossy().into_owned(),
        }
    }
}

struct NativeDocument(*mut c_void);
impl Drop for NativeDocument {
    fn drop(&mut self) {
        // 临时原生文档独占所有重建节点，任何错误路径都通过同一个析构释放。
        unsafe { wenyi_document_free(self.0) };
    }
}
fn cstring(value: &str) -> Result<CString, BookError> {
    CString::new(value).map_err(|_| source("owned XML value contains NUL"))
}
fn checked(status: c_int) -> Result<(), BookError> {
    if status == 0 {
        Ok(())
    } else {
        Err(native_error(
            status,
            "native XML reconstruction/serialization failed",
        ))
    }
}
struct Reconstruction<'a> {
    tree: &'a SourceTree,
    document: NativeDocument,
    visited: std::collections::HashSet<NodeId>,
}
impl Reconstruction<'_> {
    fn namespaces(
        &self,
        pointer: *mut c_void,
        declarations: &[NamespaceDecl],
    ) -> Result<(), BookError> {
        for declaration in declarations {
            let prefix = cstring(declaration.prefix.as_deref().unwrap_or(""))?;
            let uri = cstring(&declaration.uri)?;
            // 原生节点归临时文档所有，名称缓冲区只在同步构造调用内借用。
            checked(unsafe { wenyi_namespace(pointer, prefix.as_ptr(), uri.as_ptr()) })?;
        }
        Ok(())
    }
    fn element(
        &self,
        pointer: *mut c_void,
        element: &ElementData,
        inherited: &[NamespaceDecl],
    ) -> Result<(), BookError> {
        self.namespaces(pointer, &element.namespaces)?;
        self.namespaces(pointer, inherited)?;
        let prefix = cstring(element.name.prefix.as_deref().unwrap_or(""))?;
        let uri = cstring(element.name.namespace.as_deref().unwrap_or(""))?;
        checked(unsafe { wenyi_element_namespace(pointer, prefix.as_ptr(), uri.as_ptr()) })?;
        for attribute in &element.attributes {
            let name = cstring(&attribute.name.local)?;
            let prefix = cstring(attribute.name.prefix.as_deref().unwrap_or(""))?;
            let uri = cstring(attribute.name.namespace.as_deref().unwrap_or(""))?;
            let value = cstring(&attribute.value)?;
            checked(unsafe {
                wenyi_attribute(
                    pointer,
                    name.as_ptr(),
                    prefix.as_ptr(),
                    uri.as_ptr(),
                    value.as_ptr(),
                )
            })?;
        }
        Ok(())
    }
    fn text(
        &self,
        parent: *mut c_void,
        text: &Option<String>,
        ranges: &[std::ops::Range<usize>],
    ) -> Result<(), BookError> {
        let Some(text) = text else {
            return Ok(());
        };
        if text.is_empty() && ranges.is_empty() {
            return checked(unsafe {
                wenyi_text(self.document.0, parent, text.as_ptr().cast(), 0, 0)
            });
        }
        let mut cursor = 0;
        for range in ranges {
            if range.start < cursor || range.end < range.start {
                return Err(source("CDATA boundaries no longer match owned source text"));
            }
            let plain = text
                .get(cursor..range.start)
                .ok_or_else(|| source("invalid CDATA text boundary"))?;
            let section = text
                .get(range.clone())
                .ok_or_else(|| source("invalid CDATA section boundary"))?;
            if section.contains("]]>") {
                return Err(source("invalid owned CDATA section"));
            }
            self.segment(parent, plain, false)?;
            self.segment(parent, section, true)?;
            cursor = range.end;
        }
        self.segment(
            parent,
            text.get(cursor..)
                .ok_or_else(|| source("invalid text boundary"))?,
            false,
        )
    }
    fn segment(&self, parent: *mut c_void, text: &str, cdata: bool) -> Result<(), BookError> {
        if text.is_empty() && !cdata {
            return Ok(());
        }
        if text.contains('\0') {
            return Err(source("owned XML text contains NUL"));
        }
        let length =
            c_int::try_from(text.len()).map_err(|_| safety("native text length overflow"))?;
        checked(unsafe {
            wenyi_text(
                self.document.0,
                parent,
                text.as_ptr().cast(),
                length,
                c_int::from(cdata),
            )
        })
    }
    fn node(
        &mut self,
        id: NodeId,
        parent: *mut c_void,
        inherited: &[NamespaceDecl],
    ) -> Result<*mut c_void, BookError> {
        if !self.visited.insert(id) {
            return Err(safety("serialization tree cycle detected"));
        }
        let node = self
            .tree
            .node(id)
            .ok_or_else(|| source("invalid serialization node"))?;
        let (kind, name, value) = match &node.kind {
            NodeKind::Element(element) => (1, element.name.local.as_str(), ""),
            NodeKind::Comment(value) => {
                if value.contains("--") || value.ends_with('-') {
                    return Err(source("invalid owned XML comment"));
                }
                (7, "", value.as_str())
            }
            NodeKind::ProcessingInstruction { target, data } => {
                if target.contains([' ', '<', '>']) || data.contains("?>") {
                    return Err(source("invalid owned XML processing instruction"));
                }
                (8, target.as_str(), data.as_str())
            }
            NodeKind::EntityReference(name) => (9, name.as_str(), ""),
        };
        let name = cstring(name)?;
        let value = cstring(value)?;
        let pointer =
            unsafe { wenyi_node_new(self.document.0, parent, kind, name.as_ptr(), value.as_ptr()) };
        if pointer.is_null() {
            return Err(safety("native XML node allocation failed"));
        }
        if let NodeKind::Element(element) = &node.kind {
            self.element(pointer, element, inherited)?;
            self.text(pointer, &node.text, &node.text_cdata)?;
        } else if !node.children.is_empty() || node.text.is_some() {
            return Err(source("opaque source node has element content"));
        }
        Ok(pointer)
    }
    fn branch(&mut self, root: NodeId, inherited: &[NamespaceDecl]) -> Result<(), BookError> {
        let tree = self.tree;
        let mut pending = vec![(root, std::ptr::null_mut(), false)];
        while let Some((id, parent, tail)) = pending.pop() {
            let node = tree
                .node(id)
                .ok_or_else(|| source("invalid serialization node"))?;
            if tail {
                self.text(parent, &node.tail, &node.tail_cdata)?;
                continue;
            }
            let pointer = self.node(id, parent, if id == root { inherited } else { &[] })?;
            for &child in node.children.iter().rev() {
                if tree.node(child).is_none_or(|node| node.parent != Some(id)) {
                    return Err(source("inconsistent source parent"));
                }
                pending.push((child, pointer, true));
                pending.push((child, pointer, false));
            }
        }
        Ok(())
    }
}
fn inherited_namespaces(tree: &SourceTree, id: NodeId) -> Result<Vec<NamespaceDecl>, BookError> {
    let root = tree.node(id).ok_or_else(|| source("invalid block node"))?;
    let mut prefixes = std::collections::HashSet::new();
    if let NodeKind::Element(element) = &root.kind {
        for ns in &element.namespaces {
            prefixes.insert(ns.prefix.clone());
        }
    }
    let mut inherited = Vec::new();
    let mut parent = root.parent;
    let mut seen = std::collections::HashSet::from([id]);
    while let Some(id) = parent {
        if !seen.insert(id) {
            return Err(safety("block ancestor cycle detected"));
        }
        let owner = tree
            .node(id)
            .ok_or_else(|| source("invalid block ancestor"))?;
        if let NodeKind::Element(element) = &owner.kind {
            for ns in &element.namespaces {
                if prefixes.insert(ns.prefix.clone()) {
                    inherited.push(ns.clone());
                }
            }
        }
        parent = owner.parent;
    }
    Ok(inherited)
}
pub(super) fn serialize(
    tree: &SourceTree,
    envelope: Option<&DocumentEnvelope>,
    subtree: Option<NodeId>,
) -> Result<Vec<u8>, BookError> {
    initialize()?;
    let document = NativeDocument(unsafe { wenyi_document_new() });
    if document.0.is_null() {
        return Err(safety("native XML document allocation failed"));
    }
    let mut reconstruction = Reconstruction {
        tree,
        document,
        visited: std::collections::HashSet::new(),
    };
    if let Some(id) = subtree {
        let inherited = inherited_namespaces(tree, id)?;
        reconstruction.branch(id, &inherited)?;
    } else {
        for &id in &tree.before_root {
            reconstruction.branch(id, &[])?;
        }
        reconstruction.branch(tree.root, &[])?;
        for &id in &tree.after_root {
            reconstruction.branch(id, &[])?;
        }
    }
    let encoding = cstring(envelope.map_or("UTF-8", |value| {
        if value.mode == super::ParseMode::RecoveredHtml {
            "UTF-8"
        } else {
            value.encoding.as_str()
        }
    }))?;
    let mut output = std::ptr::null_mut();
    let mut size = 0;
    checked(unsafe {
        wenyi_document_dump(
            reconstruction.document.0,
            encoding.as_ptr(),
            &mut output,
            &mut size,
        )
    })?;
    if output.is_null() || size < 0 {
        if !output.is_null() {
            unsafe { wenyi_free(output.cast()) };
        }
        return Err(safety("invalid native serialization result"));
    }
    let bytes = unsafe { std::slice::from_raw_parts(output.cast::<u8>(), size as usize) }.to_vec();
    unsafe { wenyi_free(output.cast()) };
    Ok(bytes)
}
