#![forbid(unsafe_code)]
mod compute;
mod load;
mod select;
mod values;

use crate::archive::Archive;
use crate::error::{BookError, BookErrorKind};
use crate::source::{NodeId, SourceTree};
use lightningcss::declaration::DeclarationBlock;
use wenyi_core::SemanticRole;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CssAsset {
    pub href: String,
    pub bytes: Vec<u8>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementStyle {
    pub node: NodeId,
    pub roles: Vec<SemanticRole>,
    pub active_emphasis: Vec<(NodeId, SemanticRole)>,
    pub required: bool,
    pub decoration: bool,
    pub evidence: Vec<String>,
}
struct Rule {
    selection: select::Selection,
    declarations: DeclarationBlock<'static>,
    conditions: Vec<usize>,
    evidence: String,
    unsupported: bool,
}
pub struct SourceCss {
    assets: Vec<CssAsset>,
    rules: Vec<Rule>,
    conditions: Vec<String>,
    inline: Vec<(NodeId, DeclarationBlock<'static>)>,
}
impl SourceCss {
    pub fn load(
        archive: &mut Archive,
        tree: &SourceTree,
        resource: &str,
    ) -> Result<Self, BookError> {
        load::load(archive, tree, resource)
    }
    pub fn evidence(
        &self,
        tree: &SourceTree,
        nodes: &[NodeId],
        resource: &str,
    ) -> Result<Vec<ElementStyle>, BookError> {
        compute::evidence(self, tree, nodes, resource)
    }
    pub fn assets(&self) -> &[CssAsset] {
        &self.assets
    }
}
fn error(resource: &str, code: &'static str, detail: impl Into<String>) -> BookError {
    BookError::new(BookErrorKind::Css, code, detail).with_resource(resource)
}
