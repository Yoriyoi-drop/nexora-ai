//! NXR-VORTEX Agents Module
//!
//! Production-ready agent implementations for code analysis and debugging.
//! Rich BaseAgent implementations available as submodules:
//!   - mod code_sentinel (full analysis engine)
//!   - mod debug_phantom (full debug engine)
//!   - mod arch_weaver (full architecture engine)
//!   - mod test_forge (full test generation engine)

pub mod arch_weaver;
pub mod code_sentinel;
pub mod debug_phantom;
pub mod test_forge;

pub use arch_weaver::ArchWeaverAgent;
pub use code_sentinel::CodeSentinelAgent;
pub use debug_phantom::DebugPhantomAgent;
pub use test_forge::TestForgeAgent;

fn detect_language(code: &str) -> String {
    let extensions = vec![
        (
            "rust",
            vec![
                "fn ", "let ", "impl ", "struct ", "enum ", "pub ", "use ", "mod ", "mut ", "crate",
            ],
        ),
        (
            "python",
            vec![
                "def",
                "import",
                "class",
                "async def",
                "yield",
                "if __name__",
                "lambda",
                "raise ",
                "except",
            ],
        ),
        (
            "typescript",
            vec![
                "interface",
                "type",
                "const",
                "let",
                "function",
                "=>",
                "export",
                "import",
                "as",
                "<T>",
                ": string",
                ": number",
            ],
        ),
        (
            "javascript",
            vec![
                "function",
                "const",
                "let",
                "var",
                "=>",
                "export",
                "import",
                "require(",
                "console.log",
                "prototype",
                "this.",
            ],
        ),
        (
            "go",
            vec![
                "func",
                "package",
                "import (",
                "defer",
                "go",
                "han",
                "goroutine",
                "interface{}",
            ],
        ),
        (
            "java",
            vec![
                "public class",
                "private",
                "protected",
                "static",
                "void main",
                "extends",
                "implements",
                "@Override",
            ],
        ),
        (
            "cpp",
            vec![
                "#include",
                "int main",
                "std::",
                "::",
                "template<",
                "class",
                "virtual",
                "override",
            ],
        ),
        (
            "ruby",
            vec![
                "def ", "end", "require ", "class ", "module ", "attr_", "gem ", "do |",
            ],
        ),
    ];
    for &(lang, ref patterns) in &extensions {
        let score = patterns.iter().filter(|&p| code.contains(p)).count();
        if score >= 2 {
            return lang.to_string();
        }
    }
    "unknown".to_string()
}
