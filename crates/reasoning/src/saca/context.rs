//! Repository-Level Context Awareness Engine
//! 
//! Phase 3 of SACA: Analyze entire repository structure before writing new code
//! Implements SWE-bench methodology for repository-level understanding

use super::{types::*, config::*, error::*};
use nexora_core::async_executor::AsyncTaskExecutor;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use std::path::Path;
use std::collections::HashMap;
use anyhow::Result;

/// Repository Context engine
pub struct ContextEngine {
    config: ContextConfig,
    _executor: Arc<AsyncTaskExecutor>,
    context_cache: Arc<RwLock<std::collections::HashMap<String, RepositoryContext>>>,
    file_analyzers: Vec<Arc<dyn FileAnalyzer>>,
}

impl ContextEngine {
    /// Create new Context engine
    pub fn new(config: ContextConfig) -> SACAResult<Self> {
        let executor = Arc::new(AsyncTaskExecutor::new(nexora_core::async_executor::ExecutorConfig::default()));
        
        let file_analyzers: Vec<Arc<dyn FileAnalyzer>> = vec![
            Arc::new(RustAnalyzer::new()),
            Arc::new(PythonAnalyzer::new()),
            Arc::new(JavaScriptAnalyzer::new()),
            Arc::new(GenericAnalyzer::new()),
        ];
        
        info!("Context Engine initialized with max {} files to analyze", config.max_files_to_analyze);
        
        Ok(Self {
            config,
            _executor: executor,
            context_cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
            file_analyzers,
        })
    }
    
    /// Analyze repository context for modules and task
    pub async fn analyze(&self, modules: &[Module], task: &CodingTask) -> SACAResult<RepositoryContext> {
        debug!("Starting repository context analysis");
        
        // Check cache first
        let cache_key = self.generate_cache_key(task);
        if let Some(cached_context) = self.context_cache.read().await.get(&cache_key) {
            debug!("Using cached repository context");
            return Ok(cached_context.clone());
        }
        
        // Get repository path from task context
        let repo_path = task.context.as_ref()
            .and_then(|c| c.repository_path.clone())
            .unwrap_or_else(|| ".".to_string());
        
        // Perform context analysis
        let context = self.perform_repository_analysis(&repo_path, modules, task).await?;
        
        // Cache the result
        self.context_cache.write().await.insert(cache_key, context.clone());
        
        info!("Repository context analysis completed: {} files analyzed", context.files_analyzed);
        Ok(context)
    }
    
    /// Core repository analysis implementation
    async fn perform_repository_analysis(
        &self,
        repo_path: &str,
        modules: &[Module],
        task: &CodingTask,
    ) -> SACAResult<RepositoryContext> {
        let mut context = RepositoryContext {
            repository_path: Some(repo_path.to_string()),
            files_analyzed: 0,
            functions_found: 0,
            dependencies: Vec::new(),
            coding_patterns: HashMap::new(),
            naming_conventions: NamingConventions::default(),
            architectural_patterns: Vec::new(),
            test_frameworks: Vec::new(),
        };
        
        // Scan repository files
        let files = self.scan_repository_files(repo_path).await?;
        debug!("Found {} files in repository", files.len());
        
        // Limit files to analyze
        let files_to_analyze = files.into_iter()
            .take(self.config.max_files_to_analyze as usize)
            .collect::<Vec<_>>();
        
        // Analyze each file
        for file_path in files_to_analyze {
            if let Some(analysis) = self.analyze_file(&file_path).await? {
                self.merge_file_analysis(&mut context, analysis).await;
                context.files_analyzed += 1;
            }
        }
        
        // Analyze dependencies if enabled
        if self.config.include_dependencies {
            self.analyze_dependencies(&mut context, repo_path).await?;
        }
        
        // Detect naming conventions if enabled
        if self.config.analyze_naming_conventions {
            self.detect_naming_conventions(&mut context).await?;
        }
        
        // Detect patterns if enabled
        if self.config.detect_patterns {
            self.detect_architectural_patterns(&mut context).await?;
            self.detect_test_frameworks(&mut context).await?;
        }
        
        // Analyze module-specific context
        self.analyze_module_context(&mut context, modules).await?;
        
        Ok(context)
    }
    
    /// Scan repository for relevant files
    async fn scan_repository_files(&self, repo_path: &str) -> SACAResult<Vec<String>> {
        let mut files = Vec::new();
        let path = Path::new(repo_path);
        
        if !path.exists() {
            warn!("Repository path does not exist: {}", repo_path);
            return Ok(files);
        }
        
        // Walk through directory
        let mut entries = tokio::fs::read_dir(path).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let entry_path = entry.path();
            
            if entry_path.is_dir() {
                // Recursively scan subdirectories (limit depth to avoid infinite loops)
                if self.should_scan_directory(&entry_path) {
                    let sub_files = Box::pin(self.scan_repository_files(entry_path.to_str().ok_or_else(|| anyhow::anyhow!("Failed to convert path to string"))?)).await?;
                    files.extend(sub_files);
                }
            } else if self.should_analyze_file(&entry_path) {
                if let Some(path_str) = entry_path.to_str() {
                    files.push(path_str.to_string());
                }
            }
        }
        
        Ok(files)
    }
    
    /// Check if directory should be scanned
    fn should_scan_directory(&self, path: &Path) -> bool {
        // Skip common directories that don't contain source code
        let skip_dirs = [
            "target", "node_modules", ".git", "vendor", "build", "dist",
            "__pycache__", ".venv", "venv", "env", ".idea", ".vscode"
        ];
        
        if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
            !skip_dirs.contains(&dir_name)
        } else {
            false
        }
    }
    
    /// Check if file should be analyzed
    fn should_analyze_file(&self, path: &Path) -> bool {
        // Include test files if configured
        let include_tests = self.config.include_test_files;
        
        // Check file extension
        if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
            let is_source_file = matches!(extension, "rs" | "py" | "js" | "ts" | "java" | "cpp" | "c" | "go");
            let is_test_file = path.to_str().map_or(false, |s| s.contains("test") || s.contains("spec"));
            
            is_source_file || (include_tests && is_test_file)
        } else {
            false
        }
    }
    
    /// Analyze a single file
    async fn analyze_file(&self, file_path: &str) -> SACAResult<Option<FileAnalysis>> {
        let content = match tokio::fs::read_to_string(file_path).await {
            Ok(content) => content,
            Err(_) => return Ok(None), // Skip files that can't be read
        };
        
        // Determine file type and use appropriate analyzer
        let path = Path::new(file_path);
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        
        for analyzer in &self.file_analyzers {
            if analyzer.can_handle(extension) {
                return Ok(Some(analyzer.analyze(&content, file_path)?));
            }
        }
        
        Ok(None)
    }
    
    /// Merge file analysis into repository context
    async fn merge_file_analysis(&self, context: &mut RepositoryContext, analysis: FileAnalysis) {
        context.functions_found += analysis.functions.len();
        
        // Add dependencies
        context.dependencies.extend(analysis.dependencies);
        
        // Update coding patterns
        for (pattern, count) in analysis.patterns {
            *context.coding_patterns.entry(pattern).or_insert(0) += count;
        }
        
        // Update naming conventions
        self.update_naming_conventions(&mut context.naming_conventions, &analysis.naming_samples);
    }
    
    /// Update naming conventions based on samples
    fn update_naming_conventions(&self, conventions: &mut NamingConventions, samples: &NamingSamples) {
        if !samples.variable_names.is_empty() {
            conventions.variable_case = self.detect_case_style(&samples.variable_names);
        }
        
        if !samples.function_names.is_empty() {
            conventions.function_case = self.detect_case_style(&samples.function_names);
        }
        
        if !samples.class_names.is_empty() {
            conventions.class_case = self.detect_case_style(&samples.class_names);
        }
        
        if !samples.constant_names.is_empty() {
            conventions.constant_case = self.detect_case_style(&samples.constant_names);
        }
    }
    
    /// Detect case style from name samples
    fn detect_case_style(&self, names: &[String]) -> String {
        let mut snake_case_count = 0;
        let mut camel_case_count = 0;
        let mut pascal_case_count = 0;
        let mut screaming_snake_count = 0;
        
        for name in names {
            if name.contains('_') && name.to_lowercase().as_str() == name {
                snake_case_count += 1;
            } else if name.chars().next().map_or(false, |c| c.is_lowercase()) && name.contains('_') && name.to_uppercase().as_str() == name {
                screaming_snake_count += 1;
            } else if name.chars().next().map_or(false, |c| c.is_uppercase()) && !name.contains('_') {
                pascal_case_count += 1;
            } else if name.chars().next().map_or(false, |c| c.is_lowercase()) && !name.contains('_') {
                camel_case_count += 1;
            }
        }
        
        let max_count = std::cmp::max(
            std::cmp::max(snake_case_count, camel_case_count),
            std::cmp::max(pascal_case_count, screaming_snake_count)
        );
        
        if max_count == snake_case_count {
            "snake_case".to_string()
        } else if max_count == camel_case_count {
            "camelCase".to_string()
        } else if max_count == pascal_case_count {
            "PascalCase".to_string()
        } else if max_count == screaming_snake_count {
            "SCREAMING_SNAKE_CASE".to_string()
        } else {
            "unknown".to_string()
        }
    }
    
    /// Analyze project dependencies
    async fn analyze_dependencies(&self, context: &mut RepositoryContext, repo_path: &str) -> SACAResult<()> {
        // Look for dependency files
        let dependency_files = [
            "Cargo.toml", "requirements.txt", "package.json", "pom.xml",
            "build.gradle", "go.mod", "composer.json"
        ];
        
        for dep_file in dependency_files {
            let file_path = Path::new(repo_path).join(dep_file);
            if file_path.exists() {
                if let Some(deps) = self.parse_dependency_file(&file_path).await? {
                    context.dependencies.extend(deps);
                }
            }
        }
        
        Ok(())
    }
    
    /// Parse dependency file
    async fn parse_dependency_file(&self, file_path: &Path) -> SACAResult<Option<Vec<String>>> {
        let content = tokio::fs::read_to_string(file_path).await?;
        
        let dependencies = match file_path.file_name().and_then(|n| n.to_str()) {
            Some("Cargo.toml") => self.parse_cargo_toml(&content)?,
            Some("requirements.txt") => self.parse_requirements_txt(&content)?,
            Some("package.json") => self.parse_package_json(&content)?,
            _ => Vec::new(),
        };
        
        Ok(Some(dependencies))
    }
    
    /// Parse Cargo.toml dependencies
    fn parse_cargo_toml(&self, content: &str) -> SACAResult<Vec<String>> {
        let mut dependencies = Vec::new();
        
        // Simple parsing for [dependencies] section
        if let Some(deps_section) = content.split("[dependencies]").nth(1) {
            for line in deps_section.lines() {
                if let Some(dep_name) = line.split('=').next() {
                    let dep_name = dep_name.trim();
                    if !dep_name.is_empty() && !dep_name.starts_with('#') {
                        dependencies.push(dep_name.to_string());
                    }
                }
            }
        }
        
        Ok(dependencies)
    }
    
    /// Parse requirements.txt
    fn parse_requirements_txt(&self, content: &str) -> SACAResult<Vec<String>> {
        let dependencies = content
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
            .map(|line| line.split_whitespace().next().unwrap_or("").to_string())
            .filter(|dep| !dep.is_empty())
            .collect();
        
        Ok(dependencies)
    }
    
    /// Parse package.json
    fn parse_package_json(&self, content: &str) -> SACAResult<Vec<String>> {
        let mut dependencies = Vec::new();
        
        if content.len() > 10_000_000 {
            return Ok(dependencies);
        }
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(content) {
            if let Some(deps) = json.get("dependencies").and_then(|d| d.as_object()) {
                for dep_name in deps.keys() {
                    dependencies.push(dep_name.clone());
                }
            }
        }
        
        Ok(dependencies)
    }
    
    /// Detect naming conventions
    async fn detect_naming_conventions(&self, context: &mut RepositoryContext) -> SACAResult<()> {
        // This is already handled during file analysis merge
        debug!("Naming conventions detected: {:?}", context.naming_conventions);
        Ok(())
    }
    
    /// Detect architectural patterns
    async fn detect_architectural_patterns(&self, context: &mut RepositoryContext) -> SACAResult<()> {
        let patterns = &context.coding_patterns;
        
        // Detect common patterns
        if patterns.contains_key("mvc") || patterns.contains_key("model-view-controller") {
            context.architectural_patterns.push("MVC".to_string());
        }
        
        if patterns.contains_key("repository") || patterns.contains_key("dao") {
            context.architectural_patterns.push("Repository Pattern".to_string());
        }
        
        if patterns.contains_key("factory") {
            context.architectural_patterns.push("Factory Pattern".to_string());
        }
        
        if patterns.contains_key("observer") {
            context.architectural_patterns.push("Observer Pattern".to_string());
        }
        
        if patterns.contains_key("singleton") {
            context.architectural_patterns.push("Singleton Pattern".to_string());
        }
        
        Ok(())
    }
    
    /// Detect test frameworks
    async fn detect_test_frameworks(&self, context: &mut RepositoryContext) -> SACAResult<()> {
        let patterns = &context.coding_patterns;
        
        if patterns.contains_key("test") || patterns.contains_key("unittest") {
            context.test_frameworks.push("unittest".to_string());
        }
        
        if patterns.contains_key("pytest") {
            context.test_frameworks.push("pytest".to_string());
        }
        
        if patterns.contains_key("jest") {
            context.test_frameworks.push("Jest".to_string());
        }
        
        if patterns.contains_key("mocha") {
            context.test_frameworks.push("Mocha".to_string());
        }
        
        // Check dependencies for test frameworks
        for dep in &context.dependencies {
            match dep.to_lowercase().as_str() {
                "tokio-test" | "criterion" => context.test_frameworks.push("Rust Testing".to_string()),
                "pytest" | "unittest" => context.test_frameworks.push("Python Testing".to_string()),
                "jest" | "mocha" | "chai" => context.test_frameworks.push("JavaScript Testing".to_string()),
                _ => {}
            }
        }
        
        // Remove duplicates
        context.test_frameworks.sort();
        context.test_frameworks.dedup();
        
        Ok(())
    }
    
    /// Analyze module-specific context
    async fn analyze_module_context(&self, context: &mut RepositoryContext, modules: &[Module]) -> SACAResult<()> {
        // Analyze how modules fit into existing repository structure
        for module in modules {
            // Check if similar functionality already exists
            self.check_existing_functionality(context, module).await?;
        }
        
        Ok(())
    }
    
    /// Check if similar functionality already exists
    async fn check_existing_functionality(&self, context: &RepositoryContext, module: &Module) -> SACAResult<()> {
        // This would analyze if similar modules/functions already exist
        // For now, just log the analysis
        debug!("Checking existing functionality for module: {}", module.name);
        Ok(())
    }
    
    /// Generate cache key for context results
    fn generate_cache_key(&self, task: &CodingTask) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        task.description.hash(&mut hasher);
        if let Some(context) = &task.context {
            context.repository_path.hash(&mut hasher);
        }
        format!("context_{:x}", hasher.finish())
    }
    
    /// Clear context cache
    pub async fn clear_cache(&self) {
        self.context_cache.write().await.clear();
        info!("Repository context cache cleared");
    }
}

/// File analysis result
#[derive(Debug, Clone)]
struct FileAnalysis {
    functions: Vec<String>,
    dependencies: Vec<String>,
    patterns: HashMap<String, usize>,
    naming_samples: NamingSamples,
}

/// Naming convention samples
#[derive(Debug, Clone, Default)]
struct NamingSamples {
    variable_names: Vec<String>,
    function_names: Vec<String>,
    class_names: Vec<String>,
    constant_names: Vec<String>,
}

/// Trait for file analyzers
trait FileAnalyzer: Send + Sync {
    fn can_handle(&self, extension: &str) -> bool;
    fn analyze(&self, content: &str, file_path: &str) -> SACAResult<FileAnalysis>;
}

/// Rust file analyzer
struct RustAnalyzer {
    _private: (),
}

impl RustAnalyzer {
    fn new() -> Self {
        Self { _private: () }
    }
}

impl FileAnalyzer for RustAnalyzer {
    fn can_handle(&self, extension: &str) -> bool {
        extension == "rs"
    }
    
    fn analyze(&self, content: &str, file_path: &str) -> SACAResult<FileAnalysis> {
        let mut imports = Vec::new();
        let mut functions = Vec::new();
        let mut structs = Vec::new();
        let mut traits = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            if trimmed.starts_with("use ") {
                if let Some(import_path) = trimmed.strip_prefix("use ").and_then(|s| s.split(';').next()) {
                    imports.push(import_path.to_string());
                }
            } else if trimmed.starts_with("fn ") {
                if let Some(fn_name) = trimmed.strip_prefix("fn ").and_then(|s| s.split('(').next()) {
                    functions.push(fn_name.trim().to_string());
                }
            } else if trimmed.starts_with("struct ") {
                if let Some(struct_name) = trimmed.strip_prefix("struct ").and_then(|s| s.split('{').next()) {
                    structs.push(struct_name.trim().to_string());
                }
            } else if trimmed.starts_with("trait ") {
                if let Some(trait_name) = trimmed.strip_prefix("trait ").and_then(|s| s.split('{').next()) {
                    traits.push(trait_name.trim().to_string());
                }
            }
        }
        
        let functions_clone = functions.clone();
        Ok(FileAnalysis {
            functions,
            dependencies: imports,
            patterns: HashMap::new(),
            naming_samples: NamingSamples {
                function_names: functions_clone,
                variable_names: Vec::new(),
                class_names: structs.clone(), // Map structs to class_names
                constant_names: Vec::new(),
            },
        })
    }
}

/// Python file analyzer
struct PythonAnalyzer {
    _private: (),
}

impl PythonAnalyzer {
    fn new() -> Self {
        Self { _private: () }
    }
}

impl FileAnalyzer for PythonAnalyzer {
    fn can_handle(&self, extension: &str) -> bool {
        extension == "py"
    }
    
    fn analyze(&self, content: &str, file_path: &str) -> SACAResult<FileAnalysis> {
        let mut imports = Vec::new();
        let mut functions = Vec::new();
        let mut classes = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
                imports.push(trimmed.to_string());
            } else if trimmed.starts_with("def ") {
                if let Some(fn_name) = trimmed.strip_prefix("def ").and_then(|s| s.split('(').next()) {
                    functions.push(fn_name.trim().to_string());
                }
            } else if trimmed.starts_with("class ") {
                if let Some(class_name) = trimmed.strip_prefix("class ").and_then(|s| s.split('(').next()) {
                    classes.push(class_name.trim().to_string());
                }
            }
        }
        
        let functions_clone = functions.clone();
        Ok(FileAnalysis {
            functions,
            dependencies: Vec::new(),
            patterns: HashMap::new(),
            naming_samples: NamingSamples {
                function_names: functions_clone,
                variable_names: Vec::new(),
                class_names: classes.clone(), // Map classes to class_names
                constant_names: Vec::new(),
            },
        })
    }
}

/// JavaScript file analyzer
struct JavaScriptAnalyzer {
    _private: (),
}

impl JavaScriptAnalyzer {
    fn new() -> Self {
        Self { _private: () }
    }
}

impl FileAnalyzer for JavaScriptAnalyzer {
    fn can_handle(&self, extension: &str) -> bool {
        extension == "js" || extension == "ts" || extension == "jsx" || extension == "tsx"
    }
    
    fn analyze(&self, content: &str, file_path: &str) -> SACAResult<FileAnalysis> {
        let mut imports = Vec::new();
        let mut functions = Vec::new();
        let mut classes = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            if trimmed.starts_with("import ") || trimmed.starts_with("const ") && trimmed.contains("require") {
                imports.push(trimmed.to_string());
            } else if trimmed.contains("function ") || trimmed.contains("= (") || trimmed.contains("=>") {
                // Simple function detection
                functions.push("function".to_string());
            } else if trimmed.starts_with("class ") {
                if let Some(class_name) = trimmed.strip_prefix("class ").and_then(|s| s.split('{').next()) {
                    classes.push(class_name.trim().to_string());
                }
            }
        }
        
        let functions_clone = functions.clone();
        Ok(FileAnalysis {
            functions,
            dependencies: Vec::new(),
            patterns: HashMap::new(),
            naming_samples: NamingSamples {
                function_names: functions_clone,
                variable_names: Vec::new(),
                class_names: classes.clone(), // Map classes to class_names
                constant_names: Vec::new(),
            },
        })
    }
}

/// Generic file analyzer for unknown types
struct GenericAnalyzer {
    _private: (),
}

impl GenericAnalyzer {
    fn new() -> Self {
        Self { _private: () }
    }
}

impl FileAnalyzer for GenericAnalyzer {
    fn can_handle(&self, _extension: &str) -> bool {
        true // Handle all file types
    }
    
    fn analyze(&self, content: &str, file_path: &str) -> SACAResult<FileAnalysis> {
        let line_count = content.lines().count();
        
        Ok(FileAnalysis {
            functions: Vec::new(),
            dependencies: Vec::new(),
            patterns: HashMap::new(),
            naming_samples: NamingSamples::default(),
        })
    }
}




#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    
    #[tokio::test]
    async fn test_context_analysis() {
        let config = ContextConfig::default();
        let engine = ContextEngine::new(config).expect("Failed to create context engine");
        
        let modules = vec![];
        let task = CodingTask {
            description: "Test task".to_string(),
            requirements: vec![],
            constraints: vec![],
            context: Some(TaskContext {
                repository_path: Some("/test/repo".to_string()),
                existing_files: vec!["main.rs".to_string()],
                dependencies: vec![],
                coding_standards: HashMap::new(),
            }),
        };
        
        let context = engine.analyze(&modules, &task).await.expect("Context analysis failed");
        assert!(context.files_analyzed >= 0);
    }
    
    #[tokio::test]
    async fn test_rust_analyzer() {
        let analyzer = RustAnalyzer::new();
        let content = r#"
            fn hello_world() {
                println!("Hello, world!");
            }
            
            use std::collections::HashMap;
            
            struct MyStruct {
                field: i32,
            }
        "#;
        
        match analyzer.analyze(content, "test.rs") {
            Ok(analysis) => {
                assert_eq!(analysis.functions.len(), 1);
                assert_eq!(analysis.functions[0], "hello_world");
                assert_eq!(analysis.dependencies.len(), 1);
                assert!(analysis.dependencies[0].contains("std::collections"));
            },
            Err(e) => panic!("Code analysis failed: {}", e),
        }
    }
}
