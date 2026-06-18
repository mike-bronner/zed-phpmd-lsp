use anyhow::Result;
use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;
use std::time::Instant;
use tokio::io::{stdin, stdout};
use tokio::process::Command as ProcessCommand;
use tokio::sync::Semaphore;
use tokio::time::{timeout, Duration};
use tower_lsp::jsonrpc::Result as LspResult;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use url::Url;
use uuid::Uuid;

#[derive(Debug, Deserialize, Serialize, Clone)]
struct InitializationOptions {
    rulesets: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct PhpmdSettings {
    rulesets: Option<String>,
}

#[derive(Debug, Clone)]
struct CompressedDocument {
    compressed_data: Vec<u8>,
    original_size: usize,
    checksum: String,
    compression_ratio: f32,
}

#[derive(Debug, Clone)]
struct CachedResults {
    diagnostics: Vec<Diagnostic>,
    result_id: String,
    generated_at: Instant,
    content_checksum: String, // Track content version to detect changes
}

#[derive(Debug, Clone)]
struct PhpmdLanguageServer {
    client: Client,
    // Compressed document storage to reduce memory usage
    open_docs: std::sync::Arc<std::sync::RwLock<HashMap<Url, CompressedDocument>>>,
    // Cache PHPMD results to avoid redundant analysis
    results_cache: std::sync::Arc<std::sync::RwLock<HashMap<Url, CachedResults>>>,
    // Memory tracking
    total_memory_usage: std::sync::Arc<AtomicUsize>,
    rulesets: std::sync::Arc<std::sync::RwLock<Option<String>>>, // None means use PHPMD defaults
    phpmd_path: std::sync::Arc<std::sync::RwLock<Option<String>>>,
    workspace_root: std::sync::Arc<std::sync::RwLock<Option<std::path::PathBuf>>>,
    // Limit concurrent PHPMD processes to prevent system overload
    process_semaphore: std::sync::Arc<Semaphore>,
}

/// Matches the true line number PHPMD embeds in a violation description, e.g.
/// "Remove error control operator '@' on line 264." — the reported
/// `beginLine`/`endLine` span the enclosing method, but the description points
/// at the real offending line.
static ON_LINE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"on line (\d+)").expect("on-line regex is valid"));

/// Parse the actual line number from a PHPMD violation description using the
/// `on line N` pattern. Returns `None` when the description carries no such
/// marker.
fn parse_line_from_description(description: &str) -> Option<u32> {
    ON_LINE_RE
        .captures(description)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse::<u32>().ok())
}

/// Determine the highlight range for an `ErrorControlOperator` violation.
///
/// PHPMD reports this rule with `beginLine`/`endLine` spanning the enclosing
/// method, so the default range logic underlines the whole method body. The
/// description, however, embeds the real operator line ("… on line N."). When
/// that line can be recovered the range collapses to that single line;
/// otherwise it falls back to the first reported line rather than propagating
/// the full method span.
fn error_control_operator_range(description: &str, begin_line: u32) -> (u32, u32) {
    match parse_line_from_description(description) {
        Some(line) => (line, line),
        None => (begin_line, begin_line),
    }
}

/// Locate the column at which the `@` operator begins on the given line so the
/// squiggle underlines the operator token rather than the leading whitespace.
/// Falls back to the first non-whitespace column when the line contains no `@`.
fn error_control_operator_start_char(line_content: &str) -> usize {
    match line_content.find('@') {
        Some(offset) => offset,
        None => line_content.len() - line_content.trim_start().len(),
    }
}

impl PhpmdLanguageServer {
    fn new(client: Client) -> Self {
        Self {
            client,
            open_docs: std::sync::Arc::new(std::sync::RwLock::new(HashMap::with_capacity(100))),
            results_cache: std::sync::Arc::new(std::sync::RwLock::new(HashMap::with_capacity(100))),
            total_memory_usage: std::sync::Arc::new(AtomicUsize::new(0)),
            rulesets: std::sync::Arc::new(std::sync::RwLock::new(None)), // Let PHPMD use its defaults
            phpmd_path: std::sync::Arc::new(std::sync::RwLock::new(None)),
            workspace_root: std::sync::Arc::new(std::sync::RwLock::new(None)),
            // Limit to 4 concurrent PHPMD processes to avoid overwhelming the system
            process_semaphore: std::sync::Arc::new(Semaphore::new(4)),
        }
    }

    fn compress_document(&self, content: &str) -> CompressedDocument {
        let start = Instant::now();
        let original_size = content.len();

        // Use LZ4 for fast compression
        let compressed_data = compress_prepend_size(content.as_bytes());
        let compressed_size = compressed_data.len();
        let compression_ratio = compressed_size as f32 / original_size as f32;

        // Compute checksum for cache invalidation
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let checksum = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();

        let elapsed = start.elapsed();
        eprintln!(
            "📦 PHPMD LSP: Compressed in {:.2}ms: {}KB → {}KB ({:.1}% ratio)",
            elapsed.as_secs_f64() * 1000.0,
            original_size / 1024,
            compressed_size / 1024,
            compression_ratio * 100.0
        );

        // Update memory tracking
        self.total_memory_usage
            .fetch_add(compressed_size, Ordering::Relaxed);

        CompressedDocument {
            compressed_data,
            original_size,
            checksum,
            compression_ratio,
        }
    }

    fn decompress_document(&self, doc: &CompressedDocument) -> Result<String> {
        let start = Instant::now();
        let decompressed = decompress_size_prepended(&doc.compressed_data)
            .map_err(|e| anyhow::anyhow!("Decompression failed: {}", e))?;

        let content = String::from_utf8(decompressed)
            .map_err(|e| anyhow::anyhow!("UTF-8 conversion failed: {}", e))?;

        let elapsed = start.elapsed();
        if elapsed.as_millis() > 5 {
            eprintln!(
                "⚠️ PHPMD LSP: Slow decompression: {:.2}ms for {}KB",
                elapsed.as_secs_f64() * 1000.0,
                doc.original_size / 1024
            );
        }

        Ok(content)
    }

    fn get_memory_usage_mb(&self) -> f32 {
        self.total_memory_usage.load(Ordering::Relaxed) as f32 / 1_048_576.0
    }

    fn log_memory_stats(&self) {
        if let Ok(docs) = self.open_docs.read() {
            let doc_count = docs.len();
            let total_original: usize = docs.values().map(|d| d.original_size).sum();
            let total_compressed: usize = docs.values().map(|d| d.compressed_data.len()).sum();
            let avg_ratio = if doc_count > 0 {
                docs.values().map(|d| d.compression_ratio).sum::<f32>() / doc_count as f32
            } else {
                0.0
            };

            eprintln!("📊 PHPMD LSP Memory Stats:");
            eprintln!("  📁 Documents: {}", doc_count);
            eprintln!(
                "  💾 Compressed: {:.1}MB (from {:.1}MB original)",
                total_compressed as f32 / 1_048_576.0,
                total_original as f32 / 1_048_576.0
            );
            eprintln!("  📉 Average compression: {:.1}%", avg_ratio * 100.0);
            eprintln!(
                "  🗄️ Results cached: {}",
                self.results_cache.read().map(|c| c.len()).unwrap_or(0)
            );
        }
    }

    fn get_phpmd_path(&self) -> String {
        // First check the cache
        if let Ok(guard) = self.phpmd_path.read() {
            if let Some(cached_path) = &*guard {
                eprintln!("📂 PHPMD LSP: Using cached PHPMD path: {}", cached_path);
                return cached_path.clone();
            }
        }

        eprintln!("🔍 PHPMD LSP: Detecting PHPMD path...");

        // Not cached, find and cache it
        let phpmd_path = {
            // First priority: Check for project-local vendor/bin/phpmd
            if let Ok(workspace_guard) = self.workspace_root.read() {
                if let Some(ref workspace_root) = *workspace_guard {
                    let vendor_phpmd = workspace_root.join("vendor/bin/phpmd");
                    eprintln!(
                        "🔍 PHPMD LSP: Checking for project PHPMD at: {}",
                        vendor_phpmd.display()
                    );

                    if vendor_phpmd.exists() {
                        eprintln!("✅ PHPMD LSP: Found project-local PHPMD");
                        vendor_phpmd.to_string_lossy().to_string()
                    } else {
                        eprintln!("❌ PHPMD LSP: No project-local PHPMD found");
                        self.get_bundled_or_system_phpmd()
                    }
                } else {
                    eprintln!("❌ PHPMD LSP: No workspace root available");
                    self.get_bundled_or_system_phpmd()
                }
            } else {
                eprintln!("❌ PHPMD LSP: Could not access workspace root");
                self.get_bundled_or_system_phpmd()
            }
        };

        eprintln!("🎯 PHPMD LSP: Final PHPMD path: {}", phpmd_path);

        // Cache the result
        if let Ok(mut guard) = self.phpmd_path.write() {
            *guard = Some(phpmd_path.clone());
        }

        phpmd_path
    }

    fn get_bundled_or_system_phpmd(&self) -> String {
        // Second priority: Check for bundled PHPMD
        if let Ok(current_exe) = std::env::current_exe() {
            if let Some(exe_dir) = current_exe.parent() {
                let bundled_phpmd = exe_dir.join("phpmd.phar");
                eprintln!(
                    "🔍 PHPMD LSP: Checking for bundled PHPMD at: {}",
                    bundled_phpmd.display()
                );

                if bundled_phpmd.exists() {
                    eprintln!("✅ PHPMD LSP: Found bundled PHPMD PHAR");
                    return bundled_phpmd.to_string_lossy().to_string();
                } else {
                    eprintln!("❌ PHPMD LSP: No bundled PHPMD found");
                }
            } else {
                eprintln!("❌ PHPMD LSP: Could not get LSP directory");
            }
        } else {
            eprintln!("❌ PHPMD LSP: Could not get current executable path");
        }

        // Third priority: Fall back to system phpmd
        eprintln!("🔄 PHPMD LSP: Using system phpmd");
        "phpmd".to_string()
    }

    fn discover_rulesets(&self, workspace_root: Option<&std::path::Path>) {
        eprintln!("🔍 PHPMD LSP: Discovering PHPMD configuration files...");

        if let Some(root) = workspace_root {
            let config_files = [
                "phpmd.xml",
                "phpmd.xml.dist",
                ".phpmd.xml",
                ".phpmd.xml.dist",
            ];

            for config_file in &config_files {
                let config_path = root.join(config_file);

                if config_path.exists() {
                    eprintln!(
                        "📄 PHPMD LSP: Checking potential config file: {}",
                        config_file
                    );

                    // Validate it's a valid XML file
                    if let Ok(contents) = fs::read_to_string(&config_path) {
                        // Basic XML validation - check if it contains ruleset definition
                        if contents.contains("<ruleset") && contents.contains("</ruleset>") {
                            if let Some(path_str) = config_path.to_str() {
                                eprintln!(
                                    "✅ PHPMD LSP: Using valid PHPMD config file: {}",
                                    path_str
                                );
                                eprintln!(
                                    "📋 PHPMD LSP: Config file contains {} bytes",
                                    contents.len()
                                );
                                if let Ok(mut rulesets_guard) = self.rulesets.write() {
                                    // Store the full path to the config file
                                    *rulesets_guard = Some(path_str.to_string());
                                }
                                return;
                            }
                        } else {
                            eprintln!("⚠️ PHPMD LSP: File {} exists but doesn't appear to be a valid PHPMD ruleset XML", config_file);
                        }
                    } else {
                        eprintln!("⚠️ PHPMD LSP: Could not read config file: {}", config_file);
                    }
                }
            }

            eprintln!("🔍 PHPMD LSP: No valid PHPMD config files found in project root");
        }

        // No config file found - use ALL available PHPMD rulesets for comprehensive analysis
        eprintln!("🎯 PHPMD LSP: Using all PHPMD rulesets as fallback (cleancode, codesize, controversial, design, naming, unusedcode)");
        if let Ok(mut rulesets_guard) = self.rulesets.write() {
            // Use all available PHPMD rulesets for maximum coverage
            *rulesets_guard =
                Some("cleancode,codesize,controversial,design,naming,unusedcode".to_string());
        }
    }

    fn find_project_root(&self, uri: &Url) -> std::path::PathBuf {
        if let Ok(file_path) = uri.to_file_path() {
            let mut current = file_path.parent();

            while let Some(dir) = current {
                // Check for project markers (in order of likelihood)
                if dir.join("composer.json").exists()
                    || dir.join("phpmd.xml").exists()
                    || dir.join("phpmd.xml.dist").exists()
                    || dir.join(".phpmd.xml").exists()
                    || dir.join(".git").exists()
                {
                    eprintln!("🎯 PHPMD LSP: Found project root at: {}", dir.display());
                    return dir.to_path_buf();
                }
                current = dir.parent();
            }
        }

        // Fallback to workspace root or current directory
        let fallback = self
            .workspace_root
            .read()
            .ok()
            .and_then(|g| g.clone())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        eprintln!(
            "⚠️ PHPMD LSP: No project markers found, using fallback: {}",
            fallback.display()
        );
        fallback
    }

    async fn run_phpmd(
        &self,
        uri: &Url,
        _file_path: &str,
        content: Option<&str>,
    ) -> Result<Vec<Diagnostic>> {
        let start_time = Instant::now();
        let file_name = uri
            .path_segments()
            .and_then(|mut segments| segments.next_back())
            .unwrap_or("unknown");

        eprintln!(
            "🔍 PHPMD LSP: Starting analysis for file: {} (URI: {})",
            file_name, uri
        );

        // Debug: Show content details
        if let Some(text) = content {
            let lines: Vec<&str> = text.lines().collect();
            eprintln!("📊 PHPMD LSP: Content has {} lines", lines.len());

            // Show first 10 lines with line numbers
            eprintln!("📝 PHPMD LSP: First 10 lines of content:");
            for (i, line) in lines.iter().take(10).enumerate() {
                eprintln!("  Line {}: {:?}", i + 1, line);
            }

            // Check for special characters
            if text.contains('\r') {
                eprintln!("⚠️ PHPMD LSP: Content contains \\r characters (Windows line endings)");
            }
            if text.starts_with('\u{feff}') {
                eprintln!("⚠️ PHPMD LSP: Content starts with BOM (Byte Order Mark)");
            }
        }

        // Acquire semaphore permit to limit concurrent PHPMD processes
        let available_permits = self.process_semaphore.available_permits();
        let _permit = self
            .process_semaphore
            .acquire()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to acquire process semaphore: {}", e))?;
        eprintln!(
            "🎫 PHPMD LSP: Acquired process slot for {} (slots in use: {}/4)",
            file_name,
            4 - available_permits
        );

        // Use cached PHPMD path
        let phpmd_path = self.get_phpmd_path();

        // Always use stdin for content to avoid file system reads
        if content.is_none() {
            eprintln!("❌ PHPMD LSP: No content provided for {}", file_name);
            return Ok(vec![]);
        }

        let text = content.unwrap();
        eprintln!(
            "📝 PHPMD LSP: Content size: {} bytes, {} chars",
            text.len(),
            text.chars().count()
        );

        // Debug: Calculate line count and show line ending style
        let line_count = text.lines().count();
        let has_final_newline = text.ends_with('\n') || text.ends_with("\r\n");
        eprintln!(
            "📝 PHPMD LSP: Line count: {}, has final newline: {}",
            line_count, has_final_newline
        );

        // Find the project root for this specific file
        let project_root = self.find_project_root(uri);
        eprintln!(
            "📁 PHPMD LSP: Using project root: {}",
            project_root.display()
        );

        // Check if we need to discover config files (if none set or using fallback)
        let should_discover = if let Ok(rulesets_guard) = self.rulesets.read() {
            match &*rulesets_guard {
                None => true,
                Some(rulesets) => {
                    // Re-discover if we're using the fallback rulesets
                    rulesets == "cleancode,codesize,controversial,design,naming,unusedcode"
                }
            }
        } else {
            false
        };

        if should_discover {
            eprintln!("🔍 PHPMD LSP: Checking for config files in project root...");
            self.discover_rulesets(Some(&project_root));
        }

        // Check if PHPMD is a PHAR file that needs PHP invocation for proper error suppression
        let mut cmd = if phpmd_path.ends_with(".phar") {
            eprintln!(
                "🐘 PHPMD LSP: Detected PHAR file, invoking through PHP with error suppression"
            );
            let mut php_cmd = ProcessCommand::new("php");
            php_cmd
                .arg("-d")
                .arg("error_reporting=0") // Suppress all error reporting
                .arg("-d")
                .arg("display_errors=0") // Don't display errors to output
                .arg("-d")
                .arg("display_startup_errors=0") // Don't display startup errors
                .arg("-d")
                .arg("log_errors=0") // Don't log errors
                .arg(&phpmd_path); // Add the PHAR file path

            php_cmd
        } else {
            eprintln!("⚙️ PHPMD LSP: Using direct execution for: {}", phpmd_path);
            ProcessCommand::new(&phpmd_path)
        };

        eprintln!("🚀 PHPMD LSP: Running PHPMD on {}", file_name);

        // Create a temporary file for the PHP content
        // Using a file instead of stdin ensures complete isolation between analyses
        let temp_file_name = format!("phpmd-{}.php", Uuid::new_v4());
        let temp_file_path = std::env::temp_dir().join(&temp_file_name);

        // Write content to temporary file
        if let Err(e) = std::fs::write(&temp_file_path, text) {
            eprintln!("❌ PHPMD LSP: Failed to write temp file: {}", e);
            return Err(anyhow::anyhow!("Failed to write temp file: {}", e));
        }
        eprintln!(
            "📁 PHPMD LSP: Created temporary file: {}",
            temp_file_path.display()
        );
        eprintln!("📝 PHPMD LSP: Wrote {} bytes to temp file", text.len());

        // Add PHPMD arguments
        cmd.arg(&temp_file_path) // Analyze the temp file
            .arg("json") // Use JSON output format
            .arg("--error-file")
            .arg("/dev/null") // Redirect PHPMD errors
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true); // Ensure process is killed if dropped

        // Add rulesets or config file path after the file path and format
        if let Ok(rulesets_guard) = self.rulesets.read() {
            if let Some(ref rulesets) = *rulesets_guard {
                // Check if this is a path to a config file or ruleset names
                if rulesets.ends_with(".xml") || rulesets.ends_with(".xml.dist") {
                    eprintln!("📋 PHPMD LSP: Using config file: {}", rulesets);
                    cmd.arg(rulesets);
                } else {
                    eprintln!("📋 PHPMD LSP: Using rulesets: {}", rulesets);
                    cmd.arg(rulesets);
                }
            } else {
                eprintln!("📋 PHPMD LSP: Using all default rulesets");
                cmd.arg("cleancode,codesize,controversial,design,naming,unusedcode");
            }
        }

        eprintln!(
            "🔍 PHPMD LSP: Running PHPMD on temp file: {}",
            temp_file_name
        );

        let child = match cmd.spawn() {
            Ok(child) => {
                eprintln!("✅ PHPMD LSP: Successfully spawned PHPMD process");
                child
            }
            Err(e) => {
                eprintln!(
                    "❌ PHPMD LSP: Failed to spawn PHPMD for {}: {}",
                    file_name, e
                );
                // Clean up temp file on error
                let _ = std::fs::remove_file(&temp_file_path);
                return Err(anyhow::anyhow!("PHPMD error: {}", e));
            }
        };

        // Wait for output with timeout (10 seconds for PHPMD execution)
        let output = match timeout(Duration::from_secs(10), child.wait_with_output()).await {
            Ok(Ok(output)) => {
                let elapsed = start_time.elapsed();
                eprintln!(
                    "⚡ PHPMD LSP: Process completed for {} in {:.2}s",
                    file_name,
                    elapsed.as_secs_f64()
                );
                output
            }
            Ok(Err(e)) => {
                let elapsed = start_time.elapsed();
                eprintln!(
                    "❌ PHPMD LSP: PHPMD process error for {} after {:.2}s: {}",
                    file_name,
                    elapsed.as_secs_f64(),
                    e
                );
                return Err(anyhow::anyhow!(
                    "PHPMD process error for {}: {}",
                    file_name,
                    e
                ));
            }
            Err(_) => {
                eprintln!(
                    "⏱️ PHPMD LSP: PHPMD timeout for {} (>10s) with {} bytes of content",
                    file_name,
                    text.len()
                );
                // Process will be killed automatically due to kill_on_drop(true)
                return Err(anyhow::anyhow!(
                    "PHPMD execution timeout for {} after 10 seconds",
                    file_name
                ));
            }
        };

        let raw_output = String::from_utf8_lossy(&output.stdout);

        // Debug: Show raw PHPMD output (first 500 chars)
        let output_preview = if raw_output.len() > 500 {
            format!("{}...", &raw_output[..500])
        } else {
            raw_output.to_string()
        };
        eprintln!(
            "🔬 PHPMD LSP: Raw PHPMD output for {}: {}",
            file_name, output_preview
        );

        // Clean up temporary file
        if let Err(e) = std::fs::remove_file(&temp_file_path) {
            eprintln!("⚠️ PHPMD LSP: Failed to clean up temp file: {}", e);
        }

        // Permit is automatically released when it goes out of scope
        drop(_permit);
        let available_after = self.process_semaphore.available_permits();
        eprintln!(
            "🎫 PHPMD LSP: Released process slot for {} (slots available: {}/4)",
            file_name, available_after
        );

        // Extract JSON from raw output (PHPMD might output debug info before JSON)
        let json_output = self.extract_json_from_output(&raw_output);
        let diagnostics = self.parse_phpmd_output(&json_output, uri).await?;

        // Log results with timing
        let total_time = start_time.elapsed();
        let issue_count = diagnostics.len();
        if issue_count == 0 {
            eprintln!(
                "✅ PHPMD LSP: {} is clean! No issues found (took {:.2}s)",
                file_name,
                total_time.as_secs_f64()
            );
        } else {
            let errors = diagnostics
                .iter()
                .filter(|d| d.severity == Some(DiagnosticSeverity::ERROR))
                .count();
            let warnings = diagnostics
                .iter()
                .filter(|d| d.severity == Some(DiagnosticSeverity::WARNING))
                .count();
            let infos = diagnostics
                .iter()
                .filter(|d| d.severity == Some(DiagnosticSeverity::INFORMATION))
                .count();

            eprintln!("📊 PHPMD LSP: {} issues found in {}: {} errors, {} warnings, {} info (took {:.2}s)",
                issue_count, file_name, errors, warnings, infos, total_time.as_secs_f64());
        }

        Ok(diagnostics)
    }

    fn extract_json_from_output(&self, output: &str) -> String {
        // PHPMD might output debug information before the JSON
        // Find the first '{' and last '}' to extract the JSON object

        if let Some(start) = output.find('{') {
            // Find the matching closing brace by counting braces
            let mut brace_count = 0;
            let mut in_string = false;
            let mut escape_next = false;
            let bytes = output.as_bytes();

            for i in start..bytes.len() {
                let ch = bytes[i] as char;

                if escape_next {
                    escape_next = false;
                    continue;
                }

                if ch == '\\' && in_string {
                    escape_next = true;
                    continue;
                }

                if ch == '"' && !in_string {
                    in_string = true;
                } else if ch == '"' && in_string {
                    in_string = false;
                }

                if !in_string {
                    if ch == '{' {
                        brace_count += 1;
                    } else if ch == '}' {
                        brace_count -= 1;
                        if brace_count == 0 {
                            // Found the matching closing brace
                            let json_str = &output[start..=i];
                            eprintln!(
                                "📋 PHPMD LSP: Extracted JSON from position {} to {}",
                                start, i
                            );
                            return json_str.to_string();
                        }
                    }
                }
            }
        }

        // If no valid JSON found, return the original output
        eprintln!("⚠️ PHPMD LSP: Could not extract JSON from output, using raw output");
        output.to_string()
    }

    async fn parse_phpmd_output(&self, json_output: &str, uri: &Url) -> Result<Vec<Diagnostic>> {
        // Early return if empty output
        if json_output.trim().is_empty() {
            return Ok(vec![]);
        }

        let file_name = uri
            .path_segments()
            .and_then(|mut segments| segments.next_back())
            .unwrap_or("unknown");

        // Debug: Parse and show violations
        eprintln!(
            "🔬 PHPMD LSP: Parsing {} bytes of JSON output for {}",
            json_output.len(),
            file_name
        );

        let mut diagnostics = Vec::with_capacity(10); // Pre-allocate for common case

        // Parse PHPMD JSON output
        let phpmd_result: serde_json::Value = match serde_json::from_str(json_output) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("❌ PHPMD LSP: Failed to parse JSON output: {}", e);
                eprintln!("Raw output: {}", json_output);
                return Ok(vec![]);
            }
        };

        // PHPMD JSON structure has "files" array
        if let Some(files) = phpmd_result.get("files").and_then(|f| f.as_array()) {
            eprintln!(
                "📁 PHPMD LSP: Found {} file(s) in PHPMD output",
                files.len()
            );

            // Log ALL files in the output to debug contamination
            eprintln!("🔍 PHPMD LSP: === FILES IN PHPMD OUTPUT ===");
            for (idx, file) in files.iter().enumerate() {
                if let Some(path) = file.get("file").and_then(|f| f.as_str()) {
                    let violation_count = file
                        .get("violations")
                        .and_then(|v| v.as_array())
                        .map(|v| v.len())
                        .unwrap_or(0);
                    eprintln!(
                        "  File #{}: {} ({} violations)",
                        idx + 1,
                        path,
                        violation_count
                    );
                }
            }
            eprintln!("🔍 PHPMD LSP: === END FILES LIST ===");

            for (file_idx, file_entry) in files.iter().enumerate() {
                // Get the file path from the JSON
                let json_file_path = file_entry
                    .get("file")
                    .and_then(|f| f.as_str())
                    .unwrap_or("unknown");

                eprintln!(
                    "📄 PHPMD LSP: Processing file #{}: {}",
                    file_idx + 1,
                    json_file_path
                );

                // With temp file approach, PHPMD should report the actual temp file path
                // Log the file path for debugging
                eprintln!(
                    "📄 PHPMD LSP: Processing violations from file: {}",
                    json_file_path
                );

                if let Some(violations) = file_entry.get("violations").and_then(|v| v.as_array()) {
                    eprintln!(
                        "🔍 PHPMD LSP: Processing {} violations for stdin (target: {})",
                        violations.len(),
                        file_name
                    );

                    for (idx, violation) in violations.iter().enumerate() {
                        if let Some(diagnostic) =
                            self.convert_violation_to_diagnostic(violation, uri).await
                        {
                            diagnostics.push(diagnostic);
                            eprintln!("✅ PHPMD LSP: Successfully converted violation #{} to diagnostic for {}", idx + 1, file_name);
                        } else {
                            eprintln!("⚠️ PHPMD LSP: Failed to convert violation #{} to diagnostic for {}", idx + 1, file_name);
                        }
                    }
                } else {
                    eprintln!("ℹ️ PHPMD LSP: No violations found for {}", json_file_path);
                }
            }
        } else {
            eprintln!("⚠️ PHPMD LSP: No 'files' array found in PHPMD output");
        }

        eprintln!(
            "📊 PHPMD LSP: Total diagnostics generated for {}: {}",
            file_name,
            diagnostics.len()
        );
        Ok(diagnostics)
    }

    fn find_property_line(&self, property_name: &str, content: &str) -> Option<u32> {
        // Search for the property declaration in the file content
        let lines: Vec<&str> = content.lines().collect();
        let property_with_dollar = format!("${}", property_name);

        eprintln!(
            "🔍 PHPMD LSP: Searching for property '{}'",
            property_with_dollar
        );

        for (line_num, line) in lines.iter().enumerate() {
            // Skip comment lines
            let trimmed = line.trim();
            if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with("*") {
                continue;
            }

            // Check if this line contains the property
            if line.contains(&property_with_dollar) {
                // Check if it's a property declaration (has visibility modifier or var, or just $prop = value)
                // Don't match usage like $this->property_name or function($property_name)

                // Check it's not a parameter in a function signature
                if line.contains("function") && line.contains("(") {
                    continue;
                }

                // Check it's not usage with $this-> or self::$
                if line.contains(&format!("$this->{}", property_name))
                    || line.contains(&format!("self::${}", property_name))
                {
                    continue;
                }

                // It's likely a property declaration if:
                // 1. Has visibility modifier (private, protected, public)
                // 2. Has static keyword
                // 3. Has var keyword
                // 4. Is directly assigned with =
                let has_visibility = line.contains("private")
                    || line.contains("protected")
                    || line.contains("public")
                    || line.contains("var")
                    || line.contains("static");
                let has_assignment = line.contains(&format!("{} =", property_with_dollar))
                    || line.contains(&format!("{}=", property_with_dollar));
                let has_semicolon = line.contains(&format!("{};", property_with_dollar));

                if has_visibility || has_assignment || has_semicolon {
                    eprintln!(
                        "✅ PHPMD LSP: Found property '{}' at line {} in: {}",
                        property_with_dollar,
                        line_num + 1,
                        line.trim()
                    );
                    return Some((line_num + 1) as u32); // Convert to 1-based line number
                }
            }
        }

        eprintln!(
            "⚠️ PHPMD LSP: Could not find property '{}' in file",
            property_with_dollar
        );
        None
    }

    fn determine_diagnostic_range(
        &self,
        begin_line: u32,
        end_line: u32,
        rule: &str,
        violation: &serde_json::Value,
        uri: &Url,
    ) -> (u32, u32) {
        // Class-level rules that should only highlight the class declaration line
        const CLASS_LEVEL_RULES: &[&str] = &[
            "TooManyPublicMethods",
            "TooManyMethods",
            "TooManyFields",
            "ExcessivePublicCount",
            "ExcessiveClassComplexity",
            "ExcessiveClassLength",
            "CouplingBetweenObjects",
            "NumberOfChildren",
            "DepthOfInheritance",
            "CamelCaseClassName",
            "CamelCasePropertyName",
            "CamelCaseParameterName",
            "CamelCaseVariableName",
        ];

        // Property-specific rules that need special handling
        const PROPERTY_RULES: &[&str] = &["CamelCasePropertyName", "CamelCaseParameterName"];

        // Method-level rules that should highlight the method signature
        const METHOD_LEVEL_RULES: &[&str] = &[
            "CyclomaticComplexity",
            "NPathComplexity",
            "ExcessiveMethodLength",
            "ExcessiveParameterList",
            "UnusedFormalParameter",
            "ShortMethodName",
            "ConstructorWithNameAsEnclosingClass",
            "CamelCaseMethodName",
        ];

        // Check if this is a property-specific rule that PHPMD incorrectly reports at class level
        if PROPERTY_RULES.contains(&rule) {
            // Try to extract the property name from the description and find its actual line
            if let Some(description) = violation.get("description").and_then(|v| v.as_str()) {
                // Extract property name from description like "The property $property_name is not named in camelCase."
                if let Some(start) = description.find("$") {
                    let prop_start = start + 1; // Skip the $
                    let prop_end = description[prop_start..]
                        .find(|c: char| !c.is_alphanumeric() && c != '_')
                        .map(|i| prop_start + i)
                        .unwrap_or(description.len());

                    let property_name = &description[prop_start..prop_end];
                    eprintln!(
                        "🔍 PHPMD LSP: Extracted property name '{}' from rule {}",
                        property_name, rule
                    );

                    // Try to find the actual property line in the file content
                    if let Ok(docs) = self.open_docs.read() {
                        if let Some(compressed_doc) = docs.get(uri) {
                            if let Ok(content) = self.decompress_document(compressed_doc) {
                                if let Some(actual_line) =
                                    self.find_property_line(property_name, &content)
                                {
                                    eprintln!("✅ PHPMD LSP: Found actual property line {} for ${} (was reported as {})",
                                        actual_line, property_name, begin_line);
                                    return (actual_line, actual_line);
                                }
                            }
                        }
                    }
                }
            }

            // Fallback: treat as class-level if we can't find the property
            eprintln!(
                "⚠️ PHPMD LSP: Could not find property line for {}, using class line",
                rule
            );
            return (begin_line, begin_line);
        }

        // Check if this is a class-level violation
        if CLASS_LEVEL_RULES.contains(&rule) && !PROPERTY_RULES.contains(&rule) {
            // For class-level violations, only highlight the class declaration line
            return (begin_line, begin_line);
        }

        // Check if this is a method-level violation
        if METHOD_LEVEL_RULES.contains(&rule) {
            // For method-level violations, check if it spans multiple lines
            // If so, only highlight the method signature (first line)
            if end_line > begin_line && (end_line - begin_line) > 5 {
                // Likely a full method range, just highlight the signature
                return (begin_line, begin_line);
            }
        }

        // Special handling for specific rules
        match rule {
            // Else expression rules - highlight the else line only
            "ElseExpression" => {
                // The begin_line is usually the else line itself
                (begin_line, begin_line)
            }

            // Variable rules on a single line with multiple parameters
            "ShortVariable" | "LongVariable" => {
                // These are usually on the parameter line
                (begin_line, begin_line)
            }

            // Goto statements - just the goto line
            "GotoStatement" => (begin_line, begin_line),

            // Exit/Eval expressions - just that line
            "ExitExpression" | "EvalExpression" => (begin_line, begin_line),

            // Error control operator (@): PHPMD reports the enclosing method's
            // range, but the description points at the real operator line.
            // Collapse to that single line; fall back to the first reported
            // line when the description can't be parsed.
            "ErrorControlOperator" => {
                let description = violation
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                error_control_operator_range(description, begin_line)
            }

            // Default: If the range is very large (likely a class/method), limit it
            _ => {
                // If the range spans more than 10 lines, it's likely a block-level issue
                // Only highlight the first line to avoid overwhelming the user
                if end_line > begin_line && (end_line - begin_line) > 10 {
                    return (begin_line, begin_line);
                }
                // Otherwise, use the full range
                (begin_line, end_line)
            }
        }
    }

    async fn convert_violation_to_diagnostic(
        &self,
        violation: &serde_json::Value,
        uri: &Url,
    ) -> Option<Diagnostic> {
        // PHPMD JSON violation structure:
        // {
        //   "beginLine": 10,
        //   "endLine": 20,
        //   "package": "SomePackage",
        //   "function": "someFunction",
        //   "class": "SomeClass",
        //   "method": "someMethod",
        //   "description": "The method someMethod() has a Cyclomatic Complexity of 11.",
        //   "rule": "CyclomaticComplexity",
        //   "ruleSet": "Code Size Rules",
        //   "priority": 3
        // }

        let file_name = uri
            .path_segments()
            .and_then(|mut segments| segments.next_back())
            .unwrap_or("unknown");

        eprintln!(
            "🎯 PHPMD LSP: Converting violation to diagnostic for URI: {}",
            file_name
        );

        // With temp file approach, each analysis is isolated so no validation needed

        let begin_line = violation.get("beginLine")?.as_u64()? as u32;
        let end_line = violation
            .get("endLine")
            .and_then(|v| v.as_u64())
            .unwrap_or(begin_line as u64) as u32;
        let description = violation.get("description")?.as_str()?;
        let rule = violation.get("rule")?.as_str().unwrap_or("");
        let rule_set = violation
            .get("ruleSet")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let priority = violation
            .get("priority")
            .and_then(|v| v.as_u64())
            .unwrap_or(3);

        // Debug: Show raw line numbers from PHPMD
        eprintln!(
            "🔍 PHPMD LSP: [{}] Raw PHPMD violation - beginLine: {}, endLine: {}, rule: {}",
            file_name, begin_line, end_line, rule
        );

        // Map priority to severity (1-2: error, 3-4: warning, 5+: info)
        let severity = match priority {
            1..=2 => DiagnosticSeverity::ERROR,
            3..=4 => DiagnosticSeverity::WARNING,
            _ => DiagnosticSeverity::INFORMATION,
        };

        // Rule-based range handling
        let (effective_begin_line, effective_end_line) =
            self.determine_diagnostic_range(begin_line, end_line, rule, violation, uri);

        // Convert to 0-based indexing for LSP
        let lsp_begin_line = if effective_begin_line > 0 {
            effective_begin_line - 1
        } else {
            0
        };
        let lsp_end_line = if effective_end_line > 0 {
            effective_end_line - 1
        } else {
            0
        };

        // Debug logging for line number mapping
        eprintln!(
            "🔍 PHPMD LSP: [{}] Line mapping - PHPMD lines {}-{} -> LSP lines {}-{} (rule: {})",
            file_name, begin_line, end_line, lsp_begin_line, lsp_end_line, rule
        );

        // Calculate the actual character positions to avoid underlining leading whitespace
        let (start_char, end_char) = if let Ok(docs) = self.open_docs.read() {
            if let Some(compressed_doc) = docs.get(uri) {
                if let Ok(content) = self.decompress_document(compressed_doc) {
                    let lines: Vec<&str> = content.lines().collect();

                    // Debug logging for the line content
                    if (effective_begin_line as usize) <= lines.len() && effective_begin_line > 0 {
                        let phpmd_line_content = lines
                            .get((effective_begin_line - 1) as usize)
                            .map(|l| {
                                if l.len() > 80 {
                                    format!("{}...", &l[..80])
                                } else {
                                    l.to_string()
                                }
                            })
                            .unwrap_or_else(|| "LINE NOT FOUND".to_string());
                        eprintln!(
                            "📍 PHPMD LSP: [{}] Content at PHPMD line {}: {:?}",
                            file_name, effective_begin_line, phpmd_line_content
                        );
                    }

                    // Calculate start and end character positions
                    if (effective_begin_line as usize) <= lines.len() && effective_begin_line > 0 {
                        let start_line_content = lines[(effective_begin_line - 1) as usize];
                        // Find the start character. For ErrorControlOperator,
                        // anchor the squiggle at the `@` token; otherwise use
                        // the first non-whitespace character.
                        let start_char = if rule == "ErrorControlOperator" {
                            error_control_operator_start_char(start_line_content)
                        } else {
                            start_line_content.len() - start_line_content.trim_start().len()
                        };

                        // Calculate end character position
                        let end_char = if lsp_begin_line == lsp_end_line {
                            // Same line - use the actual line length
                            start_line_content.len()
                        } else if (effective_end_line as usize) <= lines.len()
                            && effective_end_line > 0
                        {
                            // Different end line - get its actual length
                            lines[(effective_end_line - 1) as usize].len()
                        } else {
                            // Fallback to large number if line not found
                            999
                        };

                        (start_char as u32, end_char as u32)
                    } else {
                        (0, 999)
                    }
                } else {
                    (0, 999)
                }
            } else {
                (0, 999)
            }
        } else {
            (0, 999)
        };

        // Create range with proper boundaries
        let range = Range {
            start: Position {
                line: lsp_begin_line,
                character: start_char,
            },
            end: Position {
                line: lsp_end_line,
                character: end_char,
            },
        };

        eprintln!("📐 PHPMD LSP: [{}] Final LSP Range - start: (line: {}, char: {}), end: (line: {}, char: {})",
            file_name, lsp_begin_line, start_char, lsp_end_line, end_char);

        // Store additional data for potential future features
        let data = serde_json::json!({
            "phpmd_rule": rule,
            "phpmd_ruleset": rule_set,
            "phpmd_priority": priority,
            "phpmd_class": violation.get("class"),
            "phpmd_method": violation.get("method"),
            "phpmd_function": violation.get("function")
        });

        Some(Diagnostic {
            range,
            severity: Some(severity),
            code: if !rule.is_empty() {
                Some(NumberOrString::String(rule.to_string()))
            } else {
                None
            },
            source: Some("phpmd".to_string()),
            message: description.to_string(),
            related_information: None,
            tags: None,
            code_description: if !rule_set.is_empty() {
                let ruleset_slug = rule_set
                    .to_lowercase()
                    .replace(" ", "")
                    .trim_end_matches("rules")
                    .to_string();
                let url_str = if !rule.is_empty() {
                    format!(
                        "https://phpmd.org/rules/{}.html#{}",
                        ruleset_slug,
                        rule.to_lowercase()
                    )
                } else {
                    format!("https://phpmd.org/rules/{}.html", ruleset_slug)
                };
                Some(CodeDescription {
                    href: Url::parse(&url_str).ok()?,
                })
            } else {
                None
            },
            data: Some(data),
        })
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for PhpmdLanguageServer {
    async fn initialize(&self, params: InitializeParams) -> LspResult<InitializeResult> {
        eprintln!("🚀 PHPMD LSP: Server initializing...");
        eprintln!("🔧 PHPMD LSP: Client info: {:?}", params.client_info);

        // Determine workspace root for config file lookup
        let workspace_root = params
            .root_uri
            .as_ref()
            .and_then(|uri| uri.to_file_path().ok());

        if let Some(ref root) = workspace_root {
            eprintln!("📁 PHPMD LSP: Workspace root: {}", root.display());
        } else {
            eprintln!("❌ PHPMD LSP: No workspace root detected");
        }

        // Store workspace root for PHPMD path detection
        if let Ok(mut workspace_guard) = self.workspace_root.write() {
            *workspace_guard = workspace_root.clone();
        }

        let mut should_discover = true;

        if let Some(options) = params.initialization_options {
            // Parse initialization options
            eprintln!("📦 PHPMD LSP: Processing initialization options from extension");
            match serde_json::from_value::<InitializationOptions>(options.clone()) {
                Ok(init_options) => {
                    if let Some(rulesets) = init_options.rulesets {
                        eprintln!("⚙️ PHPMD LSP: Extension provided rulesets: '{}'", rulesets);
                        if let Ok(mut rulesets_guard) = self.rulesets.write() {
                            *rulesets_guard = Some(rulesets.clone());
                        }
                        should_discover = false; // Don't discover if rulesets were explicitly provided
                    } else {
                        eprintln!("🎯 PHPMD LSP: No rulesets provided by extension - will discover from workspace");
                    }
                }
                Err(e) => {
                    eprintln!(
                        "❌ PHPMD LSP: Failed to parse initialization options: {}",
                        e
                    );
                }
            }
        } else {
            eprintln!(
                "📋 PHPMD LSP: No initialization options provided - will discover from workspace"
            );
        }

        // Discover from workspace if no explicit rulesets were provided
        if should_discover {
            self.discover_rulesets(workspace_root.as_deref());
        }

        // Log final initialization state
        if let Ok(rulesets_guard) = self.rulesets.read() {
            match &*rulesets_guard {
                Some(rulesets) => {
                    if rulesets.ends_with(".xml") || rulesets.ends_with(".xml.dist") {
                        eprintln!("🎯 PHPMD LSP: Initialized with config file: '{}'", rulesets);
                        eprintln!(
                            "📋 PHPMD LSP: Configuration source: Project-specific XML ruleset"
                        );
                    } else {
                        eprintln!("🎯 PHPMD LSP: Initialized with rulesets: '{}'", rulesets);
                        if rulesets == "cleancode,codesize,controversial,design,naming,unusedcode" {
                            eprintln!("📋 PHPMD LSP: Configuration source: Fallback (all available rulesets)");
                        } else {
                            eprintln!(
                                "📋 PHPMD LSP: Configuration source: Custom ruleset configuration"
                            );
                        }
                    }
                }
                None => {
                    eprintln!("🎯 PHPMD LSP: Initialized with default rulesets");
                    eprintln!("📋 PHPMD LSP: Configuration source: Built-in defaults");
                }
            }
        }

        eprintln!("✅ PHPMD LSP: Server initialization complete!");

        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                diagnostic_provider: Some(DiagnosticServerCapabilities::Options(
                    DiagnosticOptions {
                        identifier: Some("phpmd".to_string()),
                        inter_file_dependencies: false,
                        workspace_diagnostics: false,
                        ..Default::default()
                    },
                )),
                workspace: Some(WorkspaceServerCapabilities {
                    workspace_folders: Some(WorkspaceFoldersServerCapabilities {
                        supported: Some(true),
                        change_notifications: Some(OneOf::Left(true)),
                    }),
                    file_operations: None,
                }),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    async fn initialized(&self, _params: InitializedParams) {
        eprintln!("🎉 PHPMD LSP: Server is ready and operational!");
        // Pre-cache the PHPMD path on initialization
        let _ = self.get_phpmd_path();
        eprintln!("🚀 PHPMD LSP: Ready to analyze PHP files!");
    }

    async fn shutdown(&self) -> LspResult<()> {
        eprintln!("🔄 PHPMD LSP: Shutting down, clearing caches...");

        // Clear all cached data on shutdown
        if let Ok(mut docs) = self.open_docs.write() {
            docs.clear();
        }
        if let Ok(mut cache) = self.results_cache.write() {
            cache.clear();
        }

        // Reset memory counter
        self.total_memory_usage.store(0, Ordering::Relaxed);

        eprintln!("✅ PHPMD LSP: Shutdown complete");
        Ok(())
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        // Clear document from memory to prevent memory leaks
        let uri = params.text_document.uri;

        // Remove compressed document and update memory tracking
        if let Ok(mut docs) = self.open_docs.write() {
            if let Some(doc) = docs.remove(&uri) {
                let freed_memory = doc.compressed_data.len();
                self.total_memory_usage
                    .fetch_sub(freed_memory, Ordering::Relaxed);
                eprintln!(
                    "🗑️ PHPMD LSP: Closed file, freed {}KB, total memory: {:.1}MB",
                    freed_memory / 1024,
                    self.get_memory_usage_mb()
                );
            }
        }

        // Clear cached results
        if let Ok(mut cache) = self.results_cache.write() {
            let removed = cache.remove(&uri);
            eprintln!(
                "🗑️ PHPMD LSP: Cache cleared on close for URI: {} - removed: {}",
                uri,
                removed.is_some()
            );
        }

        // Clear diagnostics for closed file
        let _ = self.client.publish_diagnostics(uri, vec![], None).await;
    }

    async fn did_change_workspace_folders(&self, _params: DidChangeWorkspaceFoldersParams) {
        // Clear cached PHPMD path when workspace changes
        if let Ok(mut guard) = self.phpmd_path.write() {
            *guard = None;
        }

        // Clear results cache as paths may have changed
        if let Ok(mut cache) = self.results_cache.write() {
            cache.clear();
        }

        eprintln!("🔄 PHPMD LSP: Workspace changed, cleared caches");

        // Re-detect PHPMD configuration for new workspace
        // This will be done lazily on next PHPMD run
    }

    async fn did_change_configuration(&self, params: DidChangeConfigurationParams) {
        eprintln!("🔄 PHPMD LSP: Configuration change detected!");

        // Clear cached PHPMD path to force re-detection
        if let Ok(mut guard) = self.phpmd_path.write() {
            *guard = None;
            eprintln!("🗑️ PHPMD LSP: Cleared cached PHPMD path - will re-detect on next use");
        }

        // Parse the settings
        if let Some(settings) = params.settings.as_object() {
            // Look for phpmd settings
            if let Some(phpmd_settings) = settings.get("phpmd") {
                // Try to parse as PhpmdSettings
                if let Ok(parsed_settings) =
                    serde_json::from_value::<PhpmdSettings>(phpmd_settings.clone())
                {
                    // Update the rulesets if provided
                    if let Some(new_rulesets) = parsed_settings.rulesets {
                        eprintln!(
                            "⚙️ PHPMD LSP: Configuration changed via settings: '{}'",
                            new_rulesets
                        );
                        if let Ok(mut rulesets_guard) = self.rulesets.write() {
                            *rulesets_guard = Some(new_rulesets);
                        }
                    }
                }
            }

            // Also check for rulesets directly in settings (for compatibility)
            if let Some(rulesets_value) = settings.get("rulesets") {
                if let Some(new_rulesets) = rulesets_value.as_str() {
                    eprintln!(
                        "⚙️ PHPMD LSP: Configuration changed via direct rulesets setting: '{}'",
                        new_rulesets
                    );
                    if let Ok(mut rulesets_guard) = self.rulesets.write() {
                        *rulesets_guard = Some(new_rulesets.to_string());
                    }
                }
            }
        }

        // Clear results cache to force re-analysis with new config
        if let Ok(mut cache) = self.results_cache.write() {
            cache.clear();
            eprintln!("🗑️ PHPMD LSP: Cleared results cache after config change");
        }

        // Note: Documents will be re-analyzed on next diagnostic() call
        // No need to proactively re-run PHPMD on all files
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        let text = params.text_document.text;

        let file_name = uri
            .path_segments()
            .and_then(|mut segments| segments.next_back())
            .unwrap_or("unknown");

        eprintln!(
            "📂 PHPMD LSP: File opened: {} ({} bytes)",
            file_name,
            text.len()
        );

        // Debug: Show first few lines of opened file
        let lines: Vec<&str> = text.lines().collect();
        eprintln!("📊 PHPMD LSP: Opened file has {} lines", lines.len());
        for (i, line) in lines.iter().take(5).enumerate() {
            eprintln!("  Line {}: {:?}", i + 1, line);
        }

        // Compress and store the document
        let compressed_doc = self.compress_document(&text);

        {
            let mut docs = self.open_docs.write().unwrap();
            docs.insert(uri.clone(), compressed_doc);

            // Log memory stats on significant changes
            if docs.len().is_multiple_of(25) {
                drop(docs); // Release lock before logging
                self.log_memory_stats();
            }
        }

        // Invalidate any cached results for this file
        if let Ok(mut cache) = self.results_cache.write() {
            let removed = cache.remove(&uri);
            eprintln!(
                "🗑️ PHPMD LSP: Cache invalidated for {} (URI: {}) - removed: {}",
                file_name,
                uri,
                removed.is_some()
            );
        }

        // Log memory stats periodically (every 10 files)
        if let Ok(docs) = self.open_docs.read() {
            if docs.len() % 10 == 0 {
                drop(docs); // Release lock before logging
                self.log_memory_stats();
            }
        }

        // Note: Analysis is only triggered when Zed explicitly calls diagnostic()
        // This prevents overlapping analyses and cross-file contamination
        eprintln!("📝 PHPMD LSP: Document stored, waiting for diagnostic request from Zed");
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri.clone();

        let file_name = uri
            .path_segments()
            .and_then(|mut segments| segments.next_back())
            .unwrap_or("unknown");

        // With FULL sync, we always get the complete document content
        if let Some(change) = params.content_changes.first() {
            // Debug: Show change details
            let lines: Vec<&str> = change.text.lines().collect();
            eprintln!(
                "📝 PHPMD LSP: File changed: {} - now has {} lines, {} bytes",
                file_name,
                lines.len(),
                change.text.len()
            );

            // Show first 3 lines after change
            for (i, line) in lines.iter().take(3).enumerate() {
                eprintln!("  Line {}: {:?}", i + 1, line);
            }
            // Remove old compressed document to update memory tracking
            let old_size = if let Ok(docs) = self.open_docs.read() {
                docs.get(&uri).map(|doc| doc.compressed_data.len())
            } else {
                None
            };

            if let Some(size) = old_size {
                self.total_memory_usage.fetch_sub(size, Ordering::Relaxed);
            }

            // Compress and store new content
            let compressed_doc = self.compress_document(&change.text);

            let mut docs = self.open_docs.write().unwrap();
            docs.insert(uri.clone(), compressed_doc);

            // Invalidate cached results since content changed
            if let Ok(mut cache) = self.results_cache.write() {
                let removed = cache.remove(&uri);
                eprintln!(
                    "🗑️ PHPMD LSP: Cache invalidated after change for {} (URI: {}) - removed: {}",
                    file_name,
                    uri,
                    removed.is_some()
                );
            }
        }

        // Diagnostics will be provided via diagnostic() method
        // This reduces unnecessary PHPMD runs during rapid typing
    }

    async fn did_save(&self, params: DidSaveTextDocumentParams) {
        let uri = params.text_document.uri;

        let file_name = uri
            .path_segments()
            .and_then(|mut segments| segments.next_back())
            .unwrap_or("unknown");

        eprintln!("💾 PHPMD LSP: File saved: {}", file_name);

        // Note: Diagnostics will be provided via diagnostic() method calls from Zed
        // We don't need to proactively run PHPMD here to avoid duplicate analysis
    }

    async fn diagnostic(
        &self,
        params: DocumentDiagnosticParams,
    ) -> LspResult<DocumentDiagnosticReportResult> {
        let uri = params.text_document.uri;
        let file_name = uri
            .path_segments()
            .and_then(|mut segments| segments.next_back())
            .unwrap_or("unknown");

        if let Ok(file_path) = uri.to_file_path() {
            if let Some(path_str) = file_path.to_str() {
                // First check if we have cached results
                // Get current document checksum first
                let current_checksum = {
                    let docs = self.open_docs.read().unwrap();
                    docs.get(&uri).map(|doc| doc.checksum.clone())
                };

                if let Ok(cache) = self.results_cache.read() {
                    eprintln!(
                        "🔍 PHPMD LSP: Checking cache for {} (URI: {})",
                        file_name, uri
                    );
                    eprintln!(
                        "🔍 PHPMD LSP: Cache currently contains {} entries",
                        cache.len()
                    );

                    if let Some(cached) = cache.get(&uri) {
                        eprintln!("⚡ PHPMD LSP: Found cached results for {} (URI: {}) with {} diagnostics (age: {:.1}s)",
                            file_name,
                            uri,
                            cached.diagnostics.len(),
                            cached.generated_at.elapsed().as_secs_f64()
                        );

                        // Validate cache is still valid by checking content checksum
                        if let Some(ref checksum) = current_checksum {
                            if cached.content_checksum != *checksum {
                                eprintln!("🔄 PHPMD LSP: Cache invalidated for {} - content changed (old: {}, new: {})",
                                    file_name, &cached.content_checksum[..8], &checksum[..8]);
                                // Content has changed, need to re-analyze
                                drop(cache); // Release read lock before we try to write
                                if let Ok(mut cache_write) = self.results_cache.write() {
                                    cache_write.remove(&uri);
                                }
                            } else {
                                // Checksum matches, cache is valid
                                eprintln!(
                                    "✅ PHPMD LSP: Cache valid for {} - checksum matches",
                                    file_name
                                );

                                // Check if client has the same version
                                if let Some(previous_result_id) = params.previous_result_id {
                                    if previous_result_id == cached.result_id {
                                        eprintln!(
                                            "✅ PHPMD LSP: Client has current version for {}",
                                            file_name
                                        );
                                        return Ok(DocumentDiagnosticReportResult::Report(
                                            DocumentDiagnosticReport::Unchanged(
                                                RelatedUnchangedDocumentDiagnosticReport {
                                                    unchanged_document_diagnostic_report:
                                                        UnchangedDocumentDiagnosticReport {
                                                            result_id: cached.result_id.clone(),
                                                        },
                                                    related_documents: None,
                                                },
                                            ),
                                        ));
                                    }
                                }

                                // Return cached diagnostics
                                return Ok(DocumentDiagnosticReportResult::Report(
                                    DocumentDiagnosticReport::Full(
                                        RelatedFullDocumentDiagnosticReport {
                                            full_document_diagnostic_report:
                                                FullDocumentDiagnosticReport {
                                                    result_id: Some(cached.result_id.clone()),
                                                    items: cached.diagnostics.clone(),
                                                },
                                            related_documents: None,
                                        },
                                    ),
                                ));
                            }
                        } else {
                            eprintln!("⚠️ PHPMD LSP: No current document checksum available, invalidating cache");
                            drop(cache); // Release read lock
                            if let Ok(mut cache_write) = self.results_cache.write() {
                                cache_write.remove(&uri);
                            }
                        }
                    }
                }

                // No cached results, need to get content and run PHPMD
                let compressed_doc = {
                    let docs = self.open_docs.read().unwrap();
                    docs.get(&uri).cloned()
                };

                // Handle missing document (rare edge case)
                let compressed_doc = if compressed_doc.is_none() {
                    // Try to read from disk as fallback
                    match fs::read_to_string(path_str) {
                        Ok(file_content) => {
                            eprintln!(
                                "⚠️ PHPMD LSP: Document not in memory, reading from disk: {}",
                                file_name
                            );
                            let compressed = self.compress_document(&file_content);
                            let mut docs = self.open_docs.write().unwrap();
                            docs.insert(uri.clone(), compressed.clone());
                            Some(compressed)
                        }
                        Err(e) => {
                            eprintln!("❌ PHPMD LSP: Failed to read file {}: {}", file_name, e);
                            None
                        }
                    }
                } else {
                    compressed_doc
                };

                if let Some(compressed_doc) = compressed_doc {
                    // Decompress content
                    let content = match self.decompress_document(&compressed_doc) {
                        Ok(content) => {
                            // Log content details to verify we're analyzing the right file
                            eprintln!(
                                "📄 PHPMD LSP: Retrieved content for {} (URI: {})",
                                file_name, uri
                            );
                            eprintln!("📄 PHPMD LSP: Content size: {} bytes", content.len());

                            // Show first few lines to identify which file's content this is
                            let lines: Vec<&str> = content.lines().collect();
                            eprintln!("📄 PHPMD LSP: Content preview (first 5 lines):");
                            for (i, line) in lines.iter().take(5).enumerate() {
                                eprintln!("    Line {}: {}", i + 1, line);
                            }

                            // Check for specific identifiers to verify content
                            if content.contains("property_with_underscore") {
                                eprintln!("🔍 PHPMD LSP: Content contains 'property_with_underscore' (File A marker)");
                            }
                            if content.contains("bad_property_name") {
                                eprintln!("🔍 PHPMD LSP: Content contains 'bad_property_name' (File B marker)");
                            }
                            if content.contains("another_underscore_prop") {
                                eprintln!("🔍 PHPMD LSP: Content contains 'another_underscore_prop' (File A marker)");
                            }
                            if content.contains("another_bad_name") {
                                eprintln!("🔍 PHPMD LSP: Content contains 'another_bad_name' (File B marker)");
                            }

                            content
                        }
                        Err(e) => {
                            eprintln!("❌ PHPMD LSP: Failed to decompress {}: {}", file_name, e);
                            return Ok(DocumentDiagnosticReportResult::Report(
                                DocumentDiagnosticReport::Full(
                                    RelatedFullDocumentDiagnosticReport {
                                        full_document_diagnostic_report:
                                            FullDocumentDiagnosticReport {
                                                result_id: None,
                                                items: vec![],
                                            },
                                        related_documents: None,
                                    },
                                ),
                            ));
                        }
                    };

                    let version_id = compressed_doc.checksum.clone();
                    eprintln!(
                        "📋 PHPMD LSP: Running PHPMD for {} with version: {}",
                        file_name,
                        &version_id[..16]
                    );
                    eprintln!(
                        "📋 PHPMD LSP: About to analyze {} with {} bytes of content",
                        file_name,
                        content.len()
                    );

                    // Run PHPMD
                    if let Ok(diagnostics) = self.run_phpmd(&uri, path_str, Some(&content)).await {
                        eprintln!(
                            "📊 PHPMD LSP: Generated {} diagnostics for {}",
                            diagnostics.len(),
                            file_name
                        );

                        // Get the content checksum from the compressed document
                        let content_checksum = {
                            let docs = self.open_docs.read().unwrap();
                            docs.get(&uri)
                                .map(|doc| doc.checksum.clone())
                                .unwrap_or_else(|| String::from("unknown"))
                        };

                        // Cache the results with content checksum
                        let cached_results = CachedResults {
                            diagnostics: diagnostics.clone(),
                            result_id: version_id.clone(),
                            generated_at: Instant::now(),
                            content_checksum,
                        };

                        if let Ok(mut cache) = self.results_cache.write() {
                            eprintln!(
                                "💾 PHPMD LSP: Storing {} diagnostics in cache for {} (URI: {})",
                                diagnostics.len(),
                                file_name,
                                uri
                            );
                            eprintln!(
                                "💾 PHPMD LSP: Cache size before insert: {} entries",
                                cache.len()
                            );

                            // Log existing cache entries for debugging
                            for (cached_uri, cached_result) in cache.iter() {
                                let cached_file = cached_uri
                                    .path_segments()
                                    .and_then(|mut s| s.next_back())
                                    .unwrap_or("unknown");
                                eprintln!(
                                    "    - {} has {} cached diagnostics",
                                    cached_file,
                                    cached_result.diagnostics.len()
                                );
                            }

                            cache.insert(uri.clone(), cached_results);
                            eprintln!(
                                "💾 PHPMD LSP: Cache size after insert: {} entries",
                                cache.len()
                            );
                        }

                        return Ok(DocumentDiagnosticReportResult::Report(
                            DocumentDiagnosticReport::Full(RelatedFullDocumentDiagnosticReport {
                                full_document_diagnostic_report: FullDocumentDiagnosticReport {
                                    result_id: Some(version_id),
                                    items: diagnostics,
                                },
                                related_documents: None,
                            }),
                        ));
                    }
                }
            }
        }

        // Fallback: return empty diagnostics with no version
        eprintln!(
            "⚠️ PHPMD LSP: Unable to generate diagnostics for {}",
            file_name
        );
        Ok(DocumentDiagnosticReportResult::Report(
            DocumentDiagnosticReport::Full(RelatedFullDocumentDiagnosticReport {
                full_document_diagnostic_report: FullDocumentDiagnosticReport {
                    result_id: None,
                    items: vec![],
                },
                related_documents: None,
            }),
        ))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let stdin = stdin();
    let stdout = stdout();

    let (service, socket) = LspService::new(PhpmdLanguageServer::new);
    Server::new(stdin, stdout, socket).serve(service).await;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_line_number_from_description() {
        let description = "Remove error control operator '@' on line 264.";
        assert_eq!(parse_line_from_description(description), Some(264));
    }

    #[test]
    fn returns_none_when_description_has_no_line_marker() {
        let description = "The method openGzipFile() uses an error control operator.";
        assert_eq!(parse_line_from_description(description), None);
    }

    #[test]
    fn collapses_multi_line_method_span_to_description_line() {
        // PHPMD reports beginLine/endLine spanning the whole enclosing method
        // (line 260 through 290), but the description points at the real
        // operator line. These values become the diagnostic's start.line and
        // end.line, so they must be equal and match the parsed line.
        let description = "Remove error control operator '@' on line 264.";
        let begin_line = 260;
        let end_line = 290;
        assert!(end_line - begin_line > 10, "fixture must span a method");

        let (start_line, range_end_line) = error_control_operator_range(description, begin_line);

        assert_eq!(start_line, range_end_line);
        assert_eq!(start_line, 264);
    }

    #[test]
    fn falls_back_to_begin_line_when_description_unparsable() {
        let description = "Some message without a line reference.";
        let begin_line = 260;
        assert_eq!(
            error_control_operator_range(description, begin_line),
            (begin_line, begin_line)
        );
    }

    #[test]
    fn start_char_anchors_at_the_at_operator() {
        let line = "        $handle = @gzopen($path, 'wb9');";
        let at_offset = line.find('@').unwrap();
        assert_eq!(error_control_operator_start_char(line), at_offset);
    }

    #[test]
    fn start_char_falls_back_to_first_non_whitespace_without_at() {
        let line = "        $handle = gzopen($path, 'wb9');";
        // 8 leading spaces, then the first non-whitespace character.
        assert_eq!(error_control_operator_start_char(line), 8);
    }
}
