use std::env;
use std::fs;
use zed_extension_api::{self as zed, settings::LspSettings, Result};

// Constants
const PHPMD_CONFIG_FILES: &[&str] = &["phpmd.xml", "phpmd.xml.dist", ".phpmd.xml"];
const VERSION: &str = env!("CARGO_PKG_VERSION");

struct PhpmdLspExtension {
    phpmd_lsp: Option<PhpmdLspServer>,
}

struct PhpmdLspServer {
    cached_binary_path: Option<String>,
}

impl PhpmdLspServer {
    const LANGUAGE_SERVER_ID: &'static str = "phpmd";

    fn new() -> Self {
        Self {
            cached_binary_path: None,
        }
    }

    fn language_server_command(
        &mut self,
        _language_server_id: &zed::LanguageServerId,
        worktree: &zed::Worktree,
    ) -> Result<zed::Command> {
        let binary_path = self.language_server_binary_path(worktree)?;
        Ok(zed::Command {
            command: binary_path,
            args: vec![],
            env: Default::default(),
        })
    }

    fn language_server_binary_path(&mut self, worktree: &zed::Worktree) -> Result<String> {
        // Check if we have a cached binary path
        if let Some(cached_path) = &self.cached_binary_path {
            if fs::metadata(cached_path).is_ok() {
                return Ok(cached_path.clone());
            }
        }

        // Try to find the binary locally first (for development)
        let binary_name = Self::get_platform_binary_name();
        if let Some(path) = worktree.which(&binary_name) {
            self.cached_binary_path = Some(path.clone());
            return Ok(path);
        }

        // Download the binary from GitHub
        let downloaded_path = self.download_binary(&binary_name)?;
        self.cached_binary_path = Some(downloaded_path.clone());
        Ok(downloaded_path)
    }

    fn download_binary(&self, binary_name: &str) -> Result<String> {
        // Use the same pattern as Gleam extension
        let version_dir = format!("phpmd-{}", VERSION);
        let binary_path = format!("{}/{}", version_dir, binary_name);

        // Check if binary already exists
        if fs::metadata(&binary_path).is_ok() {
            return Ok(binary_path);
        }

        // Try to download from release assets first
        let (os, _arch) = zed::current_platform();
        let archive_ext = match os {
            zed::Os::Windows => "zip",
            _ => "tar.gz",
        };
        let archive_name = format!("{}.{}", binary_name, archive_ext);

        let release_url = format!(
            "https://github.com/GeneaLabs/zed-phpmd-lsp/releases/download/{}/{}",
            VERSION, archive_name
        );

        // Try downloading from release
        let file_type = match os {
            zed::Os::Windows => zed::DownloadedFileType::Zip,
            _ => zed::DownloadedFileType::GzipTar,
        };

        // Download the archive from release to version directory
        zed::download_file(&release_url, &version_dir, file_type)
            .map_err(|e| format!("Failed to download binary from release: {}. Please ensure the release {} exists with assets.", e, VERSION))?;

        // After extraction, the file should be in the bin directory
        if fs::metadata(&binary_path).is_err() {
            return Err(format!(
                "Binary not found after extraction. Expected at: {}",
                binary_path
            ));
        }

        // Make the binary executable on Unix-like systems
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(metadata) = fs::metadata(&binary_path) {
                let mut perms = metadata.permissions();
                perms.set_mode(0o755);
                fs::set_permissions(&binary_path, perms)
                    .map_err(|e| format!("Failed to set binary permissions: {}", e))?;
            }
        }

        Ok(binary_path)
    }

    fn get_platform_binary_name() -> String {
        let (os, arch) = zed::current_platform();
        match (os, arch) {
            (zed::Os::Windows, zed::Architecture::X8664) => {
                "phpmd-lsp-server-windows-x64.exe".to_string()
            }
            (zed::Os::Windows, zed::Architecture::Aarch64) => {
                "phpmd-lsp-server-windows-arm64.exe".to_string()
            }
            (zed::Os::Windows, _) => "phpmd-lsp-server.exe".to_string(),
            (zed::Os::Mac, zed::Architecture::Aarch64) => {
                "phpmd-lsp-server-macos-arm64".to_string()
            }
            (zed::Os::Mac, zed::Architecture::X8664) => "phpmd-lsp-server-macos-x64".to_string(),
            (zed::Os::Mac, _) => "phpmd-lsp-server".to_string(),
            (zed::Os::Linux, zed::Architecture::X8664) => "phpmd-lsp-server-linux-x64".to_string(),
            (zed::Os::Linux, zed::Architecture::Aarch64) => {
                "phpmd-lsp-server-linux-arm64".to_string()
            }
            (zed::Os::Linux, _) => "phpmd-lsp-server".to_string(),
        }
    }
}

impl zed::Extension for PhpmdLspExtension {
    fn new() -> Self {
        Self { phpmd_lsp: None }
    }

    fn language_server_command(
        &mut self,
        language_server_id: &zed::LanguageServerId,
        worktree: &zed::Worktree,
    ) -> Result<zed::Command> {
        match language_server_id.as_ref() {
            PhpmdLspServer::LANGUAGE_SERVER_ID => {
                let phpmd_lsp = self.phpmd_lsp.get_or_insert_with(PhpmdLspServer::new);
                phpmd_lsp.language_server_command(language_server_id, worktree)
            }
            language_server_id => Err(format!("unknown language server: {language_server_id}")),
        }
    }

    fn language_server_initialization_options(
        &mut self,
        language_server_id: &zed::LanguageServerId,
        worktree: &zed::Worktree,
    ) -> Result<Option<zed::serde_json::Value>> {
        // Check if this is our language server
        if language_server_id.as_ref() != PhpmdLspServer::LANGUAGE_SERVER_ID {
            return Ok(None);
        }
        let mut options = zed::serde_json::Map::new();

        // Try to get user-configured settings first
        let user_settings = LspSettings::for_worktree(language_server_id.as_ref(), worktree)
            .ok()
            .and_then(|lsp_settings| lsp_settings.settings.clone());

        // Download PHPMD PHAR to LSP server directory - LSP server will find it automatically
        Self::download_phar_if_needed("phpmd.phar").ok();

        // Determine rulesets to use (priority order: config file -> settings -> env -> default)
        let mut rulesets_to_use: Option<String> = None;

        // Try to find phpmd configuration file first (highest priority)
        if let Some(config_file) = Self::find_phpmd_config(worktree) {
            rulesets_to_use = Some(config_file);
        }

        // Check for user-configured rulesets from settings.json
        if rulesets_to_use.is_none() {
            if let Some(settings) = user_settings.as_ref() {
                // Support both string and array formats for rulesets
                if let Some(rulesets_value) = settings.get("rulesets") {
                    match rulesets_value {
                        // Single ruleset as string
                        zed::serde_json::Value::String(rulesets) if !rulesets.trim().is_empty() => {
                            rulesets_to_use = Some(rulesets.clone());
                        }
                        // Multiple rulesets as array
                        zed::serde_json::Value::Array(rulesets) => {
                            let ruleset_strings: Vec<String> = rulesets
                                .iter()
                                .filter_map(|v| v.as_str())
                                .filter(|s| !s.trim().is_empty())
                                .map(|s| s.to_string())
                                .collect();

                            if !ruleset_strings.is_empty() {
                                let combined_rulesets = ruleset_strings.join(",");
                                rulesets_to_use = Some(combined_rulesets);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Fall back to environment variable for rulesets
        if rulesets_to_use.is_none() {
            if let Ok(env_rulesets) = env::var("PHPMD_RULESETS") {
                if !env_rulesets.trim().is_empty() {
                    rulesets_to_use = Some(env_rulesets);
                }
            }
        }

        // If still no rulesets, use sensible defaults
        if rulesets_to_use.is_none() {
            // Default to common rulesets
            rulesets_to_use =
                Some("cleancode,codesize,controversial,design,naming,unusedcode".to_string());
        }

        // Pass the rulesets to the LSP server
        if let Some(rulesets) = rulesets_to_use {
            options.insert(
                "rulesets".to_string(),
                zed::serde_json::Value::String(rulesets.clone()),
            );
        }

        if options.is_empty() {
            Ok(None)
        } else {
            let json_value = zed::serde_json::Value::Object(options);
            Ok(Some(json_value))
        }
    }
}

impl PhpmdLspExtension {
    fn download_phar_if_needed(phar_name: &str) -> Result<String> {
        // Use the same pattern as Gleam extension for consistency
        let version_dir = format!("phpmd-{}", VERSION);
        let phar_path = format!("{}/{}", version_dir, phar_name);

        // Check if PHAR already exists
        if fs::metadata(&phar_path).is_ok() {
            return Ok(phar_path);
        }

        // Try to download from release assets first
        let archive_name = format!("{}.tar.gz", phar_name);

        let release_url = format!(
            "https://github.com/GeneaLabs/zed-phpmd-lsp/releases/download/{}/{}",
            VERSION, archive_name
        );

        // Download the archive from release to version directory
        zed::download_file(&release_url, &version_dir, zed::DownloadedFileType::GzipTar)
            .map_err(|e| format!("Failed to download {} from release: {}. Please ensure the release {} exists with assets.", phar_name, e, VERSION))?;

        // After extraction, the file should be in the bin directory
        if fs::metadata(&phar_path).is_err() {
            return Err(format!(
                "{} not found after extraction. Expected at: {}",
                phar_name, phar_path
            ));
        }

        // Make the PHAR executable on Unix-like systems
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(metadata) = fs::metadata(&phar_path) {
                let mut perms = metadata.permissions();
                perms.set_mode(0o755);
                fs::set_permissions(&phar_path, perms)
                    .map_err(|e| format!("Failed to set {} permissions: {}", phar_name, e))?;
            }
        }

        Ok(phar_path)
    }

    fn find_phpmd_config(worktree: &zed::Worktree) -> Option<String> {
        let root_path = std::path::PathBuf::from(worktree.root_path());

        for config_file in PHPMD_CONFIG_FILES {
            let config_path = root_path.join(config_file);

            if config_path.exists() {
                if let Some(path_str) = config_path.to_str() {
                    return Some(path_str.to_string());
                }
            }
        }

        None
    }
}

zed::register_extension!(PhpmdLspExtension);
