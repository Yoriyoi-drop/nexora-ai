//! File utilities untuk Nexora

use std::path::{Path, PathBuf};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write, BufRead, BufReader};
use anyhow::Result;

pub struct FileUtils;

impl FileUtils {
    /// Read file to string
    pub fn read_file_to_string(path: &Path) -> Result<String> {
        let mut file = File::open(path)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        Ok(content)
    }
    
    /// Read file to bytes
    pub fn read_file_to_bytes(path: &Path) -> Result<Vec<u8>> {
        let mut file = File::open(path)?;
        let mut content = Vec::new();
        file.read_to_end(&mut content)?;
        Ok(content)
    }
    
    /// Write string to file
    pub fn write_string_to_file(path: &Path, content: &str) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }
    
    /// Write bytes to file
    pub fn write_bytes_to_file(path: &Path, content: &[u8]) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(content)?;
        Ok(())
    }
    
    /// Append string to file
    pub fn append_to_file(path: &Path, content: &str) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }
    
    /// Check if file exists
    pub fn file_exists(path: &Path) -> bool {
        path.exists()
    }
    
    /// Check if path is a file
    pub fn is_file(path: &Path) -> bool {
        path.is_file()
    }
    
    /// Check if path is a directory
    pub fn is_directory(path: &Path) -> bool {
        path.is_dir()
    }
    
    /// Get file size in bytes
    pub fn get_file_size(path: &Path) -> Result<u64> {
        let metadata = fs::metadata(path)?;
        Ok(metadata.len())
    }
    
    /// Get file size as human readable string
    pub fn get_file_size_human(path: &Path) -> Result<String> {
        let size = Self::get_file_size(path)?;
        Ok(Self::format_bytes(size))
    }
    
    /// Format bytes to human readable string
    pub fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;
        
        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }
        
        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.2} {}", size, UNITS[unit_index])
        }
    }
    
    /// Create directory if it doesn't exist
    pub fn create_directory(path: &Path) -> Result<()> {
        if !path.exists() {
            fs::create_dir_all(path)?;
        }
        Ok(())
    }
    
    /// Create directory recursively
    pub fn create_directory_recursive(path: &Path) -> Result<()> {
        fs::create_dir_all(path)?;
        Ok(())
    }
    
    /// Delete file
    pub fn delete_file(path: &Path) -> Result<()> {
        if path.exists() {
            fs::remove_file(path)?;
        }
        Ok(())
    }
    
    /// Delete directory and its contents
    pub fn delete_directory(path: &Path) -> Result<()> {
        if path.exists() {
            fs::remove_dir_all(path)?;
        }
        Ok(())
    }
    
    /// Copy file
    pub fn copy_file(from: &Path, to: &Path) -> Result<()> {
        fs::copy(from, to)?;
        Ok(())
    }
    
    /// Move file
    pub fn move_file(from: &Path, to: &Path) -> Result<()> {
        fs::rename(from, to)?;
        Ok(())
    }
    
    /// List files in directory
    pub fn list_files(path: &Path) -> Result<Vec<PathBuf>> {
        if !path.is_dir() {
            return Err(anyhow::anyhow!("Path is not a directory"));
        }
        
        let mut files = Vec::new();
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                files.push(path);
            }
        }
        Ok(files)
    }
    
    /// List directories in directory
    pub fn list_directories(path: &Path) -> Result<Vec<PathBuf>> {
        if !path.is_dir() {
            return Err(anyhow::anyhow!("Path is not a directory"));
        }
        
        let mut dirs = Vec::new();
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
            }
        }
        Ok(dirs)
    }
    
    /// List all entries in directory (files and directories)
    pub fn list_all(path: &Path) -> Result<Vec<PathBuf>> {
        if !path.is_dir() {
            return Err(anyhow::anyhow!("Path is not a directory"));
        }
        
        let mut entries = Vec::new();
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            entries.push(entry.path());
        }
        Ok(entries)
    }
    
    /// Find files by pattern
    pub fn find_files_by_pattern(path: &Path, pattern: &str) -> Result<Vec<PathBuf>> {
        let regex = regex::Regex::new(pattern)?;
        let mut matching_files = Vec::new();
        
        Self::find_files_recursive(path, &mut matching_files)?;
        
        let filtered: Vec<PathBuf> = matching_files.into_iter()
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| regex.is_match(name))
                    .unwrap_or(false)
            })
            .collect();
        
        Ok(filtered)
    }
    
    /// Find files recursively
    fn find_files_recursive(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
        if path.is_file() {
            files.push(path.to_path_buf());
            return Ok(());
        }
        
        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let entry_path = entry.path();
                Self::find_files_recursive(&entry_path, files)?;
            }
        }
        
        Ok(())
    }
    
    /// Read file line by line
    pub fn read_lines(path: &Path) -> Result<Vec<String>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = Vec::new();
        
        for line in reader.lines() {
            lines.push(line?);
        }
        
        Ok(lines)
    }
    
    /// Read file line by line with callback
    pub fn read_lines_callback<F>(path: &Path, mut callback: F) -> Result<()>
    where
        F: FnMut(&str) -> Result<()>,
    {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        for line in reader.lines() {
            callback(&line?)?;
        }
        
        Ok(())
    }
    
    /// Write lines to file
    pub fn write_lines(path: &Path, lines: &[String]) -> Result<()> {
        let content = lines.join("\n");
        Self::write_string_to_file(path, &content)
    }
    
    /// Append lines to file
    pub fn append_lines(path: &Path, lines: &[String]) -> Result<()> {
        let content = lines.join("\n");
        if !content.is_empty() {
            Self::append_to_file(path, &format!("{}\n", content))?;
        }
        Ok(())
    }
    
    /// Get file extension
    pub fn get_file_extension(path: &Path) -> Option<String> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase())
    }
    
    /// Get file name without extension
    pub fn get_file_stem(path: &Path) -> Option<String> {
        path.file_stem()
            .and_then(|name| name.to_str())
            .map(|s| s.to_string())
    }
    
    /// Get file name with extension
    pub fn get_file_name(path: &Path) -> Option<String> {
        path.file_name()
            .and_then(|name| name.to_str())
            .map(|s| s.to_string())
    }
    
    /// Get parent directory
    pub fn get_parent_directory(path: &Path) -> Option<PathBuf> {
        path.parent().map(|p| p.to_path_buf())
    }
    
    /// Join paths
    pub fn join_paths(base: &Path, paths: &[&str]) -> PathBuf {
        let mut result = base.to_path_buf();
        for path in paths {
            result.push(path);
        }
        result
    }
    
    /// Get absolute path
    pub fn get_absolute_path(path: &Path) -> Result<PathBuf> {
        path.canonicalize()
            .map_err(|e| anyhow::anyhow!("Failed to get absolute path: {}", e))
    }
    
    /// Get relative path from base to target
    pub fn get_relative_path(base: &Path, target: &Path) -> Result<PathBuf> {
        let base_abs = Self::get_absolute_path(base)?;
        let target_abs = Self::get_absolute_path(target)?;
        
        pathdiff::diff_paths(&target_abs, &base_abs)
            .ok_or_else(|| anyhow::anyhow!("Failed to get relative path"))
    }
    
    /// Check if path is absolute
    pub fn is_absolute_path(path: &Path) -> bool {
        path.is_absolute()
    }
    
    /// Make path absolute
    pub fn make_absolute_path(path: &Path, base: &Path) -> PathBuf {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            base.join(path)
        }
    }
    
    /// Get file permissions
    pub fn get_file_permissions(path: &Path) -> Result<std::fs::Permissions> {
        let metadata = fs::metadata(path)?;
        Ok(metadata.permissions())
    }
    
    /// Set file permissions
    pub fn set_file_permissions(path: &Path, permissions: std::fs::Permissions) -> Result<()> {
        fs::set_permissions(path, permissions)?;
        Ok(())
    }
    
    /// Make file executable
    pub fn make_executable(path: &Path) -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = Self::get_file_permissions(path)?;
        perms.set_mode(perms.mode() | 0o755);
        Self::set_file_permissions(path, perms)?;
        Ok(())
    }
    
    /// Get file modification time
    pub fn get_file_modified_time(path: &Path) -> Result<std::time::SystemTime> {
        let metadata = fs::metadata(path)?;
        Ok(metadata.modified()?)
    }
    
    /// Get file creation time
    pub fn get_file_created_time(path: &Path) -> Result<std::time::SystemTime> {
        let metadata = fs::metadata(path)?;
        Ok(metadata.created()?)
    }
    
    /// Get file access time
    pub fn get_file_access_time(path: &Path) -> Result<std::time::SystemTime> {
        let metadata = fs::metadata(path)?;
        Ok(metadata.accessed()?)
    }
    
    /// Touch file (update modification time)
    pub fn touch_file(path: &Path) -> Result<()> {
        if path.exists() {
            let now = std::time::SystemTime::now();
            let file = OpenOptions::new()
                .write(true)
                .open(path)?;
            file.set_modified(now)?;
        } else {
            File::create(path)?;
        }
        Ok(())
    }
    
    /// Calculate file checksum (SHA-256)
    pub fn calculate_checksum(path: &Path) -> Result<String> {
        use sha2::{Sha256, Digest};
        
        let mut file = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0; 8192];
        
        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }
        
        let result = hasher.finalize();
        Ok(hex::encode(result))
    }
    
    /// Verify file checksum
    pub fn verify_checksum(path: &Path, expected: &str) -> Result<bool> {
        let actual = Self::calculate_checksum(path)?;
        Ok(actual == expected)
    }
    
    /// Compress file (simple gzip)
    pub fn compress_file(path: &Path, output_path: &Path) -> Result<()> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        
        let input_file = File::open(path)?;
        let output_file = File::create(output_path)?;
        let encoder = GzEncoder::new(output_file, Compression::default());
        
        let mut reader = std::io::BufReader::new(input_file);
        let mut writer = std::io::BufWriter::new(encoder);
        
        std::io::copy(&mut reader, &mut writer)?;
        Ok(())
    }
    
    /// Decompress file (simple gzip)
    pub fn decompress_file(path: &Path, output_path: &Path) -> Result<()> {
        use flate2::read::GzDecoder;
        
        let input_file = File::open(path)?;
        let output_file = File::create(output_path)?;
        
        let decoder = GzDecoder::new(input_file);
        let mut reader = std::io::BufReader::new(decoder);
        let mut writer = std::io::BufWriter::new(output_file);
        
        std::io::copy(&mut reader, &mut writer)?;
        Ok(())
    }
    
    /// Count lines in file
    pub fn count_lines(path: &Path) -> Result<usize> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(reader.lines().count())
    }
    
    /// Count words in file
    pub fn count_words(path: &Path) -> Result<usize> {
        let content = Self::read_file_to_string(path)?;
        Ok(content.split_whitespace().count())
    }
    
    /// Count characters in file
    pub fn count_characters(path: &Path) -> Result<usize> {
        let content = Self::read_file_to_string(path)?;
        Ok(content.chars().count())
    }
    
    /// Get file statistics
    pub fn get_file_stats(path: &Path) -> Result<FileStats> {
        let metadata = fs::metadata(path)?;
        let size = metadata.len();
        let modified = metadata.modified()?;
        let created = metadata.created()?;
        let accessed = metadata.accessed()?;
        
        let lines = if path.is_file() && Self::is_text_file(path)? {
            Self::count_lines(path).unwrap_or(0)
        } else {
            0
        };
        
        let words = if path.is_file() && Self::is_text_file(path)? {
            Self::count_words(path).unwrap_or(0)
        } else {
            0
        };
        
        let characters = if path.is_file() && Self::is_text_file(path)? {
            Self::count_characters(path).unwrap_or(0)
        } else {
            0
        };
        
        Ok(FileStats {
            path: path.to_path_buf(),
            size,
            size_human: Self::format_bytes(size),
            modified,
            created,
            accessed,
            is_file: path.is_file(),
            is_directory: path.is_dir(),
            lines,
            words,
            characters,
            extension: Self::get_file_extension(path),
            filename: Self::get_file_name(path),
        })
    }
    
    /// Check if file is text file (heuristic)
    pub fn is_text_file(path: &Path) -> Result<bool> {
        let mut file = File::open(path)?;
        let mut buffer = [0; 1024];
        let bytes_read = file.read(&mut buffer)?;
        
        if bytes_read == 0 {
            return Ok(true); // Empty file is considered text
        }
        
        // Check if buffer contains mostly printable ASCII
        let printable_count = buffer[..bytes_read]
            .iter()
            .filter(|&&b| b >= 32 && b <= 126 || b == b'\n' || b == b'\r' || b == b'\t')
            .count();
        
        let ratio = printable_count as f64 / bytes_read as f64;
        Ok(ratio > 0.7)
    }
    
    /// Create backup of file
    pub fn create_backup(path: &Path) -> Result<PathBuf> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let backup_name = format!("{}.backup.{}", 
            Self::get_file_name(path).unwrap_or("file".to_string()),
            timestamp
        );
        
        let backup_path = path.parent()
            .unwrap_or_else(|| Path::new("."))
            .join(backup_name);
        
        Self::copy_file(path, &backup_path)?;
        Ok(backup_path)
    }
    
    /// Restore from backup
    pub fn restore_from_backup(backup_path: &Path, target_path: &Path) -> Result<()> {
        Self::copy_file(backup_path, target_path)?;
        Ok(())
    }
    
    /// Find duplicate files in directory
    pub fn find_duplicates(path: &Path) -> Result<Vec<Vec<PathBuf>>> {
        let mut files = Vec::new();
        Self::find_files_recursive(path, &mut files)?;
        
        let mut checksum_map: std::collections::HashMap<String, Vec<PathBuf>> = std::collections::HashMap::with_capacity(files.len());
        
        for file in files {
            if let Ok(checksum) = Self::calculate_checksum(&file) {
                checksum_map.entry(checksum).or_insert_with(Vec::new).push(file);
            }
        }
        
        let duplicates: Vec<Vec<PathBuf>> = checksum_map.into_values()
            .filter(|files| files.len() > 1)
            .collect();
        
        Ok(duplicates)
    }
}

/// File statistics
#[derive(Debug, Clone)]
pub struct FileStats {
    pub path: PathBuf,
    pub size: u64,
    pub size_human: String,
    pub modified: std::time::SystemTime,
    pub created: std::time::SystemTime,
    pub accessed: std::time::SystemTime,
    pub is_file: bool,
    pub is_directory: bool,
    pub lines: usize,
    pub words: usize,
    pub characters: usize,
    pub extension: Option<String>,
    pub filename: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use tempfile::tempdir;
    
    #[test]
    fn test_file_utils() {
        let temp_dir = tempdir().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        let test_content = "Hello, World!\nThis is a test file.";
        
        // Test write and read
        FileUtils::write_string_to_file(&test_file, test_content).unwrap();
        let read_content = FileUtils::read_file_to_string(&test_file).unwrap();
        assert_eq!(read_content, test_content);
        
        // Test file exists
        assert!(FileUtils::file_exists(&test_file));
        assert!(FileUtils::is_file(&test_file));
        assert!(!FileUtils::is_directory(&test_file));
        
        // Test file size
        let size = FileUtils::get_file_size(&test_file).unwrap();
        assert_eq!(size, test_content.len() as u64);
        
        // Test file stats
        let stats = FileUtils::get_file_stats(&test_file).unwrap();
        assert_eq!(stats.size, test_content.len() as u64);
        assert_eq!(stats.lines, 2);
        assert_eq!(stats.words, 7);
        assert_eq!(stats.characters, test_content.len());
        
        // Test file extension
        assert_eq!(FileUtils::get_file_extension(&test_file), Some("txt".to_string()));
        assert_eq!(FileUtils::get_file_stem(&test_file), Some("test".to_string()));
        assert_eq!(FileUtils::get_file_name(&test_file), Some("test.txt".to_string()));
        
        // Test checksum
        let checksum = FileUtils::calculate_checksum(&test_file).unwrap();
        assert!(!checksum.is_empty());
        
        // Test backup
        let backup_path = FileUtils::create_backup(&test_file).unwrap();
        assert!(FileUtils::file_exists(&backup_path));
        
        // Test lines
        let lines = FileUtils::read_lines(&test_file).unwrap();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "Hello, World!");
        assert_eq!(lines[1], "This is a test file.");
        
        // Test append
        FileUtils::append_to_file(&test_file, "\nAppended line.").unwrap();
        let updated_content = FileUtils::read_file_to_string(&test_file).unwrap();
        assert!(updated_content.contains("Appended line."));
        
        // Test copy
        let copy_path = temp_dir.path().join("copy.txt");
        FileUtils::copy_file(&test_file, &copy_path).unwrap();
        assert!(FileUtils::file_exists(&copy_path));
        
        let copy_content = FileUtils::read_file_to_string(&copy_path).unwrap();
        assert_eq!(copy_content, updated_content);
        
        // Test delete
        FileUtils::delete_file(&copy_path).unwrap();
        assert!(!FileUtils::file_exists(&copy_path));
    }
    
    #[test]
    fn test_directory_operations() {
        let temp_dir = tempdir().unwrap();
        let test_dir = temp_dir.path().join("test_dir");
        
        // Test create directory
        FileUtils::create_directory(&test_dir).unwrap();
        assert!(FileUtils::file_exists(&test_dir));
        assert!(FileUtils::is_directory(&test_dir));
        
        // Test create subdirectory
        let sub_dir = test_dir.join("sub_dir");
        FileUtils::create_directory(&sub_dir).unwrap();
        assert!(FileUtils::file_exists(&sub_dir));
        
        // Test list directories
        let dirs = FileUtils::list_directories(&test_dir).unwrap();
        assert_eq!(dirs.len(), 1);
        assert_eq!(dirs[0].file_name().unwrap(), "sub_dir");
        
        // Test delete directory
        FileUtils::delete_directory(&test_dir).unwrap();
        assert!(!FileUtils::file_exists(&test_dir));
    }
    
    #[test]
    fn test_path_operations() {
        let temp_dir = tempdir().unwrap();
        let base_path = temp_dir.path();
        let file_path = base_path.join("test.txt");
        
        // Test path operations
        assert!(!FileUtils::is_absolute_path(&file_path));
        
        let absolute_path = FileUtils::get_absolute_path(&file_path).unwrap();
        assert!(FileUtils::is_absolute_path(&absolute_path));
        
        let joined_path = FileUtils::join_paths(base_path, &["subdir", "file.txt"]);
        assert_eq!(joined_path, base_path.join("subdir").join("file.txt"));
        
        let parent = FileUtils::get_parent_directory(&file_path).unwrap();
        assert_eq!(parent, base_path);
    }
}
