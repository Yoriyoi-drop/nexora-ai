//! Test Runner
//! 
//! Executes test cases and collects results.

use crate::reasoning::saca::{types::*, error::*};
use super::generator::{TestCase, TestType};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

/// Test runner for executing test cases
pub struct TestRunner;

impl TestRunner {
    /// Run all test cases for an implementation
    pub async fn run_tests(&self, implementation: &str, test_cases: Vec<TestCase>) -> SACAResult<Vec<TestResult>> {
        let mut test_results = Vec::new();
        
        for test_case in test_cases {
            let result = self.run_single_test(test_case, implementation).await?;
            test_results.push(result);
        }
        
        Ok(test_results)
    }
    
    /// Run a single test case
    async fn run_single_test(&self, test_case: TestCase, implementation: &str) -> SACAResult<TestResult> {
        let start_time = std::time::Instant::now();
        
        let (passed, actual_output, error_message) = self.run_test_execution(&test_case, implementation);
        let execution_time = start_time.elapsed();
        
        Ok(TestResult {
            test_id: test_case.id,
            test_type: test_case.test_type,
            passed,
            execution_time,
            input: test_case.input,
            expected_output: test_case.expected_output,
            actual_output,
            error_message,
        })
    }
    
    /// Run a test case against the implementation and compare output
    fn run_test_execution(
        &self,
        test_case: &TestCase,
        implementation: &str,
    ) -> (bool, String, Option<String>) {
        if implementation.contains("unimplemented!") {
            return (false, "Not implemented".to_string(), Some("Implementation contains unimplemented! macro".to_string()));
        }

        let temp_dir = std::env::temp_dir().join(format!("saca_test_{}_{}", std::process::id(), uuid::Uuid::new_v4()));
        let _ = std::fs::create_dir_all(&temp_dir);
        let file_path = temp_dir.join("test_impl.py");

        if let Err(e) = std::fs::write(&file_path, implementation) {
            let _ = std::fs::remove_dir_all(&temp_dir);
            return (false, String::new(), Some(format!("Failed to write temp file: {}", e)));
        }

        let output = match Command::new("python3")
            .arg(&file_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(mut child) => {
                if let Some(mut stdin) = child.stdin.take() {
                    let _ = stdin.write_all(test_case.input.as_bytes());
                    let _ = stdin.flush();
                }
                match child.wait_with_output() {
                    Ok(output) => output,
                    Err(e) => {
                        let _ = std::fs::remove_dir_all(&temp_dir);
                        return (false, String::new(), Some(format!("Failed to read child output: {}", e)));
                    }
                }
            }
            Err(e) => {
                let _ = std::fs::remove_dir_all(&temp_dir);
                return (false, String::new(), Some(format!("Failed to spawn python3: {}", e)));
            }
        };

        let _ = std::fs::remove_dir_all(&temp_dir);

        let actual_output = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if !output.status.success() {
            let err_msg = if stderr.is_empty() {
                format!("Process exited with code {:?}", output.status.code())
            } else {
                stderr
            };
            return (false, actual_output, Some(err_msg));
        }

        if actual_output == test_case.expected_output.trim() {
            (true, actual_output, None)
        } else {
            (false, actual_output.clone(), Some(format!("Expected '{}', got '{}'", test_case.expected_output, actual_output)))
        }
    }
}

/// Test result
#[derive(Debug, Clone)]
pub struct TestResult {
    pub test_id: String,
    pub test_type: TestType,
    pub passed: bool,
    pub execution_time: Duration,
    pub input: String,
    pub expected_output: String,
    pub actual_output: String,
    pub error_message: Option<String>,
}
