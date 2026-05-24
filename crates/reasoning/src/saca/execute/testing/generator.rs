//! Test Generator
//!
//! Generates test cases for implementations by analyzing function signatures and logic.

use crate::saca::error::*;

/// Test generator for implementations
pub struct TestGenerator;

impl TestGenerator {
    /// Generate test cases by analyzing the implementation's structure
    pub async fn generate_test_cases(&self, implementation: &str) -> SACAResult<Vec<TestCase>> {
        let mut test_cases = Vec::new();
        let lower = implementation.to_lowercase();

        // Detect return type for expected output generation
        let returns_result = lower.contains("-> result<")
            || lower.contains("-> core::result::")
            || lower.contains("-> std::result::");
        let returns_option = lower.contains("-> option<")
            || lower.contains("-> core::option::")
            || lower.contains("-> std::option::");
        let returns_vec = lower.contains("-> vec<")
            || lower.contains("-> std::vec::");
        let returns_bool = lower.contains("-> bool");
        let returns_int = lower.contains("-> i32")
            || lower.contains("-> i64")
            || lower.contains("-> usize")
            || lower.contains("-> u32")
            || lower.contains("-> u64");
        let returns_float = lower.contains("-> f32") || lower.contains("-> f64");
        let returns_string = lower.contains("-> string")
            || lower.contains("-> &str")
            || lower.contains("-> std::string::string");

        // Detect parameter types for input generation
        let takes_slice = lower.contains("&[") || lower.contains("&mut [") || lower.contains("slice");
        let takes_vec_param = lower.contains("vec<") || lower.contains("vector");
        let takes_string_param = lower.contains("string") || lower.contains("&str");
        let takes_int_param = lower.contains("i32") || lower.contains("i64") || lower.contains("usize") || lower.contains("u32") || lower.contains("u64");
        let takes_float_param = lower.contains("f32") || lower.contains("f64");

        // Detect operation types from function/type names and comments
        let is_sort = lower.contains("sort")
            || lower.contains("order")
            || lower.contains("cmp")
            || lower.contains("compare");
        let is_search = lower.contains("search")
            || lower.contains("find")
            || lower.contains("locate")
            || lower.contains("index");
        let is_filter = lower.contains("filter")
            || lower.contains("select")
            || lower.contains("where");
        let is_map = lower.contains("map")
            || lower.contains("transform")
            || lower.contains("convert");
        let is_parse = lower.contains("parse")
            || lower.contains("tokenize")
            || lower.contains("lex");
        let is_validate = lower.contains("validate")
            || lower.contains("check")
            || lower.contains("verify")
            || lower.contains("assert");
        let is_aggregate = lower.contains("sum")
            || lower.contains("count")
            || lower.contains("average")
            || lower.contains("total");
        let is_io = lower.contains("read")
            || lower.contains("write")
            || lower.contains("open")
            || lower.contains("load")
            || lower.contains("save");

        // Generate tests based on detected patterns
        if is_sort && takes_slice {
            self.add_sort_tests(&mut test_cases, returns_result);
        } else if is_search {
            self.add_search_tests(&mut test_cases, returns_option, returns_result);
        } else if is_filter {
            self.add_filter_tests(&mut test_cases);
        } else if is_map {
            self.add_map_tests(&mut test_cases);
        } else if is_parse {
            self.add_parse_tests(&mut test_cases, returns_result);
        } else if is_validate {
            self.add_validate_tests(&mut test_cases, returns_result, returns_bool);
        } else if is_aggregate {
            self.add_aggregate_tests(&mut test_cases, returns_int, returns_float);
        } else if is_io {
            self.add_io_tests(&mut test_cases, returns_result);
        } else {
            self.add_type_driven_tests(
                &mut test_cases,
                returns_result,
                returns_option,
                returns_vec,
                returns_bool,
                returns_string,
            );
        }

        // Add edge cases based on parameter types
        if takes_slice || takes_vec_param {
            test_cases.push(TestCase {
                id: "empty_input".to_string(),
                description: "Handle empty input".to_string(),
                input: "[]".to_string(),
                expected_output: self.default_output(returns_result, returns_option, returns_vec, 0, 0.0, ""),
                test_type: TestType::EdgeCase,
            });
            test_cases.push(TestCase {
                id: "single_element".to_string(),
                description: "Handle single element".to_string(),
                input: "[1]".to_string(),
                expected_output: self.default_output(returns_result, returns_option, returns_vec, 1, 1.0, "single"),
                test_type: TestType::EdgeCase,
            });
        }
        if takes_string_param {
            test_cases.push(TestCase {
                id: "empty_string".to_string(),
                description: "Handle empty string input".to_string(),
                input: "\"\"".to_string(),
                expected_output: self.default_output(returns_result, returns_option, returns_vec, 0, 0.0, ""),
                test_type: TestType::EdgeCase,
            });
            test_cases.push(TestCase {
                id: "unicode_string".to_string(),
                description: "Handle unicode/UTF-8 string".to_string(),
                input: "\"héllo wörld 🚀\"".to_string(),
                expected_output: self.default_output(returns_result, returns_option, returns_vec, 0, 0.0, "processed_unicode"),
                test_type: TestType::EdgeCase,
            });
        }
        if takes_int_param {
            test_cases.push(TestCase {
                id: "zero_value".to_string(),
                description: "Handle zero input value".to_string(),
                input: "0".to_string(),
                expected_output: self.default_output(returns_result, returns_option, returns_vec, 0, 0.0, "zero"),
                test_type: TestType::EdgeCase,
            });
            test_cases.push(TestCase {
                id: "negative_value".to_string(),
                description: "Handle negative input".to_string(),
                input: "-1".to_string(),
                expected_output: self.default_output(returns_result, returns_option, returns_vec, -1, -1.0, "negative"),
                test_type: TestType::EdgeCase,
            });
            test_cases.push(TestCase {
                id: "max_boundary".to_string(),
                description: "Handle maximum boundary input".to_string(),
                input: "usize::MAX".to_string(),
                expected_output: self.default_output(returns_result, returns_option, returns_vec, 0, 0.0, "boundary"),
                test_type: TestType::EdgeCase,
            });
        }
        if takes_float_param {
            test_cases.push(TestCase {
                id: "float_precision".to_string(),
                description: "Handle floating point precision".to_string(),
                input: "0.1 + 0.2".to_string(),
                expected_output: self.default_output(returns_result, returns_option, returns_vec, 0, 0.30000000000000004_f64, "precision"),
                test_type: TestType::EdgeCase,
            });
        }

        Ok(test_cases)
    }

    fn add_sort_tests(&self, tests: &mut Vec<TestCase>, returns_result: bool) {
        let ok = |v: &str| -> String {
            if returns_result { format!("Ok({})", v) } else { v.to_string() }
        };
        tests.extend(vec![
            TestCase {
                id: "sort_empty".to_string(),
                description: "Sort empty array".to_string(),
                input: "vec![]".to_string(),
                expected_output: ok("vec![]"),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "sort_single".to_string(),
                description: "Sort single element".to_string(),
                input: "vec![1]".to_string(),
                expected_output: ok("vec![1]"),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "sort_sorted".to_string(),
                description: "Sort already sorted array".to_string(),
                input: "vec![1, 2, 3, 4, 5]".to_string(),
                expected_output: ok("vec![1, 2, 3, 4, 5]"),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "sort_reverse".to_string(),
                description: "Sort reverse sorted array".to_string(),
                input: "vec![5, 4, 3, 2, 1]".to_string(),
                expected_output: ok("vec![1, 2, 3, 4, 5]"),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "sort_duplicates".to_string(),
                description: "Sort array with duplicates".to_string(),
                input: "vec![3, 1, 4, 1, 5, 9, 2, 6, 5]".to_string(),
                expected_output: ok("vec![1, 1, 2, 3, 4, 5, 5, 6, 9]"),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "sort_stability".to_string(),
                description: "Verify stable sort preserves original order of equal elements".to_string(),
                input: "vec![(2, 'b'), (1, 'a'), (2, 'a')]".to_string(),
                expected_output: ok("vec![(1, 'a'), (2, 'b'), (2, 'a')]"),
                test_type: TestType::Unit,
            },
        ]);
    }

    fn add_search_tests(&self, tests: &mut Vec<TestCase>, returns_option: bool, returns_result: bool) {
        let found = |i: &str| -> String {
            if returns_option { format!("Some({})", i) }
            else if returns_result { format!("Ok({})", i) }
            else { i.to_string() }
        };
        let not_found = || -> String {
            if returns_option { "None".to_string() }
            else if returns_result { "Err(...)".to_string() }
            else { "-1".to_string() }
        };
        tests.extend(vec![
            TestCase {
                id: "search_found".to_string(),
                description: "Search for existing element".to_string(),
                input: "vec![1, 2, 3, 4, 5], 3".to_string(),
                expected_output: found("2"),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "search_not_found".to_string(),
                description: "Search for non-existing element".to_string(),
                input: "vec![1, 2, 3, 4, 5], 6".to_string(),
                expected_output: not_found(),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "search_empty".to_string(),
                description: "Search in empty array".to_string(),
                input: "vec![], 1".to_string(),
                expected_output: not_found(),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "search_first".to_string(),
                description: "Search for first element".to_string(),
                input: "vec![1, 2, 3], 1".to_string(),
                expected_output: found("0"),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "search_last".to_string(),
                description: "Search for last element".to_string(),
                input: "vec![1, 2, 3], 3".to_string(),
                expected_output: found("2"),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "search_duplicates".to_string(),
                description: "Search when duplicates exist (returns first match)".to_string(),
                input: "vec![1, 2, 2, 3], 2".to_string(),
                expected_output: found("1"),
                test_type: TestType::Unit,
            },
        ]);
    }

    fn add_filter_tests(&self, tests: &mut Vec<TestCase>) {
        tests.extend(vec![
            TestCase {
                id: "filter_all_match".to_string(),
                description: "Filter where all elements match".to_string(),
                input: "vec![2, 4, 6], even".to_string(),
                expected_output: "vec![2, 4, 6]".to_string(),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "filter_none_match".to_string(),
                description: "Filter where no elements match".to_string(),
                input: "vec![1, 3, 5], even".to_string(),
                expected_output: "vec![]".to_string(),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "filter_empty".to_string(),
                description: "Filter empty collection".to_string(),
                input: "vec![], even".to_string(),
                expected_output: "vec![]".to_string(),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "filter_some_match".to_string(),
                description: "Filter where some elements match".to_string(),
                input: "vec![1, 2, 3, 4, 5], even".to_string(),
                expected_output: "vec![2, 4]".to_string(),
                test_type: TestType::Unit,
            },
        ]);
    }

    fn add_map_tests(&self, tests: &mut Vec<TestCase>) {
        tests.extend(vec![
            TestCase {
                id: "map_empty".to_string(),
                description: "Map empty collection".to_string(),
                input: "vec![], |x| x * 2".to_string(),
                expected_output: "vec![]".to_string(),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "map_identity".to_string(),
                description: "Map with identity function".to_string(),
                input: "vec![1, 2, 3], |x| x".to_string(),
                expected_output: "vec![1, 2, 3]".to_string(),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "map_transform".to_string(),
                description: "Map with transformation".to_string(),
                input: "vec![1, 2, 3], |x| x * 2".to_string(),
                expected_output: "vec![2, 4, 6]".to_string(),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "map_type_change".to_string(),
                description: "Map changing element type".to_string(),
                input: "vec![1, 2, 3], |x| x.to_string()".to_string(),
                expected_output: "vec![\"1\", \"2\", \"3\"]".to_string(),
                test_type: TestType::Unit,
            },
        ]);
    }

    fn add_parse_tests(&self, tests: &mut Vec<TestCase>, returns_result: bool) {
        let ok_out = |v: &str| -> String {
            if returns_result { format!("Ok({})", v) } else { v.to_string() }
        };
        let err_out = || -> String {
            if returns_result { "Err(...)".to_string() } else { "None".to_string() }
        };
        tests.extend(vec![
            TestCase {
                id: "parse_valid".to_string(),
                description: "Parse valid input".to_string(),
                input: "\"valid_input\"".to_string(),
                expected_output: ok_out("parsed_value"),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "parse_empty".to_string(),
                description: "Parse empty input".to_string(),
                input: "\"\"".to_string(),
                expected_output: err_out(),
                test_type: TestType::EdgeCase,
            },
            TestCase {
                id: "parse_malformed".to_string(),
                description: "Parse malformed input".to_string(),
                input: "\"!@#$%^\"".to_string(),
                expected_output: err_out(),
                test_type: TestType::EdgeCase,
            },
            TestCase {
                id: "parse_partial".to_string(),
                description: "Parse partial / incomplete input".to_string(),
                input: "\"prefix_\"".to_string(),
                expected_output: err_out(),
                test_type: TestType::EdgeCase,
            },
        ]);
    }

    fn add_validate_tests(&self, tests: &mut Vec<TestCase>, returns_result: bool, returns_bool: bool) {
        let pass = || -> String {
            if returns_result { "Ok(())".to_string() } else if returns_bool { "true".to_string() } else { "pass".to_string() }
        };
        let fail = || -> String {
            if returns_result { "Err(...)".to_string() } else if returns_bool { "false".to_string() } else { "fail".to_string() }
        };
        tests.extend(vec![
            TestCase {
                id: "validate_valid".to_string(),
                description: "Validate correct input passes".to_string(),
                input: "\"valid_data\"".to_string(),
                expected_output: pass(),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "validate_invalid".to_string(),
                description: "Validate incorrect input fails".to_string(),
                input: "\"\"".to_string(),
                expected_output: fail(),
                test_type: TestType::EdgeCase,
            },
            TestCase {
                id: "validate_boundary".to_string(),
                description: "Validate boundary values".to_string(),
                input: "MAX".to_string(),
                expected_output: pass(),
                test_type: TestType::EdgeCase,
            },
        ]);
    }

    fn add_aggregate_tests(&self, tests: &mut Vec<TestCase>, returns_int: bool, returns_float: bool) {
        let out = |v: &str| -> String {
            if returns_int || returns_float { v.to_string() } else { format!("\"{}\"", v) }
        };
        tests.extend(vec![
            TestCase {
                id: "aggregate_empty".to_string(),
                description: "Aggregate empty collection".to_string(),
                input: "vec![]".to_string(),
                expected_output: if returns_int { "0".to_string() } else { "0.0".to_string() },
                test_type: TestType::EdgeCase,
            },
            TestCase {
                id: "aggregate_single".to_string(),
                description: "Aggregate single element".to_string(),
                input: "vec![42]".to_string(),
                expected_output: out("42"),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "aggregate_multiple".to_string(),
                description: "Aggregate multiple elements".to_string(),
                input: "vec![1, 2, 3, 4, 5]".to_string(),
                expected_output: out("15"),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "aggregate_negative".to_string(),
                description: "Aggregate with negative values".to_string(),
                input: "vec![-5, -3, 8]".to_string(),
                expected_output: out("0"),
                test_type: TestType::Unit,
            },
        ]);
    }

    fn add_io_tests(&self, tests: &mut Vec<TestCase>, returns_result: bool) {
        let ok_out = |v: &str| -> String {
            if returns_result { format!("Ok({})", v) } else { v.to_string() }
        };
        let err_out = || -> String {
            if returns_result { "Err(...)".to_string() } else { "None".to_string() }
        };
        tests.extend(vec![
            TestCase {
                id: "io_file_exists".to_string(),
                description: "Read existing file".to_string(),
                input: "\"existing_file.txt\"".to_string(),
                expected_output: ok_out("file_content"),
                test_type: TestType::Integration,
            },
            TestCase {
                id: "io_file_not_found".to_string(),
                description: "Handle missing file".to_string(),
                input: "\"nonexistent.txt\"".to_string(),
                expected_output: err_out(),
                test_type: TestType::Integration,
            },
            TestCase {
                id: "io_empty_path".to_string(),
                description: "Handle empty file path".to_string(),
                input: "\"\"".to_string(),
                expected_output: err_out(),
                test_type: TestType::EdgeCase,
            },
        ]);
    }

    fn add_type_driven_tests(
        &self,
        tests: &mut Vec<TestCase>,
        returns_result: bool,
        returns_option: bool,
        returns_vec: bool,
        returns_bool: bool,
        returns_string: bool,
    ) {
        let default_ok = || -> String {
            if returns_option { "Some(value)".to_string() }
            else if returns_result { "Ok(value)".to_string() }
            else if returns_bool { "true".to_string() }
            else if returns_string { "\"result\"".to_string() }
            else if returns_vec { "vec![...]".to_string() }
            else { "value".to_string() }
        };
        let default_err = || -> String {
            if returns_option { "None".to_string() }
            else if returns_result { "Err(...)".to_string() }
            else if returns_bool { "false".to_string() }
            else { "error".to_string() }
        };
        tests.extend(vec![
            TestCase {
                id: "basic_functionality".to_string(),
                description: "Basic functionality test".to_string(),
                input: "valid_input".to_string(),
                expected_output: default_ok(),
                test_type: TestType::Unit,
            },
            TestCase {
                id: "empty_input".to_string(),
                description: "Edge case: empty input".to_string(),
                input: "".to_string(),
                expected_output: default_err(),
                test_type: TestType::EdgeCase,
            },
            TestCase {
                id: "null_input".to_string(),
                description: "Edge case: null/none input".to_string(),
                input: "None".to_string(),
                expected_output: default_err(),
                test_type: TestType::EdgeCase,
            },
            TestCase {
                id: "large_input".to_string(),
                description: "Stress test with large input".to_string(),
                input: "iter::repeat(0).take(10000).collect()".to_string(),
                expected_output: default_ok(),
                test_type: TestType::Performance,
            },
        ]);
    }

    fn default_output(&self, returns_result: bool, returns_option: bool, returns_vec: bool, _int_val: i32, _float_val: f64, _desc: &str) -> String {
        if returns_option { "None".to_string() }
        else if returns_result { "Err(...)".to_string() }
        else if returns_vec { "vec![]".to_string() }
        else { "default".to_string() }
    }
}

/// Test case definition
#[derive(Debug, Clone)]
pub struct TestCase {
    pub id: String,
    pub description: String,
    pub input: String,
    pub expected_output: String,
    pub test_type: TestType,
}

/// Test type enumeration
#[derive(Debug, Clone)]
pub enum TestType {
    Unit,
    Integration,
    Performance,
    EdgeCase,
}
