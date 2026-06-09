use super::*;

#[test]
fn test_system_info_default() {
    let info = SystemInfo {
        cpu_usage: 0.0,
        memory_usage: 0.0,
        total_memory: 0,
        used_memory: 0,
        processes: 0,
        uptime: "00:00:00".to_string(),
    };
    assert_eq!(info.cpu_usage, 0.0);
    assert_eq!(info.uptime, "00:00:00");
}

#[test]
fn test_threshold_alert_none() {
    assert!(matches!(ThresholdAlert::from_cpu(10.0), ThresholdAlert::None));
    assert!(matches!(ThresholdAlert::from_mem(30.0), ThresholdAlert::None));
}

#[test]
fn test_threshold_alert_warning() {
    assert!(matches!(ThresholdAlert::from_cpu(60.0), ThresholdAlert::Warning));
    assert!(matches!(ThresholdAlert::from_mem(80.0), ThresholdAlert::Warning));
}

#[test]
fn test_threshold_alert_critical() {
    assert!(matches!(ThresholdAlert::from_cpu(85.0), ThresholdAlert::Critical));
    assert!(matches!(ThresholdAlert::from_mem(95.0), ThresholdAlert::Critical));
}

#[test]
fn test_test_result_serialization() {
    let tr = TestResult {
        name: "test_foo".to_string(),
        status: "✓ PASSED".to_string(),
        duration: "0.05s".to_string(),
        error: None,
    };
    let json = serde_json::to_string(&tr).unwrap();
    assert!(json.contains("test_foo"));
    assert!(json.contains("PASSED"));
}

#[test]
fn test_test_result_with_error() {
    let tr = TestResult {
        name: "test_bar".to_string(),
        status: "✗ FAILED".to_string(),
        duration: "0.10s".to_string(),
        error: Some("assertion failed".to_string()),
    };
    let json = serde_json::to_string(&tr).unwrap();
    assert!(json.contains("assertion failed"));
}

#[test]
fn test_log_entry_new() {
    let log = LogEntry {
        timestamp: "12:00:00".to_string(),
        level: "INFO".to_string(),
        message: "test message".to_string(),
    };
    assert_eq!(log.level, "INFO");
    assert_eq!(log.message, "test message");
}

#[test]
fn test_nextest_output_parse() {
    let json = r#"{
        "test_run": {
            "test_list": [
                {"test_name": "test_a", "status": "passed", "exec_time": 0.05}
            ]
        }
    }"#;
    let parsed: NextestOutput = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.test_run.test_list.len(), 1);
    assert_eq!(parsed.test_run.test_list[0].test_name, "test_a");
}

#[test]
fn test_app_new() {
    let app = App::new();
    assert!(!app.should_quit);
    assert!(!app.is_running_tests);
    assert!(!app.show_health_detail);
}

#[test]
fn test_app_add_log() {
    let mut app = App::new();
    app.add_log("ERROR", "something broke");
    assert!(app.logs.len() >= 2);
    assert_eq!(app.logs.last().unwrap().level, "ERROR");
    assert!(app.dirty);
}

#[test]
fn test_app_log_limit() {
    let mut app = App::new();
    for i in 0..150 {
        app.add_log("INFO", &format!("log {}", i));
    }
    assert!(app.logs.len() <= LOG_CAPACITY);
}

#[test]
fn test_app_on_key_quit() {
    let mut app = App::new();
    app.on_key(KeyCode::Char('q'));
    assert!(app.should_quit);
}

#[test]
fn test_app_on_key_up_down() {
    let mut app = App::new();
    app.on_key(KeyCode::Up);
    assert_eq!(app.selected_test, 0);
    app.on_key(KeyCode::Down);
    assert_eq!(app.selected_test, 0);
}

#[test]
fn test_app_on_key_refresh() {
    let mut app = App::new();
    app.on_key(KeyCode::Char('r'));
    assert!(app.dirty);
}

#[test]
fn test_app_on_key_health_toggle() {
    let mut app = App::new();
    app.on_key(KeyCode::Char('h'));
    assert!(app.show_health_detail);
    app.on_key(KeyCode::Char('h'));
    assert!(!app.show_health_detail);
}

#[test]
fn test_app_dirty_after_new() {
    let app = App::new();
    assert!(app.dirty);
}

#[test]
fn test_parse_nextest_output_single() {
    let mut app = App::new();
    let output = r#"{"test_name": "my_test", "status": "passed", "exec_time": 0.05}"#;
    app.parse_nextest_output(output).unwrap();
    assert_eq!(app.test_results.len(), 1);
    assert_eq!(app.test_results[0].name, "my_test");
}

#[test]
fn test_parse_nextest_output_non_json() {
    let mut app = App::new();
    let output = "running 1 test\n some warning\n error: test failed";
    app.parse_nextest_output(output).unwrap();
    assert!(app.logs.len() > 1);
}

#[test]
fn test_parse_nextest_output_failed() {
    let mut app = App::new();
    let output = r#"{"test_name": "fail_test", "status": "failed", "exec_time": 0.1, "stderr": "assertion failed"}"#;
    app.parse_nextest_output(output).unwrap();
    assert_eq!(app.test_results[0].status, "✗ FAILED");
    assert!(app.test_results[0].error.is_some());
}

#[test]
fn test_parse_nextest_output_mixed() {
    let mut app = App::new();
    let output = "{\"test_name\": \"pass_test\", \"status\": \"passed\", \"exec_time\": 0.01}\nregular output\n{\"test_name\": \"fail_test\", \"status\": \"failed\", \"exec_time\": 0.02}";
    app.parse_nextest_output(output).unwrap();
    assert_eq!(app.test_results.len(), 2);
    assert!(app.logs.len() > 2);
}

#[test]
fn test_app_update_system_info() {
    let mut app = App::new();
    app.update_system_info();
    assert!(app.dirty);
    assert!(app.system_info.cpu_usage >= 0.0);
}

#[test]
fn test_test_result_default_values() {
    let json = r#"{"name": "", "status": "", "duration": "", "error": null}"#;
    let tr: TestResult = serde_json::from_str(json).unwrap();
    assert_eq!(tr.name, "");
    assert_eq!(tr.status, "");
}

#[test]
fn test_on_key_ignores_unknown() {
    let mut app = App::new();
    let before = app.dirty;
    app.on_key(KeyCode::Char('z'));
    assert_eq!(app.dirty, before);
}

#[test]
fn test_gauge_color() {
    assert_eq!(gauge_color(30.0, 50.0, 80.0), Color::Green);
    assert_eq!(gauge_color(60.0, 50.0, 80.0), Color::Yellow);
    assert_eq!(gauge_color(90.0, 50.0, 80.0), Color::Red);
}

#[test]
fn test_format_memory() {
    assert_eq!(format_memory(500), "500 B");
    assert!(format_memory(2_000_000).contains("MB"));
    assert!(format_memory(2_000_000_000).contains("GB"));
}

#[test]
fn test_monitoring_bridge_new() {
    let bridge = MonitoringBridge::new();
    assert_eq!(bridge.last_cpu_pct, 0.0);
    assert_eq!(bridge.last_mem_pct, 0.0);
}

#[test]
fn test_monitoring_bridge_update() {
    let mut bridge = MonitoringBridge::new();
    let sys = SystemInfo {
        cpu_usage: 45.0,
        memory_usage: 60.0,
        total_memory: 16_000_000_000,
        used_memory: 9_600_000_000,
        processes: 128,
        uptime: "01:00:00".to_string(),
    };
    let events = bridge.update(&sys);
    assert_eq!(bridge.last_cpu_pct, 45.0);
    assert_eq!(bridge.last_mem_pct, 60.0);
    assert!(events.is_empty());
}

#[test]
fn test_monitoring_bridge_health_transition() {
    let mut bridge = MonitoringBridge::new();
    let sys = SystemInfo {
        cpu_usage: 96.0,
        memory_usage: 50.0,
        total_memory: 16_000_000_000,
        used_memory: 8_000_000_000,
        processes: 128,
        uptime: "01:00:00".to_string(),
    };
    let events = bridge.update(&sys);
    let has_health_event = events.iter().any(|e| e.message.contains("Health status"));
    assert!(has_health_event);
}

#[test]
fn test_nextest_test_case() {
    let tc = TestCase {
        test_name: "t".to_string(),
        status: "passed".to_string(),
        exec_time: Some(0.5),
        stdout: Some("ok".to_string()),
        stderr: None,
    };
    assert_eq!(tc.test_name, "t");
    assert_eq!(tc.exec_time, Some(0.5));
}

#[test]
fn test_nextest_test_case_empty() {
    let tc = TestCase {
        test_name: String::new(),
        status: String::new(),
        exec_time: None,
        stdout: None,
        stderr: None,
    };
    assert!(tc.exec_time.is_none());
    assert!(tc.stdout.is_none());
}

#[test]
fn test_log_capacity_constant() {
    assert_eq!(LOG_CAPACITY, 100);
}

#[test]
fn test_threshold_constants() {
    assert_eq!(CPU_WARN, 50.0);
    assert_eq!(CPU_CRIT, 80.0);
    assert_eq!(MEM_WARN, 70.0);
    assert_eq!(MEM_CRIT, 90.0);
}
