//! Data models for runtime isolation specification. Awaiting implementation.

use serde::{Deserialize, Serialize};

use crate::config::SandboxKind;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeIsolationSpec {
    pub sandbox: SandboxRuntime,
    pub container: ContainerRuntime,
    pub os_restrictions: OsRestrictions,
    pub resource_limits: ResourceLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxRuntime {
    pub kind: SandboxKind,
    pub version: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerRuntime {
    pub kind: ContainerKind,
    pub image: String,
    pub read_only_rootfs: bool,
    pub privileged: bool,
    pub drop_capabilities: Vec<String>,
    pub seccomp_profile: String,
    pub apparmor_profile: String,
    pub allowed_syscalls: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContainerKind {
    Docker,
    Podman,
    Containerd,
    Firecracker,
    Kata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsRestrictions {
    pub user_namespace: bool,
    pub pid_namespace: bool,
    pub net_namespace: bool,
    pub ipc_namespace: bool,
    pub uts_namespace: bool,
    pub cgroup_namespace: bool,
    pub chroot_enabled: bool,
    pub no_new_privileges: bool,
    pub disable_core_dumps: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub cpu_shares: u64,
    pub cpu_quota_us: i64,
    pub cpu_period_us: u64,
    pub memory_max_bytes: u64,
    pub memory_swap_max_bytes: u64,
    pub pids_max: i64,
    pub io_read_bps_max: u64,
    pub io_write_bps_max: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationLevel {
    Pod,
    MicroVM,
    Container,
    Process,
    Thread,
}

pub type RuntimeProfile = RuntimeIsolationSpec;

impl RuntimeIsolationSpec {
    pub fn gvisor_default() -> Self {
        Self {
            sandbox: SandboxRuntime {
                kind: SandboxKind::GVisor,
                version: "latest".into(),
                enabled: true,
            },
            container: ContainerRuntime {
                kind: ContainerKind::Containerd,
                image: "nexora/sandbox:latest".into(),
                read_only_rootfs: true,
                privileged: false,
                drop_capabilities: vec!["ALL".into()],
                seccomp_profile: "restricted.json".into(),
                apparmor_profile: "nexora-restricted".into(),
                allowed_syscalls: vec![
                    "read".into(), "write".into(), "exit".into(), "exit_group".into(),
                    "futex".into(), "clock_gettime".into(), "nanosleep".into(),
                ],
            },
            os_restrictions: OsRestrictions {
                user_namespace: true,
                pid_namespace: true,
                net_namespace: true,
                ipc_namespace: true,
                uts_namespace: true,
                cgroup_namespace: true,
                chroot_enabled: true,
                no_new_privileges: true,
                disable_core_dumps: true,
            },
            resource_limits: ResourceLimits {
                cpu_shares: 1024,
                cpu_quota_us: 100000,
                cpu_period_us: 100000,
                memory_max_bytes: 4 * 1024 * 1024 * 1024,
                memory_swap_max_bytes: 0,
                pids_max: 256,
                io_read_bps_max: 50 * 1024 * 1024,
                io_write_bps_max: 30 * 1024 * 1024,
            },
        }
    }

    pub fn firecracker_microvm() -> Self {
        Self {
            sandbox: SandboxRuntime {
                kind: SandboxKind::Firecracker,
                version: "1.7".into(),
                enabled: true,
            },
            container: ContainerRuntime {
                kind: ContainerKind::Firecracker,
                image: "nexora/microvm:latest".into(),
                read_only_rootfs: true,
                privileged: false,
                drop_capabilities: vec!["ALL".into()],
                seccomp_profile: "firecracker-default.json".into(),
                apparmor_profile: "nexora-microvm".into(),
                allowed_syscalls: vec!["read".into(), "write".into(), "exit".into(), "exit_group".into()],
            },
            os_restrictions: OsRestrictions {
                user_namespace: true,
                pid_namespace: true,
                net_namespace: true,
                ipc_namespace: true,
                uts_namespace: true,
                cgroup_namespace: true,
                chroot_enabled: true,
                no_new_privileges: true,
                disable_core_dumps: true,
            },
            resource_limits: ResourceLimits {
                cpu_shares: 512,
                cpu_quota_us: 50000,
                cpu_period_us: 100000,
                memory_max_bytes: 512 * 1024 * 1024,
                memory_swap_max_bytes: 0,
                pids_max: 64,
                io_read_bps_max: 20 * 1024 * 1024,
                io_write_bps_max: 10 * 1024 * 1024,
            },
        }
    }

    pub fn kata_container() -> Self {
        Self {
            sandbox: SandboxRuntime {
                kind: SandboxKind::KataContainers,
                version: "3.0".into(),
                enabled: true,
            },
            container: ContainerRuntime {
                kind: ContainerKind::Kata,
                image: "nexora/kata:latest".into(),
                read_only_rootfs: true,
                privileged: false,
                drop_capabilities: vec!["ALL".into()],
                seccomp_profile: "kata-default.json".into(),
                apparmor_profile: "nexora-kata".into(),
                allowed_syscalls: vec![
                    "read".into(), "write".into(), "open".into(), "close".into(),
                    "mmap".into(), "munmap".into(), "brk".into(), "exit".into(),
                ],
            },
            os_restrictions: OsRestrictions {
                user_namespace: true,
                pid_namespace: true,
                net_namespace: true,
                ipc_namespace: true,
                uts_namespace: true,
                cgroup_namespace: true,
                chroot_enabled: true,
                no_new_privileges: true,
                disable_core_dumps: true,
            },
            resource_limits: ResourceLimits {
                cpu_shares: 1024,
                cpu_quota_us: 100000,
                cpu_period_us: 100000,
                memory_max_bytes: 2 * 1024 * 1024 * 1024,
                memory_swap_max_bytes: 0,
                pids_max: 128,
                io_read_bps_max: 30 * 1024 * 1024,
                io_write_bps_max: 20 * 1024 * 1024,
            },
        }
    }

    pub fn validate(&self) -> Vec<String> {
        let mut violations = Vec::with_capacity(4);
        if self.container.privileged {
            violations.push("Privileged container is not allowed".into());
        }
        if self.resource_limits.memory_max_bytes < 64 * 1024 * 1024 {
            violations.push("Memory limit too low (< 64MB)".into());
        }
        if !self.os_restrictions.no_new_privileges {
            violations.push("no_new_privileges must be enabled".into());
        }
        if !self.os_restrictions.user_namespace {
            violations.push("User namespace isolation required".into());
        }
        violations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gvisor_default_validation() {
        let spec = RuntimeIsolationSpec::gvisor_default();
        let violations = spec.validate();
        assert!(violations.is_empty());
    }

    #[test]
    fn test_privileged_container_rejected() {
        let mut spec = RuntimeIsolationSpec::gvisor_default();
        spec.container.privileged = true;
        let violations = spec.validate();
        assert!(violations.iter().any(|v| v.contains("Privileged")));
    }
}
