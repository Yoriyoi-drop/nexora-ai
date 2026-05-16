.PHONY: all check build test lint fmt clean deny audit

all: check test lint

check:
	cargo check

build:
	cargo build

build-release:
	cargo build --release

test:
	cargo nextest run

test-all:
	cargo nextest run --all-features

lint:
	cargo clippy --all-targets -- -D warnings

fmt:
	cargo fmt --all --check

fmt-fix:
	cargo fmt --all

clean:
	cargo clean

deny:
	cargo deny check all

audit:
	cargo audit

ci: check fmt lint test deny
