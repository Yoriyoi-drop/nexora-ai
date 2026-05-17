fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let output = std::process::Command::new("pkg-config")
        .args(["--libs", "--cflags", "openblas"])
        .output()
        .expect("pkg-config for openblas failed. Install libopenblas-dev.");
    let link_args = String::from_utf8_lossy(&output.stdout);
    for arg in link_args.split_whitespace() {
        if let Some(lib) = arg.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={}", lib);
        } else if let Some(path) = arg.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={}", path);
        }
    }
}
