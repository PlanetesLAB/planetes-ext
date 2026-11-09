//! Downloads from the GitHub Releases of `PlanetesLAB` repositories with `curl` and `tar`:
//! a repository's data folder (its `setup` command) and its latest release executable
//! (its `update` command). Every command is printed before it runs.

use std::cmp::Ordering;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The address of the releases of a `PlanetesLAB` repository.
fn releases(repo: &str) -> String {
    format!("https://github.com/PlanetesLAB/{repo}/releases")
}

/// The name of this platform in release assets: `linux-x86_64` or `macos-arm64`.
///
/// # Errors
/// Returns an error on other platforms, which have no release executable.
pub fn platform() -> io::Result<&'static str> {
    match (env::consts::OS, env::consts::ARCH) {
        ("linux", "x86_64") => Ok("linux-x86_64"),
        ("macos", "aarch64") => Ok("macos-arm64"),
        (os, arch) => Err(io::Error::other(format!(
            "no release executable for {os} {arch}"
        ))),
    }
}

/// Downloads `<repo>-resources-<version>.tar.gz` of release `v<version>` and puts its
/// `resources/` folder at `target`, replacing what is there (a symlink is removed, not
/// followed) only after the archive is unpacked.
///
/// # Errors
/// Returns an error if `curl` or `tar` fails, the archive has no `resources/`, or the
/// folder cannot be replaced.
pub fn install_resources(repo: &str, version: &str, target: &Path) -> io::Result<()> {
    let home = target
        .parent()
        .ok_or_else(|| io::Error::other("resources folder without a parent"))?;
    let work = fresh_dir(&home.join(".setup"))?;
    let file = format!("{repo}-resources-{version}.tar.gz");
    let archive = work.join(&file);
    download(
        &format!("{}/download/v{version}/{file}", releases(repo)),
        &archive,
    )?;
    unpack(&archive, &work)?;
    let unpacked = work.join("resources");
    if !unpacked.is_dir() {
        return Err(io::Error::other(format!("{file} has no resources/ folder")));
    }
    if let Ok(meta) = fs::symlink_metadata(target) {
        if meta.is_dir() {
            fs::remove_dir_all(target)?;
        } else {
            fs::remove_file(target)?;
        }
        println!("{repo}: replaced {}", target.display());
    }
    fs::rename(&unpacked, target)?;
    fs::remove_dir_all(&work)?;
    println!("{repo}: data {version} in {}", target.display());
    Ok(())
}

/// The version of the latest release, from the tag `v<version>` that
/// `<releases>/latest` redirects to.
///
/// # Errors
/// Returns an error if `curl` fails or the address has no tag.
pub fn latest_version(repo: &str) -> io::Result<String> {
    let latest = format!("{}/latest", releases(repo));
    println!("curl -fsSL -o /dev/null -w %{{url_effective}} {latest}");
    let out = Command::new("curl")
        .args(["-fsSL", "-o", "/dev/null", "-w", "%{url_effective}"])
        .arg(&latest)
        .output()?;
    let url = String::from_utf8_lossy(&out.stdout);
    match url.rsplit_once("/tag/v") {
        Some((_, version)) if out.status.success() => Ok(version.trim().to_owned()),
        _ => Err(io::Error::other(format!(
            "finding the latest release at {latest} failed"
        ))),
    }
}

/// Replaces the running executable with the latest release of `repo` for this
/// platform, `<repo>-<version>-<platform>.tar.gz`, when that is newer than `current`.
/// Returns the path of the replaced executable, or `None` when there is nothing newer.
///
/// # Errors
/// Returns an error on a platform without release executables, or if a download, the
/// unpacking or the replacement fails.
pub fn update_executable(repo: &str, current: &str) -> io::Result<Option<PathBuf>> {
    let platform = platform()?;
    let latest = latest_version(repo)?;
    match parse(&latest)?.cmp(&parse(current)?) {
        Ordering::Equal => {
            println!("{repo} {current} is the latest release");
            return Ok(None);
        }
        Ordering::Less => {
            println!("{repo} {current} is newer than the latest release, {latest}");
            return Ok(None);
        }
        Ordering::Greater => {}
    }

    let exe = fs::canonicalize(env::current_exe()?)?;
    let dir = exe
        .parent()
        .ok_or_else(|| io::Error::other("executable without a directory"))?;
    let work = fresh_dir(&dir.join(format!(".{repo}.update")))?;
    let file = format!("{repo}-{latest}-{platform}.tar.gz");
    let archive = work.join(&file);
    download(
        &format!("{}/download/v{latest}/{file}", releases(repo)),
        &archive,
    )?;
    unpack(&archive, &work)?;
    let new = work.join(repo);
    if !new.is_file() {
        return Err(io::Error::other(format!("{file} has no {repo} executable")));
    }
    // On the same file system, the rename replaces the executable in one step.
    fs::rename(&new, &exe)?;
    fs::remove_dir_all(&work)?;
    println!("{repo}: {current} -> {latest} in {}", exe.display());
    Ok(Some(exe))
}

/// `MAJOR.MINOR.PATCH` as numbers.
fn parse(version: &str) -> io::Result<(u64, u64, u64)> {
    let invalid = || io::Error::other(format!("unexpected version {version}"));
    let parts: Vec<u64> = version
        .split('.')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .map_err(|_| invalid())?;
    match parts[..] {
        [major, minor, patch] => Ok((major, minor, patch)),
        _ => Err(invalid()),
    }
}

/// An empty directory at `path`.
fn fresh_dir(path: &Path) -> io::Result<PathBuf> {
    if path.exists() {
        fs::remove_dir_all(path)?;
    }
    fs::create_dir_all(path)?;
    Ok(path.to_path_buf())
}

/// Downloads `url` to `archive` with curl.
fn download(url: &str, archive: &Path) -> io::Result<()> {
    println!("curl -fL --retry 3 -o {} {url}", archive.display());
    let status = Command::new("curl")
        .args(["-fL", "--retry", "3", "-o"])
        .arg(archive)
        .arg(url)
        .status()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "downloading {url} failed; the release may not exist or not be public"
        )));
    }
    Ok(())
}

/// Unpacks `archive` into `dir` with tar.
fn unpack(archive: &Path, dir: &Path) -> io::Result<()> {
    println!("tar -xzf {} -C {}", archive.display(), dir.display());
    let status = Command::new("tar")
        .arg("-xzf")
        .arg(archive)
        .arg("-C")
        .arg(dir)
        .status()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "unpacking {} failed",
            archive.display()
        )));
    }
    Ok(())
}
