/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Model cache scanner.
//!
//! Discovers GGUF models the user *already has* on disk — the no-lock-in
//! adoption hook. Scans known cache locations (Hugging Face hub, LM Studio) plus
//! any user-added folders, reads each file's GGUF metadata, and flags continuation
//! shards of split models so the library shows one entry per model.

use std::path::{Path, PathBuf};

use serde::Serialize;
use walkdir::WalkDir;

use crate::gguf::{load_blocker, read_gguf_metadata, GgufMetadata};

/// A directory the scanner looks in, with a human label for its origin.
#[derive(Debug, Clone, Serialize)]
pub struct ScanRoot {
    pub path: String,
    pub source: String,
    pub exists: bool,
}

/// One discovered model file.
#[derive(Debug, Clone, Serialize)]
pub struct ModelEntry {
    pub path: String,
    pub file_name: String,
    pub size_bytes: u64,
    /// Origin: "huggingface" | "lm-studio" | "ollama" | "folder".
    pub source: String,
    /// Ollama stores weights as extensionless content-addressed blobs, so the
    /// file name is a hash. This carries the `name:tag` to show instead.
    pub display_name: Option<String>,
    /// True for shard 2..N of a split model — hidden from the primary list.
    pub is_shard_continuation: bool,
    pub shard_total: Option<u32>,
    /// True for multimodal projector companion files (mmproj-*, arch "clip") —
    /// grouped under their parent model rather than listed standalone.
    pub is_mmproj: bool,
    pub metadata: Option<GgufMetadata>,
    pub parse_error: Option<String>,
    /// Set when the header parsed fine but stock llama.cpp will refuse to load
    /// the file anyway. Distinct from `parse_error`: this model is readable,
    /// just not runnable. See `gguf::load_blocker`.
    pub load_blocker: Option<String>,
}

/// Build the default set of roots to scan, honoring `HF_HOME` if set.
pub fn default_roots() -> Vec<(PathBuf, String)> {
    let mut roots: Vec<(PathBuf, String)> = Vec::new();

    // Hugging Face hub cache. Default is ~/.cache/huggingface/hub unless HF_HOME
    // (or the legacy HUGGINGFACE_HUB_CACHE) overrides it.
    if let Some(hf_home) = std::env::var_os("HF_HOME") {
        roots.push((PathBuf::from(hf_home).join("hub"), "huggingface".into()));
    }
    if let Some(cache) = std::env::var_os("HUGGINGFACE_HUB_CACHE") {
        roots.push((PathBuf::from(cache), "huggingface".into()));
    }
    if let Some(home) = dirs::home_dir() {
        roots.push((
            home.join(".cache").join("huggingface").join("hub"),
            "huggingface".into(),
        ));
        // LM Studio: current (~/.lmstudio/models) and legacy (~/.cache/lm-studio/models).
        roots.push((home.join(".lmstudio").join("models"), "lm-studio".into()));
        roots.push((
            home.join(".cache").join("lm-studio").join("models"),
            "lm-studio".into(),
        ));
    }

    // Tokamak's own download folder. Registered here rather than as a user
    // folder so in-app downloads show up in the library with no extra step.
    if let Some(config) = dirs::config_dir() {
        roots.push((
            config.join("tokamak").join("models"),
            "downloaded".into(),
        ));
    }

    dedupe_roots(roots)
}

/// Canonicalize + de-duplicate root paths, keeping the first source label seen.
fn dedupe_roots(roots: Vec<(PathBuf, String)>) -> Vec<(PathBuf, String)> {
    let mut seen: Vec<PathBuf> = Vec::new();
    let mut out: Vec<(PathBuf, String)> = Vec::new();
    for (path, source) in roots {
        let key = path.canonicalize().unwrap_or_else(|_| path.clone());
        if seen.iter().any(|p| p == &key) {
            continue;
        }
        seen.push(key);
        out.push((path, source));
    }
    out
}

/// Report the default roots and whether each currently exists (for the UI).
/// The Ollama store is listed here even though it is not walked like the
/// others — it is read via manifests in `scan_models`.
pub fn default_roots_info() -> Vec<ScanRoot> {
    let mut roots: Vec<ScanRoot> = default_roots()
        .into_iter()
        .map(|(path, source)| ScanRoot {
            exists: path.is_dir(),
            path: path.to_string_lossy().into_owned(),
            source,
        })
        .collect();
    if let Some(o) = crate::ollama::models_root() {
        roots.push(ScanRoot {
            exists: o.is_dir(),
            path: o.to_string_lossy().into_owned(),
            source: "ollama".into(),
        });
    }
    roots
}

/// What the scanner is doing right now, so the UI can say so.
#[derive(Debug, Clone, Serialize)]
pub struct ScanProgress {
    /// The root being walked, as a path, or "Ollama" for the manifest pass.
    pub root: String,
    /// Which root this is, 1-based, out of `total_roots`.
    pub root_index: usize,
    pub total_roots: usize,
    /// Models found so far, across every root.
    pub found: usize,
}

/// Scan the default roots plus any `extra_dirs` (labeled "folder") for GGUF models.
pub fn scan_models(extra_dirs: &[String]) -> Vec<ModelEntry> {
    scan_models_reporting(extra_dirs, &mut |_| {})
}

/// `scan_models`, reporting what it is doing as it goes.
///
/// `report` fires when a root is opened and again on each model found, so the
/// window can name the folder being walked rather than only claim that
/// something is happening. On a warm local disk this flashes past; it earns its
/// keep when a root is a network mount or a cold drive, which is precisely when
/// the user most needs telling that the app has not hung.
pub fn scan_models_reporting(
    extra_dirs: &[String],
    report: &mut dyn FnMut(ScanProgress),
) -> Vec<ModelEntry> {
    let mut roots = default_roots();
    for d in extra_dirs {
        roots.push((PathBuf::from(d), "folder".into()));
    }
    walk_roots(dedupe_roots(roots), true, report)
}

/// The scan proper, over an explicit root list.
///
/// Taking the roots as an argument rather than building them is what lets the
/// progress test scan one directory it planted instead of the whole machine's
/// caches: a test that walks the user's real library is both slow and unable to
/// assert an exact count, since the answer depends on whose disk it runs on.
fn walk_roots(
    roots: Vec<(PathBuf, String)>,
    include_ollama: bool,
    report: &mut dyn FnMut(ScanProgress),
) -> Vec<ModelEntry> {
    let mut entries: Vec<ModelEntry> = Vec::new();
    let mut seen_paths: Vec<PathBuf> = Vec::new();
    // The Ollama pass is one more step than there are directories, when it runs
    // at all — counting it keeps the progress fraction honest.
    let total_roots = roots.len() + usize::from(include_ollama);

    for (i, (root, source)) in roots.iter().enumerate() {
        report(ScanProgress {
            root: root.to_string_lossy().into_owned(),
            root_index: i + 1,
            total_roots,
            found: entries.len(),
        });
        if !root.is_dir() {
            continue;
        }
        for entry in WalkDir::new(root).follow_links(false).into_iter().flatten() {
            let path = entry.path();
            if !is_gguf(path) {
                continue;
            }
            // A model may live under overlapping roots; de-dup by canonical path.
            let key = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
            if seen_paths.iter().any(|p| p == &key) {
                continue;
            }
            seen_paths.push(key);

            entries.push(build_entry(path, source));
            report(ScanProgress {
                root: root.to_string_lossy().into_owned(),
                root_index: i + 1,
                total_roots,
                found: entries.len(),
            });
        }
    }

    if !include_ollama {
        entries.sort_by_cached_key(|e| e.file_name.to_lowercase());
        return entries;
    }

    report(ScanProgress {
        root: "Ollama".into(),
        root_index: total_roots,
        total_roots,
        found: entries.len(),
    });
    // Ollama's store is content-addressed, so it needs manifest lookups rather
    // than a file walk. Skip any blob already reached through another root.
    for m in crate::ollama::discover() {
        let key = m
            .blob_path
            .canonicalize()
            .unwrap_or_else(|_| m.blob_path.clone());
        if seen_paths.iter().any(|p| p == &key) {
            continue;
        }
        seen_paths.push(key);
        let mut entry = build_entry(&m.blob_path, "ollama");
        entry.file_name = m.name.clone();
        entry.display_name = Some(m.name);
        entry.size_bytes = m.size_bytes;
        entries.push(entry);
        report(ScanProgress {
            root: "Ollama".into(),
            root_index: total_roots,
            total_roots,
            found: entries.len(),
        });
    }

    entries.sort_by_cached_key(|e| e.file_name.to_lowercase());
    entries
}

fn build_entry(path: &Path, source: &str) -> ModelEntry {
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let size_bytes = path.metadata().map(|m| m.len()).unwrap_or(0);
    let shard = shard_index(&file_name);

    let (metadata, parse_error) = match read_gguf_metadata(path) {
        Ok(m) => (Some(m), None),
        Err(e) => (None, Some(e.to_string())),
    };

    let is_mmproj = is_mmproj_name(&file_name)
        || metadata
            .as_ref()
            .map(|m| m.architecture.as_deref() == Some("clip"))
            .unwrap_or(false);

    ModelEntry {
        path: path.to_string_lossy().into_owned(),
        file_name,
        size_bytes,
        source: source.to_string(),
        display_name: None,
        is_shard_continuation: shard.map(|(i, _)| i > 1).unwrap_or(false),
        shard_total: shard.map(|(_, t)| t),
        // A projector is never launched on its own, so llama.cpp's model
        // requirements do not apply to it — flagging one would be noise.
        load_blocker: if is_mmproj {
            None
        } else {
            metadata.as_ref().and_then(load_blocker)
        },
        is_mmproj,
        metadata,
        parse_error,
    }
}

/// llama.cpp convention: multimodal projector files are named `mmproj-…`.
fn is_mmproj_name(file_name: &str) -> bool {
    file_name.to_ascii_lowercase().starts_with("mmproj")
}

fn is_gguf(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
}

/// Parse a `...-00001-of-00003.gguf` shard suffix into (index, total), 1-based.
fn shard_index(file_name: &str) -> Option<(u32, u32)> {
    let stem = file_name.strip_suffix(".gguf")?;
    let (left, total) = stem.rsplit_once("-of-")?;
    let total: u32 = total.parse().ok()?;
    let idx: u32 = left.rsplit('-').next()?.parse().ok()?;
    Some((idx, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_shard_suffix() {
        assert_eq!(shard_index("model-00001-of-00003.gguf"), Some((1, 3)));
        assert_eq!(shard_index("model-00002-of-00003.gguf"), Some((2, 3)));
        assert_eq!(shard_index("plain-model.gguf"), None);
        assert_eq!(shard_index("not-a-model.txt"), None);
    }

    #[test]
    fn detects_mmproj_by_name() {
        assert!(is_mmproj_name("mmproj-Qwen3.6-27B-BF16.gguf"));
        assert!(is_mmproj_name("MMPROJ-model.gguf"));
        assert!(!is_mmproj_name("Qwen3.6-27B-Q4_K_M.gguf"));
    }

    /// Progress has to be usable as a fraction: it may never count backwards,
    /// and it must finish on the number actually returned. The UI draws a bar
    /// from these, and a bar that retreats or stops short reads as a bug in the
    /// scan rather than in the reporting.
    #[test]
    fn progress_counts_up_to_exactly_what_was_found() {
        let dir = std::env::temp_dir().join(format!("tokamak-scan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("nested")).unwrap();
        // Empty files: `build_entry` records a parse error and lists them
        // anyway, which is what we want. This is a test about counting.
        std::fs::write(dir.join("alpha.gguf"), b"").unwrap();
        std::fs::write(dir.join("nested").join("beta.gguf"), b"").unwrap();
        std::fs::write(dir.join("notes.txt"), b"not a model").unwrap();

        let mut seen: Vec<ScanProgress> = Vec::new();
        // One root, no Ollama pass: the machine's own caches are irrelevant
        // here, and including them would make the count depend on whose disk
        // the suite runs on.
        let entries = walk_roots(vec![(dir.clone(), "folder".into())], false, &mut |p| {
            seen.push(p)
        });

        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(entries.len(), 2, "the .txt must not be counted as a model");
        // Opening the root, then one report per model found.
        assert_eq!(
            seen.iter().map(|p| p.found).collect::<Vec<_>>(),
            vec![0, 1, 2],
            "every found model should move the count by exactly one"
        );
        for p in &seen {
            assert_eq!((p.root_index, p.total_roots), (1, 1));
        }
        assert_eq!(
            seen.last().unwrap().found,
            entries.len(),
            "the last report must agree with what was returned"
        );
    }

    /// Real end-to-end scan of whatever models are on this machine's caches.
    /// Ignored by default (machine-dependent); run with:
    ///   cargo test -- --ignored --nocapture scan_real_caches
    #[test]
    #[ignore]
    fn scan_real_caches() {
        let entries = scan_models(&[]);
        println!("\n--- scanned {} GGUF file(s) ---", entries.len());
        for e in &entries {
            let md = e.metadata.as_ref();
            println!(
                "{:<55} arch={:<10} quant={:<8} ctx={:<8} {} [{}]{}{}",
                e.file_name,
                md.and_then(|m| m.architecture.clone()).unwrap_or("?".into()),
                md.and_then(|m| m.quant_label.clone()).unwrap_or("?".into()),
                md.and_then(|m| m.context_length).unwrap_or(0),
                human_size(e.size_bytes),
                e.source,
                if e.is_shard_continuation { " (shard)" } else { "" },
                e.parse_error
                    .as_ref()
                    .map(|s| format!(" ERR: {s}"))
                    .or_else(|| e.load_blocker.as_ref().map(|_| " *** WILL NOT LOAD".into()))
                    .unwrap_or_default(),
            );
        }
    }

    fn human_size(n: u64) -> String {
        let gb = n as f64 / 1024.0 / 1024.0 / 1024.0;
        format!("{gb:.2}GB")
    }
}

#[cfg(test)]
mod perf {
    /// Times a real scan against this machine's own model roots.
    ///
    /// Kept because this was once ten seconds, every one of them spent on the
    /// UI thread. The cause was `gguf::skip` seeking past each string in a
    /// tokenizer vocab, which threw away the `BufReader` buffer ~150k times
    /// per array. If this creeps back into seconds, that is the first place to
    /// look. The bound is deliberately loose: it is here to catch a return to
    /// the old order of magnitude, not to police milliseconds.
    #[test]
    #[ignore] // walks the machine's real model dirs; run with --ignored
    fn scan_is_not_pathological() {
        let t = std::time::Instant::now();
        let entries = super::scan_models(&[]);
        let elapsed = t.elapsed();
        let parsed = entries.iter().filter(|e| e.metadata.is_some()).count();
        eprintln!(
            "scan_models: {elapsed:?} for {} entries ({parsed} headers parsed)",
            entries.len()
        );
        assert!(
            elapsed.as_secs() < 3,
            "scan took {elapsed:?}; it should be well under a second"
        );
    }
}
