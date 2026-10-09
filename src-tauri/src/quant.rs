/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Measuring what a model would weigh at another quant.
//!
//! llama.cpp ships its whole toolset in the archive Tokamak already downloads
//! for `llama-server`, so `llama-quantize` is on disk the moment a runtime is
//! installed: no second download, and no Python, which is what a
//! safetensors-to-GGUF conversion would have cost.
//!
//! Its `--dry-run` reads a model's tensor list and reports exactly how large
//! each target quant would come out, without doing the work. That is under
//! half a second even on a 15GB file, because it never reads tensor data — so
//! measuring a whole ladder is cheap enough to do on request.
//!
//! Why ask at all, when `estimator::quant_advice` already answers the same
//! question: the estimator works from a table of average bits-per-weight and a
//! flat 5% allowance. That is close for the common K-quants and drifts
//! elsewhere. Measured against a 27B model, Q2_K came out 14% larger than the
//! table predicts and IQ4_XS 6% denser per weight. Near the edge of a card's
//! budget, that difference is the difference between loading and not.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;

/// What `llama-quantize --dry-run` reports for one target type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DryRun {
    /// The source file as llama.cpp measures it, which is the sum of its
    /// tensors rather than the size on disk.
    pub source_bytes: u64,
    pub source_bpw: f32,
    pub target_bytes: u64,
    pub target_bpw: f32,
}

/// One rung of the ladder, measured rather than estimated.
#[derive(Debug, Clone, Serialize)]
pub struct MeasuredRung {
    pub label: String,
    pub weights_bytes: u64,
    pub bpw: f32,
    pub fits: bool,
    pub headroom_bytes: u64,
    pub is_current: bool,
    /// True when some higher-quality rung produces a file this size or smaller,
    /// making this one a strictly worse choice: less quality for no less space.
    ///
    /// Only measurement finds these. On a 30B MoE, Q3_K_M measured 18.48 GiB
    /// against IQ4_XS at 16.95 GiB — nominally a step down, actually a step up
    /// in size. The bits-per-weight table cannot see it, because the answer
    /// depends on which tensors this particular model keeps at high precision.
    pub dominated: bool,
}

/// The answer to "what could this model become, and would it fit".
#[derive(Debug, Clone, Serialize)]
pub struct QuantMeasurement {
    /// Absent when no `llama-quantize` could be found, which is not an error:
    /// the estimated ladder is still shown, just not refined.
    pub tool: Option<String>,
    pub source_bytes: u64,
    pub source_bpw: f32,
    pub source_label: Option<String>,
    /// True when the source is itself quantized, so producing any of these
    /// locally would be a requantize. llama.cpp refuses that by default, and
    /// for good reason: quality drops further than the target quant implies.
    pub requantize: bool,
    pub rungs: Vec<MeasuredRung>,
}

const EXE: &str = if cfg!(windows) {
    "llama-quantize.exe"
} else {
    "llama-quantize"
};

/// Find `llama-quantize`.
///
/// Looked for beside a discovered `llama-server` first. That one lookup covers
/// the managed runtime, an LM Studio install and anything on PATH, because
/// llama.cpp distributes the tools together and `resolve_binaries` already
/// knows every place a server turns up.
pub fn tool_path() -> Option<PathBuf> {
    for b in crate::llama::resolve_binaries() {
        if let Some(dir) = Path::new(&b.path).parent() {
            let candidate = dir.join(EXE);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    // A runtime whose server is missing or unresolvable may still have the
    // rest of the toolset unpacked.
    if let Ok(dir) = crate::runtime::runtime_dir() {
        let direct = dir.join(EXE);
        if direct.is_file() {
            return Some(direct);
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path().join(EXE);
                if p.is_file() {
                    return Some(p);
                }
            }
        }
    }
    None
}

/// Pull the byte count and bits-per-weight out of one `… = 1234.56 MiB (7.89 BPW)`.
///
/// Hand-parsed because this is two lines of a known format and the crate does
/// not otherwise need a regex engine.
fn size_and_bpw(line: &str) -> Option<(u64, f32)> {
    let mib: f64 = line.split('=').nth(1)?.split_whitespace().next()?.parse().ok()?;
    let bpw: f32 = line
        .rsplit_once('(')?
        .1
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    Some(((mib * 1024.0 * 1024.0) as u64, bpw))
}

/// Read the two summary lines out of a dry run's output.
///
/// Everything before them is a per-tensor log some hundreds of lines long,
/// which is why this looks for the summary rather than parsing positionally.
pub fn parse_dry_run(out: &str) -> Option<DryRun> {
    let mut source = None;
    let mut target = None;
    for line in out.lines() {
        // Both lines share a prefix and differ only in the words before `=`,
        // so match on those rather than on the prefix.
        if source.is_none() && line.contains("model size") {
            source = size_and_bpw(line);
        } else if target.is_none() && line.contains("quant size") {
            target = size_and_bpw(line);
        }
    }
    let (source_bytes, source_bpw) = source?;
    let (target_bytes, target_bpw) = target?;
    Some(DryRun {
        source_bytes,
        source_bpw,
        target_bytes,
        target_bpw,
    })
}

/// Ask llama.cpp how big `model` would be at `quant`.
///
/// `--dry-run` writes no file, so this is safe to run over a whole ladder. It
/// also does not require `--allow-requantize`: it will happily plan a
/// requantize that a real run would refuse, which is exactly what we want here
/// — the numbers are worth showing even when the conversion would need an
/// explicit opt-in.
pub fn dry_run(tool: &Path, model: &Path, quant: &str) -> Result<DryRun, String> {
    let out = Command::new(tool)
        .arg("--dry-run")
        .arg(model)
        .arg(quant)
        .output()
        .map_err(|e| format!("could not run {}: {e}", tool.display()))?;

    // llama.cpp logs to stderr, but which stream carries the summary has moved
    // between releases, so both are searched.
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push('\n');
    text.push_str(&String::from_utf8_lossy(&out.stderr));

    parse_dry_run(&text).ok_or_else(|| {
        // A refusal is reported in prose, so the tail is more use than a code.
        let tail: Vec<&str> = text.lines().rev().take(3).collect();
        let tail = tail.into_iter().rev().collect::<Vec<_>>().join(" / ");
        format!("{quant}: llama-quantize reported no size ({tail})")
    })
}

/// Is this label a full-precision source, i.e. not already quantized?
fn is_full_precision(label: &str) -> bool {
    let up = label.to_ascii_uppercase();
    up.starts_with("F32") || up.starts_with("F16") || up.starts_with("BF16")
}

/// Measure every ladder rung worth producing from this model.
///
/// Rungs at or above the source's own density are skipped: quantizing upward
/// cannot recover detail the source already threw away, so offering it would
/// invite someone to spend twenty minutes making a bigger file that is no
/// better. `report` fires per rung, since each is a process spawn.
pub fn measure_ladder(
    tool: &Path,
    model: &Path,
    shape: &crate::estimator::ModelShape,
    source_label: Option<&str>,
    gpu_total: u64,
    kv: crate::estimator::KvType,
    report: &mut dyn FnMut(usize, usize, &str),
) -> Result<QuantMeasurement, String> {
    let labels: Vec<&str> = crate::estimator::QUANT_LADDER
        .iter()
        .map(|(l, _)| *l)
        .collect();

    // One run establishes the source's own measured size and density, which is
    // what decides the rest of the ladder. The first rung is measured anyway,
    // so nothing is wasted.
    let first = dry_run(tool, model, labels[0])?;
    let source_up = source_label.map(|s| s.to_ascii_uppercase());

    let mut rungs: Vec<MeasuredRung> = Vec::new();
    for (i, label) in labels.iter().enumerate() {
        report(i + 1, labels.len(), label);
        let d = if i == 0 {
            first
        } else {
            match dry_run(tool, model, label) {
                Ok(d) => d,
                // One unsupported type must not cost the other rungs their
                // numbers; the ladder is still useful with a gap in it.
                Err(_) => continue,
            }
        };
        // Including the source's own density, which is a no-op conversion, and
        // anything denser, which cannot recover what the source already threw
        // away. Nominal names are no guide here: on an MoE source, "upgrading"
        // Q4_K_M to Q5_K_M measured *larger* than the file it came from.
        if d.target_bpw >= d.source_bpw {
            continue;
        }
        let (fits, headroom) = crate::estimator::judge_weights(shape, d.target_bytes, gpu_total, kv);
        // The ladder is walked best-quality-first, so anything already pushed
        // is higher quality than this rung.
        let dominated = rungs
            .iter()
            .any(|r: &MeasuredRung| r.weights_bytes <= d.target_bytes);
        rungs.push(MeasuredRung {
            label: (*label).to_string(),
            weights_bytes: d.target_bytes,
            bpw: d.target_bpw,
            fits,
            headroom_bytes: headroom,
            is_current: source_up
                .as_deref()
                .map(|c| c.starts_with(label))
                .unwrap_or(false),
            dominated,
        });
    }

    Ok(QuantMeasurement {
        tool: Some(tool.display().to_string()),
        source_bytes: first.source_bytes,
        source_bpw: first.source_bpw,
        source_label: source_label.map(str::to_string),
        requantize: !source_label.map(is_full_precision).unwrap_or(false),
        rungs,
    })
}

// ---- conversion ----

/// How far a conversion has got, and how it ended.
#[derive(Debug, Clone, Serialize)]
pub struct ConvertProgress {
    /// Tensors written, out of the model's total.
    pub done: u32,
    pub total: u32,
    /// The finished file, set only once it is in place under its real name.
    pub output: Option<String>,
    pub finished: bool,
    pub cancelled: bool,
    pub error: Option<String>,
}

impl ConvertProgress {
    fn at(done: u32, total: u32) -> Self {
        ConvertProgress {
            done,
            total,
            output: None,
            finished: false,
            cancelled: false,
            error: None,
        }
    }
}

/// One conversion at a time.
///
/// Quantizing saturates disk and CPU, so a second run would make both slower
/// and could fill the disk twice over while doing it. A single slot also means
/// the UI can report progress honestly instead of summing two unrelated runs.
/// Holds an `Arc` rather than the mutex directly so the worker thread can own a
/// handle and clear the slot itself when it finishes, without reaching back into
/// Tauri's managed state.
#[derive(Default, Clone)]
pub struct QuantState {
    slot: Arc<Mutex<Option<Arc<AtomicBool>>>>,
}

impl QuantState {
    /// Claim the slot, or refuse because it is taken.
    fn arm(&self) -> Result<Arc<AtomicBool>, String> {
        let mut slot = self.slot.lock().unwrap();
        if slot.is_some() {
            return Err("a conversion is already running".into());
        }
        let flag = Arc::new(AtomicBool::new(false));
        *slot = Some(flag.clone());
        Ok(flag)
    }

    fn release(&self) {
        *self.slot.lock().unwrap() = None;
    }

    pub fn cancel(&self) {
        if let Some(f) = self.slot.lock().unwrap().as_ref() {
            f.store(true, Ordering::Relaxed);
        }
    }
}

/// Parse `[ 304/ 310]` off the front of a per-tensor line.
///
/// This is the only progress llama-quantize offers, and it is worth having: a
/// real conversion runs for minutes, and a bar that moves is the difference
/// between waiting and force-quitting.
pub fn parse_tensor_progress(line: &str) -> Option<(u32, u32)> {
    let inner = line.trim_start().strip_prefix('[')?;
    let (inner, _) = inner.split_once(']')?;
    let (done, total) = inner.split_once('/')?;
    Some((done.trim().parse().ok()?, total.trim().parse().ok()?))
}

/// Where a conversion's result goes: beside the source, with the quant named.
///
/// Beside the source because that is the least surprising answer to "convert
/// this model", and because every directory the scanner walks is one the library
/// already shows, so the result appears without another step. When the source's
/// own quant is in its file name it is replaced rather than appended, so a
/// Q4_K_M converted to Q3_K_M does not come out as `…-Q4_K_M-Q3_K_M`.
pub fn output_path(model: &Path, source_label: Option<&str>, quant: &str) -> PathBuf {
    let stem = model
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let base = match source_label {
        // The label is ASCII, so trimming it off a matched suffix cannot land
        // inside a multi-byte character.
        Some(l)
            if !l.is_empty()
                && stem
                    .to_ascii_uppercase()
                    .ends_with(&l.to_ascii_uppercase()) =>
        {
            stem[..stem.len() - l.len()]
                .trim_end_matches(['-', '_', '.'])
                .to_string()
        }
        _ => stem,
    };
    model.with_file_name(format!("{base}-{quant}.gguf"))
}

/// Convert `model` to `quant`, reporting progress and leaving nothing behind.
///
/// Writes to `<output>.part` and renames only on success. This is not caution
/// for its own sake: a refused or interrupted run **does** leave a truncated
/// GGUF where its output was pointed. One was observed at 5.9 MB, with eight
/// zero bytes where the magic belongs — so it would scan as a parse error
/// rather than masquerade as a model, but it would still sit in the library
/// until someone deleted it by hand.
#[allow(clippy::too_many_arguments)]
pub fn convert(
    tool: &Path,
    model: &Path,
    quant: &str,
    output: &Path,
    allow_requantize: bool,
    cancel: &AtomicBool,
    report: &mut dyn FnMut(ConvertProgress),
) -> Result<PathBuf, String> {
    if output.exists() {
        return Err(format!("{} already exists", output.display()));
    }
    let part = output.with_extension("gguf.part");
    let _ = std::fs::remove_file(&part);

    let mut cmd = Command::new(tool);
    if allow_requantize {
        cmd.arg("--allow-requantize");
    }
    cmd.arg(model).arg(&part).arg(quant);
    // Progress arrives on stderr; stdout is captured too so a build that moves
    // it does not silently lose the bar.
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("could not run {}: {e}", tool.display()))?;

    let stderr = child.stderr.take().ok_or("no stderr from llama-quantize")?;
    let mut tail: Vec<String> = Vec::new();
    let mut killed = false;

    for line in BufReader::new(stderr).lines().map_while(Result::ok) {
        // Checked between lines rather than on a timer. llama-quantize emits one
        // per tensor, so the gap is short in practice, and this needs no second
        // thread to own the child handle.
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            killed = true;
            break;
        }
        if let Some((done, total)) = parse_tensor_progress(&line) {
            report(ConvertProgress::at(done, total));
        }
        // Kept for the error message: llama.cpp explains a refusal in prose,
        // and the exit code alone cannot say "requantizing is disabled".
        tail.push(line);
        if tail.len() > 40 {
            tail.remove(0);
        }
    }

    let status = child.wait().map_err(|e| e.to_string())?;

    if killed || cancel.load(Ordering::Relaxed) {
        let _ = std::fs::remove_file(&part);
        return Err("cancelled".into());
    }
    if !status.success() {
        let _ = std::fs::remove_file(&part);
        let why = tail
            .iter()
            .rev()
            .find(|l| l.contains("failed") || l.contains("error"))
            .cloned()
            .unwrap_or_else(|| format!("llama-quantize exited with {status}"));
        return Err(why);
    }

    std::fs::rename(&part, output).map_err(|e| {
        let _ = std::fs::remove_file(&part);
        format!("converted, but could not move into place: {e}")
    })?;
    Ok(output.to_path_buf())
}

/// Run a conversion on a worker thread, emitting `quant-convert`.
pub fn start(
    window: tauri::Window,
    state: &QuantState,
    model: PathBuf,
    quant: String,
    source_label: Option<String>,
    allow_requantize: bool,
) -> Result<String, String> {
    use tauri::Emitter;
    let tool = tool_path().ok_or("no llama-quantize found; install a runtime first")?;
    let output = output_path(&model, source_label.as_deref(), &quant);
    let cancel = state.arm()?;

    let shown = output.display().to_string();
    std::thread::spawn({
        let output = output.clone();
        let slot = state.clone();
        move || {
            let emit = |p: ConvertProgress| {
                let _ = window.emit("quant-convert", &p);
            };
            let result = convert(
                &tool,
                &model,
                &quant,
                &output,
                allow_requantize,
                &cancel,
                &mut |p| emit(p),
            );
            // Released before the terminal event, so a UI that immediately
            // starts another conversion on "finished" is not refused.
            slot.release();
            match result {
                Ok(path) => emit(ConvertProgress {
                    done: 0,
                    total: 0,
                    output: Some(path.display().to_string()),
                    finished: true,
                    cancelled: false,
                    error: None,
                }),
                Err(e) if e == "cancelled" => emit(ConvertProgress {
                    done: 0,
                    total: 0,
                    output: None,
                    finished: true,
                    cancelled: true,
                    error: None,
                }),
                Err(e) => emit(ConvertProgress {
                    done: 0,
                    total: 0,
                    output: None,
                    finished: true,
                    cancelled: false,
                    error: Some(e),
                }),
            }
        }
    });
    Ok(shown)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Captured verbatim from `llama-quantize --dry-run` on a 15.41GB
    /// Qwen3.6-27B-Q4_K_M, trimmed to the shape the parser has to cope with:
    /// hundreds of per-tensor lines, then the two that matter.
    const REAL_OUTPUT: &str = "\
llama_model_loader: - type  f32:  121 tensors
llama_model_loader: - type q4_K:  434 tensors
[   1/ 579] output_norm.weight                   - [  5376,     1,     1,     1], type =    f32, size =    0.021 MiB
[   2/ 579] token_embd.weight                    - [  5376, 151936,     1,     1], type =   q4_K, size =   438.75 MiB ->   329.06 MiB (q3_K)
llama_model_quantize_impl: model size  = 15770.35 MiB (4.92 BPW)
llama_model_quantize_impl: quant size  = 12674.76 MiB (3.95 BPW)

llama_quantize: quantize time =   174.74 ms
llama_quantize:    total time =   174.74 ms
";

    #[test]
    fn reads_both_sizes_out_of_a_real_dry_run() {
        let d = parse_dry_run(REAL_OUTPUT).expect("should parse");
        assert_eq!(d.source_bpw, 4.92);
        assert_eq!(d.target_bpw, 3.95);
        // 15770.35 MiB, to the byte.
        assert_eq!(d.source_bytes, (15770.35 * 1024.0 * 1024.0) as u64);
        assert_eq!(d.target_bytes, (12674.76 * 1024.0 * 1024.0) as u64);
        // Sanity: the target must be the smaller of the two, or the ladder
        // filter built on this would keep the wrong rungs.
        assert!(d.target_bytes < d.source_bytes);
    }

    /// The per-tensor lines also contain `size =` and `MiB`, and one of them
    /// even contains a parenthesised type. Picking the first match of a looser
    /// pattern would read a single tensor's size as the model's.
    #[test]
    fn is_not_fooled_by_the_per_tensor_lines() {
        let d = parse_dry_run(REAL_OUTPUT).unwrap();
        let tensor_mib = (438.75 * 1024.0 * 1024.0) as u64;
        assert_ne!(d.source_bytes, tensor_mib);
        assert!(d.source_bytes > tensor_mib);
    }

    #[test]
    fn a_refusal_carries_no_sizes() {
        let refused = "\
[   2/ 310] token_embd.weight - [ 1024, 151936, 1, 1], type =   q8_0, \
llama_model_quantize: failed to quantize: requantizing from type q8_0 is disabled
llama_quantize: failed to quantize model from 'C:/models/x.gguf'
";
        assert!(parse_dry_run(refused).is_none());
    }

    #[test]
    fn half_an_answer_is_no_answer() {
        // A truncated run that reported the source but died before the target.
        let partial = "llama_model_quantize_impl: model size  = 604.15 MiB (8.50 BPW)\n";
        assert!(parse_dry_run(partial).is_none());
    }

    #[test]
    fn nothing_useful_in_nothing() {
        assert!(parse_dry_run("").is_none());
        assert!(parse_dry_run("some unrelated chatter\n").is_none());
    }

    #[test]
    fn full_precision_sources_are_the_only_non_requantize() {
        assert!(is_full_precision("F16"));
        assert!(is_full_precision("f32"));
        assert!(is_full_precision("BF16"));
        // The labels that actually appear on quantized files.
        assert!(!is_full_precision("Q4_K_M"));
        assert!(!is_full_precision("IQ4_XS"));
        assert!(!is_full_precision("Q8_0"));
    }

    #[test]
    fn reads_the_tensor_counter() {
        // The real spacing, which pads the numbers to align the column.
        assert_eq!(parse_tensor_progress("[   1/ 310] output_norm.weight"), Some((1, 310)));
        assert_eq!(parse_tensor_progress("[ 304/ 310] blk.27.attn_q.weight"), Some((304, 310)));
        assert_eq!(parse_tensor_progress("[1/2] x"), Some((1, 2)));
    }

    #[test]
    fn ignores_lines_that_are_not_the_counter() {
        for line in [
            "llama_model_quantize_impl: model size  = 604.15 MiB (8.50 BPW)",
            "",
            "[not/numbers] x",
            // A bracket with no slash, and a slash with no bracket.
            "[310] done",
            "1/310 tensors",
        ] {
            assert_eq!(parse_tensor_progress(line), None, "should ignore {line:?}");
        }
    }

    #[test]
    fn output_replaces_the_source_quant_in_the_name() {
        let p = Path::new("/m/Qwen3.6-27B-Q4_K_M.gguf");
        assert_eq!(
            output_path(p, Some("Q4_K_M"), "Q3_K_M").file_name().unwrap(),
            "Qwen3.6-27B-Q3_K_M.gguf"
        );
    }

    /// Backslash-separated paths, which only `Path` on Windows splits.
    ///
    /// Gated because a `C:\…` literal is not a path on Linux at all — it is one
    /// long file name with no separators, so `file_stem` returns the whole
    /// thing. An earlier version of the test above used one and passed on
    /// Windows while failing in CI, which is the job that check exists to do.
    #[test]
    #[cfg(windows)]
    fn output_handles_windows_separators() {
        let p = Path::new(r"C:\models\org\Qwen3.6-27B-Q4_K_M.gguf");
        let out = output_path(p, Some("Q4_K_M"), "Q3_K_M");
        assert_eq!(out.file_name().unwrap(), "Qwen3.6-27B-Q3_K_M.gguf");
        assert_eq!(out.parent(), p.parent());
    }

    #[test]
    fn output_appends_when_the_name_does_not_carry_a_quant() {
        let p = Path::new("/m/mystery-model.gguf");
        assert_eq!(
            output_path(p, Some("Q4_K_M"), "Q3_K_M").file_name().unwrap(),
            "mystery-model-Q3_K_M.gguf"
        );
        // And when the source quant is unknown entirely.
        assert_eq!(
            output_path(p, None, "Q3_K_M").file_name().unwrap(),
            "mystery-model-Q3_K_M.gguf"
        );
    }

    /// The label on disk is not always the case the metadata reports, and a
    /// mismatch would leave both quants in the name.
    #[test]
    fn output_matches_the_source_quant_case_insensitively() {
        let p = Path::new("/m/qwen2.5-coder-32b-instruct-q5_k_m.gguf");
        assert_eq!(
            output_path(p, Some("Q5_K_M"), "Q3_K_M").file_name().unwrap(),
            "qwen2.5-coder-32b-instruct-Q3_K_M.gguf"
        );
    }

    #[test]
    fn output_stays_beside_the_source() {
        let p = Path::new("/models/org/repo/m-Q8_0.gguf");
        let out = output_path(p, Some("Q8_0"), "Q4_K_M");
        assert_eq!(out.parent(), p.parent());
    }

    /// Two conversions at once would fight over the disk, so the second is
    /// refused rather than queued.
    #[test]
    fn only_one_conversion_holds_the_slot() {
        let state = QuantState::default();
        let first = state.arm().expect("slot should be free");
        assert!(state.arm().is_err(), "a second claim must be refused");

        // Cancelling signals the holder; the slot stays taken until the worker
        // releases it, or a cancel would let a second run start while the first
        // is still winding down.
        state.cancel();
        assert!(first.load(Ordering::Relaxed));
        assert!(state.arm().is_err(), "still held until the worker releases");

        state.release();
        assert!(state.arm().is_ok(), "the slot should be reusable");
    }

    /// Pick the smallest real model on this machine, for tests that convert.
    #[cfg(test)]
    fn smallest_model() -> Option<crate::scanner::ModelEntry> {
        crate::scanner::scan_models(&[])
            .into_iter()
            .filter(|m| !m.is_shard_continuation && !m.is_mmproj && m.metadata.is_some())
            .min_by_key(|m| m.size_bytes)
    }

    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("tokamak-q-{}-{name}", std::process::id()))
    }

    /// Converts a real model and checks the result is actually loadable.
    ///
    /// The whole point of this module is producing a file llama.cpp will run, so
    /// the assertion is that Tokamak's own GGUF parser can read the output and
    /// agrees about what it is — not merely that a file appeared.
    /// `cargo test -- --ignored --nocapture converts_a_real_model`
    #[test]
    #[ignore]
    fn converts_a_real_model() {
        let (Some(tool), Some(m)) = (tool_path(), smallest_model()) else {
            eprintln!("no tool or no models; skipping");
            return;
        };
        let src_label = m.metadata.as_ref().and_then(|d| d.quant_label.clone());
        let out = scratch("converted.gguf");
        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_file(out.with_extension("gguf.part"));

        eprintln!(
            "converting {} ({:.2} GiB, {:?}) -> Q4_K_M",
            m.file_name,
            m.size_bytes as f64 / 1024.0f64.powi(3),
            src_label
        );

        let cancel = AtomicBool::new(false);
        let mut seen: Vec<(u32, u32)> = Vec::new();
        let t = std::time::Instant::now();
        let result = convert(
            &tool,
            Path::new(&m.path),
            "Q4_K_M",
            &out,
            // The source is quantized, so without this llama.cpp refuses.
            true,
            &cancel,
            &mut |p| seen.push((p.done, p.total)),
        );
        let elapsed = t.elapsed();

        let path = match result {
            Ok(p) => p,
            Err(e) => {
                let _ = std::fs::remove_file(&out);
                panic!("conversion failed: {e}");
            }
        };
        eprintln!("done in {elapsed:?}, {} progress reports", seen.len());

        assert!(!seen.is_empty(), "a conversion should report progress");
        let (_, total) = seen[0];
        assert!(total > 0, "the counter needs a denominator");
        // Monotonic, and it reaches the end.
        for w in seen.windows(2) {
            assert!(w[1].0 >= w[0].0, "counter went {} -> {}", w[0].0, w[1].0);
        }
        assert_eq!(seen.last().unwrap().0, total, "should finish every tensor");

        // Nothing left over, and the result is under its real name.
        assert!(path.is_file(), "output should exist");
        assert!(
            !out.with_extension("gguf.part").exists(),
            "the .part must be gone"
        );

        // The real check: it parses, and it is the quant that was asked for.
        let md = crate::gguf::read_gguf_metadata(&path).expect("output should be a valid GGUF");
        eprintln!(
            "output {:.2} GiB, quant {:?}, arch {:?}",
            std::fs::metadata(&path).unwrap().len() as f64 / 1024.0f64.powi(3),
            md.quant_label,
            md.architecture
        );
        assert_eq!(
            md.architecture,
            m.metadata.as_ref().unwrap().architecture,
            "conversion must not change the architecture"
        );
        assert!(
            md.quant_label
                .as_deref()
                .map(|q| q.to_ascii_uppercase().contains("Q4"))
                .unwrap_or(false),
            "expected a Q4 label, got {:?}",
            md.quant_label
        );
        assert!(
            std::fs::metadata(&path).unwrap().len() < m.size_bytes,
            "Q4_K_M of a Q8_0 should be smaller"
        );

        let _ = std::fs::remove_file(&path);
    }

    /// Cancelling must leave the disk exactly as it was.
    ///
    /// This is the behaviour the module exists to guarantee: llama-quantize
    /// itself leaves a truncated GGUF behind when a run ends early, and that
    /// stub would sit in whichever model folder the output was pointed at.
    /// `cargo test -- --ignored --nocapture cancelling_leaves_nothing_behind`
    #[test]
    #[ignore]
    fn cancelling_leaves_nothing_behind() {
        let (Some(tool), Some(m)) = (tool_path(), smallest_model()) else {
            eprintln!("no tool or no models; skipping");
            return;
        };
        let out = scratch("cancelled.gguf");
        let part = out.with_extension("gguf.part");
        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_file(&part);

        // Cancelled from inside the progress callback rather than on a timer.
        // A sleep races the conversion, and on a small model the conversion
        // wins: an earlier version of this test cancelled after all 112 tensors
        // had already gone by, so it proved the cleanup but never exercised
        // killing a child mid-write.
        const STOP_AFTER: usize = 5;
        let cancel = Arc::new(AtomicBool::new(false));
        let trip = cancel.clone();
        let mut count = 0usize;
        let mut total = 0u32;

        let err = convert(
            &tool,
            Path::new(&m.path),
            "Q4_K_M",
            &out,
            true,
            &cancel,
            &mut |p| {
                count += 1;
                total = p.total;
                if count == STOP_AFTER {
                    trip.store(true, Ordering::Relaxed);
                }
            },
        )
        .expect_err("a cancelled conversion must not report success");

        eprintln!("cancelled after {count} of {total} tensors: {err}");
        assert_eq!(err, "cancelled");
        assert!(
            count < total as usize,
            "cancelled at {count} of {total}: the child was not killed mid-run, \
             so this did not test what it claims"
        );
        assert!(!part.exists(), "the .part must be cleaned up");
        assert!(!out.exists(), "no output should appear under the real name");
    }

    /// Refusing to overwrite is checked before anything is spawned, so a typo
    /// cannot cost someone a model they already have.
    #[test]
    fn will_not_overwrite_an_existing_file() {
        let out = scratch("existing.gguf");
        std::fs::write(&out, b"not really a model").unwrap();
        let err = convert(
            Path::new("llama-quantize-does-not-need-to-exist"),
            Path::new("whatever.gguf"),
            "Q4_K_M",
            &out,
            false,
            &AtomicBool::new(false),
            &mut |_| {},
        )
        .expect_err("should refuse");
        assert!(err.contains("already exists"), "got {err}");
        // And the file it refused to touch is untouched.
        assert_eq!(std::fs::read(&out).unwrap(), b"not really a model");
        let _ = std::fs::remove_file(&out);
    }

    /// Machine-dependent: proves the tool is where this module claims it is.
    /// Run with `cargo test -- --ignored --nocapture finds_the_real_tool`.
    #[test]
    #[ignore]
    fn finds_the_real_tool() {
        match tool_path() {
            Some(p) => eprintln!("llama-quantize: {}", p.display()),
            None => eprintln!("no llama-quantize found (no runtime installed?)"),
        }
    }

    /// Prints measured against estimated for the largest model on this machine.
    ///
    /// This is the claim the module is built on, kept runnable rather than
    /// written down: if the table in `estimator` were accurate for every quant,
    /// spawning a process per rung would be waste. Run with
    /// `cargo test -- --ignored --nocapture measured_against_estimated`.
    #[test]
    #[ignore]
    fn measured_against_estimated() {
        let Some(tool) = tool_path() else {
            eprintln!("no llama-quantize; skipping");
            return;
        };
        let models = crate::scanner::scan_models(&[]);
        let Some(m) = models
            .iter()
            .filter(|m| !m.is_shard_continuation && !m.is_mmproj && m.metadata.is_some())
            .max_by_key(|m| m.size_bytes)
        else {
            eprintln!("no usable models; skipping");
            return;
        };
        let md = m.metadata.as_ref().unwrap();
        let mut notes = Vec::new();
        let Some(shape) = crate::estimator::shape_from_metadata(md, m.size_bytes, &mut notes) else {
            eprintln!("{} has too little metadata; skipping", m.file_name);
            return;
        };
        let telemetry = crate::telemetry::TelemetryState::new();
        let snap = telemetry.snapshot();
        let gpu_total = snap
            .gpus
            .first()
            .map(|g| g.vram_total_bytes)
            .unwrap_or(16 * 1024 * 1024 * 1024);
        let kv = crate::estimator::KvType::F16;

        let measured = measure_ladder(
            &tool,
            Path::new(&m.path),
            &shape,
            md.quant_label.as_deref(),
            gpu_total,
            kv,
            &mut |_, _, _| {},
        )
        .expect("ladder should measure");

        let estimated = crate::estimator::quant_advice(
            &shape,
            md.quant_label.as_deref(),
            md.parameter_count,
            gpu_total,
            kv,
        );

        eprintln!("\n{}", m.file_name);
        eprintln!(
            "source {:.2} GiB at {:.2} BPW, requantize={}",
            measured.source_bytes as f64 / 1024.0f64.powi(3),
            measured.source_bpw,
            measured.requantize
        );
        eprintln!(
            "\n{:<9} {:>12} {:>12} {:>8}  {:<5} note",
            "rung", "measured", "estimated", "delta", "fits"
        );
        for r in &measured.rungs {
            let est = estimated.as_ref().and_then(|a| {
                a.options
                    .iter()
                    .find(|o| o.label == r.label)
                    .map(|o| o.est_weights_bytes)
            });
            let gib = |b: u64| b as f64 / 1024.0f64.powi(3);
            let note = if r.dominated { "dominated" } else { "" };
            let fits = if r.fits { "yes" } else { "no" };
            // Built as strings so a rung the estimator has no row for prints a
            // placeholder in the same columns as one it does.
            let (est_col, delta_col) = match est {
                Some(e) => (
                    format!("{:.2}G", gib(e)),
                    format!(
                        "{:.1}%",
                        (e as f64 - r.weights_bytes as f64) / r.weights_bytes as f64 * 100.0
                    ),
                ),
                None => ("-".to_string(), "-".to_string()),
            };
            eprintln!(
                "{:<9} {:>11.2}G {est_col:>12} {delta_col:>8}  {fits:<5} {note}",
                r.label,
                gib(r.weights_bytes),
            );
        }

        // The contract, rather than a count: a rung that is no smaller than the
        // source is a conversion with nothing to gain, and offering one would
        // invite someone to spend real time producing a file that is not
        // better. An empty ladder is a legitimate answer for an already-minimal
        // source, so emptiness is not the thing being asserted.
        for r in &measured.rungs {
            assert!(
                r.bpw < measured.source_bpw,
                "{} at {:.2} BPW is not below the source's {:.2}",
                r.label,
                r.bpw,
                measured.source_bpw
            );
        }
    }
}
