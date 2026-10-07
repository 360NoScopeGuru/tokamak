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

use std::path::{Path, PathBuf};
use std::process::Command;

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
