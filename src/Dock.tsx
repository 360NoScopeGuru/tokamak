/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  BenchResult,
  InferenceMetrics,
  MeasuredRung,
  ModelEntry,
  QuantAdvice,
  QuantMeasurement,
  VramEstimate,
  ctxLabel,
  gb,
  modelLabel,
} from "./types";

// Containment dock (232px under the console): the arithmetic behind every
// recommendation. Cold + selected → VRAM budget | context ladder | quant
// advisor. Live → live budget | session stats | context ladder. After a
// single-model bench → the measured per-config table.

/// The conversion, owned by `App` rather than by the advisor column.
///
/// A conversion runs for minutes, and this column unmounts whenever another
/// model is selected or the dock switches to the live view. Holding the state
/// here would take the event listener down with it: the completion event would
/// be lost, the library would never be told to rescan, and the backend slot
/// would stay claimed with nothing on screen able to release or stop it.
export interface QuantConvertUi {
  running: { label: string; done: number; total: number } | null;
  /// How the last one ended, until something replaces it.
  result: { output: string | null; error: string | null } | null;
  onMake: (quant: string, sourceLabel: string | null, allowRequantize: boolean) => void;
  onStop: () => void;
}

interface DockProps {
  selected: ModelEntry | null;
  selectedEst: VramEstimate | null;
  liveModel: ModelEntry | null;
  liveEst: VramEstimate | null;
  liveCfg: { ngl: number; ctx: number } | null;
  metrics: InferenceMetrics | null;
  uptimeMs: number | null;
  benchDetail: { name: string; expected: number; results: BenchResult[]; running: boolean } | null;
  onCloseBench: () => void;
  quant: QuantConvertUi;
}

function uptime(ms: number): string {
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function BudgetCol({
  est,
  kvRatio,
}: {
  est: VramEstimate;
  kvRatio: number | null; // null = cold (single kv segment)
}) {
  const budget = est.budget_bytes || 1;
  const pct = (b: number) => `${Math.min(100, (b / budget) * 100).toFixed(1)}%`;
  const headroom = est.budget_bytes - est.est_total_bytes;
  const kvUsed = kvRatio != null ? est.est_kv_bytes * kvRatio : 0;
  const kvRest = est.est_kv_bytes - kvUsed;
  return (
    <div className="dock-col">
      <div className="budget-bar">
        <span className="seg-w" style={{ width: pct(est.est_weights_bytes) }} />
        {kvRatio != null && <span className="seg-ku" style={{ width: pct(kvUsed) }} />}
        <span className="seg-k" style={{ width: pct(kvRatio != null ? kvRest : est.est_kv_bytes) }} />
        <span className="seg-o" style={{ width: pct(est.est_overhead_bytes) }} />
      </div>
      <div className="budget-legend">
        <span className="li">
          <span className="sw w" />
          weights
          <span className="val">{gb(est.est_weights_bytes, 2)} GB</span>
        </span>
        {kvRatio != null ? (
          <>
            <span className="li">
              <span className="sw ku" />
              kv used
              <span className="val">{gb(kvUsed, 2)} GB</span>
            </span>
            <span className="li">
              <span className="sw k" />
              kv reserved
              <span className="val">{gb(kvRest, 2)} GB</span>
            </span>
          </>
        ) : (
          <span className="li">
            <span className="sw k" />
            kv cache · {ctxLabel(est.ctx_size)}
            <span className="val">{gb(est.est_kv_bytes, 2)} GB</span>
          </span>
        )}
        <span className="li">
          <span className="sw o" />
          overhead
          <span className="val">{gb(est.est_overhead_bytes, 2)} GB</span>
        </span>
        <span className="li head-room">
          <span className="sw h" />
          headroom
          <span className={`val ${headroom >= 0 ? "good" : "bad"}`}>
            {headroom >= 0 ? "" : "−"}
            {gb(Math.abs(headroom), 2)} GB
          </span>
        </span>
      </div>
    </div>
  );
}

function ContextCol({ est }: { est: VramEstimate }) {
  return (
    <div className="dock-col">
      <div className="lbl faint">Context Ladder</div>
      <div className="ladder">
        {est.context_options.map((o) => {
          const rec = o.ctx === est.ctx_size;
          const tight = o.fits && o.est_total_bytes > est.budget_bytes * 0.92;
          // A rung is usable whenever any layer fits, not only on full offload.
          // Marking partial rungs "✗" made the ladder claim nothing worked on
          // models that were running perfectly well.
          const usable = o.fits || o.n_gpu_layers > 0;
          return (
            <span key={o.ctx} className="rung">
              <span className={usable ? (o.fits ? "ok" : "part") : "no"}>
                {o.fits ? "✓" : usable ? "◐" : "✗"}
              </span>
              <span className={`k ${rec ? "hot" : ""}`} style={{ width: 36 }}>
                {ctxLabel(o.ctx)}
              </span>
              <span className="v">
                {o.fits ? `${gb(o.est_total_bytes)} GB` : `${o.n_gpu_layers}L`}
              </span>
              {rec ? (
                <span className="tag rec">● rec</span>
              ) : tight ? (
                <span className="tag tight">tight</span>
              ) : null}
            </span>
          );
        })}
      </div>
    </div>
  );
}

/// The estimated ladder, refinable into a measured one.
///
/// The estimate is arithmetic over an average bits-per-weight table, so it is
/// free and always on screen. Measuring spawns `llama-quantize --dry-run` per
/// rung, which is llama.cpp's own answer for this model's actual tensors, and
/// costs a few hundred milliseconds each — hence a button rather than an
/// automatic refresh. The two disagreed by as much as 24% on a 30B MoE, in
/// both directions, so the refinement is not cosmetic.
function AdvisorCol({
  advice,
  modelPath,
  quant,
}: {
  advice: QuantAdvice;
  modelPath: string;
  quant: QuantConvertUi;
}) {
  const rec = advice.recommended;
  const [measured, setMeasured] = useState<QuantMeasurement | null>(null);
  const [busy, setBusy] = useState<{ done: number; total: number; label: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  /// Which rung is awaiting confirmation. Purely local: nothing has been asked
  /// of the backend yet.
  const [asking, setAsking] = useState<MeasuredRung | null>(null);

  // A measurement belongs to one model. Selecting another must not leave the
  // previous model's exact sizes sitting under the new one's name. The
  // conversion is *not* reset here, because it does not belong to this
  // component — see `App`.
  useEffect(() => {
    setMeasured(null);
    setErr(null);
    setBusy(null);
    setAsking(null);
  }, [modelPath]);

  useEffect(() => {
    let dead = false;
    let un: (() => void) | undefined;
    listen<{ done: number; total: number; label: string }>("quant-progress", (e) => {
      // Only while a run of ours is open, so a late event cannot bring the
      // counter back after the ladder has been replaced. Measuring is bounded
      // by a few seconds, so losing it on unmount costs nothing.
      setBusy((b) => (b ? e.payload : b));
    })
      .then((u) => (dead ? u() : (un = u)))
      .catch(() => {});
    return () => {
      dead = true;
      un?.();
    };
  }, []);

  async function measure() {
    setErr(null);
    setBusy({ done: 0, total: 0, label: "" });
    try {
      setMeasured(await invoke<QuantMeasurement>("quant_measure", { modelPath }));
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(null);
    }
  }

  const exact = measured?.tool ? measured : null;

  return (
    <div className="dock-col">
      <div className="lbl faint">
        {exact ? "Quant Advisor · measured" : `Quant Advisor · ~${advice.est_params_b.toFixed(0)}B`}{" "}
        · this GPU
      </div>

      {exact ? (
        <div className="ladder">
          {exact.rungs.length === 0 && (
            <span className="rung">
              <span className="tag dim">nothing below {exact.source_label ?? "this quant"}</span>
            </span>
          )}
          {exact.rungs.map((r) => (
            <span key={r.label} className={`rung ${r.is_current ? "current" : ""}`}>
              <span className={r.fits ? "ok" : "no"}>{r.fits ? "✓" : "✗"}</span>
              <span className={`k ${r.is_current ? "hot" : ""}`}>{r.label}</span>
              <span className="v">{gb(r.weights_bytes)} GB</span>
              {r.dominated ? (
                <span className="tag bad" title="a higher-quality rung is this size or smaller">
                  dominated
                </span>
              ) : r.fits ? (
                <span className="tag rec">+{gb(r.headroom_bytes)} GB</span>
              ) : (
                <span className="tag bad">over w/ kv</span>
              )}
              {/* Offered on every rung, including the ones that will not fit
                  and the dominated ones: the verdict is already stated beside
                  it, and someone converting for another machine is entitled to
                  a size this GPU cannot hold. */}
              <button
                className="rung-make"
                disabled={!!quant.running || r.is_current}
                onClick={() => setAsking(r)}
                title={`write a ${r.label} copy beside this model`}
              >
                make
              </button>
            </span>
          ))}
        </div>
      ) : (
        <div className="ladder">
          {advice.options.map((o) => {
            const isRec = !!rec && o.label === rec;
            return (
              <span key={o.label} className={`rung ${o.is_current ? "current" : ""}`}>
                <span className={o.fits ? "ok" : "no"}>{o.fits ? "✓" : "✗"}</span>
                <span className={`k ${o.is_current || isRec ? "hot" : ""}`}>{o.label}</span>
                <span className="v">{gb(o.est_weights_bytes)} GB</span>
                {isRec ? (
                  <span className="tag rec">● sweet spot</span>
                ) : o.fits ? (
                  <span className="tag dim">+{gb(o.headroom_bytes)} GB</span>
                ) : (
                  <span className="tag bad">over w/ kv</span>
                )}
              </span>
            );
          })}
        </div>
      )}

      <div className="advisor-foot">
        {asking ? (
          // Stated before anything is written, because this is the one action
          // here that costs real time and real disk. The requantize warning is
          // llama.cpp's own default talking: it refuses this without a flag.
          <span className="confirm">
            <span className="faint">
              write {asking.label} · {gb(asking.weights_bytes)} GB
              {exact?.requantize ? ` · requantizing from ${exact.source_label ?? "a quant"} loses quality` : ""}
            </span>
            <button
              onClick={() => {
                const r = asking;
                setAsking(null);
                quant.onMake(r.label, measured?.source_label ?? null, !!measured?.requantize);
              }}
            >
              write it
            </button>
            <button onClick={() => setAsking(null)}>cancel</button>
          </span>
        ) : quant.running ? (
          <span className="confirm">
            <span className="faint">
              writing {quant.running.label}
              {quant.running.total
                ? ` · ${quant.running.done}/${quant.running.total} tensors`
                : ""}
            </span>
            <button onClick={quant.onStop}>stop</button>
          </span>
        ) : quant.result?.error ? (
          <span className="bad" title={quant.result.error}>
            {quant.result.error.length > 48
              ? `${quant.result.error.slice(0, 48)}…`
              : quant.result.error}
          </span>
        ) : quant.result?.output ? (
          <span className="faint" title={quant.result.output}>
            ✓ wrote {quant.result.output.split(/[\\/]/).pop()}
          </span>
        ) : busy ? (
          <span className="faint">
            measuring{busy.total ? ` ${busy.done}/${busy.total}` : ""}
            {busy.label ? ` · ${busy.label}` : ""}
          </span>
        ) : err ? (
          <span className="bad" title={err}>
            {err.length > 48 ? `${err.slice(0, 48)}…` : err}
          </span>
        ) : measured && !measured.tool ? (
          <span className="faint">install a runtime to measure exactly</span>
        ) : exact ? (
          <span className="faint">
            {exact.source_label ?? "source"} at {exact.source_bpw.toFixed(2)} BPW
            {exact.requantize ? " · requantizing costs quality beyond the target" : ""}
          </span>
        ) : (
          <button onClick={measure} title="ask llama.cpp the exact size of each rung">
            ⟐ measure exactly
          </button>
        )}
      </div>
    </div>
  );
}

function SessionCol({
  metrics,
  uptimeMs,
}: {
  metrics: InferenceMetrics | null;
  uptimeMs: number | null;
}) {
  return (
    <div className="dock-col">
      <div className="lbl faint">This Session</div>
      <div className="kv-list">
        <span className="row">
          <span className="k">decode</span>
          <span className="v">{metrics ? `${metrics.predicted_tokens_per_sec.toFixed(1)} tok/s` : "—"}</span>
        </span>
        <span className="row">
          <span className="k">prefill</span>
          <span className="v">{metrics ? `${Math.round(metrics.prompt_tokens_per_sec).toLocaleString()} tok/s` : "—"}</span>
        </span>
        <span className="row">
          <span className="k">total tok</span>
          <span className="v">{metrics ? metrics.predicted_tokens_total.toLocaleString() : "—"}</span>
        </span>
        <span className="row">
          <span className="k">kv tokens</span>
          <span className="v">{metrics ? metrics.kv_cache_tokens.toLocaleString() : "—"}</span>
        </span>
        <span className="row">
          <span className="k">in flight</span>
          <span className="v">{metrics ? metrics.requests_processing : "—"}</span>
        </span>
        <span className="row">
          <span className="k">uptime</span>
          <span className="v">{uptimeMs != null ? uptime(uptimeMs) : "—"}</span>
        </span>
      </div>
    </div>
  );
}

function BenchDetail({
  detail,
  onClose,
}: {
  detail: { name: string; expected: number; results: BenchResult[]; running: boolean };
  onClose: () => void;
}) {
  const best = detail.results.reduce((m, r) => (r.loaded ? Math.max(m, r.decode_tok_s) : m), 0);
  return (
    <>
      <div className="dock-head">
        <span className="lbl">Bench Detail</span>
        <span className="name">{detail.name}</span>
        <span className="right">
          {detail.running
            ? `measuring ${detail.results.length + 1} / ${detail.expected}…`
            : `${detail.results.length} configs measured`}
        </span>
        {!detail.running && <button onClick={onClose}>✕</button>}
      </div>
      <div style={{ flex: 1, padding: "0 14px 10px", overflowY: "auto" }}>
        <div className="board-cols" style={{ borderBottom: "1px solid var(--hair2)", padding: "4px 0" }}>
          <span style={{ width: 130 }}>Config</span>
          <span style={{ width: 90, textAlign: "right" }}>Decode</span>
          <span style={{ width: 90, textAlign: "right" }}>Prefill</span>
          <span style={{ width: 70, textAlign: "right" }}>Load</span>
          <span style={{ width: 90, textAlign: "right" }}>Peak VRAM</span>
          <span style={{ flex: 1 }} />
        </div>
        {detail.results.map((r, i) => (
          <div
            key={i}
            className="board-row"
            style={{ padding: "5px 0", borderBottom: "1px solid var(--hair3)", fontSize: 11 }}
          >
            <span style={{ width: 130, color: "var(--hi)" }}>
              {r.n_gpu_layers}L · {ctxLabel(r.ctx_size)}
            </span>
            {r.loaded ? (
              <>
                <span
                  style={{
                    width: 90,
                    textAlign: "right",
                    color: r.decode_tok_s === best && best > 0 ? "var(--plasma)" : "var(--hi)",
                    fontWeight: r.decode_tok_s === best && best > 0 ? 600 : 400,
                  }}
                >
                  {r.decode_tok_s.toFixed(1)}
                </span>
                <span style={{ width: 90, textAlign: "right", color: "var(--mid)" }}>
                  {Math.round(r.prefill_tok_s).toLocaleString()}
                </span>
                <span style={{ width: 70, textAlign: "right", color: "var(--mid)" }}>
                  {(r.load_ms / 1000).toFixed(1)}s
                </span>
                <span style={{ width: 90, textAlign: "right", color: "var(--mid)" }}>
                  {gb(r.peak_vram_bytes, 1)} GB
                </span>
                <span style={{ flex: 1, color: "var(--faint)", paddingLeft: 8 }}>
                  {r.decode_tok_s === best && best > 0 ? "fastest" : ""}
                </span>
              </>
            ) : (
              <span style={{ flex: 1, color: "var(--danger)" }}>{r.error ?? "failed"}</span>
            )}
          </div>
        ))}
        {detail.running && (
          <div className="board-row pending" style={{ padding: "6px 0" }}>
            measuring — loading model on the bench port…
          </div>
        )}
        {!detail.running && detail.results.length > 0 && (
          <div className="board-foot">measured on your GPU — real generation, not estimated</div>
        )}
      </div>
    </>
  );
}

export function Dock(p: DockProps) {
  if (p.benchDetail) {
    return (
      <div className="dock">
        <BenchDetail detail={p.benchDetail} onClose={p.onCloseBench} />
      </div>
    );
  }

  // Selecting a different model than the running one wins over the live view.
  const showSelected =
    p.selected && p.selectedEst && (!p.liveModel || p.selected.path !== p.liveModel.path);

  if (showSelected && p.selected && p.selectedEst) {
    const est = p.selectedEst;
    const fitTag = est.full_offload ? (
      <span className="state-tag" style={{ color: "var(--good)" }}>● FITS FULLY</span>
    ) : est.fits ? (
      <span className="state-tag" style={{ color: "var(--warn)" }}>◐ PARTIAL OFFLOAD</span>
    ) : (
      <span className="state-tag" style={{ color: "var(--danger)" }}>○ CPU ONLY</span>
    );
    return (
      <div className="dock">
        <div className="dock-head">
          <span className="lbl">Containment Budget</span>
          <span className="name">{modelLabel(p.selected)}</span>
          {fitTag}
          <span className="right">
            recommended{" "}
            <b>
              {est.n_gpu_layers}
              {p.selected.metadata?.block_count ? `/${p.selected.metadata.block_count}` : ""} layers ·{" "}
              {ctxLabel(est.ctx_size)} ctx
            </b>
          </span>
        </div>
        <div className="dock-grid">
          <BudgetCol est={est} kvRatio={null} />
          <ContextCol est={est} />
          {est.quant_advice ? (
            <AdvisorCol advice={est.quant_advice} modelPath={p.selected.path} quant={p.quant} />
          ) : (
            <div className="dock-col" />
          )}
        </div>
      </div>
    );
  }

  if (p.liveModel && p.liveEst) {
    return (
      <div className="dock">
        <div className="dock-head">
          <span className="lbl">Containment Budget</span>
          <span className="name">{modelLabel(p.liveModel)}</span>
          <span className="state-tag" style={{ color: "var(--plasma)" }}>▶ LIVE</span>
          <span className="right">
            {p.liveCfg ? (
              <b>
                {p.liveCfg.ngl}
                {p.liveModel.metadata?.block_count ? `/${p.liveModel.metadata.block_count}` : ""} layers ·{" "}
                {ctxLabel(p.liveCfg.ctx)} ctx
              </b>
            ) : null}
          </span>
        </div>
        <div className="dock-grid">
          <BudgetCol est={p.liveEst} kvRatio={p.metrics?.kv_cache_usage_ratio ?? 0} />
          <SessionCol metrics={p.metrics} uptimeMs={p.uptimeMs} />
          <ContextCol est={p.liveEst} />
        </div>
      </div>
    );
  }

  return (
    <div className="dock">
      <div className="dock-head">
        <span className="lbl">Containment Budget</span>
      </div>
      <div className="dock-empty">
        <div className="inner">
          <span className="sub">
            Select a model to see recommended layers &amp; context, the VRAM budget, the context
            ladder and the quant advisor.
          </span>
        </div>
      </div>
    </div>
  );
}
