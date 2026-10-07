/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! What the app is doing while it works out what you have.
//!
//! Deliberately not a modal. Opening the app used to lock the window for about
//! ten seconds, and the obvious reading of that was "we need a loading screen"
//! — but the lock was a bug in the GGUF reader, not an unavoidable wait, and
//! once it was fixed a gate across the screen would have invented a wait that
//! no longer exists. So this is a strip, not a screen: it says which folder is
//! being walked and how far along the fit checks are, and it takes nothing
//! away while it does. Everything else on screen stays live.

import { ScanPhase } from "./types";

/// Last two path segments, which is enough to tell two model folders apart
/// without letting a deep cache path push the counts off the strip.
function shortRoot(path: string): string {
  if (path === "Ollama") return "Ollama";
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts.slice(-2).join("/") || path;
}

export function ScanStatus({ phase }: { phase: ScanPhase }) {
  if (phase.kind === "idle") return null;

  // Roots are the only honest denominator during the walk: the number of
  // models is not known until it finishes, so a bar drawn from `found` would
  // have no ceiling to approach.
  const [label, detail, frac] =
    phase.kind === "scanning"
      ? ([
          "Looking for models",
          // The first report arrives before any root is open, so the location
          // is genuinely unknown for a moment. Joining the known parts avoids
          // announcing a separator with nothing on one side of it.
          [shortRoot(phase.root), `${phase.found} found`].filter(Boolean).join(" · "),
          phase.totalRoots > 0 ? phase.rootIndex / phase.totalRoots : 0,
        ] as const)
      : ([
          "Checking what fits",
          `${phase.done} of ${phase.total}`,
          phase.total > 0 ? phase.done / phase.total : 0,
        ] as const);

  return (
    <div
      className="scan-status"
      role="status"
      aria-live="polite"
      aria-label={`${label}: ${detail}`}
    >
      <div className="scan-status-line">
        <span className="scan-status-label">{label}</span>
        <span className="scan-status-detail">{detail}</span>
      </div>
      <div className="scan-status-track">
        <div
          className="scan-status-fill"
          style={{ width: `${Math.round(Math.min(1, Math.max(0, frac)) * 100)}%` }}
        />
      </div>
    </div>
  );
}
