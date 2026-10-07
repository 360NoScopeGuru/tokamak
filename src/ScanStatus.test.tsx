/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Tests for the startup progress strip.
//!
//! Rendered to a string rather than into a DOM: this component has no
//! behaviour, only a mapping from phase to words and a bar width, and that
//! mapping is the whole thing worth pinning. Checking it this way needs no
//! jsdom and no testing library.
//!
//! These were run by hand once, which is how the dangling-separator bug below
//! was found. Running them by hand again next time is how it comes back.

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ScanStatus } from "./ScanStatus";
import { ScanPhase } from "./types";

/// The strip's visible text and how full the bar is: everything a user can see.
function render(phase: ScanPhase) {
  const html = renderToStaticMarkup(<ScanStatus phase={phase} />);
  return {
    html,
    label: html.match(/aria-label="([^"]*)"/)?.[1] ?? null,
    width: html.match(/width:(\d+)%/)?.[1] ?? null,
  };
}

const scanning = (over: Partial<Extract<ScanPhase, { kind: "scanning" }>> = {}): ScanPhase => ({
  kind: "scanning",
  root: "/home/me/.cache/huggingface/hub",
  rootIndex: 3,
  totalRoots: 6,
  found: 14,
  ...over,
});

describe("ScanStatus", () => {
  it("renders nothing at all when there is nothing to report", () => {
    expect(render({ kind: "idle" }).html).toBe("");
  });

  it("names the folder it is walking and what it has found", () => {
    const { label, width } = render(scanning());
    expect(label).toBe("Looking for models: huggingface/hub · 14 found");
    expect(width).toBe("50");
  });

  it("shortens a Windows path the same way as a Unix one", () => {
    const win = String.raw`C:\Users\me\.cache\huggingface\hub`;
    expect(render(scanning({ root: win })).label).toBe(
      "Looking for models: huggingface/hub · 14 found"
    );
  });

  it("leaves a root with fewer than two segments alone", () => {
    expect(render(scanning({ root: String.raw`D:\models` })).label).toBe(
      "Looking for models: D:/models · 14 found"
    );
  });

  it("calls the Ollama pass by name instead of treating it as a path", () => {
    const { label, width } = render(
      scanning({ root: "Ollama", rootIndex: 6, totalRoots: 6, found: 22 })
    );
    expect(label).toBe("Looking for models: Ollama · 22 found");
    expect(width).toBe("100");
  });

  /// The bug this pins: the first report arrives before any root is open, so
  /// the location is genuinely unknown for a moment. The strip used to render
  /// a dangling `" · 0 found"`, with a separator and nothing on one side.
  it("does not announce a separator with nothing before it", () => {
    const { label } = render(scanning({ root: "", rootIndex: 0, totalRoots: 0, found: 0 }));
    expect(label).toBe("Looking for models: 0 found");
    expect(label).not.toContain("· 0 found");
  });

  it("counts the fit checks off against their total", () => {
    const { label, width } = render({ kind: "estimating", done: 9, total: 22 });
    expect(label).toBe("Checking what fits: 9 of 22");
    expect(width).toBe("41");
  });

  it("fills the bar when the fit checks finish", () => {
    expect(render({ kind: "estimating", done: 22, total: 22 }).width).toBe("100");
  });

  describe("never emits a width that is not a width", () => {
    const cases: [string, ScanPhase][] = [
      ["no models to check", { kind: "estimating", done: 0, total: 0 }],
      ["no roots yet", scanning({ rootIndex: 0, totalRoots: 0 })],
      ["more done than the total", { kind: "estimating", done: 30, total: 22 }],
      ["past the last root", scanning({ rootIndex: 9, totalRoots: 6 })],
    ];
    for (const [name, phase] of cases) {
      it(name, () => {
        const { width } = render(phase);
        const n = Number(width);
        expect(Number.isFinite(n)).toBe(true);
        expect(n).toBeGreaterThanOrEqual(0);
        expect(n).toBeLessThanOrEqual(100);
      });
    }
  });

  it("stays announceable by a screen reader while it changes", () => {
    const { html } = render(scanning());
    expect(html).toContain('role="status"');
    expect(html).toContain('aria-live="polite"');
  });
});
