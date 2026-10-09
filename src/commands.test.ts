/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Every `invoke` has to name a command the backend actually registered.
//!
//! This is the one seam neither side's type system covers. The Rust commands
//! are checked by the compiler and the React code by `tsc`, but the string in
//! between is just a string: a typo, or a command added to `lib.rs` without
//! being listed in `generate_handler!`, fails only when a user clicks the thing.
//! That is exactly how it would have gone unnoticed here, since the quant
//! conversion cannot be exercised by a unit test.
//!
//! Reads the Rust source rather than running it, so it costs nothing.

import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

// Resolved from the project root, which is where vitest runs, rather than from
// `import.meta.dirname` — that needs a newer module target than this app builds
// against and would only break at typecheck time.
const SRC = join(process.cwd(), "src");
const LIB = join(process.cwd(), "src-tauri", "src", "lib.rs");

/// Command names passed to `invoke(...)` anywhere in the frontend.
function invokedNames(): Map<string, string[]> {
  const found = new Map<string, string[]>();
  for (const file of readdirSync(SRC).filter(
    (f: string) => /\.tsx?$/.test(f) && !f.includes(".test.")
  )) {
    const text = readFileSync(join(SRC, file), "utf8");
    // invoke<T>("name" | 'name'), with or without the type argument.
    for (const m of text.matchAll(/\binvoke\s*(?:<[^>]*>)?\s*\(\s*["']([^"']+)["']/g)) {
      const list = found.get(m[1]) ?? [];
      list.push(file);
      found.set(m[1], list);
    }
  }
  return found;
}

/// The identifiers inside `tauri::generate_handler![ ... ]`.
function registeredNames(): Set<string> {
  const rust = readFileSync(LIB, "utf8");
  const start = rust.indexOf("generate_handler![");
  expect(start, "lib.rs should have a generate_handler! block").toBeGreaterThan(-1);
  const end = rust.indexOf("]", start);
  const block = rust.slice(start + "generate_handler![".length, end);
  return new Set(
    block
      .split(",")
      .map((s: string) => s.trim())
      .filter((s: string) => /^[a-z_][a-z0-9_]*$/.test(s))
  );
}

describe("tauri command wiring", () => {
  const invoked = invokedNames();
  const registered = registeredNames();

  it("finds both sides of the seam", () => {
    // Guards the parsing itself: if either regex stopped matching, every
    // assertion below would pass vacuously.
    expect(invoked.size).toBeGreaterThan(20);
    expect(registered.size).toBeGreaterThan(20);
    expect(registered.has("scan_models")).toBe(true);
    expect([...invoked.keys()]).toContain("scan_models");
  });

  it("registers every command the frontend invokes", () => {
    const missing = [...invoked.entries()]
      .filter(([name]) => !registered.has(name))
      .map(([name, files]) => `${name} (from ${[...new Set(files)].join(", ")})`);
    expect(missing, "invoked but not in generate_handler!").toEqual([]);
  });

  it("names the quant commands this release added", () => {
    // Spelled out because they are the ones no test can reach at runtime: a
    // conversion writes gigabytes and takes minutes.
    for (const name of ["quant_measure", "quant_convert_start", "quant_convert_cancel"]) {
      expect(registered.has(name), `${name} should be registered`).toBe(true);
    }
    expect([...invoked.keys()]).toContain("quant_measure");
    expect([...invoked.keys()]).toContain("quant_convert_start");
    expect([...invoked.keys()]).toContain("quant_convert_cancel");
  });
});
