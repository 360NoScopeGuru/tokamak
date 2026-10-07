/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Tests for the conversation tree.
//!
//! `history.rs` has a hundred of these for the same walks on the Rust side;
//! this file is the half that was missing. Rev G shipped three bugs that were
//! pure-function bugs in exactly these shapes — a legacy pool collapsing to one
//! turn, a sibling picker that only looked at one role, a fork that discarded
//! the turn it forked from — and each is pinned below by the case that would
//! have caught it.

import { describe, expect, it } from "vitest";
import { Turn, chain, childrenOf, leafUnder, newTurnId, pathTo } from "./branch";

/// A turn whose content is its own id, so a path reads as the ids it visited.
function t(id: string, parent: string | null, extra: Partial<Turn> = {}): Turn {
  return { id, parent, role: "user", content: id, ...extra };
}

const ids = (turns: { id?: string | null }[]) => turns.map((x) => x.id);

describe("pathTo", () => {
  it("has no path through an empty pool", () => {
    expect(pathTo([], null)).toEqual([]);
    expect(pathTo([], "nope")).toEqual([]);
  });

  /// The bug this pins: `pathTo` once keyed its map with `t.id ?? ""`, so every
  /// turn in a pre-Rev-G session collided on the empty string and the console
  /// rendered a single turn where a whole conversation should have been. A pool
  /// with no ids is already a transcript in order, not a tree to walk.
  it("returns a pool with no ids as the ordered transcript it is", () => {
    // The keys are declared but never set, which is what a pre-Rev-G turn
    // deserialised from disk actually looks like: absent, not null.
    const legacy: { content: string; id?: string; parent?: string | null }[] = [
      { content: "one" },
      { content: "two" },
      { content: "three" },
    ];
    expect(pathTo(legacy, null).map((x) => x.content)).toEqual(["one", "two", "three"]);
    // Also when a head is supplied that cannot mean anything in that pool.
    expect(pathTo(legacy, "whatever").map((x) => x.content)).toEqual(["one", "two", "three"]);
  });

  it("walks a straight line, and a null head means the last turn", () => {
    const pool = [t("a", null), t("b", "a"), t("c", "b")];
    expect(ids(pathTo(pool, null))).toEqual(["a", "b", "c"]);
    expect(ids(pathTo(pool, "c"))).toEqual(["a", "b", "c"]);
  });

  it("stops at the head rather than running to the end", () => {
    const pool = [t("a", null), t("b", "a"), t("c", "b")];
    expect(ids(pathTo(pool, "b"))).toEqual(["a", "b"]);
    expect(ids(pathTo(pool, "a"))).toEqual(["a"]);
  });

  it("falls back to the last turn when the head is not in the pool", () => {
    const pool = [t("a", null), t("b", "a")];
    expect(ids(pathTo(pool, "ghost"))).toEqual(["a", "b"]);
  });

  it("follows one branch of a fork and not the other", () => {
    //    a - b - c   (left)
    //      \ d - e   (right)
    const pool = [t("a", null), t("b", "a"), t("c", "b"), t("d", "a"), t("e", "d")];
    expect(ids(pathTo(pool, "c"))).toEqual(["a", "b", "c"]);
    expect(ids(pathTo(pool, "e"))).toEqual(["a", "d", "e"]);
    // A null head still means the newest turn, which is down the right branch.
    expect(ids(pathTo(pool, null))).toEqual(["a", "d", "e"]);
  });

  it("does not hang on a cycle in a hand-edited file", () => {
    const pool = [t("a", "b"), t("b", "a")];
    const path = pathTo(pool, "a");
    expect(path.length).toBeLessThanOrEqual(2);
    expect(new Set(ids(path)).size).toBe(path.length);
  });

  it("terminates when only some turns carry ids", () => {
    // Half-migrated files should not be reachable, but a short transcript is a
    // better failure than a spin.
    const pool: Turn[] = [t("a", null), { ...t("b", "a"), id: "" }, t("c", "b")];
    expect(() => pathTo(pool, "c")).not.toThrow();
    expect(ids(pathTo(pool, "c"))).toContain("c");
  });
});

describe("childrenOf", () => {
  const pool = [t("a", null), t("b", "a"), t("c", "a"), t("d", "b"), t("r2", null)];

  it("finds roots with a null parent", () => {
    expect(ids(childrenOf(pool, null))).toEqual(["a", "r2"]);
  });

  it("keeps siblings in the order they were created", () => {
    expect(ids(childrenOf(pool, "a"))).toEqual(["b", "c"]);
  });

  it("gives a leaf no children", () => {
    expect(childrenOf(pool, "d")).toEqual([]);
    expect(childrenOf(pool, "ghost")).toEqual([]);
  });

  /// The `◂ n/m ▸` picker rendered only on assistant turns, so a `⑂ branch`
  /// off a *user* turn created siblings with no way to reach them. Sibling
  /// lookup must not care about role.
  it("does not care what role the siblings are", () => {
    const mixed = [
      t("u", null, { role: "user" }),
      t("a1", "u", { role: "assistant" }),
      t("a2", "u", { role: "assistant" }),
      t("u2", "u", { role: "user" }),
    ];
    expect(ids(childrenOf(mixed, "u"))).toEqual(["a1", "a2", "u2"]);
  });
});

describe("leafUnder", () => {
  it("returns the turn itself when nothing follows it", () => {
    expect(leafUnder([t("a", null)], "a")).toBe("a");
  });

  it("runs to the bottom of a straight line", () => {
    expect(leafUnder([t("a", null), t("b", "a"), t("c", "b")], "a")).toBe("c");
  });

  it("takes the newest child at each fork, which is the branch you were last on", () => {
    //    a - b - c
    //      \ d - e
    const pool = [t("a", null), t("b", "a"), t("c", "b"), t("d", "a"), t("e", "d")];
    expect(leafUnder(pool, "a")).toBe("e");
    // Asking under an older sibling still follows that sibling.
    expect(leafUnder(pool, "b")).toBe("c");
  });

  it("does not spin on a cycle", () => {
    expect(() => leafUnder([t("a", "b"), t("b", "a")], "a")).not.toThrow();
  });
});

describe("chain", () => {
  const body = { role: "user" as const, content: "hi" };

  it("roots the first turn of an empty conversation", () => {
    const out = chain([], body);
    expect(out).toHaveLength(1);
    expect(out[0].parent).toBeNull();
    expect(out[0].id).toBeTruthy();
  });

  it("links each added turn onto the one before it", () => {
    const out = chain([], body, { role: "assistant", content: "yo" }, body);
    expect(out[1].parent).toBe(out[0].id);
    expect(out[2].parent).toBe(out[1].id);
  });

  it("appends onto the tail it is given, which is what forking relies on", () => {
    const prev = [t("a", null), t("b", "a")];
    const out = chain(prev, body);
    expect(out[2].parent).toBe("b");
  });

  it("leaves the array it was given alone", () => {
    const prev = [t("a", null)];
    chain(prev, body);
    expect(ids(prev)).toEqual(["a"]);
  });

  it("gives every turn a distinct id", () => {
    const out = chain([], body, body, body, body, body);
    expect(new Set(ids(out)).size).toBe(out.length);
  });
});

describe("newTurnId", () => {
  it("does not repeat itself when called in a tight loop", () => {
    // Ids are minted per turn inside `chain`, so same-millisecond collisions
    // are the realistic failure, not long-run uniqueness.
    const seen = new Set(Array.from({ length: 2000 }, newTurnId));
    expect(seen.size).toBe(2000);
  });
});
