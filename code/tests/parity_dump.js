// Loads the tool's real <script> in a stubbed DOM and dumps the outputs we need
// to compare against the Python side. Nothing here is a reimplementation — it
// runs the shipped code.
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const HTML = process.argv[2];
const PUZZLE = process.argv[3];

const html = fs.readFileSync(HTML, "utf8");
const js = /<script>([\s\S]*)<\/script>/.exec(html)[1];

// ── minimal DOM stub ──
function el(id) {
  return {
    id, value: "", checked: id === "narcNarrative" || id === "narcSizes" || id === "batchSkipDone",
    innerHTML: "", textContent: "", disabled: false, style: {}, files: [],
    addEventListener() {}, removeEventListener() {}, click() {}, appendChild() {},
    getContext: () => ({
      fillRect() {}, strokeRect() {}, beginPath() {}, moveTo() {}, lineTo() {},
      stroke() {}, fillText() {}, save() {}, restore() {}, clip() {}, rect() {},
      scale() {}, setLineDash() {}, measureText: () => ({ width: 10 }),
    }),
    toDataURL: () => "data:image/png;base64,STUB",
    width: 0, height: 0,
  };
}
const nodes = new Map();
const store = new Map();

const sandbox = {
  console,
  performance: { now: () => 0 },
  setTimeout, clearTimeout, Blob: class {}, URL: { createObjectURL: () => "", revokeObjectURL() {} },
  DOMException: class extends Error { constructor(m, n) { super(m); this.name = n; } },
  AbortController: class { constructor() { this.signal = { aborted: false, addEventListener() {}, removeEventListener() {} }; } abort() {} },
  indexedDB: { open: () => ({ onupgradeneeded: null, onsuccess: null, onerror: null, result: null }) },
  localStorage: {
    getItem: k => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, v), removeItem: k => store.delete(k),
  },
  document: {
    getElementById(id) { if (!nodes.has(id)) nodes.set(id, el(id)); return nodes.get(id); },
    createElement: id => el(id),
    addEventListener() {},
  },
  fetch: () => Promise.reject(new Error("no network in parity harness")),
};
sandbox.addEventListener = () => {};
sandbox.removeEventListener = () => {};
sandbox.devicePixelRatio = 1;
sandbox.window = sandbox;
sandbox.globalThis = sandbox;

vm.createContext(sandbox);
// Top-level `const` lives in the script's lexical scope, not on the global
// object, so re-export the constants we want to inspect.
const EXPORTS = ["NARC_RECOVER_PROMPT", "SCHEMA_VERSION", "ARC_COLORS", "ARC_COLOR_NAMES",
                 "RECOVER_PROMPT", "SOLVE_PROMPT_IMAGE"];
vm.runInContext(js + "\n;" + EXPORTS.map(n => `globalThis.${n}=${n};`).join(""),
                sandbox, { filename: "tool.js" });

// ── exercise it ──
const puzzle = JSON.parse(fs.readFileSync(PUZZLE, "utf8"));
const masked = sandbox.narcMaskedPositions(puzzle);
const answers = puzzle.answer_grids || {};
const dims = {};
masked.forEach(p => { const a = answers[String(p)]; if (a) dims[String(p)] = `${a.length}x${a[0].length}`; });

const visible = sandbox.narcVisibleFrames(puzzle);
const seqJson = JSON.stringify({ title: puzzle.title, sequence: visible });

const out = {
  masked_positions: masked.map(Number),
  dims,
  visible_positions: visible.map(f => f.position),
  prompt_image: sandbox.narcSolvePromptImage(puzzle.narrative, masked, dims),
  prompt_image_no_narrative: sandbox.narcSolvePromptImage("", masked, dims),
  prompt_image_no_dims: sandbox.narcSolvePromptImage(puzzle.narrative, masked, null),
  prompt_recover: sandbox.NARC_RECOVER_PROMPT,
  prompt_text: sandbox.narcSolvePromptText(seqJson, puzzle.narrative, masked, dims),
  run_key: sandbox.runKeyFor(puzzle.puzzle_id, "qwen/qwen3.6-35b", false, true, true, "reconstruct"),
  run_key_ablated: sandbox.runKeyFor(puzzle.puzzle_id, "m", true, false, false, "blind_solve"),
  record: sandbox.baseJsonlRecord(
    puzzle, path.basename(PUZZLE), "qwen/qwen3.6-35b",
    { thinking: false, includeNarrative: true, includeSizes: true },
    "reconstruct", `${puzzle.puzzle_id}.png`, "folder"),
  // extraction tolerance: the off-spec wrappers models actually emit
  extract_sequence: {
    spec: sandbox.extractSequenceMap({ sequence: [{ position: 0, grid: [[1]] }, { position: 2, grid: [[2]] }] }),
    bare_list: sandbox.extractSequenceMap([[[1]], [[2]]]),
    keyed: sandbox.extractSequenceMap({ "frame 0": [[1]], "frame 1": [[2]] }),
    frames_key: sandbox.extractSequenceMap({ frames: [{ position: 5, grid: [[9]] }] }),
  },
  extract_answers: {
    spec: sandbox.extractAnswersMap({ answers: { "3": [[1]] } }, [3]),
    bare_single: sandbox.extractAnswersMap([[7]], [3]),
    output_key: sandbox.extractAnswersMap({ output: [[7]] }, [3]),
    prefixed: sandbox.extractAnswersMap({ "position_3": [[7]] }, [3]),
  },
  parse_repair: {
    clean: sandbox.parseWithRepairFlag('{"a":1}'),
    fenced: sandbox.parseWithRepairFlag('```json\n{"a":1}\n```'),
    chatty: sandbox.parseWithRepairFlag('Sure! Here you go:\n{"a":1}\nHope that helps.'),
    broken: sandbox.parseWithRepairFlag('not json at all'),
  },
};
process.stdout.write(JSON.stringify(out, null, 1));
