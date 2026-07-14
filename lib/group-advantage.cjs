'use strict';

/**
 * GRPO group-relative advantage (critic-free): Â_k = (R_k − μ) / σ
 * Population statistics — shared reward-normalization primitive.
 * Pure, deterministic, zero dependencies, no I/O.
 */

/** Coerce any value to a finite number; NaN / ±Infinity → 0. */
function toNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

/** Arithmetic mean; empty → 0; non-numeric entries coerced to 0. */
function mean(nums) {
  if (!Array.isArray(nums) || nums.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < nums.length; i++) {
    sum += toNumber(nums[i]);
  }
  return sum / nums.length;
}

/** Population standard deviation (÷N); empty or single → 0. */
function std(nums) {
  if (!Array.isArray(nums) || nums.length <= 1) return 0;

  const n = nums.length;
  const values = new Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    values[i] = toNumber(nums[i]);
    sum += values[i];
  }

  const mu = sum / n;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const d = values[i] - mu;
    sumSq += d * d;
  }

  return Math.sqrt(sumSq / n);
}

/**
 * GRPO group-relative advantage: Â_k = (R_k − μ) / σ.
 * Returns array of same length. Guards:
 *   [] → [], single → [0], σ===0 → all zeros, non-numeric coerced to 0.
 * Never produces NaN / Infinity / undefined.
 */
function groupAdvantage(rewards) {
  if (!Array.isArray(rewards) || rewards.length === 0) return [];

  const n = rewards.length;
  const values = new Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    values[i] = toNumber(rewards[i]);
    sum += values[i];
  }

  if (n === 1) return [0];

  const mu = sum / n;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const d = values[i] - mu;
    sumSq += d * d;
  }

  if (sumSq === 0) return new Array(n).fill(0);

  const sigma = Math.sqrt(sumSq / n);
  if (sigma === 0) return new Array(n).fill(0);

  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = (values[i] - mu) / sigma;
  }
  return out;
}

/** Index of max reward (first on tie); empty → -1. */
function bestIndex(rewards) {
  if (!Array.isArray(rewards) || rewards.length === 0) return -1;

  let best = 0;
  let bestVal = toNumber(rewards[0]);
  for (let i = 1; i < rewards.length; i++) {
    const v = toNumber(rewards[i]);
    if (v > bestVal) {
      bestVal = v;
      best = i;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Self-test — `node lib/group-advantage.cjs` is the smoke test.
// ---------------------------------------------------------------------------

function assertDeepEqual(actual, expected, label) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a !== e) {
    throw new Error(`${label}: expected ${e}, got ${a}`);
  }
}

function assertFinite(value, label) {
  if (!Number.isFinite(value)) {
    throw new Error(`${label}: non-finite value ${String(value)}`);
  }
}

function selfTest() {
  // --- mean ---
  assertDeepEqual(mean([]), 0, 'mean([])');
  assertDeepEqual(mean([1, 2, 3]), 2, 'mean([1,2,3])');
  assertDeepEqual(mean([1, 'x', 3]), 4 / 3, 'mean coerces non-numeric to 0');

  // --- std (population, ÷N) ---
  assertDeepEqual(std([]), 0, 'std([])');
  assertDeepEqual(std([5]), 0, 'std single');
  assertDeepEqual(std([1, 0]), 0.5, 'std([1,0]) = 0.5');
  assertDeepEqual(std([2, 4, 4, 4, 5, 5, 7, 9]), 2, 'std classic = 2');

  // --- groupAdvantage contract cases ---
  assertDeepEqual(groupAdvantage([]), [], 'groupAdvantage([])');
  assertDeepEqual(groupAdvantage([5]), [0], 'groupAdvantage single');
  assertDeepEqual(groupAdvantage([1, 1, 1]), [0, 0, 0], 'groupAdvantage σ=0');
  assertDeepEqual(groupAdvantage([1, 0]), [1, -1], 'groupAdvantage([1,0])');
  assertDeepEqual(
    groupAdvantage([2, 4, 4, 4, 5, 5, 7, 9]),
    [-1.5, -0.5, -0.5, -0.5, 0, 0, 1, 2],
    'groupAdvantage classic z-scores'
  );

  // --- bestIndex ---
  assertDeepEqual(bestIndex([]), -1, 'bestIndex([])');
  assertDeepEqual(bestIndex([0.2, 0.9, 0.9]), 1, 'bestIndex first-on-tie');

  // --- no NaN / Infinity / undefined in any output path ---
  assertFinite(mean([NaN, 'bad', null, undefined]), 'mean all-invalid');
  assertFinite(std([NaN, 'bad', null, 1]), 'std with invalid entries');
  for (const v of groupAdvantage([NaN, 'bad', 1, 0])) {
    assertFinite(v, 'groupAdvantage with invalid entries');
  }
  for (const v of groupAdvantage([3, 3, 3, 3])) {
    assertFinite(v, 'groupAdvantage all-equal');
  }
  assertDeepEqual(bestIndex([NaN, undefined, null]), 0, 'bestIndex all-invalid');

  // --- σ===0 guard must not swallow legitimately small spreads ---
  assertDeepEqual(groupAdvantage([0, 1e-12]), [-1, 1], 'groupAdvantage small spread');

  console.log('group-advantage OK');
}

if (require.main === module) {
  selfTest();
}

module.exports = {
  mean,
  std,
  groupAdvantage,
  bestIndex,
};
