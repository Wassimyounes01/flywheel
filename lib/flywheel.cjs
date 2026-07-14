#!/usr/bin/env node
/**
 * flywheel.cjs — a tiny task-level reinforcement-learning brain.
 *
 * Generalizes proven multi-armed-bandit math (win-rate·0.6 + avgReward·0.4 + exploration + count-decay) from
 * "which model wins" to "which ACTION/approach wins for THIS task type" — so every outcome your agent records
 * updates a policy that biases the next decision. Recall the best past approach for a task; reward outcomes;
 * exploit proven winners while still exploring under-sampled ones.
 *
 * It's a leaf: no framework, no lesson store — just the task-level policy. $0, no deps, atomic writes, never throws.
 *
 * API:
 *   record(taskType, action, reward[, lesson])   // reward 0..1; logs an observation + updates the policy
 *   recommend(taskType, actions[])                 // → { action, score, why } : best learned action (explore-aware)
 *   policyFor(taskType)                            // → ranked [{action, score, n, avgReward, winRate, stale}]
 *   recent(n)                                      // → last n observations
 */
'use strict';
const fs = require('fs');
const path = require('path');

const MEM = path.join(__dirname, '..', 'data');
try { fs.mkdirSync(MEM, { recursive: true }); } catch {}
const LOG = path.join(MEM, 'flywheel.jsonl');
const POLICY = path.join(MEM, 'flywheel-policy.json');
const N_CAP = 50;                 // bounded memory; decay old counts so the policy stays adaptive
const WIN = 0.6;                  // reward >= WIN counts as a "win"
const MIN_CONF = 0.55;            // below this a judgment logs but does NOT update the policy (a fuzzy judge → no drift)

function load() { try { return JSON.parse(fs.readFileSync(POLICY, 'utf8')); } catch { return { tasks: {}, updated_at: null }; } }
function save(s) { try { const t = POLICY + '.tmp'; fs.writeFileSync(t, JSON.stringify(s, null, 2)); fs.renameSync(t, POLICY); } catch {} }

/** record an outcome and update the policy. reward in [0,1]. */
function record(taskType, action, reward, lesson, opts = {}) {
  const r = Math.max(0, Math.min(1, Number(reward)));
  const tt = String(taskType || 'general').toLowerCase();
  const ac = String(action || 'default');
  // fail-open confidence read: a bad value must never throw or silently drop a legitimate update
  let conf = null;
  try {
    const o = (opts && typeof opts === 'object') ? opts : (typeof opts === 'number' ? { confidence: opts } : {});
    const raw = (o.confidence != null) ? o.confidence : (o.conf != null ? o.conf : null);
    // accept only numbers or non-empty numeric strings — Number(''/false/[]) === 0 would wrongly gate
    if (typeof raw === 'number' || (typeof raw === 'string' && raw.trim() !== '')) {
      const c = Number(raw);
      if (Number.isFinite(c)) conf = c;
    }
  } catch {}
  const lowConf = conf != null && conf < MIN_CONF;
  const confR = conf != null ? +conf.toFixed(3) : undefined;
  // append observation (the recursive-learning trace) — ALWAYS, even when low-conf (audit trail)
  try { fs.appendFileSync(LOG, JSON.stringify({ ts: new Date().toISOString(), task: tt, action: ac, reward: +r.toFixed(3), lesson: lesson ? String(lesson).slice(0, 280) : undefined, ...(conf != null ? { confidence: confR } : {}), ...(lowConf ? { low_conf: true, policy_skipped: true } : {}) }) + '\n'); } catch {}
  // a low-confidence judgment is recorded (audit trail) but must not drift the calibration
  if (lowConf) return { task: tt, action: ac, reward: +r.toFixed(3), policy_updated: false, low_conf: true };
  // update the policy (the reinforcement)
  const s = load();
  const T = (s.tasks[tt] = s.tasks[tt] || {});
  const m = (T[ac] = T[ac] || { n: 0, wins: 0, sum_r: 0, updated_at: null });
  if (m.n >= N_CAP) { const k = (N_CAP - 1) / N_CAP; m.n *= k; m.wins *= k; m.sum_r *= k; } // count-decay
  m.n += 1; m.sum_r += r; if (r >= WIN) m.wins += 1;
  m.updated_at = new Date().toISOString();
  s.updated_at = m.updated_at;
  save(s);
  const out = { task: tt, action: ac, reward: +r.toFixed(3), policy_updated: true };
  if (conf != null) out.confidence = confR;
  return out;
}

function scoreAction(m) {
  if (!m || !m.n) return { score: 0.55, n: 0, avgReward: null, winRate: null, unknown: true }; // unknown → sampled
  const winRate = m.wins / m.n, avgReward = m.sum_r / m.n;
  const explore = 0.15 / Math.sqrt(m.n + 1);                                  // UCB-ish: try the under-sampled
  const staleDays = m.updated_at ? (Date.now() - Date.parse(m.updated_at)) / 864e5 : null;
  const freshExplore = staleDays != null ? Math.min(0.08, staleDays / 550) : 0; // re-sample frozen winners
  return { score: +(winRate * 0.6 + avgReward * 0.4 + explore + freshExplore).toFixed(3), n: Math.round(m.n), avgReward: +avgReward.toFixed(2), winRate: +winRate.toFixed(2), stale: staleDays != null && staleDays > 7 };
}

function policyFor(taskType) {
  const s = load(); const T = s.tasks[String(taskType || 'general').toLowerCase()] || {};
  return Object.entries(T).map(([action, m]) => ({ action, ...scoreAction(m) })).sort((a, b) => b.score - a.score);
}

/** recommend the best action for a task. If `actions` given, only rank those (unknowns get a sampling score). */
function recommend(taskType, actions) {
  const tt = String(taskType || 'general').toLowerCase();
  const s = load(); const T = s.tasks[tt] || {};
  const pool = (Array.isArray(actions) && actions.length) ? actions : Object.keys(T);
  if (!pool.length) return { action: null, score: 0, why: 'no-policy-yet' };
  const ranked = pool.map(a => ({ action: a, ...scoreAction(T[a]) })).sort((x, y) => y.score - x.score);
  const top = ranked[0];
  return { action: top.action, score: top.score, why: top.unknown ? 'explore (untried)' : `exploit win${top.winRate}·avg${top.avgReward}·n${top.n}${top.stale ? '·stale' : ''}`, ranked };
}

function recent(n = 20) { try { return fs.readFileSync(LOG, 'utf8').trim().split('\n').slice(-n).map(l => JSON.parse(l)); } catch { return []; } }

/**
 * GRPO group layer — record K variants tried on the SAME task (a real group).
 * Each item goes through the existing record() UNCHANGED (absolute reward still
 * drives the winRate*0.6 + avgReward*0.4 policy math), then ONE sibling
 * append-only log line captures the group-relative advantages. Existing log
 * lines are never rewritten: the log is a shared cross-process append-only
 * jsonl that can exceed the runtime's max string size, so a read-patch-truncate
 * pass would both throw at scale and race concurrent appenders into silent data loss.
 * @param {string} taskType
 * @param {Array<{action:string, reward:number, lesson?:string}>} batch
 * @returns {Array<{action:string, reward:number, advantage:number}>}
 */
function recordGroup(taskType, batch) {
  const items = (Array.isArray(batch) ? batch : []).filter((b) => b && typeof b === 'object');
  if (!items.length) return [];

  const rewards = items.map((b) => {
    const r = Number(b.reward);
    return Number.isFinite(r) ? r : 0;
  });

  let advantages;
  let degraded = null;
  try {
    const { groupAdvantage } = require('./group-advantage.cjs');
    advantages = groupAdvantage(rewards);
    if (!Array.isArray(advantages) || advantages.length !== items.length) {
      throw new Error('groupAdvantage returned bad shape');
    }
  } catch (e) {
    // fail-open: zero advantages, absolute rewards still recorded below
    advantages = rewards.map(() => 0);
    degraded = String((e && e.message) || e);
  }

  const results = items.map((b, i) => {
    record(taskType, b.action, rewards[i], b.lesson);
    return {
      action: b.action,
      reward: rewards[i],
      advantage: Number.isFinite(advantages[i]) ? advantages[i] : 0
    };
  });

  try {
    require('fs').appendFileSync(
      LOG,
      JSON.stringify({
        ts: new Date().toISOString(),
        type: 'group_advantage',
        taskType,
        k: results.length,
        degraded,
        group: results
      }) + '\n'
    );
  } catch (e) {
    // fail-open: the per-item record() entries above are already persisted
  }

  return results;
}

module.exports = { record, recordGroup, recommend, policyFor, recent, LOG_PATH: LOG, POLICY_PATH: POLICY };
function auditPolicy(opts) {
  const o = opts || {};
  const minN = Number.isFinite(o.minN) ? o.minN : 5;
  const staleDays = Number.isFinite(o.staleDays) ? o.staleDays : 7;
  const weakReward = Number.isFinite(o.weakReward) ? o.weakReward : 0.5;
  const s = load();
  const tasks = s.tasks || {};
  const flagged = [];
  let actions = 0;
  for (const [task, T] of Object.entries(tasks)) {
    for (const [action, m] of Object.entries(T || {})) {
      actions += 1;
      const n = m && m.n ? m.n : 0;
      const avgReward = n ? m.sum_r / n : null;
      const parsed = m && m.updated_at ? Date.parse(m.updated_at) : NaN;
      const ageDays = Number.isFinite(parsed) ? (Date.now() - parsed) / 864e5 : null;
      const reasons = [];
      if (n < minN) reasons.push('starved');
      if (ageDays != null && ageDays > staleDays) reasons.push('stale');
      if (avgReward != null && avgReward < weakReward) reasons.push('weak');
      if (reasons.length) flagged.push({ task, action, n: Math.round(n), avgReward: avgReward == null ? null : +avgReward.toFixed(2), ageDays: ageDays == null ? null : +ageDays.toFixed(1), reasons });
    }
  }
  flagged.sort((a, b) => b.reasons.length - a.reasons.length);
  return { tasks: Object.keys(tasks).length, actions, flagged, healthy: flagged.length === 0, updated_at: s.updated_at || null };
}
module.exports.auditPolicy = auditPolicy;


// CLI: inspect the brain
if (require.main === module) {
  const [cmd, ...rest] = process.argv.slice(2);
  if (cmd === 'policy') console.log(JSON.stringify(policyFor(rest[0] || 'general'), null, 2));
  else if (cmd === 'recommend') console.log(JSON.stringify(recommend(rest[0], rest.slice(1)), null, 2));
  else if (cmd === 'record') console.log(JSON.stringify(record(rest[0], rest[1], parseFloat(rest[2]), rest.slice(3).join(' ')), null, 2));
  else console.log('usage: flywheel.cjs [policy <task> | recommend <task> [actions...] | record <task> <action> <reward0..1> [lesson]]');
}
