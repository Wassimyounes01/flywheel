'use strict';
// demo.cjs — reward a few outcomes, watch recommend() shift. Run: node examples/demo.cjs
const { record, recommend, policyFor } = require('../lib/flywheel.cjs');
const task = 'choose-framework';
for (const [a, r] of [['react', 0.9], ['vue', 0.5], ['react', 0.8], ['svelte', 0.3], ['react', 0.85]]) {
  record(task, a, r);
  console.log(`  record(${a}, ${r})`);
}
console.log('\npolicy:', JSON.stringify(policyFor(task)));
const rec = recommend(task, ['react', 'vue', 'svelte', 'solid']);
console.log(`\nrecommend -> ${rec.action}  (${rec.why})`);
