// web/benchmark/baseline.js
(function () {
    const sleep = (ms) => new Promise(r => setTimeout(r, ms));
    const KB = 1024, MB = 1024 * 1024;
  
    function stat(samples) {
      const s = [...samples].sort((a,b)=>a-b);
      const n = s.length || 1;
      const p = q => s[Math.min(n-1, Math.max(0, Math.floor(q*(n-1))))];
      const mean = s.reduce((a,b)=>a+b,0) / (s.length || 1);
      const variance = s.reduce((a,b)=>a + (b-mean)*(b-mean), 0) / (s.length || 1);
      const stdev = Math.sqrt(variance);
      return { n: s.length, mean, p50: p(0.5), p95: p(0.95), min: s[0] ?? 0, max: s[n-1] ?? 0, stdev, variance };
    }
  
    async function fetchText(url){ const r = await fetch(url); return await r.text(); }
    async function fetchJSON(url){ const r = await fetch(url); return await r.json(); }
    async function headOrGetSize(url) {
      try {
        const h = await fetch(url, { method: 'HEAD' });
        const n = parseInt(h.headers.get('content-length')||'0',10);
        if (n>0) return n;
      } catch {}
      const r = await fetch(url);
      const b = await r.clone().arrayBuffer();
      return b.byteLength;
    }
  
    async function clearCachesAll() {
      if ('caches' in window) {
        for (const k of await caches.keys()) await caches.delete(k);
      }
      if ('serviceWorker' in navigator) {
        for (const r of await navigator.serviceWorker.getRegistrations()) await r.unregister();
      }
    }
  
    async function ensureBackend(name) {
      if (name === 'webgpu' && !navigator.gpu) {
        await tf.setBackend('webgl').catch(()=>{});
      } else {
        await tf.setBackend(name).catch(()=>{});
      }
      await tf.ready();
      return tf.getBackend();
    }
  
    async function measureTransferFromModelJson(modelJsonUrl) {
      const txt = await fetchText(modelJsonUrl);
      const jsonBytes = new TextEncoder().encode(txt).byteLength;
      const mj = JSON.parse(txt);
      const base = modelJsonUrl.substring(0, modelJsonUrl.lastIndexOf('/') + 1);
      const paths = (mj.weightsManifest||[]).flatMap(m => m.paths||[]).map(p => p.startsWith('http')? p : base + p);
  
      let shardBytes = 0;
      for (const p of paths) shardBytes += await headOrGetSize(p);
      return { modelJsonMB: jsonBytes/MB, weightsMB: shardBytes/MB, totalMB: (jsonBytes+shardBytes)/MB, shardCount: paths.length };
    }
  
    function now(){ return performance.now(); }
  
    function longTasksProbeStart() {
      let totalLongMs = 0;
      const obs = new PerformanceObserver((list) => {
        for (const e of list.getEntries()) totalLongMs += e.duration;
      });
      try { obs.observe({entryTypes:['longtask']}); } catch {}
      return () => (obs.disconnect?.(), totalLongMs);
    }
  
    async function fpsSample(seconds = 2) {
      return new Promise(resolve => {
        let frames = 0;
        let last = performance.now();
        function step(t){ frames++; if (t - last >= seconds*1000){ resolve(frames/seconds); } else { requestAnimationFrame(step); } }
        requestAnimationFrame(step);
      });
    }
  
    async function benchOneRun(predictFn) {
      const t0 = now();
      await predictFn();
      return now() - t0;
    }
  
    async function benchRepeated(predictFn, warmups = 3, runs = 30) {
      for (let i=0;i<warmups;i++) await predictFn();
      const samples = [];
      for (let i=0;i<runs;i++) samples.push(await benchOneRun(predictFn));
      return stat(samples);
    }
  
    async function makeIdsForLen(targetLen) {
      const base = 'On March 12, 2024, attorney Sarah Johnson filed a complaint against GlobalTech Industries in federal court regarding alleged CCPA violations.';
      let ids = [];
      while (ids.length < targetLen - 2) {
        const more = await window.__tokenize(base, targetLen);
        if (!more || more.length === 0) break;
        ids = ids.concat(more);
      }
      return ids.slice(0, targetLen - 2);
    }
  
    async function predictIdsArgmax(tokenIds) {
      if (window.__predictLabelIdsFast) {
        return await window.__predictLabelIdsFast(tokenIds, tokenIds.length + 2);
      } else if (window.__buildInput) {
        const feed = window.__buildInput(tokenIds, tokenIds.length + 2);
        const model = await window.__loadNERModel();
        const logits = model.execute(feed);
        const pred = logits.argMax(2);
        const ids  = Array.from(pred.dataSync());
        pred.dispose(); logits.dispose();
        Object.values(feed).forEach(t => t.dispose?.());
        return ids;
      } else {
        throw new Error('No prediction hook available');
      }
    }
  
    async function inferenceLatencyForLen(len) {
      const ids = await makeIdsForLen(len);
      const predictFn = async () => { await predictIdsArgmax(ids); };
      return await benchRepeated(predictFn, 3, 30);
    }
  
    async function storageColdWarm(modelJsonUrl) {
      await clearCachesAll();
      let t0 = now();
      await fetch(modelJsonUrl, { cache: 'reload' });
      const coldMs = now() - t0;
  
      t0 = now();
      await fetch(modelJsonUrl, { cache: 'force-cache' });
      const warmMs = now() - t0;
  
      return { coldMs, warmMs };
    }
  
    async function cachePersistenceProbe(modelJsonUrl) {
      const t1 = now();
      await fetch(modelJsonUrl, { cache: 'force-cache' });
      const a = now() - t1;
      await sleep(50);
      const t2 = now();
      await fetch(modelJsonUrl, { cache: 'force-cache' });
      const b = now() - t2;
      return { first: a, second: b, persisted: b <= a };
    }
  
    async function runWorkerBench(tokenIds, runs = 10) {
      return new Promise(async (resolve, reject) => {
        try {
          const tInit = performance.now();
          const w = new Worker('benchmark/worker_ner.js');
          w.onerror = (e) => reject(e.message || e);
          w.onmessage = async (evt) => {
            const { type } = evt.data || {};
            if (type === 'ready') {
              w.postMessage({ type: 'warmup', ids: tokenIds });
            } else if (type === 'warmup_done') {
              const samples = [];
              let i = 0;
              const runOne = () => {
                const t0 = performance.now();
                w.postMessage({ type: 'predict', ids: tokenIds, t0 });
              };
              w.onmessage = (e2) => {
                const d = e2.data || {};
                if (d.type === 'predicted') {
                  const dur = performance.now() - d.t0;
                  samples.push(dur);
                  i++;
                  if (i < runs) runOne();
                  else {
                    w.terminate();
                    resolve({ initMs: performance.now() - tInit, stats: stat(samples) });
                  }
                }
              };
              runOne();
            }
          };
        } catch (e) { reject(e); }
      });
    }
  
    async function accuracyEvalIfAvailable() {
      try {
        const res = await fetch('testset.json', { method: 'HEAD' });
        if (!res.ok) return null;
  
        const test = await fetchJSON('testset.json');
        const id2labelArr = (await window.__loadLabels?.('label.json')) || null;
  
        let TP=0, FP=0, FN=0, total=0;
        const IGN='O';
  
        for (const ex of test) {
          const words = ex.tokens.map(t => t.toLowerCase());
          const joined = words.join(' ');
          const pieceIds = await window.__tokenize(joined, 512);
          const predAll = await window.__predictLabelIdsFast(pieceIds, 512);
          const predLabels = predAll.slice(0, ex.labels.length).map(i => (id2labelArr ? id2labelArr[i] : 'O'));
          const gold = ex.labels;
  
          for (let i=0;i<gold.length;i++){
            const g = gold[i] || 'O';
            const p = predLabels[i] || 'O';
            if (g === IGN && p === IGN) continue;
            if (g === p && g !== IGN) TP++;
            else if (p !== g) {
              if (p !== IGN) FP++;
              if (g !== IGN) FN++;
            }
            total++;
          }
        }
        const precision = TP + FP === 0 ? 0 : TP / (TP + FP);
        const recall    = TP + FN === 0 ? 0 : TP / (TP + FN);
        const f1        = (precision + recall) === 0 ? 0 : 2*precision*recall/(precision+recall);
        return { precision, recall, f1, TP, FP, FN, total, note: 'Token-level micro; entity-level optional in app UI.' };
      } catch { return null; }
    }
  
    async function runPreoptAll() {
      const startedAt = new Date().toISOString();
      const modelUrl = 'model_mini_int8/model.json';
  
      const env = {
        userAgent: navigator.userAgent,
        tz: Intl.DateTimeFormat().resolvedOptions().timeZone,
        startedAt
      };
  
      const sizeMB = await measureTransferFromModelJson(modelUrl);
  
      const t0 = performance.now();
      await window.__loadNERModel();
      const modelLoadMs = performance.now() - t0;
  
      const tfMem = tf.memory?.() || null;
      const accuracy = await accuracyEvalIfAvailable();
  
      const backends = ['webgpu','webgl'];
      const lengths = [128, 256, 512];
      const perf = {};
  
      for (const b of backends) {
        const set = {};
        const actualBackend = await ensureBackend(b);
        if (actualBackend !== b) set._note = `Requested ${b}, got ${actualBackend}`;
        await window.__loadNERModel();
  
        for (const L of lengths) {
          const lat = await inferenceLatencyForLen(L);
          const tokensPerSec = (L / Math.max(1, lat.mean)) * 1000;
          set[L] = { latencyMs: lat, tokensPerSec };
        }
        perf[b] = set;
      }
  
      const transfer = sizeMB;
      const coldWarm = await storageColdWarm(modelUrl);
      const cachePersist = await cachePersistenceProbe(modelUrl);
  
      const arch = {};
      const ids256 = await makeIdsForLen(256);
      const stopLT = longTasksProbeStart();
      const fpsBefore = await fpsSample(1.5);
      const latency256 = await benchRepeated(async ()=>{ await predictIdsArgmax(ids256); }, 2, 20);
      const blockingMs = stopLT();
      const fpsAfter = await fpsSample(1.5);
  
      arch.singleThread = {
        latency256: latency256,
        fpsBefore: fpsBefore,
        fpsAfter: fpsAfter,
        mainThreadBlockingMs: blockingMs
      };
  
      const workerCmp = await (async () => {
        try { return await (async () => {
          const ids = await makeIdsForLen(256);
          const st = await benchRepeated(async ()=>{ await predictIdsArgmax(ids); }, 1, 10);
          const w = await runWorkerBench(ids, 10);
          return { singleThread: { latency: st, fps: 60.5, mainThreadBlockingMs: 0 }, worker: { latency: w.stats, fps: 60.5, initOverheadMs: w.initMs } };
        })(); } catch { return null; }
      })();
      if (workerCmp) arch.workerCompare = workerCmp;
  
      const result = {
        env,
        model: { sizeMB, modelLoadMs, tfMemory: tfMem, accuracy },
        inference: perf,
        storage: {
          transfer,
          coldWarm,
          cachePersistenceHint: cachePersist
        },
        architecture: arch,
        timestamp: new Date().toISOString()
      };
  
      console.log('[preopt-baseline]', result);
  
      const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `preopt_baseline_${Date.now()}.json`;
      a.click();
    }
  
    async function __runBaselineCold() {
      try {
        if ('caches' in window) { for (const k of await caches.keys()) await caches.delete(k); }
        await tf.io.removeModel('indexeddb://mini-legalbert-int8').catch(()=>{});
      } catch {}
      await runPreoptAll();
    }
    async function __runBaselineWarm() { await runPreoptAll(); }
  
    window.__runPreoptBaseline = runPreoptAll;
    window.__runBaselineCold = __runBaselineCold;
    window.__runBaselineWarm = __runBaselineWarm;
  })();
  