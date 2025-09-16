// web/app.js
let model;
let vocab, tokenizerCfg, id2label, label2id;
let MAX_LEN = 128;
let CLS_ID, SEP_ID, PAD_ID, UNK_ID;
let IDS_NAME, MASK_NAME, TYPE_NAME, OUT_NAME;
let ready = false;
const memo = new Map();

window.__loadNERModel = async function () {
  if (window.__nerModel) return window.__nerModel;
  let m = await tf.loadGraphModel('indexeddb://mini-legalbert-int8').catch(()=>null);
  if (!m) {
    m = await tf.loadGraphModel('model_mini_int8/model.json');
    try { await m.save('indexeddb://mini-legalbert-int8'); } catch {}
  }
  window.__nerModel = m;
  return m;
};

async function loadAll() {
  const status = document.getElementById('status');
  try {
    model = await window.__loadNERModel();

    const vocabJson = await (await fetch('tokenizer/vocab.json')).json();
    tokenizerCfg = await (await fetch('tokenizer/config.json')).json();

    vocab = vocabJson.vocab || {};
    id2label = tokenizerCfg.id2label || {};
    label2id = tokenizerCfg.label2id || {};
    if (typeof tokenizerCfg.max_length === 'number') {
      MAX_LEN = Math.max(8, Math.min(512, tokenizerCfg.max_length));
    }

    const tok = (name) => vocab[name];
    UNK_ID = tok('[UNK]'); CLS_ID = tok('[CLS]'); SEP_ID = tok('[SEP]'); PAD_ID = tok('[PAD]');
    if ([UNK_ID, CLS_ID, SEP_ID, PAD_ID].some(x => typeof x !== 'number')) {
      throw new Error('Missing special tokens in tokenizer/vocab.json');
    }

    IDS_NAME  = (model.inputs.find(i => /input_ids/i.test(i.name)) || model.inputs[0])?.name;
    MASK_NAME = (model.inputs.find(i => /attention_mask/i.test(i.name)) || model.inputs[1])?.name;
    TYPE_NAME = (model.inputs.find(i => /token_type_ids/i.test(i.name)) || null)?.name || null;
    OUT_NAME  = model.outputs[0].name;
    if (!IDS_NAME || !MASK_NAME) throw new Error('Could not find input_ids / attention_mask');

    ready = true;
    if (status) status.textContent = 'Model loaded ✅';
    const outEl = document.getElementById('output');
    if (outEl) outEl.textContent = 'Ready.';
  } catch (e) {
    console.error(e);
    if (status) status.textContent = 'Load error';
    const outErr = document.getElementById('output');
    if (outErr) outErr.textContent = 'Load error: ' + e.message;
  }
}

const isPunc = (ch) => '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'.includes(ch);
function basicTokenizer(text) {
  text = (text || '').toLowerCase();
  const out = []; let cur = '';
  for (const ch of text) {
    if (isPunc(ch) || /\s/.test(ch)) { if (cur) out.push(cur); if (isPunc(ch)) out.push(ch); cur=''; }
    else cur += ch;
  }
  if (cur) out.push(cur);
  return out.filter(t => t.trim().length > 0);
}
function wordpiece(tokens) {
  if (!vocab) throw new Error('Vocab not loaded');
  const wp = [], align = [];
  for (let i=0;i<tokens.length;i++){
    const w=tokens[i];
    if (vocab[w] !== undefined) { align.push([wp.length, wp.length]); wp.push(w); continue; }
    const chars=w.split(''); let start=0; const pieces=[]; let bad=false;
    while(start<chars.length){
      let end=chars.length, found=false;
      while(start<end){
        let sub=chars.slice(start,end).join(''); if(start>0) sub='##'+sub;
        if (vocab[sub] !== undefined) { pieces.push(sub); start=end; found=true; break; }
        end--;
      }
      if(!found){ bad=true; break; }
    }
    if(bad){ align.push([wp.length, wp.length]); wp.push('[UNK]'); }
    else { const s=wp.length; for (const p of pieces) wp.push(p); align.push([s, wp.length-1]); }
  }
  return { pieces: wp, align };
}
const tokensToIds = (pieces) => pieces.map(t => (vocab[t] !== undefined ? vocab[t] : UNK_ID));

window.__tokenize = async function (text, maxLen = MAX_LEN) {
  try {
    const { ids } = await workerTokenize(text, maxLen);
    return ids;
  } catch {
    const words = basicTokenizer(text);
    const { pieces } = wordpiece(words);
    return tokensToIds(pieces).slice(0, maxLen - 2);
  }
};
window.__loadLabels = async function(url='label.json') {
  try { const r = await fetch(url); if (r.ok) return await r.json(); } catch {}
  if (tokenizerCfg?.id2label) {
    const arr = Object.keys(tokenizerCfg.id2label)
      .map(k => [Number(k), tokenizerCfg.id2label[k]])
      .sort((a,b) => a[0]-b[0])
      .map(x => x[1]);
    return arr;
  }
  return null;
};
window.__buildInput = function(tokenIds, maxLen = MAX_LEN) {
  const { inputIds, attentionMask, tokenTypeIds } = makeInputsFromTokenIds(tokenIds, maxLen);
  const dict = {};
  dict[IDS_NAME]  = tf.tensor2d([inputIds], [1, maxLen], 'int32');
  dict[MASK_NAME] = tf.tensor2d([attentionMask], [1, maxLen], 'int32');
  if (TYPE_NAME)  dict[TYPE_NAME] = tf.tensor2d([tokenTypeIds], [1, maxLen], 'int32');
  return dict;
};

function makeInputsFromTokenIds(pieceIds, maxLen = MAX_LEN) {
  const ids = [CLS_ID, ...pieceIds.slice(0, maxLen - 2), SEP_ID];
  const mask = new Array(ids.length).fill(1);
  while (ids.length < maxLen) { ids.push(PAD_ID); mask.push(0); }
  const type = new Array(maxLen).fill(0);
  return { inputIds: ids, attentionMask: mask, tokenTypeIds: type };
}

async function predictLabelIdsFastMain(pieceIds, maxLen = MAX_LEN) {
  const { inputIds, attentionMask, tokenTypeIds } = makeInputsFromTokenIds(pieceIds, maxLen);
  const feed = {};
  feed[IDS_NAME]  = tf.tensor([inputIds], undefined, 'int32');
  feed[MASK_NAME] = tf.tensor([attentionMask], undefined, 'int32');
  if (TYPE_NAME)  feed[TYPE_NAME] = tf.tensor([tokenTypeIds], undefined, 'int32');

  const ids = tf.tidy(() => {
    const logits = model.execute(feed, OUT_NAME);
    const pred = logits.argMax(2);
    const out = Array.from(pred.dataSync());
    pred.dispose(); logits.dispose();
    return out;
  });
  Object.values(feed).forEach(t => t.dispose?.());
  return ids;
}
window.__predictLabelIdsFast = async function(tokenIds, maxLen = MAX_LEN) {
  try { return await workerPredict(tokenIds); }
  catch { return predictLabelIdsFastMain(tokenIds, maxLen); }
};

let nerWorker = null;
function ensureWorker() {
  if (nerWorker) return nerWorker;
  if (!window.Worker) return null;
  nerWorker = new Worker('benchmarks/worker_ner.js'); // NOTE: folder name is "benchmarks"
  nerWorker.addEventListener('error', (e)=>console.warn('worker error', e.message||e));
  return nerWorker;
}
function workerTokenize(text, maxLen=MAX_LEN) {
  return new Promise((resolve, reject) => {
    const w = ensureWorker();
    if (!w) return reject(new Error('Web Worker not supported'));
    const onMsg = (e) => {
      const d = e.data || {};
      if (d.type === 'ids') { w.removeEventListener('message', onMsg); resolve(d); }
      else if (d.type === 'error') { w.removeEventListener('message', onMsg); reject(new Error(d.message)); }
    };
    w.addEventListener('message', onMsg);
    w.postMessage({ type: 'tokenize', text, maxLen });
  });
}
function workerPredict(ids) {
  return new Promise((resolve, reject) => {
    const w = ensureWorker();
    if (!w) return reject(new Error('Web Worker not supported'));
    const onMsg = (e) => {
      const d = e.data || {};
      if (d.type === 'predicted') { w.removeEventListener('message', onMsg); resolve(d.predIds); }
      else if (d.type === 'error') { w.removeEventListener('message', onMsg); reject(new Error(d.message)); }
    };
    w.addEventListener('message', onMsg);
    w.postMessage({ type: 'predict', ids });
  });
}

function mergeBIO(results, confThreshold = 0.6) {
  const spans = []; let cur = null; const flush = () => { if (cur) { spans.push(cur); cur = null; } };
  for (const r of results) {
    const lab = r.label || 'O';
    const conf = typeof r.conf === 'number' ? r.conf : 1.0;
    if (lab === 'O') { flush(); continue; }
    const isB = lab.startsWith('B-'); const isI = lab.startsWith('I-'); const type = lab.slice(2);
    if (isB || !cur || (isI && cur.type !== type)) { flush(); cur = { type, tokens: [r.word], confs: [conf] }; }
    else if (isI && cur && cur.type === type) { cur.tokens.push(r.word); cur.confs.push(conf); }
    else { flush(); }
  }
  flush();
  return spans.map(s => ({ ...s, text: s.tokens.join(' '), conf: s.confs.reduce((a,b)=>a+b,0)/s.confs.length }))
              .filter(s => s.conf >= confThreshold);
}
function render(results) {
  const out = document.getElementById('output');
  if (!results || results.length === 0) { out.textContent = 'No tokens.'; return; }
  const tokenLine = results.map(r => `${r.word}/{${r.label}}`).join(' ');
  const spans = mergeBIO(results, 0.60);
  const list = spans.length ? spans.map(s => `${s.text}  [${s.type}]  ${(s.conf*100).toFixed(1)}%`).join('\n')
                            : 'No confident entities.';
  out.textContent = tokenLine + '\n\nEntities:\n' + list;
}

document.getElementById('run')?.addEventListener('click', async () => {
  const text = document.getElementById('input')?.value?.trim() || '';
  if (!text) return;
  try {
    if (!ready) throw new Error('Model not ready yet');
    const key = text.toLowerCase();
    if (memo.has(key)) return render(memo.get(key));

    let words, align, pieceIds, predIds;
    try {
      const tok = await workerTokenize(key, MAX_LEN);
      ({ words, align } = tok);
      pieceIds = tok.ids;
      predIds = await workerPredict(pieceIds);
    } catch {
      words = basicTokenizer(key);
      const wp = wordpiece(words);
      align = wp.align;
      pieceIds = tokensToIds(wp.pieces).slice(0, MAX_LEN - 2);
      predIds = await predictLabelIdsFastMain(pieceIds, MAX_LEN);
    }

    const results = [];
    for (let wi = 0; wi < words.length; wi++) {
      const [pStart] = align[wi];
      const seqIdx = 1 + pStart;
      if (seqIdx >= predIds.length) break;
      const lab = id2label[String(predIds[seqIdx])] || 'O';
      results.push({ word: words[wi], label: lab });
    }
    memo.set(key, results);
    render(results);
  } catch (e) {
    console.error(e);
    const out = document.getElementById('output'); if (out) out.textContent = 'Inference error: ' + e.message;
  }
});

window.clearModelCaches = async function () {
  const keys = ['indexeddb://mini-legalbert-int8','indexeddb://legalbert','indexeddb://legalbert-fp16'];
  for (const k of keys) {
    try { await tf.io.removeModel(k); console.log('Removed', k); }
    catch (e) { console.warn('Skip', k, e?.message||e); }
  }
  alert('Model caches cleared. Reload the page!');
};

loadAll().then(async () => {
  try {
    const m = await window.__loadNERModel();
    const ids = await window.__tokenize("warmup warmup warmup", 128);
    await tf.tidy(() => m.execute(window.__buildInput(ids, 128)));
    console.log('Warm-up done');
  } catch (e) { console.warn('Warm-up skipped:', e?.message || e); }
});
