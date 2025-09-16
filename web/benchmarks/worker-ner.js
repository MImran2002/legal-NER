// web/benchmark/worker_ner.js
(function () {
    try { importScripts('../lib/tf.min.js'); }
    catch { importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js'); }
  
    const MODEL_JSON = '../model_mini_int8/model.json';
    const VOCAB_URL  = '../tokenizer/vocab.json';
    const CFG_URL    = '../tokenizer/config.json';
    const IDB_KEY    = 'indexeddb://mini-legalbert-int8';
    const MAX_LEN_DEFAULT = 128;
  
    let model = null;
    let vocab = null, id2label = null, label2id = null;
    let CLS_ID, SEP_ID, PAD_ID, UNK_ID;
    let IDS_NAME, MASK_NAME, TYPE_NAME, OUT_NAME;
    let MAX_LEN = MAX_LEN_DEFAULT;
  
    const isPunc = (ch) => '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'.includes(ch);
    function basicTokenizer(text) {
      const out = []; let cur = '';
      text = (text||'').toLowerCase();
      for (const ch of text) {
        if (isPunc(ch) || /\s/.test(ch)) { if (cur) out.push(cur); if (isPunc(ch)) out.push(ch); cur=''; }
        else cur += ch;
      }
      if (cur) out.push(cur);
      return out.filter(t => t.trim().length>0);
    }
    function wordpiece(tokens) {
      if (!vocab) throw new Error('vocab not loaded');
      const wp = []; const align = [];
      for (let i=0;i<tokens.length;i++){
        const w = tokens[i];
        if (vocab[w] !== undefined) { align.push([wp.length, wp.length]); wp.push(w); continue; }
        const chars = w.split(''); let start=0; const pieces=[]; let bad=false;
        while (start < chars.length) {
          let end = chars.length; let found=false;
          while (start < end) {
            let sub = chars.slice(start, end).join(''); if (start>0) sub = '##'+sub;
            if (vocab[sub] !== undefined) { pieces.push(sub); start=end; found=true; break; }
            end--;
          }
          if (!found) { bad=true; break; }
        }
        if (bad) { align.push([wp.length, wp.length]); wp.push('[UNK]'); }
        else { const s=wp.length; for (const p of pieces) wp.push(p); align.push([s, wp.length-1]); }
      }
      return { pieces: wp, align };
    }
    const tokensToIds = (pieces) => pieces.map(p => (vocab[p] !== undefined ? vocab[p] : UNK_ID));
  
    function makeInputsFromIds(pieceIds, maxLen=MAX_LEN) {
      const ids = [CLS_ID, ...pieceIds.slice(0, maxLen-2), SEP_ID];
      const mask = new Array(ids.length).fill(1);
      while (ids.length < maxLen) { ids.push(PAD_ID); mask.push(0); }
      const type = new Array(maxLen).fill(0);
      return { ids, mask, type };
    }
  
    async function ensureModel() {
      if (model) return model;
      try { await tf.setBackend('webgl'); } catch { try { await tf.setBackend('cpu'); } catch {} }
      await tf.ready();
  
      const vocabJson = await (await fetch(VOCAB_URL)).json();
      const cfg = await (await fetch(CFG_URL)).json();
      vocab = vocabJson.vocab || {};
      id2label = cfg.id2label || {};
      label2id = cfg.label2id || {};
      if (typeof cfg.max_length === 'number') MAX_LEN = Math.max(8, Math.min(512, cfg.max_length));
  
      const tok = (n)=>vocab[n];
      UNK_ID = tok('[UNK]'); CLS_ID = tok('[CLS]'); SEP_ID = tok('[SEP]'); PAD_ID = tok('[PAD]');
      if ([UNK_ID, CLS_ID, SEP_ID, PAD_ID].some(x => typeof x !== 'number')) throw new Error('special tokens missing');
  
      model = await tf.loadGraphModel(IDB_KEY).catch(()=>null);
      if (!model) {
        model = await tf.loadGraphModel(MODEL_JSON);
        try { await model.save(IDB_KEY); } catch {}
      }
  
      IDS_NAME  = (model.inputs.find(i => /input_ids/i.test(i.name)) || model.inputs[0])?.name;
      MASK_NAME = (model.inputs.find(i => /attention_mask/i.test(i.name)) || model.inputs[1])?.name;
      TYPE_NAME = (model.inputs.find(i => /token_type_ids/i.test(i.name)) || null)?.name || null;
      OUT_NAME  = model.outputs[0].name;
      if (!IDS_NAME || !MASK_NAME) throw new Error('input_ids/attention_mask not found');
      return model;
    }
  
    async function predictIdsArgmax(pieceIds) {
      const m = await ensureModel();
      const { ids, mask, type } = makeInputsFromIds(pieceIds, MAX_LEN);
      const feed = {};
      feed[IDS_NAME]  = tf.tensor([ids],  undefined, 'int32');
      feed[MASK_NAME] = tf.tensor([mask], undefined, 'int32');
      if (TYPE_NAME)  feed[TYPE_NAME] = tf.tensor([type], undefined, 'int32');
  
      const out = tf.tidy(() => {
        const logits = m.execute(feed, OUT_NAME);
        const pred = logits.argMax(2);
        const arr = Array.from(pred.dataSync());
        pred.dispose(); logits.dispose();
        return arr;
      });
      Object.values(feed).forEach(t => t.dispose?.());
      return out;
    }
  
    self.onmessage = async (e) => {
      const msg = e.data || {};
      try {
        if (msg.type === 'warmup') {
          await ensureModel();
          const ids = Array.isArray(msg.ids) ? msg.ids : new Array(Math.max(8, MAX_LEN-2)).fill(1);
          await predictIdsArgmax(ids);
          self.postMessage({ type: 'warmup_done' });
        } else if (msg.type === 'predict') {
          const predIds = await predictIdsArgmax(msg.ids || []);
          self.postMessage({ type: 'predicted', predIds, t0: msg.t0 });
        } else if (msg.type === 'tokenize') {
          await ensureModel();
          const words = basicTokenizer(msg.text || '');
          const { pieces, align } = wordpiece(words);
          const ids = tokensToIds(pieces).slice(0, (msg.maxLen || MAX_LEN) - 2);
          self.postMessage({ type: 'ids', ids, align, words });
        }
      } catch (err) {
        self.postMessage({ type: 'error', message: String(err?.message || err) });
      }
    };
  
    (async () => {
      try { await ensureModel(); } catch {}
      self.postMessage({ type: 'ready' });
    })();
  })();
  