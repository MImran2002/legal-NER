## Project Overview

The project is part of the bigger research on building optimization and pruning of ML models on the front-end for **fine-tuned Hugging Face model**  
The model is **Legal-bert-base-uncased** converted 
into a **browser-ready TensorFlow.js model**.  

This allows you to run advanced NLP directly in the browser — no server required — enabling **client-side inference** for privacy-sensitive domains like law, healthcare, or finance.

---

## 🔄 Conversion Pipeline

The export process works in four main steps:

1. **Load a Hugging Face model**
   - Uses `TFAutoModelForTokenClassification` to bring in a pre-trained NER model.
   - Reads the model configuration (`AutoConfig`) from the Hugging Face directory.

2. **Run a dummy forward pass**
   - Creates fake input tensors (`input_ids`, `attention_mask`, `token_type_ids`).
   - Runs to warm up the model.

3. **Save as TensorFlow `SavedModel`**
   - Exports the model into TensorFlow’s standard `SavedModel` format.
   - This format is widely supported and is the input format for the TF.js converter.
   - The SavedModel includes both:
     - The computation graph
     - The trained weights

4. **Convert to TensorFlow.js format**
   - Calls `tfjs.converters.convert_tf_saved_model`.
   - Produces:
     - `model.json` → the model architecture + metadata
     - `group1-shard*.bin` → weight shards 

---

## Why This Matters

- **Browser-ready AI** → No need for a Python backend.
- **Privacy** → Legal text never leaves the client machine.
- **Performance** → Inference runs on GPU via WebGL/WebGPU in modern browsers.
- **Portability** → Model can be dropped into any web app, desktop app (via Electron), or hybrid mobile app.

---

