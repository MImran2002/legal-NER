# Legal NER – TensorFlow.js Export Pipeline

This project demonstrates how to convert a fine-tuned Hugging Face model
(e.g. BERT for Named-Entity Recognition in legal text) into a browser-ready
TensorFlow.js model.

The pipeline:
1. Loads a Hugging Face model (`TFAutoModelForTokenClassification`)
2. Runs a dummy forward pass to build the graph
3. Saves the model as a TensorFlow `SavedModel`
4. Converts the `SavedModel` to TensorFlow.js format (`model.json` + shard files)

---

## 📦 Requirements

Install dependencies:

```bash
pip install tensorflow tensorflow-text tensorflowjs transformers numpy
