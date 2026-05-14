# संस्कृत धातु Model — Project Context for Claude Code
 
## Goal
Train a धातु identifier and derivation tracer on a **Jetson Orin Nano** (8GB unified memory).
This is Model 1 of a planned संस्कृत NLP family.
 
## Hardware Constraints (Jetson Orin Nano Super)
- **Memory:** 8GB LPDDR5 unified (CPU + GPU share this pool)
- **GPU:** 1,024 CUDA cores, 32 Tensor Cores (Ampere architecture)
- **AI Perf:** 67 TOPS
- **Power:** 7W–25W configurable
- **OS:** JetPack 6.2 / Ubuntu 22.04
### Training constraints to always respect:
- Use **FP16 mixed precision** at all times — never FP32
- Use **Adafactor optimizer**, not Adam (4× less memory)
- **Batch size: 4–8**, with gradient accumulation over 64–128 steps
- Model size cap: **~5M parameters** (keeps optimizer states within budget)
- Never load two models simultaneously
- Monitor memory with `tegrastats` — stop if unified memory > 6.5GB
---
 
## Internal Encoding
- All data and model I/O uses **SLP1** (ASCII, bijective, 1 char per phoneme)
- Expose देवनागरी/IAST only at surface (input/output layer)
- Example: `देवानाम् → devAnAm`
---
 
## Model 1: धातु Identifier + Derivation Tracer
 
### Input
A surface संस्कृत word in SLP1. Example: `paWati` (पठति)
 
### Output (structured JSON)
```json
{
  "input": "paWati",
  "dhatu": {
    "root": "paW",
    "devanagari": "पठ्",
    "meaning": "to read/recite",
    "gana": 1,
    "pada": "परस्मैपद",
    "it_markers": []
  },
  "derivation": [
    {"step": 1, "rule": "3.1.68", "operation": "पठ् + अ (शप् विकरण, गण 1)"},
    {"step": 2, "rule": "3.4.78", "operation": "+ ति (लट्, प्रथम पुरुष, एकवचन)"},
    {"step": 3, "rule": "7.3.84", "operation": "गुण of अ → अ (no change)"}
  ],
  "lakara": "लट्",
  "purusha": "प्रथम",
  "vacana": "एकवचन",
  "confidence": 1.0,
  "unambiguous": true
}
```
 
### Architecture
- **Encoder:** Character-level BiLSTM or small Transformer over SLP1 input
- **धातु head:** Classifier → root + गण
- **Rule sequence head:** Sequence decoder → ordered list of पाणिनि rule IDs
- **Target size:** ~5M parameters total
---
 
## पाणिनीय Derivation Engine (build this first)
 
The engine is the **foundation of everything** — it generates training data and validates outputs.
 
### Derivation pipeline (in order):
1. Strip अनुबन्ध (इत् markers) — rules 1.3.2–1.3.9
2. Apply विकरण प्रत्यय (stem suffix by गण) — rules 3.1.x
3. Apply लकार + पुरुष/वचन endings — rules 3.2.x, 3.3.x, 3.4.x
4. Apply आगम and लोप (augments/deletions) — rules 6.x, 7.x
5. Apply गुण/वृद्धि substitutions — rules 7.3.82–7.3.86
6. Apply त्रिपादी phonological rules (internal सन्धि) — rules 8.2–8.4
### विकरण by गण:
| गण | Name | विकरण |
|----|------|--------|
| 1 | भ्वादि | शप् (अ) |
| 2 | अदादि | none (athematic) |
| 3 | जुहोत्यादि | none + reduplication |
| 4 | दिवादि | श्यन् (य) |
| 5 | स्वादि | श्नु (नु) |
| 6 | तुदादि | श (अ, short) |
| 7 | रुधादि | श्नम् (न, infix) |
| 8 | तनादि | उ |
| 9 | क्र्यादि | श्ना (ना/नी) |
| 10 | चुरादि | णिच् (अय) |
 
### लकार to cover (in priority order):
1. लट् (present) — start here
2. लङ् (imperfect)
3. लोट् (imperative)
4. लिङ् (optative)
5. लुट्, लृट् (futures)
6. लिट् (perfect) — hardest, do last
### Known hard cases (encode as metadata on धातु entry, not learned behavior):
- णित्/ञित् roots → वृद्धि (rule 7.2.115)
- सम्प्रसारण roots: वच्, यज्, वह्, ग्रह् etc. (rule 6.1.17)
- सेट्/अनिट्/वेट् distinction for augment इ (rules 7.2.10ff)
- Reduplication in लिट् (rules 6.1.8–6.1.11)
- Defective paradigms (some roots lack certain लकार) — hard constraints
---
 
## Data Generation Strategy
 
### Sources for धातु list:
- धातुपाठ JSON (search GitHub: `dhatupatha.json` or `dhatupatha slp1`)
- ~2,000 roots total
### Synthetic generation formula:
```
~2,000 roots × गण × 9 लकार × 3 पुरुष × 3 वचन × 2 पद
= several million unique surface forms (filter to valid combinations)
```
 
### Training example schema:
```json
{
  "surface": "apaWat",
  "slp1": "apaWat",
  "devanagari": "अपठत्",
  "dhatu": "paW",
  "dhatu_devanagari": "पठ्",
  "gana": 1,
  "lakara": "लङ्",
  "purusha": "प्रथम",
  "vacana": "एकवचन",
  "pada": "परस्मैपद",
  "rules_applied": ["6.4.71", "3.1.68", "3.4.78"],
  "derivation_steps": [
    {"rule": "6.4.71", "before": "पठ्", "after": "अपठ्", "note": "अट् आगम for लङ्"},
    {"rule": "3.1.68", "before": "अपठ्", "after": "अपठ", "note": "शप् विकरण"},
    {"rule": "3.4.78", "before": "अपठ+ति", "after": "अपठत्", "note": "ति→त् in लङ्"}
  ]
}
```
 
### Phase 1 dataset (start small):
- गण 1 (भ्वादि) + लट् only
- ~500K examples
- Validates the full pipeline before expanding to other गण and लकार
---
 
## Training Objective
Multi-task loss:
```
L_total = λ₁ · L_dhatu_root + λ₂ · L_gana + λ₃ · L_rule_sequence + λ₄ · L_lakara
```
- λ values: start at [1.0, 0.5, 1.0, 0.5], tune from there
- Rule sequence loss forces the model to learn *why*, not just *what*
## Evaluation Metric (free, no annotation needed)
For every model prediction, run the predicted rule sequence through the derivation engine
and verify it reproduces the original surface form:
```python
assert derivation_engine.derive(*model.predict(surface)) == surface
```
This is the primary eval — no human labeling required.
 
---
 
## Rule Conflict Resolution (पाणिनीय priority order)
 
When two rules could apply at the same point, resolve in this order:
 
1. **अपवाद beats उत्सर्ग** — exception overrides general rule; longer/more specific match wins
2. **नित्य beats अनित्य** — obligatory rule beats optional rule
3. **अन्तरङ्ग beats बहिरङ्ग** — inner operation applies before outer; resolve stem-internal changes before boundary सन्धि
4. **पर beats पूर्व** — later rule in अष्टाध्यायी wins; use rule index as tiebreaker
### Two-phase processing (त्रिपादी असिद्ध principle):
- **Phase 1:** All main grammar rules (अध्याय 1–8.1)
- **Phase 2:** त्रिपादी phonological rules (8.2–8.4) — these do not feed each other
---
 
## Project Structure
```
sanskrit-dhatu/
├── CLAUDE.md                  ← this file
├── engine/
│   ├── dhatupatha.json        ← machine-readable धातुपाठ (SLP1 + देवनागरी)
│   ├── rules.py               ← पाणिनीय rule implementations
│   ├── derivation.py          ← DerivationEngine class
│   └── vikarana.py            ← विकरण by गण
├── data/
│   ├── generate.py            ← synthetic data generator
│   └── schema.py              ← training example dataclass
├── model/
│   ├── encoder.py             ← character-level encoder
│   ├── heads.py               ← धातु + rule sequence heads
│   └── train.py               ← training loop (Jetson-optimized)
├── eval/
│   └── rederive.py            ← re-derivation verification
└── inference.py               ← production inference entry point
```
 
---
 
## Jetson-Specific Training Setup
 
### Mixed precision + Adafactor:
```python
from transformers import Adafactor
from torch.cuda.amp import autocast, GradScaler
 
optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True)
scaler = GradScaler()
 
# Training step
with autocast():
    loss = model(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
 
### Memory monitoring:
```bash
# Run in a separate terminal while training
tegrastats --interval 1000
```
 
### Recommended batch config:
```python
BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 64  # effective batch = 256
MAX_SEQ_LEN = 32              # SLP1 words are short
```
 
---
 
## Build Order
1. `engine/dhatupatha.json` — get/clean धातुपाठ in SLP1 + देवनागरी
2. `engine/rules.py` — implement गण 1 + लट् rules only first
3. `engine/derivation.py` — DerivationEngine, verify manually on known forms
4. `data/generate.py` — generate ~500K गण-1/लट् pairs
5. `model/` — encoder + heads, keep under 5M params
6. `model/train.py` — Jetson-optimized loop with Adafactor + FP16
7. `eval/rederive.py` — re-derivation eval
8. Expand to all गण + लकार once गण 1/लट् is solid
