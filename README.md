# TakeMeter

A fine-tuned text classifier that labels posts from online content creator communities as `analysis`, `hot_take`, or `experience`. Built as a project for AI201.

---

## Community

Reddit communities: r/contentcreation, r/ContentCreators, r/creators, r/NewTubers.

These communities mix structured arguments about platform strategy, bold unsupported opinions, and personal creator stories — enough variation to make classification non-trivial, and a space I personally know well enough to annotate reliably.

---

## Labels

**`analysis`** — the post makes a structured argument about content creation strategy, platform behavior, or the industry, with specific reasoning or evidence. The point could stand on its own even without the author's personal feelings.

> *"Starting from zero is slower than it was in 2018 — the feedback loops are longer and algorithms need more data before pushing your content."*
>
> *"The best content strategy isn't a single format — use short-form to capture attention and long-form on owned channels to convert it."*

**`hot_take`** — a bold, confident opinion about content creation stated with little or no supporting argument. The claim might be true, but the post asserts rather than reasons.

> *"Content creation is the hardest business model."*
>
> *"Content Creation is Dead."*

**`experience`** — a personal story or account from someone's creator journey. The post is primarily about what happened to them, not making a general argument.

> *"Quit my corporate job 8 months ago to do content full time — here's the honest reality."*
>
> *"I wasted 3 months creating content so you don't have to."*

**Edge case rule:** if a post shares a personal story AND makes a broader argument, ask what the PRIMARY purpose is. Mostly storytelling → `experience`. Mostly arguing a general point → `analysis`.

---

## Dataset

| Split | Examples |
|-------|----------|
| Train | 232 |
| Validation | 50 |
| Test | 50 |
| **Total** | **332** |

Label distribution across the full dataset:

| Label | Count | % |
|-------|-------|---|
| experience | 157 | 47% |
| analysis | 103 | 31% |
| hot_take | 72 | 22% |

Posts were collected from Reddit via public RSS feeds (no API key required) across multiple subreddits and sort orders (top/month, top/year, top/all). Question and help-request posts were filtered out using keyword patterns. Each post was pre-labeled using Groq (llama-3.1-8b-instant) and then reviewed and corrected manually before use.

### Difficult Annotation Cases

**Case 1:** *"I treated content creation like compound interest for 2 years — here's the honest result."*
This post opens with a metaphor (compound interest → analysis framing) but spends nearly all of its length recounting what happened month by month. The takeaway is personal, not universal. Labeled **`experience`** because the primary purpose is recounting, not arguing.

**Case 2:** *"Nobody talks about how platform algorithms actively punish consistency — here's what I noticed over 18 months."*
The title reads like a `hot_take` (bold assertion, no hedging), but the body provides channel data and a timeline of observations. The evidence is personal but structured as an argument. Labeled **`analysis`** because the post is building a case, not just asserting.

**Case 3:** *"Never doing critiques again."*
Could be `hot_take` (bold declarative statement) or `experience` (the post tells a story about a bad critique that led to this decision). The body is narrative, but the title is a standalone assertion. Labeled **`hot_take`** because the primary unit — the title and opening claim — is an unqualified declaration. The story is supporting context, not the point.

---

## Models

**Zero-shot baseline:** Groq API (`llama-3.1-8b-instant`) with the following system prompt:

```
You are a text classifier for an online content creator community.
Classify each post into exactly one of these three labels:

analysis — the post makes a structured argument about content creation strategy,
platform behavior, or the industry, with specific reasoning or evidence.

hot_take — a bold, confident opinion about content creation stated with little
or no supporting argument. The claim might be true, but the post asserts rather
than reasons.

experience — a personal story or account from someone's creator journey.
The post is primarily about what happened to them, not making a general argument.

Respond with ONLY one word: analysis, hot_take, or experience. No explanation.
```

Each of the 50 test posts was sent as a user message with the prefix `"Classify this post:"`. Results were collected in the notebook (Section 5) with a 1-second delay between requests.

**Fine-tuned model:** `distilbert-base-uncased` with a 3-class classification head, fine-tuned using the HuggingFace `Trainer` API on a T4 GPU in Google Colab.

Key training decisions:

- **3 epochs:** the notebook comments note that more epochs risk overfitting on ~200 training examples. I kept the default because validation accuracy was already plateauing by epoch 2, and increasing epochs on a small dataset typically hurts generalization rather than helping it.
- **Learning rate 2e-5:** standard starting point for fine-tuning BERT-family models. Lower values are more stable on small datasets and reduce the risk of catastrophic forgetting of the pre-trained weights.
- **Batch size 16:** chosen to fit the T4 GPU without out-of-memory errors while keeping gradient updates reasonably stable.

---

## Results

| Model | Accuracy |
|-------|----------|
| Zero-shot baseline (Groq) | **0.600** |
| Fine-tuned DistilBERT | **0.560** |

Fine-tuning produced a **regression of 0.040** relative to the baseline.

### Per-class F1 scores

| Label | Baseline F1 | Fine-tuned F1 |
|-------|-------------|---------------|
| analysis | 0.58 | 0.45 |
| hot_take | 0.43 | **0.00** |
| experience | 0.65 | 0.69 |

### Confusion matrix (fine-tuned model)

![Confusion Matrix](confusion_matrix.png)

The matrix shows the core failure clearly: the fine-tuned model predicted **zero hot_takes** across all 50 test examples. Every hot_take in the test set was classified as either `analysis` (1) or `experience` (10). The model also mislabeled 11 of 16 `analysis` posts as `experience`. It essentially collapsed to predicting `experience` for anything ambiguous.

---

## Error Analysis

**Why the fine-tuned model failed on hot_take:**

`hot_take` was the smallest class at 22% of the dataset (~50 training examples after the 70/15/15 split). DistilBERT didn't see enough examples of the class to learn its surface features. Instead, it defaulted to `experience` — the majority class — whenever it was uncertain. This is a classic symptom of class imbalance in small training sets.

The zero-shot baseline handled `hot_take` much better (F1=0.43) because Groq could reason from the label definition in the prompt — "bold opinion with little supporting argument" — without needing to have seen labeled examples.

**Three representative errors from the fine-tuned model:**

1. *"Never doing critiques again"* — True: `hot_take`, Predicted: `experience`. The post opens with a personal story about a bad critique experience, but the title alone is a bold assertion. The model latched onto the narrative structure and missed the declarative claim.

2. A post asserting that a specific platform strategy is universally wrong — True: `analysis`, Predicted: `experience`. The argument was framed in first person ("I've found that…"), which likely triggered the experience pattern even though the body was building a structured case.

3. A post comparing monetization strategies across platforms with data — True: `analysis`, Predicted: `experience`. The model struggled with analytical posts written in an informal, personal register.

**Pattern:** the fine-tuned model over-relied on surface cues (first-person framing, narrative structure) rather than learning the underlying distinction between asserting vs. reasoning vs. recounting. Any post written in a personal voice got pulled toward `experience` regardless of its actual purpose.

---

## Definition of Success

I defined success as a per-class F1 of **0.65 or higher on all three labels**. The fine-tuned model did not meet this bar — it reached 0.69 on `experience` but failed entirely on `hot_take`. The baseline came closer overall (macro F1: 0.55 vs. 0.38) but also fell short of 0.65 on `analysis` and `hot_take`.

The most likely path to improvement: collect more `hot_take` examples to balance the dataset (targeting 100+ per class), and possibly weight the loss function to penalize mistakes on minority classes during training.

---

## AI Tool Usage and Spec Reflection

### AI Tool Usage

**Pre-labeling (annotation assistance):** Groq (`llama-3.1-8b-instant`) was used to generate a first-pass label for each of the 332 posts using the label definitions as a system prompt. I reviewed and corrected every label before using the CSV for training. The model consistently over-labeled posts as `analysis` — it treated any post that mentioned strategy or platforms as analytical, even when the post was just a personal story with a strategic-sounding title. I overrode roughly 30–40% of the pre-assigned `analysis` labels, reclassifying them as `experience` after reading the full post body.

**Zero-shot baseline:** The same Groq model was used as the zero-shot comparator in Section 5 of the notebook. I initially ran the baseline with `llama-3.3-70b-versatile`, which hit its daily token limit mid-run (100k tokens/day). I switched to `llama-3.1-8b-instant` for the full run — this was a practical override, not a design choice, but it means the baseline reflects a smaller model than originally planned.

**Error pattern analysis:** Claude (Anthropic) was used to help identify patterns in the wrong predictions after evaluation. I verified each suggested pattern by re-reading the actual error examples before writing them into this report.

### Spec Reflection

**Where the spec helped:** The edge case decision rule — "if a post shares a story AND makes a broader argument, ask what the PRIMARY purpose is" — turned out to be the most practically useful part of the spec. I applied it on nearly every borderline annotation. Without it, I would have labeled inconsistently across the story-with-takeaway cases that made up roughly a third of the dataset.

**Where implementation diverged:** The data collection plan in `planning.md` described manual collection — reading and copying posts into a CSV by hand. In practice, I automated the entire collection and pre-labeling pipeline using Reddit RSS feeds and Groq. The manual review step remained, but the collection itself was fully automated. The divergence happened because manual collection at the scale needed (300+ posts) was impractical given time constraints, and an automated approach with human review produced higher-quality annotations than manual collection alone would have.

---

## Files

```
README.md                          — this file
planning.md                        — label definitions, data plan, success criteria
takemeter_dataset.csv              — final labeled dataset (332 examples)
collect_and_prelabel.py            — data collection + pre-labeling script
confusion_matrix.png               — confusion matrix for fine-tuned model
evaluation_results.json            — accuracy and improvement numbers
Copy_of_ai201_project3_takemeter_starter_clean.ipynb  — training notebook
```