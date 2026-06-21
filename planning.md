# TakeMeter — Planning Document

## 1. Community

I chose r/contentcreation and r/ContentCreators because these communities are full of active creators sharing opinions, strategies, and personal stories about building an audience. The discourse varies significantly in quality — some posts make structured arguments about platform trends or strategy, while others are emotional reactions or unsupported claims. This is also a space I personally care about, which means I have strong intuitions for what good vs. weak discourse looks like here. That makes it a good fit for a classification task where annotation quality matters.

---

## 2. Labels

**`analysis`**
The post makes a structured argument about content creation strategy, platform behavior, or the industry, with specific reasoning or evidence behind the claim. The point could stand on its own even if you removed the author's personal feelings.

- Example 1: *"Starting from zero is slower than it was in 2018 — the feedback loops are longer and algorithms need more data before pushing your content."*
- Example 2: *"The best content strategy isn't a single format — use short-form to capture attention and long-form on owned channels to convert it."*

---

**`hot_take`**
A bold, confident opinion about content creation stated with little or no supporting argument. The claim might be true, but the post asserts rather than reasons.

- Example 1: *"Content creation is the hardest business model."*
- Example 2: *"Content Creation is Dead."*

---

**`experience`**
A personal story or account from someone's creator journey. The post is primarily about what happened to them, not making a general argument.

- Example 1: *"Quit my corporate job 8 months ago to do content full time — here's the honest reality."*
- Example 2: *"I wasted 3 months creating content so you don't have to."*

---

## 3. Hard Edge Cases

The hardest case is a post that shares a personal story but also makes a broader claim others can learn from.

**Decision rule:** If the post's primary purpose is to make a general argument, label it `analysis`. If it's primarily recounting what happened to them, label it `experience`. A post that's 80% storytelling with a takeaway at the end = `experience`. A post that opens with a brief story but spends most of its length arguing a general point = `analysis`.

**Example borderline post:** *"I treated content creation like compound interest for 2 years — here's the honest result."* This is primarily a personal account, so → `experience`.

**Posts to skip entirely:** Pure question/advice-request posts (e.g., "How do I start content creation?") don't fit any label cleanly and will be excluded during data collection.

---

## 4. Data Collection Plan

- **Source:** r/contentcreation and r/ContentCreators (public posts only)
- **Method:** Manual collection — read and copy posts into a CSV
- **Target:** ~70 examples per label (~210 total)
- **Columns:** `text`, `label`, `notes` (for difficult cases)
- **Imbalance check:** After every 50 examples, count per-label totals. If any label exceeds 70% of the dataset, stop collecting that label and focus on the underrepresented ones.
- **Split:** Single CSV file — the Colab notebook handles the 70/15/15 train/validation/test split automatically.

---

## 5. Evaluation Metrics

I'll report:
- **Overall accuracy** for both the fine-tuned model and the zero-shot baseline
- **Per-class F1** for each of the three labels
- **Confusion matrix** showing which labels get confused and in which direction

Accuracy alone isn't enough because a model can score high by always predicting the majority class. F1 penalizes models that ignore underrepresented labels by combining precision (how often the model is right when it predicts a label) and recall (how many real examples of that label the model actually catches). A high F1 on all three labels means the model has genuinely learned all three distinctions.

---

## 6. Definition of Success

A per-class F1 of **0.65 or higher on all three labels** would make this classifier genuinely useful. Below that threshold, the model is too unreliable to trust on unseen posts. Given that this is a subjective task trained on only ~200 examples, expecting 0.90+ would be unrealistic — 0.65+ means the model meaningfully outperforms chance (which sits around 0.33 for a 3-class task) and is consistent enough to be deployed in a real community tool.

---

## 7. AI Tool Plan

I'll use AI tools in three specific places during this project:

**Label stress-testing (before annotation):**
I'll give Claude my three label definitions and the edge case decision rule, and ask it to generate 10 posts that sit at the boundary between labels. If I can't classify them cleanly using my own rules, I'll tighten the definitions before annotating 200 examples.

**Annotation assistance (during data collection):**
I may use an LLM to pre-label batches of posts using my definitions, then review and correct every single one myself. I won't accept any pre-assigned label without reading the post. All AI-assisted labels will be disclosed in the AI usage section of the README.

**Failure analysis (after evaluation):**
After getting my list of wrong predictions from the fine-tuned model, I'll paste them into Claude and ask it to identify patterns — e.g., "do these errors share common structure, length, or topic?" I'll then verify those patterns myself by re-reading the examples before writing them up in the evaluation report.

---

*Last updated: Milestone 1–2 complete. To be updated before any stretch features.*
