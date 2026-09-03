# Readability Metrics

A collection of popular readability formulas used to estimate text complexity and target reading levels.

---

## Flesch Reading Ease

### Purpose
Measures how easy or difficult a text is to read on a scale from 0 to 100, where higher scores indicate easier reading. Scores between 60 and 70 represent standard, plain English (approximately 8th to 9th-grade reading level).

### Mathematical Formula
$$\text{Score} = 206.835 - 1.015 \left( \frac{\text{total words}}{\text{total sentences}} \right) - 84.6 \left( \frac{\text{total syllables}}{\text{total words}} \right)$$

### Limitations
* **Syllable Counting Ambiguity:** Relying strictly on syllable counts can penalize texts with multi-syllable yet common, easy words (e.g., "banana" or "community").
* **Non-Grade Scale:** Unlike other metrics, it outputs an inverted 0–100 index rather than a direct school grade level, requiring manual conversion tables to interpret target audiences.

---

## Gunning Fog

### Purpose
Estimates the formal years of education needed to comprehend a passage on first reading. A score of 12 corresponds to a US high school senior level, while 17+ indicates a college graduate level.

### Mathematical Formula
$$\text{Grade Level} = 0.4 \left[ \left( \frac{\text{total words}}{\text{total sentences}} \right) + 100 \left( \frac{\text{complex words}}{\text{total words}} \right) \right]$$

*(Where "complex words" are defined as words with 3 or more syllables, excluding proper nouns, familiar jargon, and compound words.)*

### Limitations
* **Penalizes Common Long Words:** Short, easily understood compound words or common multi-syllable vocabulary can inflate the "complex word" count unnecessarily.
* **Ignores Structural Clarity:** Evaluates average sentence length rather than syntactic complexity, treating simple multi-clause sentences the same as dense passive constructions.

---

## Coleman Liau

### Purpose
Calculates the US school grade level required to understand text using character counts instead of syllable counts. This makes it computationally fast and highly suitable for automated text analysis pipelines.

### Mathematical Formula
$$\text{Grade Level} = 0.0588 \times L - 0.296 \times S - 15.8$$

* $L$ = Average number of letters per 100 words
* $S$ = Average number of sentences per 100 words

### Limitations
* **Character-Length Bias:** Assumes longer words are inherently harder. Short, abstract words (e.g., "glib", "ax") are treated as simple, while common long words (e.g., "understanding") raise the difficulty score.
* **Sensitivity to Formatting:** Bullet points, code snippets, or missing terminal punctuation heavily distort the sentence count ($S$), drastically skewing the resulting grade.

---

## Dale Chall

### Purpose
Determines reading difficulty by comparing words against a pre-compiled lookup list of ~3,000 everyday words that 4th-grade students reliably comprehend. Any word not on the list is classified as "difficult."

### Mathematical Formula
$$\text{Raw Score} = 0.1579 \left( \frac{\text{difficult words}}{\text{total words}} \times 100 \right) + 0.0496 \left( \frac{\text{total words}}{\text{total sentences}} \right)$$

$$\text{Adjusted Score} = \text{Raw Score} + 3.6365 \quad \text{(if difficult words } > 5\%)$$

### Limitations
* **Outdated Vocabulary List:** The original 3,000-word vocabulary list reflects mid-20th-century usage. Everyday modern terms (e.g., "internet", "website", "email") are incorrectly flagged as "difficult."
* **External Dictionary Dependency:** Cannot be calculated purely through structural properties (like sentence or character length); it requires maintaining a static reference dictionary.