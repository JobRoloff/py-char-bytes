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

---

## Test Fixture Derivations

The unit tests in `tests/test_readability.py` lock expected values against the
installed `textstat` implementation, not a hand-written formula clone. These
assertions are still deterministic, but they inherit `textstat` tokenization,
syllable counting, and Dale-Chall difficult-word lookup behavior.

### Fixture Counts

| Fixture | Words | Sentences | Syllables | Letters | Fog Complex Words | Dale-Chall Difficult Words |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `The cat sat on the mat. It was sunny.` | 9 | 2 | 10 | 27 | 0 | 0 |
| `Photosynthesis converts light energy into chemical energy through chlorophyll-mediated reactions.` | 10 | 1 | 29 | 86 | 6 | 7 |
| `Word.` | 1 | 1 | 1 | 4 | 0 | 0 |

These counts can be reproduced with:

```bash
uv run python -c "import textstat; import textstat.backend.metrics._dale_chall_readability_score as dale; text='The cat sat on the mat. It was sunny.'; print(textstat.lexicon_count(text), textstat.sentence_count(text), textstat.syllable_count(text), textstat.letter_count(text), textstat.polysyllabcount(text), dale.count_difficult_words(text, 'en_US', syllable_threshold=0))"
```

### Expected Values

For `The cat sat on the mat. It was sunny.`:

```text
Flesch = 206.835 - 1.015 * (9 / 2) - 84.6 * (10 / 9)
        = 108.2675

Gunning Fog = 0.4 * ((9 / 2) + 100 * (0 / 9))
            = 1.8

Coleman-Liau = 0.058 * ((27 / 9) * 100) - 0.296 * ((2 / 9) * 100) - 15.8
             = -4.977777777777776

Dale-Chall = 0.1579 * (100 * (0 / 9)) + 0.0496 * (9 / 2)
           = 0.2232
```

For `Photosynthesis converts light energy into chemical energy through chlorophyll-mediated reactions.`:

```text
Flesch = 206.835 - 1.015 * (10 / 1) - 84.6 * (29 / 10)
        = -48.655

Gunning Fog = 0.4 * ((10 / 1) + 100 * (6 / 10))
            = 28.0

Coleman-Liau = 0.058 * ((86 / 10) * 100) - 0.296 * ((1 / 10) * 100) - 15.8
             = 31.12

Dale-Chall raw = 0.1579 * (100 * (7 / 10)) + 0.0496 * (10 / 1)
               = 11.549
Dale-Chall adjusted = 11.549 + 3.6365
                    = 15.1855
```

For `Word.`:

```text
Flesch = 206.835 - 1.015 * (1 / 1) - 84.6 * (1 / 1)
        = 121.22

Gunning Fog = 0.4 * ((1 / 1) + 100 * (0 / 1))
            = 0.4

Coleman-Liau = 0.058 * ((4 / 1) * 100) - 0.296 * ((1 / 1) * 100) - 15.8
             = -22.2

Dale-Chall = 0.1579 * (100 * (0 / 1)) + 0.0496 * (1 / 1)
           = 0.0496
```

### Implementation Notes

`textstat` can validly return Flesch Reading Ease scores below 0 or above 100
for very difficult or very simple text, so tests should assert exact reference
behavior rather than clamp Flesch to a `0..100` range.

The standard Coleman-Liau formula is often written with `0.0588 * L`, but the
installed `textstat` implementation currently uses `0.058 * L`. The tests lock
the project to `textstat` behavior because the production service is a wrapper
around that library.

For Gunning Fog, `textstat` counts difficult words using the configured
polysyllable threshold. For Dale-Chall, it calls a separate easy-word-list
lookup with `syllable_threshold=0`, which is why the technical fixture has 6
Fog complex words but 7 Dale-Chall difficult words.
