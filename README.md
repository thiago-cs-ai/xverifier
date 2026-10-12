# xverifier

**Check a judge's verdicts, fix the likely mistakes, and learn how many cases you can decide
automatically at an error level you choose.**

You bring three things for each case:

- a **verdict** from a judge (an LLM or a classifier),
- the true **label**,
- a **representation** of the case (an embedding).

xverifier learns from them when the judge tends to be right and when it tends to be wrong. It then
does two things on new cases:

1. **Keep or flip.** It keeps a verdict it trusts and flips one it doesn't.
2. **Decide or review.** For each error level you ask for (say 10%), it finds a confidence cutoff.
   Cases above the cutoff are decided automatically. Cases below it go to a person.

You control an experiment with one YAML file. This page explains that file.

---

## Quick start

```bash
cd xverifier
pip install -e .
xverifier run examples/synthetic.yaml --out runs/synthetic
```

The synthetic example uses generated rows that ship with the folder. It needs no data, no GPU and no API
key. It prints one table per seed. This table is the command's output:

```
error level   cutoff     decided    errors   error rate
0.25          0.571429   300/300    21       0.070
0.20          0.571429   300/300    21       0.070
0.15          0.571429   300/300    21       0.070
0.10          0.772064   274/300     9       0.033
0.05          review all    0/300    —        —
```

How to read it:

- **error level**: the share of automatic decisions you are willing to get wrong.
- **cutoff**: the confidence a case needs to be decided automatically at that level.
- **decided**: how many test cases cleared the cutoff. A case counts when score >= cutoff - 1e-12.
  That window decides which cases are counted, on the calibration cases that choose the cutoff and on
  the test cases. The upper bound is compared to the error level exactly.
- **errors** and **error rate**: how many of those were wrong.
- **review all**: no cutoff met that level on your calibration data, so every case goes to a person.
  That is a normal outcome. It usually means the calibration set is too small, or the judge's
  mistakes are hard to separate at that level.

Two looser levels can share one cutoff, as 0.25, 0.20 and 0.15 do here. That is expected: a cutoff
good enough for 15% is also good enough for 20% and 25%.

---

## What you bring: row files

A row file is a `.npz` with one row per case:

| Field | Shape | Meaning |
|---|---|---|
| `representation` | (n, d) float32 | The embedding of the case. |
| `label` | (n,) 0 or 1 | The true outcome. |
| `verdict` | (n,) 0 or 1 | What the judge said. |
| `group` | (n,) integer, optional | Cases that belong together, such as near-duplicates. `-1` means the case has no group. |

For IMR appeals, use `0 = Upheld` and `1 = Overturned` for both `label` and `verdict`.

**Why `group` matters.** If two near-identical cases land on different sides of a split, the
experiment looks better than it is. Cases with the same `group` always stay together. A case
with no near-duplicates gets `-1`, and xverifier treats it as a group of its own. If you
leave `group` out, each case is its own group.

### Build your row files from the cleaned IMR dataset

The dataset (`utcompliance1/imr-appeals-cleaned`) gives you two of the four fields. You produce the
other two.

| Field | Who provides it |
|---|---|
| `label` | The dataset: `decision`. |
| `group` | The dataset: `near_dup_group_id`, as it is. |
| `verdict` | **You.** Run your judge on every row. |
| `representation` | **You.** Compute one embedding per row. |

Steps:

1. **Take the rows.** Use the `train` split for the train file and the full `test` split for the test
   file. Leave out `quarantined` and `dropped_exact_duplicates`.
2. **Run your judge on every row of both splits.** Save one verdict per `row_id`. If an answer cannot
   be read as Upheld or Overturned, drop that row.
3. **Compute an embedding for every row.** One row per case, at least 4 columns, float32. Save it with
   its `row_id`.
4. **Map the label.** `{"Upheld": 0, "Overturned": 1}`, for both `label` and `verdict`.
5. **Take the group.** Use `near_dup_group_id` as it is. It is `-1` for a case with no
   near-duplicate, which is what xverifier expects.

6. **Join everything on `row_id`**, never by position, sort the rows by `row_id`, then save. The
   split depends on the row order, so a fixed order keeps your runs repeatable.

   ```python
   np.savez("rows/train.npz", representation=Z, label=y, verdict=yhat, group=group)
   ```

Three things to watch:

- **Use `text`, never `full_text`.** `full_text` contains the reviewer's conclusion.
- **If the judge was trained or chosen on rows you pass in, the error level does not apply.** The
  published DistilBERT model was trained on the `train` split. If it is your judge, cut both files
  from the `test` split.
- **Keep row files out of git.** They are large. Put them in `rows/` and add `rows/` to your
  `.gitignore`, or put them on Hugging Face. `rows/` is not git-ignored by default.

---

## The experiment file

Here is a complete file. Every key is explained below it.

```yaml
study: imr-first-run          # a name for this experiment; recorded in seed_*.json and aggregate.json
seeds: [42, 123, 456, 789, 1024]

frame:
  train: rows/train.npz
  test: rows/test.npz

split:
  mode: files                 # files | draw | roles
  calibration_fraction: 0.274

estimator:
  probe:
    family: linear
    C: 1.0
    max_components: 256
    min_rows_per_verdict: 40
  calibration:
    method: venn_abers
    cross_fit_folds: 5

flip:
  threshold: 0.5

decide:
  error_levels: [0.25, 0.20, 0.15, 0.10, 0.05]
  confidence: 0.95
```

If you are starting out, change only `study`, `frame` and maybe `seeds`. The other values are the
ones the method was validated with.

### `study`

A short name, with no spaces. It is recorded inside each `seed_*.json` and in `aggregate.json`, the
files `xverifier run` writes, so pick one you will recognize later. It is not part of the file names.

### `seeds`

A list of whole numbers. The experiment runs once per seed and then reports the average and the
spread.

A seed controls everything random in a run: which training cases are held out for calibration, and
how the model is fitted. Using several seeds shows how much your result depends on that luck. Five is
a good default. Use one seed for a quick check.

The spread across seeds tells you how sensitive the result is to the split. It is not a margin of
error for new data.

### `frame`

Where your row files are. Which keys you need depends on `split.mode`:

| `split.mode` | Keys under `frame` |
|---|---|
| `files` | `train` and `test` |
| `draw` | `path` (one file) |
| `roles` | `path` (one file that also has a `role` field) |

Optional: add `sha256` values to make sure a run uses the exact file you meant.

```yaml
frame:
  train: rows/train.npz
  test: rows/test.npz
  sha256:
    train: 3f1c…
    test: 9ab2…
```

If a file's hash does not match, the run stops before fitting anything. If you leave a hash out, or
write `null`, that file is not checked. This applies to the whole `sha256` block and to each file in
it.

### `split`

Every run uses three sets of cases:

- **training**: fits the model,
- **calibration**: picks the cutoffs,
- **test**: reports how well it worked. Test cases never influence a cutoff.

`mode` says where those sets come from.

**`mode: files`.** You already have a train file and a test file. This is the usual choice when you
want to compare against other models scored on the same test set.

```yaml
split:
  mode: files
  calibration_fraction: 0.274   # share of training groups held out to pick the cutoffs
```

**`mode: draw`.** You have one file, and xverifier splits it three ways.

```yaml
split:
  mode: draw
  calibration_fraction: 0.274   # share of all groups, taken first, to pick the cutoffs
  evaluation_fraction: 0.082    # share of all groups, taken next, as the test set
```

Both fractions are shares of the whole set of groups. Calibration is the first slice. The test
set is the next slice. Training is what remains. `calibration_fraction` is not applied again to
the groups left after the test set.

**`mode: roles`.** You name the set for each case. Add a `role` field to the row
file with the value `train`, `calibration`, or `test` for each case.

```yaml
split:
  mode: roles
```

In `files` and `draw` modes the split is drawn by group, using the seed, without looking at labels
or verdicts. Group ids are sorted as integers before that draw.

**How big should calibration be?** Big enough to support the error levels you ask for. With the
default settings, a level needs at least this many correct, confidently scored calibration cases
before any cutoff can meet it:

| Error level | Minimum |
|---|---|
| 0.25 | 20 |
| 0.20 | 26 |
| 0.15 | 36 |
| 0.10 | 55 |
| 0.05 | 112 |

The table assumes that the 15 candidate cutoffs xverifier tries, taken from quantiles of the
confidence on the training cases, are all different. If some coincide, the minimums are the same or
lower.

In practice you want many more. If a level keeps printing `review all`, a larger calibration set is
the first thing to try.

### `estimator`

How xverifier learns when the judge is right.

**`probe`.** The model that scores each case.

| Key | Default | What it does |
|---|---|---|
| `family` | `linear` | The kind of model. `linear` is the only one. |
| `C` | `1.0` | Regularization strength. Smaller means a simpler model. |
| `max_components` | `256` | The representation is compressed first, to `max(4, min(n_train // 10, max_components, columns))` dimensions, where `n_train` is the number of cases the model is fitted on. The width is at least 4, even if `max_components` is smaller. |
| `min_rows_per_verdict` | `40` | See below. |

xverifier fits two models: one for cases the judge marked 0, one for cases it marked 1. A judge's
mistakes look different in each. If one side has fewer than `min_rows_per_verdict` training cases,
xverifier uses a single shared model for that side.

**`calibration`.** Turns a raw score into a probability that the judge is right.

| Key | Default | What it does |
|---|---|---|
| `method` | `venn_abers` | The calibration method. It is the only one. |
| `cross_fit_folds` | `5` | Calibration learns from scores on cases the model was not fitted on. This is the number of rounds used to produce them. |

### `flip`

```yaml
flip:
  threshold: 0.5
```

If the probability that the judge is right falls below the threshold, xverifier flips the verdict.
At `0.5`, a verdict is flipped when it is more likely wrong than right. Leave it at `0.5`: that is
the setting the method was validated with.

### `decide`

```yaml
decide:
  error_levels: [0.25, 0.20, 0.15, 0.10, 0.05]
  confidence: 0.95
```

**`error_levels`.** The error rates you would accept among automatic decisions. You get one row of
output per level.

**`confidence`.** How sure the cutoff selection has to be. At `0.95`, the error
rate among automatically decided cases is at most the level you asked for, with 95% confidence, for
one seed's run and for all of that run's levels at once. That statement holds for
new cases that resemble your calibration cases. If the new cases come from a different source or
time period, check again. Rows that share a group are not independent draws. A test file is a finite
sample. Its error rate can sit above the level more often than 5% of the time, by chance, even when
the long-run error rate is at most the level.

---

## Run it

```bash
xverifier run my_experiment.yaml --out runs/my-study
```

What you get in `runs/my-study/`:

| File | Contents |
|---|---|
| `seed_42.json`, one per seed | The table for that seed, plus the cutoffs and the settings used. |
| `aggregate.json` | The average and spread across seeds. |

Each file records a hash of your YAML and of your row files, so a result can be traced
to the experiment that produced it.

A run never overwrites an earlier one: it stops if `--out` already exists. To run again, choose a new
`--out` folder. Changing `study` is not enough.

---

## Rules worth following

1. **Do not pick settings by looking at test results.** If you try several YAML files and keep the
   one with the best test numbers, the error levels no longer mean what they say. Decide your
   settings first. If you need to tune something, tune it on a separate held-out set of cases that
   are in none of your row files. In `files` and `draw` modes the calibration cases are drawn from
   your rows again for each seed, and the 95% refers to that draw.
2. **Keep near-duplicates together.** Fill in `group`.
3. **Do not train the judge on the cases you give to xverifier.** A judge looks better on cases it
   was trained on. xverifier would then learn a picture of the judge that is too optimistic.
4. **Change one thing at a time.** Copy the YAML, change one key, change `study`, and run again. Then
   you know what caused the difference.

---

## When something goes wrong

xverifier checks your YAML and your row files before it fits anything. Its messages say which key or
field is the problem.

| Message is about | Usual cause |
|---|---|
| an unknown key | A typo in the YAML. Key names must match this page exactly. |
| a missing section | A whole block, such as `split`, was left out. |
| values outside 0 and 1 | `label` or `verdict` holds something else, such as text or 2. |
| lengths that differ | The fields in a row file do not have the same number of rows. |
| values that are not finite | The representation contains NaN or infinity. |
| a file hash | The row file changed since you wrote its `sha256` in the YAML. |
| an output that already exists | You are reusing an `--out` folder. Pick a new one. |

---

## Common recipes

**A quick check with one seed**

```yaml
seeds: [42]
```

**Compare two judges on the same cases.** Make two row files that differ only in `verdict` (and in
`representation`, if it comes from the judge). Use two YAML files that differ only in `study` and
`frame`. Keep `seeds` and `split` the same, and make sure both
row files contain exactly the same cases. The split depends on which cases are in the file, so
dropping even one case from one file changes which cases are held out for calibration.

**Stricter error levels only**

```yaml
decide:
  error_levels: [0.10, 0.05, 0.02]
  confidence: 0.95
```

Stricter levels need more calibration cases. Expect `review all` if the calibration set is small.
