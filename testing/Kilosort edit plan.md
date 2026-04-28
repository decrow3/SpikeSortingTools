Kilosort plan

## Current status

This plan is now partly implemented, not just proposed.

* The baseline `spikeinterface` environment was restored to released `kilosort==4.0.27` for reproducible mainline analysis.
* A separate `spikeinterface-claimmask` environment now imports the editable extracted 4.0.27 source tree at:
    `/home/huklaban5/Documents/kilosort-src-4.0.27/kilosort-4.0.27/kilosort/`
* The first-pass claim-mask patch is already applied there:
    * `cross_peel_claim_ms` added to `kilosort/parameters.py`
    * `cross_peel_claim_um` added to `kilosort/parameters.py`
    * `run_matching` patched in `kilosort/template_matching.py`
* SpikeInterface propagation has been verified in the claimmask env:
    * `kilosort.parameters.DEFAULT_SETTINGS` contains both new keys
    * `spikeinterface.sorters.get_default_sorter_params("kilosort4")` contains both new keys
* Local sweep scripts have already been wired to expose two claim-mask runs:
    * `claim_tonly`
    * `claim_spatial`
* First 0302 evaluation is now complete:
    * default within-run duplicate burden: 912 flagged nearby pairs out of 1204 informative pairs, median near-zero fraction 0.189
    * `claim_tonly`: 115/568 flagged, median near-zero fraction 0.043
    * `claim_spatial`: 124/610 flagged, median near-zero fraction 0.044
    * `claim_spatial` currently looks like the best practical intervention on 0302, improving efficiency from 0.185 to 0.275 and `n_well` from 12 to 14 while preserving 32/33 good units

So the next phase is experimental evaluation, not more integration plumbing.

## Repo-specific integration check

This repo does **not** vendor Kilosort. The actual call chain in this environment is:

* [pipeline/sorting.py](/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/pipeline/sorting.py) `sort_ks4(...)` calls `run_sorter("kilosort4", ..., **sorter_params)`.
* For the experiment env, SpikeInterface's Kilosort4 wrapper lives at:
    `/home/huklaban5/anaconda3/envs/spikeinterface-claimmask/lib/python3.12/site-packages/spikeinterface/sorters/external/kilosort4.py`
* For the experiment env, Kilosort is imported from the editable extracted 4.0.27 source tree at:
    `/home/huklaban5/Documents/kilosort-src-4.0.27/kilosort-4.0.27/kilosort/`
* The separate baseline analysis env remains on the released package in:
    `/home/huklaban5/anaconda3/envs/spikeinterface/lib/python3.12/site-packages/kilosort/`

Two implementation consequences matter:

1. The repo wrapper itself is thin. It forwards `sorter_params` straight into SpikeInterface and does not impose its own KS4 parameter schema.
2. SpikeInterface's Kilosort4 wrapper dynamically builds its parameter list from `kilosort.parameters` and then filters runtime settings with:

```python
settings_ks = {k: v for k, v in params.items() if k in DEFAULT_SETTINGS}
```

So if this experiment adds new parameters to the active Kilosort install in `kilosort.parameters` and those keys appear in `DEFAULT_SETTINGS`, they should propagate through this repo **without** any changes to [pipeline/sorting.py](/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/pipeline/sorting.py).

That propagation has now been verified in the `spikeinterface-claimmask` environment.

That means the primary patch target is still Kilosort itself, not this repo. The editable-source route is now the active experiment setup and should remain the only place where claim-mask development happens.

Your current evidence is already strong enough to justify a **minimal, targeted KS4 intervention** rather than only parameter sweeps. The strategic goal should be:

> **preserve matching-pursuit overlap recovery, but block later peels from re-claiming the same event core.**

That is more principled than just forcing `max_peels = 1`, and it matches both the code and the docs.

## Why this is the right intervention

The official parameter definition says `max_peels` is the number of iterations in the matching-pursuit step, and that more iterations may detect more overlapping spikes. ([GitHub][1]) So peeling is not an accident or optional cleanup. It is central to KS4’s intended overlap recovery.

At the same time, the current code for `run_matching` does three things that matter here:

* it computes a score matrix `Cf` and takes the best-scoring template at each timepoint with `Cfmax, imax = torch.max(Cf, 0)`; ([GitHub][2])
* it enforces local temporal exclusivity **within a single peel** using `max_pool1d` over a `2*nt+1` window; ([GitHub][2])
* after accepting spikes, it subtracts them from both the residual voltage `Xres` and the template-response tensor `B`, then continues to the next peel. ([GitHub][2])

So the current implementation already says, in effect:

> one winner per local time neighborhood **within a peel**

but it does **not** obviously say:

> once an event has been claimed, later peels cannot re-fit essentially the same event in the same place.

That is the gap your intervention should target.

There is also a good reason not to rely on existing postprocessing to fix this. `duplicate_spike_ms` is explicitly defined as removing spikes within a short interval **from the same cluster**, not across different clusters. ([GitHub][1]) And the merge code uses CCG-based refractoriness criteria, which means true duplicate-fit pairs across clusters can become exactly the kind of pair the merge stage refuses to merge. ([GitHub][2])

## The intervention I would recommend first

I would implement a **cross-peel spatiotemporal claim mask** inside `run_matching`, with a clean toggle.

The design is:

* let peel 1 run normally;
* once spikes are accepted, mark their **time** and **spatial footprint** as “claimed”;
* on later peels, suppress any candidate that falls back into that same claimed spatiotemporal neighborhood;
* still allow candidates at nearly the same time if they are spatially distinct enough to plausibly be a real overlap.

That gives you a true mechanistic test of your hypothesis:

* if the main problem is residual self-refits, this should collapse the duplicate burden while preserving some overlap recovery;
* if the main problem is elsewhere, this should have a much weaker effect than `max_peels = 1`.

## Exact code touchpoints

### 1. `kilosort/template_matching.py` — primary intervention point

This is the core place to patch.

In the **installed package currently used by this repo**, `run_matching(...)` starts at about line `167`, and it reads `Th`, `nt`, and `max_peels` immediately at about lines `168` to `170`.

The candidate selection happens at about lines `202` to `209`:

* `Cfmax, imax = torch.max(Cf, 0)`
* `Cmax = max_pool1d(...)`
* `cnd1 = Cmax[0,0] > Th**2`
* `cnd2 = torch.abs(Cmax[0,0] - Cfmax) < 1e-9`
* `xs = torch.nonzero(cnd1 * cnd2)`

This is exactly where I would insert the new filter:

* **after** `iX = xs[:,:1]` and `iY = imax[iX]`
* **before** those candidates are written into `st` and subtracted.

In the installed file, that means patching just after about lines `216` to `217`.

The subtraction happens at about lines `230` to `232`:

* `Xres[:, iX[j::n] + tiwave] -= ...`
* `B[:, iX[j::n] + trange] -= ...`

That is where the newly accepted events should also be appended to the claim set for subsequent peels.

### 2. `kilosort/parameters.py` — new toggles

This is where KS4 exposes settings to the GUI and API. The docs explicitly say parameter descriptions are defined there and shown in the GUI. ([Kilosort][3])

In the **installed package currently used by this repo**:

* `nearest_templates` is defined around line `280`
* `max_peels` is defined around line `301`
* `duplicate_spike_ms` is defined around line `398`

I would add the new settings near `max_peels`, because this is part of spike detection / matching, not postprocessing.

## Minimal patch design

I would do this in two stages.

### Stage 1: clean, togglable claim mask

Status: implemented in the editable 4.0.27 source tree used by `spikeinterface-claimmask`.

Add the new settings in `parameters.py` near `max_peels`.

I would **not** add `cross_peel_claim_mode` in the first patch. In the installed Kilosort `parameters.py`, existing parameters are all `int`, `float`, or `bool`; I did not find existing `str`-typed parameters. The safer first implementation is numeric-only:

* `cross_peel_claim_ms`
* `cross_peel_claim_um`

with behavior inferred as:

* `claim_ms == 0` -> disabled
* `claim_ms > 0`, `claim_um == 0` -> time-only
* `claim_ms > 0`, `claim_um > 0` -> time+space

That should also integrate more cleanly with SpikeInterface's dynamic parameter extraction.

Conceptually:

```python
'cross_peel_claim_ms': {
    'gui_name': 'cross-peel claim ms',
    'type': float,
    'min': 0,
    'max': np.inf,
    'exclude': [],
    'default': 0.0,
    'step': 'spike detection',
    'description': """
    Prevent later matching-pursuit passes from fitting a new spike within this
    many ms of an already accepted spike if it is also spatially nearby.
    A value of 0 disables the cross-peel claim rule.
    """
},

'cross_peel_claim_um': {
    'gui_name': 'cross-peel claim um',
    'type': float,
    'min': 0,
    'max': np.inf,
    'exclude': [],
    'default': 0.0,
    'step': 'spike detection',
    'description': """
    Spatial radius used with cross_peel_claim_ms. Candidates within both the
    time and distance thresholds of previously accepted spikes are suppressed
    on later peels.
    """
},
```

### Stage 2: patch `run_matching`

Status: implemented in the editable 4.0.27 source tree used by `spikeinterface-claimmask`.

Inside `run_matching`, compute a spatial summary for each template once, before the peel loop.

The simplest robust choice is the **dominant channel** per template, using the same kind of logic already used elsewhere in the file for template-channel relationships. Because `U` already contains template weights over channels, you can compute:

```python
template_main_chan = torch.argmax((U**2).sum(1), dim=1)
template_x = torch.as_tensor(ops['xc'], device=device)[template_main_chan]
template_y = torch.as_tensor(ops['yc'], device=device)[template_main_chan]
```

This is deliberately simpler than a weighted centroid. For a first test, dominant channel is probably enough, but it is also one of the places where the patch could become too blunt. If the first experiment suppresses plausible true overlaps, this spatial summary is an obvious refinement point.

Then initialize empty claim storage before the peel loop:

```python
claimed_t = []
claimed_x = []
claimed_y = []
```

Then, after `iX` and `iY` are formed but before writing to `st`, filter candidates if `t > 0` and the claim rule is enabled.

Conceptually:

```python
if claim_enabled and t > 0 and len(iX) > 0 and len(claimed_t) > 0:
    # candidate times and template centers
    cand_t = iX[:, 0].float()
    cand_x = template_x[iY[:, 0]]
    cand_y = template_y[iY[:, 0]]

    # previous accepted events
    prev_t = torch.cat(claimed_t)
    prev_x = torch.cat(claimed_x)
    prev_y = torch.cat(claimed_y)

    dt = torch.abs(cand_t[:, None] - prev_t[None, :])

    if claim_um > 0:
        d2 = (cand_x[:, None] - prev_x[None, :])**2 + (cand_y[:, None] - prev_y[None, :])**2
        claimed = (dt <= claim_bins) & (d2 <= claim_um**2)
    else:
        claimed = (dt <= claim_bins)

    keep = ~claimed.any(dim=1)
    iX = iX[keep]
    iY = iY[keep]
```

Then, after the accepted candidates are finalized, append them to the claim storage:

```python
claimed_t.append(iX[:, 0].float())
claimed_x.append(template_x[iY[:, 0]])
claimed_y.append(template_y[iY[:, 0]])
```

That is the entire first intervention.

## Why this is the best first patch

It has several advantages.

First, it is **local**. In this repo's actual call chain, you should only need to touch the installed or forked Kilosort package: `run_matching` and `parameters.py`.

Second, it is **cleanly togglable**. Default values can preserve current behavior exactly, and because SpikeInterface pulls parameters dynamically from Kilosort, those toggles should surface automatically once added correctly.

Third, it is **mechanistically specific**. It tests your hypothesis directly, rather than crudely suppressing all later peels.

Fourth, it preserves the main intended value of matching pursuit: a second spike can still be fit at nearly the same time if it is spatially distinct enough.

## What I would not change first

I would not start by patching the merge code.

The merge logic is downstream and currently uses `acg_threshold` and `ccg_threshold` as part of its acceptance test. ([GitHub][2]) That logic may be doing the right thing given the events it is handed. If the problem begins at detection, it is cleaner to stop creating pathological duplicate pairs than to teach merging to rescue them.

I also would not first change `duplicate_spike_ms`, because both the docs and parameter definitions say it only applies within a cluster. ([GitHub][1]) That is not the central failure mode here.

## Optional second patch, if stage 1 works

If the claim-mask patch helps but seems too blunt, the next refinement would be to require not only temporal and spatial proximity, but also **same-template-family similarity**.

A simple version would be:

* suppress later-peel candidates only if they are within the claim window **and**
* their template waveform correlation to the claimed event’s template exceeds some threshold.

That is more principled, but it requires carrying claimed template ids and computing template-template similarity at runtime or precomputing it. I would only do this if the simpler time+space claim rule suppresses obvious true overlaps.

## Evaluation plan

Status: ready to run. The local sweep scripts already contain claim-mask conditions, so this section is now an execution plan rather than a design sketch.

To test this patch properly, I would compare four conditions on the same short dataset slice:

1. **baseline default KS4**
2. **`max_peels = 1`** diagnostic condition
3. **default `max_peels` + claim mask enabled**
4. **default `max_peels` + claim mask enabled + one modest `Th_learned` adjustment**

The key outcomes should be:

* within-run nearby-pair near-zero-lag burden
* KS4 good-unit count
* total unit count
* median missing-spike metric
* a few manually inspected overlap-rich examples
* debug counts of candidates suppressed per peel

The most important comparison is 2 vs 3.

In the current repo wiring, condition 3 is represented by two concrete runs:

* `claim_tonly`: `cross_peel_claim_ms=0.25`, `cross_peel_claim_um=0.0`
* `claim_spatial`: `cross_peel_claim_ms=0.25`, `cross_peel_claim_um=75.0`

These are intended to answer two separate questions:

* is time-only suppression already enough to collapse the duplicate burden?
* if not, does adding a spatial radius preserve a better overlap/duplicate tradeoff?

If condition 3 approaches condition 2 on duplicate burden while preserving more plausible overlaps or more stable unit counts, that is a very strong result. It would show that the problem is not peeling itself, but unconstrained re-claiming across peels.

I would make the acceptance criteria explicit:

* **Worked**: within-run duplicate burden moves strongly toward `max_peels = 1`, while total unit count and plausible overlap examples are less damaged than under `max_peels = 1`.
* **Too blunt**: duplicate burden falls, but unit count, overlap-rich examples, or spike recovery collapse to essentially the same endpoint as `max_peels = 1`.
* **Weak evidence**: the claim mask has much smaller effect than `max_peels = 1`, suggesting later peels are not re-claiming already-owned event cores in the way hypothesized.

## Strategic caution

One thing to say very explicitly in your notes and commit message:

> This patch does **not** test drift correction.

The docs make clear that `nblocks` governs drift correction and that drift behavior depends on probe geometry and channel count. ([Kilosort][3]) Your intervention is entirely **after** preprocessing and drift correction, inside the matching-pursuit stage. So the interpretation should be:

* this is a test of **residual re-fitting behavior conditional on the existing drift-corrected input**, not a replacement for drift analysis.

## Recommended implementation order

Updated status:

1. Patch Kilosort in a separate editable environment: complete.
2. Add numeric toggles to `parameters.py`: complete.
3. Patch `run_matching` with the simplest time+space claim rule: complete.
4. Wire the repo sweep scripts to launch claim-mask runs: complete.
5. Run the short benchmark slice and compare `default`, `claim_tonly`, `claim_spatial`, and `peel1`: next.
6. Only if the patch helps but is too blunt, add the template-similarity refinement.

I would do it in this order:

1. Patch the active Kilosort package or an editable Kilosort fork, not this repo wrapper.
2. Add the numeric toggles to `parameters.py`.
3. Patch `run_matching` with the simplest time+space claim rule.
4. Add lightweight debug logging inside `run_matching`:

   * candidates before claim filter
   * candidates after claim filter
   * per-peel suppressed count
    * optionally, suppressed counts by template id or template family
5. Run the short peel benchmark slice.
6. Only if the patch helps but is too blunt, add the template-similarity refinement.

Given the current state, steps 1 to 4 should now be treated as fixed infrastructure, not open questions.

## Bottom line

The most strategic minimal intervention is:

> **Add a togglable cross-peel spatiotemporal claim mask in `run_matching`**
> so later peels cannot fit another spike in essentially the same place at essentially the same time as an already accepted event.

The exact primary touchpoints are:

* `kilosort/template_matching.py`

    * `run_matching` around lines `167` to `170` in the installed package for setup
    * candidate selection around lines `202` to `209`
    * `iX` / `iY` formation around lines `216` to `217`
    * subtraction around lines `230` to `232`
* `kilosort/parameters.py`

    * add new spike-detection settings near `max_peels` around line `301`
* `spikeinterface/sorters/external/kilosort4.py`

    * no algorithm patch should be needed here for a first pass, but it is the integration layer to keep in mind
    * `_dynamic_params()` pulls from `kilosort.parameters`
    * `_run_from_folder()` forwards only keys in `DEFAULT_SETTINGS`


[1]: https://github.com/MouseLand/Kilosort/blob/main/kilosort/parameters.py "Kilosort/kilosort/parameters.py at main · MouseLand/Kilosort · GitHub"
[2]: https://github.com/MouseLand/Kilosort/blob/main/kilosort/template_matching.py "Kilosort/kilosort/template_matching.py at main · MouseLand/Kilosort · GitHub"
[3]: https://kilosort.readthedocs.io/en/latest/parameters.html "When to adjust default settings — Kilosort4 0.0.1 documentation"


IMPLEMENTATION

Here is a precise documentation of every line of code that was changed or added in the Kilosort patch, based on your editable source tree:

---

### 1. **kilosort/parameters.py**  
**Location:** In the `EXTRA_PARAMETERS` dictionary, near the other spike detection parameters (after `max_peels`).

**Added:**
```python
    'cross_peel_claim_ms': {
        'gui_name': 'cross-peel claim ms', 'type': float, 'min': 0, 'max': np.inf,
        'exclude': [], 'default': 0.0, 'step': 'spike detection',
        'description':
        """
        Suppress later matching-pursuit candidates that fall within this many
        milliseconds of an already accepted spike. A value of 0 disables the
        cross-peel claim rule.
        """
    },

    'cross_peel_claim_um': {
        'gui_name': 'cross-peel claim um', 'type': float, 'min': 0, 'max': np.inf,
        'exclude': [], 'default': 0.0, 'step': 'spike detection',
        'description':
        """
        Spatial radius paired with cross_peel_claim_ms. Later matching-pursuit
        candidates are suppressed only if they are also within this many
        microns of a previously accepted spike. A value of 0 applies the claim
        rule using time alone.
        """
    },
```

---

### 2. **kilosort/template_matching.py**  
**Location:** In the `run_matching` function.

**Changes:**

- **At the start of `run_matching`:**  
  **Added:**
  ```python
    claim_ms = float(ops.get('cross_peel_claim_ms', 0.0))
    claim_um = float(ops.get('cross_peel_claim_um', 0.0))
    claim_enabled = claim_ms > 0
    claim_bins = int(np.ceil(claim_ms * ops['fs'] / 1000.0)) if claim_enabled else 0
    if claim_enabled:
        template_main_chan = torch.argmax((U**2).sum(-1), dim=1)
        xc = torch.as_tensor(ops['xc'], device=device, dtype=torch.float32)
        yc = torch.as_tensor(ops['yc'], device=device, dtype=torch.float32)
        template_x = xc[template_main_chan]
        template_y = yc[template_main_chan]
        claimed_t = []
        claimed_x = []
        claimed_y = []
  ```

- **Inside the peel loop, after forming `iX` and `iY` (after candidate selection, before accepting):**  
  **Added:**
  ```python
        if claim_enabled and t > 0 and len(claimed_t) > 0:
            cand_t = iX[:, 0].to(torch.float32)
            prev_t = torch.cat(claimed_t)
            dt = torch.abs(cand_t[:, None] - prev_t[None, :])

            if claim_um > 0:
                cand_x = template_x[iY[:, 0]]
                cand_y = template_y[iY[:, 0]]
                prev_x = torch.cat(claimed_x)
                prev_y = torch.cat(claimed_y)
                d2 = (cand_x[:, None] - prev_x[None, :])**2 + (cand_y[:, None] - prev_y[None, :])**2
                claimed = (dt <= claim_bins) & (d2 <= claim_um**2)
            else:
                claimed = dt <= claim_bins

            keep = ~claimed.any(dim=1)
            if not keep.all():
                iX = iX[keep]
                iY = iY[keep]

            if len(iX) == 0:
                break
  ```

- **After accepting spikes for this peel (after updating `st`, `amps`, etc.):**  
  **Added:**
  ```python
        if claim_enabled:
            claimed_t.append(iX[:, 0].to(torch.float32))
            claimed_x.append(template_x[iY[:, 0]])
            claimed_y.append(template_y[iY[:, 0]])
  ```

---

**Summary Table:**

| File                          | Location/Context                | Change Type | Code/Description (see above for full code) |
|-------------------------------|---------------------------------|-------------|--------------------------------------------|
| kilosort/parameters.py        | EXTRA_PARAMETERS dict           | Add         | cross_peel_claim_ms, cross_peel_claim_um   |
| kilosort/template_matching.py | run_matching (start)            | Add         | claim_ms, claim_um, claim_enabled, etc.    |
| kilosort/template_matching.py | run_matching (in peel loop)     | Add         | claim mask candidate filtering             |
| kilosort/template_matching.py | run_matching (after accept)     | Add         | append to claimed_t/x/y                    |

---

**No other files or lines are changed.**  
This is a minimal, fully replicable patch. If you need the exact line numbers for your current version, let me know and I can extract them.