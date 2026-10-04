# Additive correction table

| Item | As executed / previously reported | Contract-adherence audit | Corrected reading |
|---|---|---|---|
| Primary H−F | Delta `+0.06599645095644802`; 95% interval `[0.046296547965893387, 0.0962770414589578]`; `inconclusive` | Adheres to intended H−F endpoint and unchanged ±0.05 precedence | **INCONCLUSIVE remains correct.** Interval is above zero, but not wholly above `+0.05` and not wholly inside `[-0.05,+0.05]` |
| Quality comparator | Scorer built guardrails against both F and L and required all to pass | Deviates: pre-run independent review specified **L-based** guardrails; no explicit authorized override found | Use L for the same-window quality guardrails; retain F whole-population counts as descriptive context |
| Population/yield semantics | F gate compared 461 H labels with 972 F labels and failed at `-0.5257201646090535` | Deviates: active/eligible populations were frozen as descriptive, and F is not the intended same-window quality denominator | F comparison is not an acceptance failure and must not be described as lost neurons or biological damage |
| Intended H/L yield check | H 461 versus L 478 | Adheres when evaluated with the existing 5% rule | `(461−478)/478 = -0.03556485355648536`; **PASS** |
| H/L negative short interval | H `0.0084066082028826`; L `0.00666562952502902`; difference `+0.0017409786778535804` | Same-window comparator; existing 0.01 directional and absolute limits | **PASS directional; PASS equivalence** |
| H/L flat short interval | H `0.00959781271049393`; L `0.009093505454960873`; difference `+0.0005043072555330572` | Same-window comparator; existing 0.01 directional and absolute limits | **PASS directional; PASS equivalence** |
| H/L remainder short interval | H `0.00985236471772074`; L `0.010399987800600821`; difference `-0.0005476230828800818` | Same-window comparator; existing 0.01 directional and absolute limits | **PASS directional; PASS equivalence** |
| Proxy context | H rho `0.7421635214982477`, L `0.7525836443406859`, F `0.6761670705417997` | Values are descriptive continuity proxies; H−L is non-decompositional | H is near L and above F. The observed proxy difference persisted in this run but missed the frozen criterion; no causal or across-run claim |
| Pipeline decision | Prior final review said no progression and attributed an independent failure to the F yield gate | Reasoning partly deviated; outcome does not change | **No progression**, because primary H−F remains inconclusive—not because of the unintended F population-yield gate |

No result file, threshold, hypothesis, or score is changed by this table.
