# BP: agglomeration route-provenance semantic repair

## Verdict

The local, opt-in observation adapter now represents what the inspected DARTsort
source can actually establish. It remains detached from the engine and does not
run QDA, merge units, replay a payload, sort spikes, or touch voltage.

Source identity is DARTsort Git object
`edcfe1b51d672b4136eb13cc78c0875da804b851`. In that source, QDA schedules only
the requested upper-triangle pairs (`agglomerate.py:582-587`), initializes result
arrays to zero (`agglomerate.py:558-569`), and has several early returns before
the final score is computed (`agglomerate.py:634-666`). Hierarchical clustering
consumes only the upper triangle, tolerates small negative numerical zeros,
handles positive infinity specially, and applies the threshold inclusively
(`cluster_util.py:104-164`).

## Corrected representation

- Git commits are 40-hex object IDs; 64-hex SHA-256 strings remain artifact hashes.
- Boolean matrices accept only JSON booleans. Event IDs and labels accept only
  integers; labels are at least -1 and rows must name the declared pre-stage
  namespace.
- Raw numeric observations retain direction and sign. Separate distance- and
  force-threshold consumed matrices record the symmetric upper-triangle
  representations actually supplied to SciPy linkage (their infinity
  substitutions differ because the thresholds differ).
- `finite`, `positive_infinity`, `negative_infinity`, and `missing` are distinct
  value states. Thus a source-written zero or infinity is not inferred to be
  absent data.
- QDA requested, firing-eligible, upper-triangle scheduled, completed, accepted,
  early-exit, and unknown states are separate. An early exit or unknown result is
  not counted as a completed rejection.

## Independent failure-to-inference table

| Observed artifact | Invalid inference | Additional observation required |
|---|---|---|
| Same final component labels | Same pairwise route or merge evidence | Direct route masks and pre-linkage values |
| QDA score equals zero | QDA completed and rejected the pair | Per-pair evaluation status or instrumented completion |
| Pair appears in requested mask | Pair was scheduled or computed | Firing eligibility, upper-triangle schedule, completion status |
| Symmetric linkage distance | Original distance computation was symmetric | Raw directed distance matrix |
| Null JSON numeric entry | Source value was non-finite | Explicit value state: missing versus positive/negative infinity |
| Pair is in expanded linkage mask | That pair passed the direct threshold | Direct adjacency before component expansion |
| Final labels are depth ordered | Those were the pre-reorder component IDs | Explicit pre-reorder mapping and reorder permutation |

## Three-node counterexample

Let the final component be `{A,B,C}` in both experiments.

- History 1: force edges are `A-B` and `B-C`; QDA accepts `A-B`.
- History 2: force has only `A-B`; QDA accepts `B-C`.

Both unions are the path `A-B-C`, so both produce one identical final connected
component. Nevertheless, the direct force and QDA histories differ. Final labels
therefore cannot reconstruct route attribution. The targeted synthetic test
preserves this counterexample explicitly.

## Validation

Only the BP-targeted suite was run, per the time-box: 10 tests passed. The tests
cover Git/hash identity, strict booleans, signed/asymmetric observations,
upper-triangle source behavior, numerical-zero tolerance, infinity substitution,
inclusive threshold ties, QDA unknown/completion semantics, integer event
identity and namespaces, missingness versus infinity, and the graph
counterexample.

No BH payload, production engine integration, real data, GPU work, sort, or
replay was used. Prior cumulative huklaban1 time remains 6773.22 seconds; BI is
accounted separately. BP stayed within its 600-second cap.
