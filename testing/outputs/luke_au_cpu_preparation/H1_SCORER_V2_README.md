# AZ scorer/generator packet v2

Status: **scorer qualified; donors pending; launch not permitted.** This packet
supersedes scorer v1 only. It does not overwrite v1, authorize voltage access or
declare any donor-dependent gate complete.

v2 retains every truth donor. A donor with no positive exclusive match has
`primary_label: null`, recall 0 where truth exists and undefined precision as
JSON `null`. Null associations never create a false-merge link. Empty output,
all-zero-TP labels, mixed matched/unmatched donors and multiple unmatched donors
in chance controls are explicit fixtures. Existing positive association,
global exclusivity, ambiguity and duplicate tests remain passing.

The template-stage wording is also clarified. Donors are measured and injected
without spatial whitening into the accepted post-ibllikecmr standardized
float32 background, exactly once before arm branching. A second preprocessing
pass before branching is forbidden. Normal downstream DARTsort operations,
including its configured internal whitening, remain enabled and identical in
S_h, DZ_h and D2L_h.

Donor templates, placements, immutable materialized event trains, native-clone
avoidance, query-time transition agreement and new-hybrid stage lineage remain
pending. No fourth extraction, voltage read or GPU work occurred.
