"""Public shared evaluation contract for topic clipping.

The corpus, labels, metrics, and runner here are the initial shared contract
for comparing deterministic retrieval/windows against Jev-decided selection on
identical candidates. Everything is synthetic and rights-cleared; labels are
independent of any decision scores and are never replaced by them.
"""

CONTRACT_VERSION = 1
