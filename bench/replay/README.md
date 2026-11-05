# Replay buffer (general instruction data for the train stage)

`replay.jsonl` has 1,000 general instruction samples. The trainer mixes them in at
`lora.replay_ratio` to reduce forgetting. Each line looks like this:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "source": "databricks/databricks-dolly-15k#<row_idx>", "category": "open_qa"}
```

**Source:** [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
(train split). The rows are random blocks picked with seed 1234. A row is dropped when its
response is longer than 1,500 characters or its context is longer than 2,000 characters. When
a row has a context, it is appended to the instruction. `sources.json` records the row count
and sha256.

**Licence: CC BY-SA 3.0. Attribution is required.** databricks-dolly-15k is Copyright
Databricks, Inc. It was written by Databricks employees and released under the
[Creative Commons Attribution-ShareAlike 3.0 Unported licence](https://creativecommons.org/licenses/by-sa/3.0/).
This file is a reformatted subset and is shared under the same licence. Adapters trained with
this replay data should carry this attribution in their SBOM/report (package stage).

Regenerate on the build host only:
`ZLLF_ALLOW_ONLINE=1 python3 scripts/fetch_bench.py --only replay`.
