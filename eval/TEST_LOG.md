# Locked-run log (append-only, docs/EVAL_PROTOCOL.md §5)

| date (UTC) | kind | branch | candidate | baseline | splits (n plans) | Δ correct-closed on primary [95% CI] | decision | job |
|---|---|---|---|---|---|---|---|---|
| 2026-11-24 02:06 | smoke | eval/protocol-v1 (smoke) | 88fe31c7a3 | 88fe31c7a3 | val-synth (200) | n/a (no primary split in this run) | smoke | 128277 (/home/kevin.antony/kelpod-runs/locked/20261124-020658_eval-protocol-v1) |
| 2026-11-24 02:07 | smoke | eval/protocol-v1 (smoke) | abd48f2429 | abd48f2429 | test-synth (5) | n/a (no primary split in this run) | smoke | 128277 (/home/kevin.antony/kelpod-runs/locked/20261124-020750_eval-protocol-v1) |
| 2026-11-24 11:55 | smoke | eval/learned-runner (smoke) | 971c4134fe ckpt s0:3f2f93b3157a | 376338a6c7 | val-synth (20) | +0.050 [+0.000, +0.150] (all splits, no primary in this run: val-synth) | smoke | 128277 (~/kelpod-runs/locked/20261124-115505_eval-learned-runner) |
| 2026-11-24 18:09 | locked | exp/din-web-plans | 243091343e | e64fbb4354 | test-synth (300), fresh (200) | +0.005 [-0.010, +0.020] (fresh) | FAIL | 128277 (~/kelpod-runs/locked/20261124-180913_exp-din-web-plans) |
| 2026-11-24 23:46 | locked | exp/din-web-v2 | c224833db1 | 4df2fea197 | test-synth (300), fresh (200) | +0.030 [+0.010, +0.055] (fresh) | FAIL | 128277 (~/kelpod-runs/locked/20261124-234639_exp-din-web-v2) |
| 2026-11-25 02:26 | locked | exp/din-web-v2 | ef5a483f5f | ec2323b22d | test-synth (300), fresh (200) | +0.015 [-0.010, +0.040] (fresh) | FAIL | 128277 (~/kelpod-runs/locked/20261125-022606_exp-din-web-v2) |
| 2026-11-24 23:34 | smoke | exp/learned-fast (smoke) | 92570dabc3 ckpt s1:6d1b6d399415, s2:9ed1bed6a149 | 4df2fea197 | val-synth (40) | worse seed cand0: +0.050 [-0.075, +0.175] (all splits, no primary in this run: val-synth) | smoke | 128277 (~/kelpod-runs/locked/20261124-233408_exp-learned-fast) |
| 2026-11-25 02:37 | smoke | exp/learned-fast (smoke) | 2a4a9d753b ckpt s1:6d1b6d399415, s2:9ed1bed6a149 | ec2323b22d | val-synth (200) | worse seed cand1: +0.200 [+0.130, +0.270] (all splits, no primary in this run: val-synth) | smoke | 128900 (~/kelpod-runs/locked/20261125-023739_exp-learned-fast) |
