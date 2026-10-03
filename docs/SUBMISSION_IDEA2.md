# SIH submission, idea 2: pitch, slide outline, demo video, judge Q&A

## One-line differentiator

> **Court-defensible, explainable triage of Bitcoin laundering, fully offline:** every flag carries its evidence, its uncertainty,
> an analyst feedback loop, and a tamper-evident report. It is deliberately a decision-support tool that never accuses.

Most teams will show a model plus a graph. Ours adds: real laundering heuristics with validated precision, unseen-scenario
testing, honest uncertainty, cases with feedback, and verifiable reports.

## Slide outline (10 slides)

1. **Problem:** pseudonymous laundering (ransomware, darknet, layering); investigators drown in leads.
2. **Solution in one picture:** raw metadata in -> entities -> evidence -> ranked leads -> report.
3. **Data and schema:** CSV/JSON/XML ingestion, audit of dropped rows, synthetic generator with planted typologies, offline GeoIP + Tor/VPN lists.
4. **Forensic heuristics:** CoinJoin, dust, change address, peel chain, rapid-hop layering, each validated against planted truth (numbers from `docs/FORENSICS_LAB.md`).
5. **AI/ML:** Random Forest (calibrated, with interval) + Isolation Forest + graph embeddings/propagation, fused through the risk synthesizer.
6. **Explainability:** observation / inference / prediction labels, occlusion contributions, plain-language summary, per-flag TXIDs.
7. **Network correlation:** first-seen origin, Tor/VPN, geo-hops, wallet-to-IP link with confidence interval (correlation is not attribution).
8. **Evidence of performance:** baselines, ablation, **leave-one-scenario-out**, robustness, honest limits (slide text from `docs/ETHICS_AND_LIMITS.md`).
9. **Analyst workflow:** link graph with replay, money trail, verdicts, active learning, retrain with before/after, verifiable HTML/PDF report.
10. **Deployment and scale:** Linux Docker, offline, measured 500k transactions in 67 s; ethics, limits, next steps.

## 2-minute demo video script

| Time | On screen | Say |
|---|---|---|
| 0:00 | Forensics Lab header | "This turns raw Bitcoin transaction and network metadata into ranked, explainable leads, fully offline." |
| 0:10 | Status strip | "19,000 transactions, 24,000 wallet fragments, 986 leads. Scores are cross-fitted, so none are in-sample. The data is synthetic and marked as such." |
| 0:25 | Click top lead | "Top lead: a nine-hop layering chain. It forwarded all funds within seconds, came via Tor. Every statement is labelled observation, inference or model prediction." |
| 0:45 | Signals bars + structural evidence | "Five signals fuse into the score. Structural evidence lists the exact TXIDs." |
| 1:00 | Link analysis tab, press Replay | "Replay the flow. Amber is the onward money trail ending at an exchange-style hub." |
| 1:15 | Model explanation tab | "Each bar is the change in probability if that factor were typical: explainable for this exact model." |
| 1:30 | Give a verdict, click Retrain | "Analysts confirm or reject leads; the model retrains and we measure on entities it never saw." |
| 1:45 | Tick leads, download report | "A printable report with an evidence hash. Anyone can verify it offline for tampering." |
| 1:55 | Benchmark table | "And we report where we are weak: the unseen darknet-market scenario. Triage, not verdicts." |

## Judge Q&A additions

**Is this real data?** No. The dataset is synthetic and generated to model real typologies; it is marked synthetic everywhere.
The official dataset can be loaded with `python -m scripts.analyze_dataset` (see `docs/OFFICIAL_DATASET.md`).

**Why do your numbers look so high?** Because in-distribution synthetic data is easy; plain logistic regression matches our
Random Forest and we say so. The meaningful test is leave-one-scenario-out, where performance drops and we report it.

**Does the AI actually help, or are these just rules?** Rules alone reach PR-AUC 0.88; models 0.96 to 0.99. On an unseen
darknet-market scenario the fused score (0.79) beats both rules (0.74) and the model alone (0.63). Caveat: synthetic data.

**What about real data?** On Elliptic++ (real, labelled, strict out-of-time split) the Random Forest reaches PR-AUC 0.79 and
precision 1.0 at the top 100, versus 0.30 for logistic regression and 0.07 for chance. Rules and anomaly detection do not help
there, and accuracy drops sharply in the last time window (concept drift), which we report in `reports/BENCHMARK.md`.

**How do you avoid falsely accusing someone?** Entities are wallet fragments, never people. Scores are calibrated with
intervals, anomaly-only alerts are capped, benign look-alikes (exchanges, processors, CoinJoin users) are in the test data,
and IP links are shown as correlation with a confidence range. Reports say "triage evidence, not proof".

**How do you handle Tor and VPNs?** Flagged via a local Tor exit list and hosting/VPN ASN list; used as a bounded signal
that is never decisive alone, because legitimate privacy users exist.

**Does it scale?** Measured: 509k transactions in 67 s on a laptop (1.8 GB). 1M is extrapolated, not measured.

**Why Random Forest, not deep learning or a GNN?** Tabular entity features, small labelled sets, a need for calibration,
intervals and explanations. Graph structure enters through PPMI-SVD embeddings and neighbour propagation (the closed-form
and fixed-weight versions of DeepWalk and a GraphSAGE layer). Embeddings added little in our ablation, and we say so.

**What if the dataset has no labels?** The model trains on internal synthetic data and transfers; the UI says so. Heuristic
evidence is unaffected, and analyst feedback adapts the model.

**Can the report be trusted?** It embeds its evidence and a SHA-256 digest plus dataset and model hashes; `verify` recomputes
the digest offline. This detects tampering with the report, not errors in the analysis.
