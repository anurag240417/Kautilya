# ChainTrace: YouTube video script for SIH (about 4 minutes)

Problem statement 26146 (NTRO): AI-powered monitoring and analysis of Bitcoin transaction traffic.
Live demo: https://chain-trace-dijp.onrender.com/

## Before you record (10 minutes)

1. **Wake the Render site** 3 to 5 minutes before: open the URL and wait until the Forensics Lab page shows numbers.
2. Browser at 1080p, zoom 110 to 125%, bookmarks bar hidden, one tab only.
3. Click **SYNC ALL (110)** in the top bar so the older pages are filled.
4. Do one dry run. **Do not click "Retrain with feedback"** on the free server (slow) and do not click "Regenerate synthetic data" (it changes the lead names).
5. Record voice-over after the screen capture if you stumble live; keep each scene under the time shown.
6. Numbers below match the deployed site (7,979 transactions, 497 alerts, top lead E-4091). If yours differ, say what is on screen.

## Script

### 0:00 - 0:20 | Hook (face or title slide, then Overview page)
**Show:** title slide "ChainTrace: explainable triage of Bitcoin laundering, fully offline", then the Overview page.

**Say:**
"Criminals use Bitcoin to move ransomware payments and darknet-market money, and investigators drown in millions of transactions. ChainTrace takes raw transaction and network metadata and gives analysts a ranked list of leads, each one with its evidence, its uncertainty, and a reason in plain language. It never accuses anyone. It runs completely offline."

### 0:20 - 0:50 | The two parts (Overview, then sidebar)
**Show:** Overview top alerts, then slowly pan the sidebar.

**Say:**
"There are two parts. The first is trained on Elliptic++, a real labelled Bitcoin dataset. The second, the Forensics Lab, works on the problem statement's raw schema: timestamps, IP addresses and ports, transaction IDs, input and output addresses with amounts, fees and script types. Everything here is synthetic and marked that way."

### 0:50 - 1:35 | Investigation page: the Shadow Mixer case
**Show:** Investigation, type **1001**, click Trace Entity. Show graph, then the timeline tab. Then search **1006**, open "Correlated Observations".

**Say:**
"Transaction 1001 is a fourteen-point-five bitcoin darknet deposit. It scores ninety-two, critical. The graph follows the money into a mixer, through layering hops, to a cash-out. The risk score is broken down into five separate signals: model prediction, anomaly, graph structure, timing and network. Transaction 1006 is flagged for a different reason: it was relayed from the same IP as another suspicious transaction. We show that as a correlation with a confidence, never as proof."

### 1:35 - 2:50 | Forensics Lab (the main demo)
**Show, in order:**
1. Open Forensics Lab. Pause on the stats strip (**transactions, entities, alerts, "SYNTHETIC DATA" tag**).
2. Click the top lead, **E-4091**. Evidence tab: summary, signal bars, structural evidence.
3. **Link analysis** tab: press **Replay**; point at the amber trail and the rounded-square hub.
4. **Model explanation** tab: the bars.

**Say:**
"Here is the Forensics Lab. About eight thousand transactions, ten thousand wallet fragments, and five hundred ranked leads. The top lead is a ten-hop rapid layering chain. Funds move on within minutes, and every transaction first appeared from a Tor exit node. The summary explains why, and lists the exact transaction IDs.

Replay shows the money moving through the chain until it reaches an exchange-style hub, where it would be cashed out. This is the analyst's view of layering.

And the model explanation shows what drove the score. For this lead, timing contributes the most: funds held for seconds, not days. Each bar is the change in probability if that factor were typical, so the explanation is exact for the deployed model."

### 2:50 - 3:15 | Analyst loop and report
**Show:** click **Confirm illicit**, tick two leads, click **Download report**, open the HTML, scroll to the chain-of-custody box.

**Say:**
"Analysts stay in control. Confirm or reject a lead, and that feeds the next retrain. One click produces a printable report with an evidence hash, a dataset fingerprint and the model version. Anyone can verify it offline for tampering."

### 3:15 - 3:50 | Honest evaluation (slide with two small tables)
**Show:** slide with: Elliptic++ real data table, and "what does not work".

**Say:**
"We tested honestly. On real Elliptic++ data with a strict time split, the Random Forest reaches a PR-AUC of point seven nine, with perfect precision in the top hundred, against point three for logistic regression. Simple rules and anomaly detection do not help on that data, and accuracy drops in the final time window as criminal behaviour shifts, so models need retraining. On our synthetic data, we also tested whole laundering types the model had never seen, and report where it is weakest: a darknet market."

### 3:50 - 4:10 | Offline, deployment, close
**Show:** Data Sources page ("Network Calls: 0"), then the title slide.

**Say:**
"Everything runs offline on Linux, in a container, with no network calls. This is a decision-support tool: explainable, uncertain where it should be, and always reviewed by a human. Thank you."

## YouTube metadata

**Title:** ChainTrace | Explainable AI to Trace Bitcoin Laundering (Offline) | SIH 2026

**Description:**
ChainTrace turns raw Bitcoin transaction and network metadata into ranked, explainable investigative leads, fully offline.
Problem statement 26146 (NTRO): AI-powered monitoring and analysis of Bitcoin transaction traffic.

Chapters:
0:00 Problem
0:20 What it does
0:50 Investigation: the Shadow Mixer case
1:35 Forensics Lab: entities, heuristics, graph ML, explanations
2:50 Analyst feedback and tamper-evident report
3:15 Honest evaluation
3:50 Offline deployment

Live demo: https://chain-trace-dijp.onrender.com/ (synthetic data only)
Code: <add your GitHub link>

All data in the Forensics Lab is synthetic. Scores are triage aids, not evidence of wrongdoing.

**Tags:** bitcoin forensics, blockchain analytics, anti money laundering, explainable AI, SIH, Smart India Hackathon, NTRO, machine learning, graph analysis, cybersecurity

**Pinned comment:** "Elliptic++ results, benchmarks and limits are in the repo (reports/ and docs/)."

## 60-second cut (for a short or a form field)
Hook (10 s) -> Forensics Lab top lead and Replay (30 s) -> model explanation (10 s) -> "triage, not proof, fully offline" (10 s).

## Things not to claim in the video
* Do not call the Forensics Lab data real. It is synthetic.
* Do not say "99 percent accurate". Say what was tested and on which data.
* Do not claim million-transaction scale; the tested size is about five hundred thousand.
* Do not say the Docker image or the Render deployment is "production".
