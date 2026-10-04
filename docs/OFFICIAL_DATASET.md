# The dataset for problem statement 26146

**There is no official dataset to download.** The problem statement lists "Dataset Link: Nil"; the link shown on the SIH
portal leads back to the problem-statement document itself. The statement says participants "will work with a synthetic dataset
modelled on real Bitcoin P2P/transaction fields (no real seized or live-intercept data will be provided)". So the dataset is
one you generate.

Kautilya does this:

* **Schema:** the minimum fields from the statement are all ingested: timestamp, src_ip, dst_ip, src_port, dst_port, txid,
  input_addresses[], output_addresses[], input_amounts[], output_amounts[], geo_country/asn, plus fee and script type.
* **Generator:** `python -m scripts.generate_dataset --n-tx 20000 --out data/synthetic` writes the same data as CSV, JSON and XML
  (plus a ground-truth file), with planted ransomware, darknet-market, layering and dust-attack scenarios and innocent look-alikes.
* **Elliptic++** (real, labelled, anonymised) is used separately to evaluate the supervised model on real data
  (`reports/BENCHMARK.md`). It has no IPs or per-address amounts, so it cannot feed the raw-transaction pipeline.

## Using your own file (for example one a mentor or the organisers hand you later)

1. Put it in `./data/` (git-ignored).
2. Audit the ingestion first:

   ```bash
   python -m scripts.analyze_dataset data/<file>.csv
   ```

   The **Ingestion audit** shows renamed columns, rows dropped per reason, duplicate TXIDs merged (extra network observations)
   and the detected amount unit. Check that `rows_dropped` is small.
3. If a column name is not recognised, add it to `COLUMN_ALIASES` in `backend/forensics/ingest.py`.
   Accepted list encodings: JSON arrays, `a|b|c`, `a;b`, `a,b`, or nested XML elements.
4. Amounts are assumed BTC; a median above 1,000,000 is treated as satoshis. Override with `--amount-unit btc|sat`.
5. **No labels?** The model is trained on an internally generated synthetic dataset and transferred; the status strip says
   `labels: transfer from synthetic`. Treat the ranking as a prior and expect domain shift. Heuristic evidence does not depend on the model.
6. Load it in the UI with `KAUTILYA_DATA_DIR=./data python -m backend.main`, then pick the file in the Forensics Lab status strip.
