# Running ChainTrace on the official SIH dataset

The dataset link in the problem statement was **not** downloaded or analysed during development (Google Drive, needs a
manual download). Everything below is ready for when you have the file. Expect to spend a few minutes on column mapping.

1. Download the file(s) and put them in `./data/` (git-ignored).
2. Audit the ingestion first, before trusting any result:

   ```bash
   python -m scripts.analyze_dataset data/<file>.csv
   ```

   The **Ingestion audit** shows renamed columns, rows dropped per reason, duplicate TXIDs merged (extra network
   observations), and the detected amount unit. Check that `rows_dropped` is small.
3. If a column is not recognised, add its name to `COLUMN_ALIASES` in `backend/forensics/ingest.py`.
   Accepted list encodings: JSON arrays, `a|b|c`, `a;b`, `a,b`, or nested XML elements.
4. Amounts are assumed BTC; values above 1,000,000 (median) are treated as satoshis. Override with `--amount-unit btc|sat`.
5. **No labels?** The model is then trained on an internally generated synthetic dataset and transferred. The status strip
   says `labels: transfer from synthetic`. Treat the ranking as a prior, use analyst feedback to adapt it, and expect
   domain shift. Heuristic evidence (peel chains, CoinJoin, dust, layering) does not depend on the model.
6. **Has labels?** Provide them as entity-level labels through `run_forensics(ds, labels=...)` (see `backend/forensics/cases.py`).
7. Open the UI: `CHAINTRACE_DATA_DIR=./data python -m backend.main`, then Forensics Lab -> load via
   `POST /forensics/load {"source":"file","path":"<file>.csv"}`.

Things worth reporting for the official run: ingestion audit, heuristic counts, cluster count, pipeline time and peak memory.
