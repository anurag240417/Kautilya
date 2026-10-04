import React, { useCallback, useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import RiskBadge from '../components/RiskBadge';
import ForensicDisclaimer from '../components/ForensicDisclaimer';
import FlowGraph from '../components/FlowGraph';
import { forensics } from '../api/client';

const pct = (v, d = 0) => (v === null || v === undefined ? 'n/a' : `${(v * 100).toFixed(d)}%`);
const num = (v, d = 3) => (v === null || v === undefined ? 'n/a' : Number(v).toFixed(d));

const VERDICT_STYLE = {
  confirmed: { color: '#C9453E', label: 'Confirmed' },
  false_positive: { color: '#4A7A5C', label: 'False positive' },
  unsure: { color: '#C9A93E', label: 'Unsure' },
};

function Chip({ children, color = '#3D4F63' }) {
  return (
    <span
      style={{
        fontSize: 10,
        padding: '1px 6px',
        border: `1px solid ${color}`,
        borderRadius: 2,
        color: 'var(--ct-text-secondary)',
        whiteSpace: 'nowrap',
      }}
    >
      {children}
    </span>
  );
}

function Bar({ value, color = '#4A90D9', label, right }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--ct-text-secondary)' }}>
        <span>{label}</span>
        <span style={{ fontFamily: 'var(--ct-font-mono)' }}>{right}</span>
      </div>
      <div style={{ height: 6, background: 'var(--ct-bg-elevated)', borderRadius: 2 }}>
        <div style={{ height: 6, width: `${Math.min(Math.max(value, 0), 1) * 100}%`, background: color, borderRadius: 2 }} />
      </div>
    </div>
  );
}

function Stat({ label, value, sub }) {
  return (
    <div className="ct-card" style={{ padding: '10px 14px', flex: '1 1 150px' }}>
      <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.5, color: 'var(--ct-text-muted)' }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: 'var(--ct-text-secondary)' }}>{sub}</div>}
    </div>
  );
}

function StatusStrip({ status, onReload, onRetrain, onReset, onLoadFile, files, busy }) {
  const [file, setFile] = useState('');
  if (!status || !status.loaded) {
    return <div className="ct-card" style={{ padding: 16 }}>Loading forensic dataset and models (first run takes ~10 s)...</div>;
  }
  const rank = status.metrics?.ranking?.fused_score;
  const hv = status.metrics?.heuristic_validation;
  const cq = status.metrics?.clustering;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <Stat label="Transactions" value={status.dataset.transactions.toLocaleString()} sub={`${status.dataset.addresses.toLocaleString()} addresses`} />
        <Stat label="Entities" value={status.n_entities.toLocaleString()} sub={`${status.n_active_entities.toLocaleString()} active`} />
        <Stat label="Alerts raised" value={status.n_alerts.toLocaleString()} sub={`pipeline ${status.timings_s.total_s}s`} />
        <Stat label="Fused PR-AUC" value={rank ? num(rank.pr_auc) : 'n/a'} sub={rank ? `P@50 ${pct(rank['precision@50'])} (synthetic truth)` : 'no ground truth'} />
        <Stat label="Change-detect precision" value={hv?.change_detection ? pct(hv.change_detection.precision, 1) : 'n/a'} sub={cq ? `cluster purity ${pct(cq.address_weighted_purity, 1)}` : ''} />
      </div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', fontSize: 12, color: 'var(--ct-text-secondary)' }}>
        <Chip>labels: {status.label_source.replace(/_/g, ' ')}</Chip>
        <Chip>model: {status.model_version}</Chip>
        <Chip>GeoIP: {status.geoip_source}</Chip>
        <Chip>feedback: {status.n_feedback}</Chip>
        {status.dataset.has_ground_truth && <Chip color="#C9A93E">SYNTHETIC DATA</Chip>}
        <span style={{ flex: 1 }} />
        {files.length > 0 && (
          <>
            <select value={file} onChange={(e) => setFile(e.target.value)} aria-label="Dataset file">
              <option value="">Load a data file...</option>
              {files.map((f) => <option key={f.path} value={f.path}>{f.path} ({f.size_mb} MB)</option>)}
            </select>
            <button type="button" disabled={busy || !file} onClick={() => onLoadFile(file)}>Load</button>
          </>
        )}
        <button type="button" disabled={busy} onClick={onReload}>Regenerate synthetic data</button>
        <button type="button" className="primary" disabled={busy} onClick={onRetrain}>Retrain with feedback</button>
        {status.last_retrain && <button type="button" disabled={busy} onClick={onReset}>Reset model</button>}
      </div>
      {status.last_retrain && <RetrainSummary r={status.last_retrain} />}
    </div>
  );
}

function RetrainSummary({ r }) {
  const b = r.before_feedback?.fused_score;
  const a = r.after_feedback?.fused_score;
  return (
    <div className="ct-card" style={{ padding: 12, fontSize: 12 }}>
      <b>Retrained on {r.seed_labels} seed labels + {r.feedback_labels} analyst verdicts.</b>{' '}
      {r.note}
      <div style={{ marginTop: 6, display: 'flex', gap: 24 }}>
        <span>Before feedback: PR-AUC {num(b?.pr_auc)}, P@50 {pct(b?.['precision@50'])}</span>
        <span>After feedback: PR-AUC {num(a?.pr_auc)}, P@50 {pct(a?.['precision@50'])}</span>
      </div>
    </div>
  );
}

function AlertsTable({ data, selected, onSelect, basket, toggleBasket }) {
  return (
    <table style={{ width: '100%' }}>
      <thead>
        <tr>
          <th style={{ width: 28 }} />
          <th>#</th>
          <th>Entity</th>
          <th>Priority</th>
          <th>P(illicit)</th>
          <th>Why</th>
          <th>Verdict</th>
        </tr>
      </thead>
      <tbody>
        {data.alerts.map((a) => (
          <tr
            key={a.entity_id}
            onClick={() => onSelect(a.entity_id)}
            style={{ cursor: 'pointer', background: selected === a.entity_id ? 'var(--ct-bg-elevated)' : undefined }}
          >
            <td onClick={(e) => e.stopPropagation()}>
              <input type="checkbox" checked={basket.includes(a.entity_id)} onChange={() => toggleBasket(a.entity_id)} aria-label="Add to report" />
            </td>
            <td style={{ fontFamily: 'var(--ct-font-mono)' }}>{a.rank}</td>
            <td>
              <div style={{ fontWeight: 600 }}>{a.label}</div>
              <div style={{ fontSize: 10, color: 'var(--ct-text-muted)' }}>
                {a.n_addresses} addr{a.truth_scenario && a.truth_scenario !== 'normal' ? ` | truth: ${a.truth_scenario}` : a.truth_illicit === false ? ' | truth: licit' : ''}
              </div>
            </td>
            <td><RiskBadge tier={a.tier} score={a.score} /></td>
            <td style={{ fontFamily: 'var(--ct-font-mono)' }}>{pct(a.illicit_probability)}</td>
            <td><div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>{a.reasons.slice(0, 3).map((r) => <Chip key={r}>{r}</Chip>)}</div></td>
            <td>{a.verdict ? <span style={{ color: VERDICT_STYLE[a.verdict].color, fontSize: 12 }}>{VERDICT_STYLE[a.verdict].label}</span> : <span style={{ color: 'var(--ct-text-muted)' }}>-</span>}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Evidence({ e }) {
  const s = e.score;
  const link = e.network_evidence?.wallet_ip_link;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div style={{ background: 'var(--ct-bg-elevated)', borderLeft: '3px solid #4A90D9', padding: '10px 12px', fontSize: 13 }}>{e.summary}</div>

      <div>
        <h3 style={{ marginBottom: 6 }}>Signals fused into the score</h3>
        <Bar label={`Model illicit probability (90% range ${pct(s.interval90[0])} to ${pct(s.interval90[1])})`} value={s.illicit_probability} right={pct(s.illicit_probability)} color="#C9453E" />
        <Bar label="Structural heuristics (peel / layering / dust / CoinJoin)" value={s.structural_heuristic_score} right={num(s.structural_heuristic_score, 2)} color="#D4763A" />
        <Bar label="Anomaly percentile (novelty, not guilt)" value={s.anomaly_percentile} right={pct(s.anomaly_percentile)} color="#C9A93E" />
        <Bar label="Network obfuscation (Tor / VPN / geo-hops)" value={s.network_obfuscation} right={num(s.network_obfuscation, 2)} color="#7B68AE" />
        <Bar label="Temporal burst" value={s.temporal_burst} right={num(s.temporal_burst, 2)} color="#4A90D9" />
      </div>

      <div>
        <h3 style={{ marginBottom: 6 }}>Structural evidence</h3>
        {e.heuristic_evidence.length === 0 && <div style={{ fontSize: 12, color: 'var(--ct-text-muted)' }}>No structural laundering pattern detected.</div>}
        {e.heuristic_evidence.map((h) => (
          <div key={h.code} style={{ fontSize: 12, marginBottom: 8 }}>
            <b>{h.label}</b> <Chip>{h.type}</Chip>
            <div style={{ color: 'var(--ct-text-secondary)' }}>{h.detail}</div>
            <div style={{ fontFamily: 'var(--ct-font-mono)', fontSize: 10, color: 'var(--ct-text-muted)', wordBreak: 'break-all' }}>
              {h.txids.slice(0, 2).join('  ')}
            </div>
          </div>
        ))}
      </div>

      <div>
        <h3 style={{ marginBottom: 6 }}>Network-layer evidence</h3>
        <table style={{ width: '100%', fontSize: 12 }}>
          <thead><tr><th>Origin IP</th><th>Sends</th><th>Country</th><th>ASN</th><th>Flag</th></tr></thead>
          <tbody>
            {(e.network_evidence.top_origins || []).map((o) => (
              <tr key={o.ip}>
                <td style={{ fontFamily: 'var(--ct-font-mono)' }}>{o.ip}</td><td>{o.count}</td><td>{o.country}</td><td>AS{o.asn}</td>
                <td>{o.is_tor_exit ? <Chip color="#7B68AE">Tor exit</Chip> : o.is_hosting_or_vpn ? <Chip color="#7B68AE">hosting/VPN</Chip> : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {link && (
          <div style={{ marginTop: 6 }}>
            <Bar label={`Wallet-IP link: ${link.ip} sent ${pct(link.share)} of its transactions`} value={link.share} right={`95% CI ${pct(link.ci95[0])} - ${pct(link.ci95[1])}`} color="#7B68AE" />
            <div style={{ fontSize: 11, color: 'var(--ct-text-muted)' }}>{link.interpretation}</div>
          </div>
        )}
        <div style={{ fontSize: 11, color: 'var(--ct-text-secondary)', marginTop: 4 }}>
          {e.network_evidence.distinct_ips} distinct origin IPs, {e.network_evidence.distinct_countries} countries,
          {' '}{e.network_evidence.rapid_country_changes} rapid country changes.
        </div>
      </div>

      <div>
        <h3 style={{ marginBottom: 6 }}>Key transactions</h3>
        <table style={{ width: '100%', fontSize: 11 }}>
          <thead><tr><th>TXID</th><th>Time (UTC)</th><th>Role</th><th>BTC</th><th>in/out</th></tr></thead>
          <tbody>
            {e.key_transactions.map((t) => (
              <tr key={t.txid + t.role}>
                <td style={{ fontFamily: 'var(--ct-font-mono)' }}>{t.txid.slice(0, 16)}...</td><td>{t.time}</td><td>{t.role}</td><td>{t.amount_btc}</td><td>{t.n_in}/{t.n_out}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ModelPanel({ e }) {
  const max = Math.max(...e.model_contributions.map((c) => Math.abs(c.contribution)), 0.01);
  return (
    <div>
      <h3 style={{ marginBottom: 4 }}>Why the model scored it this way</h3>
      <div style={{ fontSize: 11, color: 'var(--ct-text-muted)', marginBottom: 8 }}>
        Each bar is the change in illicit probability when that factor is replaced by the typical value (occlusion).
        Red raises risk, green lowers it.
      </div>
      {e.model_contributions.map((c) => (
        <div key={c.feature} style={{ marginBottom: 8, fontSize: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>{c.description}</span>
            <span style={{ fontFamily: 'var(--ct-font-mono)' }}>{c.contribution > 0 ? '+' : ''}{(c.contribution * 100).toFixed(1)} pts</span>
          </div>
          <div style={{ height: 6, background: 'var(--ct-bg-elevated)' }}>
            <div style={{ height: 6, width: `${(Math.abs(c.contribution) / max) * 100}%`, background: c.contribution > 0 ? '#C9453E' : '#4A7A5C' }} />
          </div>
          {c.value !== null && <div style={{ fontSize: 10, color: 'var(--ct-text-muted)' }}>this entity: {c.value} | typical: {c.typical_value}</div>}
        </div>
      ))}
      <h3 style={{ margin: '14px 0 4px' }}>Behavioural peer group</h3>
      {e.peer_group.distinguishing ? (
        <div style={{ fontSize: 12 }}>
          Group {e.peer_group.id} ({e.peer_group.size?.toLocaleString()} entities) is distinguished by:{' '}
          {e.peer_group.distinguishing.map((d) => `${d.direction} ${d.feature.replace(/_/g, ' ')}`).join(', ')}.
        </div>
      ) : <div style={{ fontSize: 12, color: 'var(--ct-text-muted)' }}>Too little activity to assign a peer group.</div>}
      {e.ground_truth_synthetic && (
        <div style={{ marginTop: 12, fontSize: 12 }}>
          <Chip color="#C9A93E">SYNTHETIC GROUND TRUTH</Chip>{' '}
          {e.ground_truth_synthetic.is_illicit ? `planted illicit actor (${e.ground_truth_synthetic.scenario})` : 'planted licit entity'}
        </div>
      )}
    </div>
  );
}

function FeedbackBox({ entityId, onSaved, existing }) {
  const [note, setNote] = useState('');
  const [saving, setSaving] = useState(false);
  const send = async (verdict) => {
    setSaving(true);
    try {
      await forensics.feedback({ entity_id: entityId, verdict, analyst: 'analyst', note });
      setNote('');
      onSaved();
    } finally {
      setSaving(false);
    }
  };
  return (
    <div className="ct-card" style={{ padding: 12 }}>
      <div style={{ fontSize: 12, marginBottom: 6 }}>
        <b>Your verdict</b> (feeds the next retrain)
        {existing?.length > 0 && <span style={{ color: 'var(--ct-text-muted)' }}> | last: {existing[0].verdict.replace('_', ' ')}</span>}
      </div>
      <input placeholder="Optional note" value={note} onChange={(ev) => setNote(ev.target.value)} style={{ width: '100%', marginBottom: 8 }} />
      <div style={{ display: 'flex', gap: 8 }}>
        <button type="button" disabled={saving} onClick={() => send('confirmed')} style={{ borderColor: '#C9453E' }}>Confirm illicit</button>
        <button type="button" disabled={saving} onClick={() => send('false_positive')} style={{ borderColor: '#4A7A5C' }}>False positive</button>
        <button type="button" disabled={saving} onClick={() => send('unsure')}>Unsure</button>
      </div>
    </div>
  );
}

export default function ForensicsLab() {
  const [status, setStatus] = useState(null);
  const [alerts, setAlerts] = useState(null);
  const [tier, setTier] = useState('');
  const [searchParams] = useSearchParams();
  const [search, setSearch] = useState(searchParams.get('search') || '');
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState(null);
  const [entity, setEntity] = useState(null);
  const [graph, setGraph] = useState(null);
  const [tab, setTab] = useState('evidence');
  const [basket, setBasket] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [uncertain, setUncertain] = useState([]);
  const [files, setFiles] = useState([]);
  const LIMIT = 25;

  const loadStatus = useCallback(async () => {
    try { setStatus(await forensics.status()); } catch (err) { setError(err.message); }
  }, []);

  const loadAlerts = useCallback(async () => {
    try {
      setError(null);
      setAlerts(await forensics.alerts({ limit: LIMIT, offset, tier, search }));
    } catch (err) { setError(err.message); }
  }, [offset, tier, search]);

  // First visit: the server builds the default dataset lazily.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      await loadStatus();
      try { const f = await forensics.files(); if (!cancelled) setFiles(f.files); } catch { /* optional */ }
      try {
        const d = await forensics.alerts({ limit: LIMIT, search: new URLSearchParams(window.location.hash.split('?')[1] || '').get('search') || '' });
        if (!cancelled) { setAlerts(d); await loadStatus(); }
      } catch (err) { if (!cancelled) setError(err.message); }
    })();
    return () => { cancelled = true; };
  }, [loadStatus]);

  useEffect(() => { if (status?.loaded) loadAlerts(); }, [loadAlerts, status?.loaded]);

  const openEntity = useCallback(async (id) => {
    setSelected(id);
    setEntity(null);
    setGraph(null);
    try {
      const [e, g] = await Promise.all([forensics.entity(id), forensics.graph(id, 2)]);
      setEntity(e);
      setGraph(g);
    } catch (err) { setError(err.message); }
  }, []);

  const toggleBasket = (id) => setBasket((b) => (b.includes(id) ? b.filter((x) => x !== id) : [...b, id]));

  const act = async (fn) => {
    setBusy(true);
    try { await fn(); await loadStatus(); await loadAlerts(); } catch (err) { setError(err.message); } finally { setBusy(false); }
  };

  const refreshAfterFeedback = async () => {
    await loadStatus();
    await loadAlerts();
    if (selected !== null) setEntity(await forensics.entity(selected));
    try { setUncertain((await forensics.uncertain(6)).entities); } catch { /* optional */ }
  };

  return (
    <div style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 16, maxWidth: 1500, margin: '0 auto' }}>
      <div>
        <h1 style={{ fontSize: 22, marginBottom: 4 }}>Forensics Lab</h1>
        <p style={{ color: 'var(--ct-text-secondary)', fontSize: 13 }}>
          Raw transaction and network metadata in, ranked explainable leads out: entity resolution, laundering heuristics,
          graph ML, Tor/VPN correlation, analyst feedback and court-ready reports. Fully offline.
        </p>
      </div>
      <ForensicDisclaimer text="Leads are triage aids. Heuristics are inferences, probabilities are model output, and IP correlation is not attribution. Nothing here proves wrongdoing or identity." />
      {error && <div style={{ color: '#C9453E', fontSize: 13 }}>Error: {error}</div>}

      <StatusStrip
        status={status}
        busy={busy}
        files={files}
        onLoadFile={(path) => act(async () => { setSelected(null); setEntity(null); setGraph(null); await forensics.load({ source: 'file', path }); })}
        onReload={() => act(async () => { setSelected(null); setEntity(null); setGraph(null); await forensics.load({ source: 'synthetic', n_tx: status?.default_n_tx || 20000, seed: Math.floor(Math.random() * 1000) }); })}
        onRetrain={() => act(async () => { await forensics.retrain(0.1); })}
        onReset={() => act(async () => { await forensics.resetModel(); })}
      />

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(460px, 1fr))', gap: 16, alignItems: 'start' }}>
        <div className="ct-card" style={{ padding: 0 }}>
          <div style={{ padding: 12, display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center', borderBottom: '1px solid var(--ct-border)' }}>
            <h2 style={{ fontSize: 15, marginRight: 'auto' }}>Ranked leads {alerts ? `(${alerts.total.toLocaleString()})` : ''}</h2>
            <select value={tier} onChange={(e) => { setOffset(0); setTier(e.target.value); }}>
              <option value="">All tiers</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
            <input placeholder="E-123, address or TXID" value={search} onChange={(e) => { setOffset(0); setSearch(e.target.value); }} style={{ width: 190 }} />
          </div>
          {alerts ? (
            <AlertsTable data={alerts} selected={selected} onSelect={openEntity} basket={basket} toggleBasket={toggleBasket} />
          ) : <div style={{ padding: 24, color: 'var(--ct-text-muted)' }}>Building the first dataset...</div>}
          <div style={{ padding: 10, display: 'flex', gap: 8, alignItems: 'center', borderTop: '1px solid var(--ct-border)', fontSize: 12 }}>
            <button type="button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - LIMIT))}>Prev</button>
            <button type="button" disabled={!alerts || offset + LIMIT >= alerts.total} onClick={() => setOffset(offset + LIMIT)}>Next</button>
            <span style={{ flex: 1 }} />
            <span style={{ color: 'var(--ct-text-secondary)' }}>{basket.length} selected for report</span>
            <a
              href={basket.length ? forensics.reportUrl(basket, 'ChainTrace investigation report', 'analyst') : undefined}
              style={{ pointerEvents: basket.length ? 'auto' : 'none', opacity: basket.length ? 1 : 0.4, textDecoration: 'underline' }}
            >
              Download report (HTML, print to PDF)
            </a>
          </div>
          {uncertain.length > 0 && (
            <div style={{ padding: 10, borderTop: '1px solid var(--ct-border)', fontSize: 12 }}>
              <b>Most useful to label next</b> (model is least sure):{' '}
              {uncertain.map((u) => (
                <button key={u.entity_id} type="button" onClick={() => openEntity(u.entity_id)} style={{ margin: '2px 4px 2px 0', padding: '1px 6px', fontSize: 11 }}>
                  {u.label} ({pct(u.illicit_probability)})
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="ct-card" style={{ padding: 16, minHeight: 400 }}>
          {!selected && <div style={{ color: 'var(--ct-text-muted)' }}>Select a lead to see its evidence, link graph and model explanation.</div>}
          {selected && !entity && <div style={{ color: 'var(--ct-text-muted)' }}>Loading evidence for E-{selected}...</div>}
          {entity && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <h2>{entity.label}</h2>
                <RiskBadge tier={entity.score.tier} score={entity.score.final} />
                <span style={{ fontSize: 12, color: 'var(--ct-text-secondary)' }}>
                  rank #{entity.score.rank} of {entity.score.of.toLocaleString()} | {entity.n_addresses} addresses | {entity.first_seen} to {entity.last_seen}
                </span>
              </div>
              <div style={{ display: 'flex', gap: 6 }}>
                {['evidence', 'graph', 'model'].map((t) => (
                  <button key={t} type="button" className={tab === t ? 'primary' : ''} onClick={() => setTab(t)}>
                    {{ evidence: 'Evidence', graph: 'Link analysis', model: 'Model explanation' }[t]}
                  </button>
                ))}
              </div>
              {tab === 'evidence' && <Evidence e={entity} />}
              {tab === 'graph' && (graph ? <FlowGraph graph={graph} onSelect={openEntity} /> : <div>Loading graph...</div>)}
              {tab === 'model' && <ModelPanel e={entity} />}
              <FeedbackBox entityId={entity.entity_id} existing={entity.feedback} onSaved={refreshAfterFeedback} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
