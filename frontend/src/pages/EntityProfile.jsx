import React, { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import SearchInput from '../components/SearchInput';
import RiskBadge from '../components/RiskBadge';
import RiskBreakdown from '../components/RiskBreakdown';
import SyntheticTag from '../components/SyntheticTag';
import ForensicDisclaimer from '../components/ForensicDisclaimer';
import InvestigationGraph from '../components/InvestigationGraph';
import { getTransaction, getWallet, getGraph } from '../api/client';
import { formatBTC, truncateIdentifier } from '../utils/formatters';
import { getEntityColor } from '../utils/colors';

/**
 * Entity Profile Page per FRONTEND_BRIEF.md §3.
 * Deep-dive on one wallet/transaction/IP.
 * Behavioral statistics, direct connections, aggregation traceability, and mini-graph.
 */
export default function EntityProfile() {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const queryEntity = searchParams.get('id') || '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa';
  const queryType = searchParams.get('type') || 'wallet';

  const [activeId, setActiveId] = useState(queryEntity);
  const [activeType, setActiveType] = useState(queryType);
  const [aggregationMethod, setAggregationMethod] = useState('max');

  const [data, setData] = useState(null);
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadEntity() {
      if (!activeId) return;
      setLoading(true);
      setError(null);
      try {
        let result = null;
        if (activeType === 'transaction') {
          result = await getTransaction(activeId);
        } else {
          result = await getWallet(activeId, aggregationMethod);
        }
        setData(result);

        // Fetch mini ego-graph
        try {
          const g = await getGraph(activeId, { depth: 1, maxNodes: 30 });
          if (g && g.nodes) {
            setGraphData({ nodes: g.nodes, edges: g.edges });
          }
        } catch {
          setGraphData({
            nodes: [{ id: activeId, type: activeType, is_synthetic: result?.is_synthetic }],
            edges: [],
          });
        }
      } catch (err) {
        console.error('Failed to load entity profile:', err);
        setError(err.message || 'Entity not found in forensic database');
        setData(null);
      } finally {
        setLoading(false);
      }
    }

    loadEntity();
  }, [activeId, activeType, aggregationMethod]);

  const handleSearch = (id, type) => {
    setActiveId(id);
    setActiveType(type);
    setSearchParams({ id, type });
  };

  const handleTrace = () => {
    navigate(`/investigation?entity=${encodeURIComponent(activeId)}&type=${encodeURIComponent(activeType)}`);
  };

  const isSynth = Boolean(data?.is_synthetic);
  const risk = data?.risk_score || (data?.aggregation ? {
    score: data.aggregation.score,
    priority_tier: data.aggregation.priority_tier,
    explanation: data.aggregation.summary,
    active_signals: ['aggregation_traceable'],
  } : null);

  return (
    <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px', maxWidth: '1440px', margin: '0 auto' }}>
      {/* Header & Search */}
      <div className="ct-card">
        <h1 style={{ fontSize: '20px', marginBottom: '8px' }}>Entity Forensic Profile</h1>
        <SearchInput onSearch={handleSearch} initialEntity={activeId} initialType={activeType} />
      </div>

      <ForensicDisclaimer text={data?.forensic_disclaimer} />

      {error && (
        <div className="ct-card" style={{ color: '#C9453E', borderLeft: '4px solid #C9453E' }}>
          <strong>Query Notice:</strong> {error}
        </div>
      )}

      {loading ? (
        <div style={{ padding: '32px', textAlign: 'center', color: 'var(--ct-text-muted)' }}>
          Loading forensic entity profile...
        </div>
      ) : data ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Main Entity Header Card */}
          <div
            className={`ct-card ${isSynth ? 'ct-synthetic-marker' : ''}`}
            style={{
              padding: '20px',
              borderLeft: isSynth ? '4px dashed #7B68AE' : `4px solid ${getEntityColor(activeType)}`,
              display: 'flex',
              flexDirection: 'column',
              gap: '14px',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                  <span
                    style={{
                      fontSize: '11px',
                      fontWeight: 700,
                      textTransform: 'uppercase',
                      color: getEntityColor(activeType),
                      backgroundColor: `${getEntityColor(activeType)}1A`,
                      padding: '2px 6px',
                      borderRadius: '2px',
                    }}
                  >
                    {activeType}
                  </span>
                  {isSynth && <SyntheticTag />}
                  <span style={{ fontSize: '12px', color: 'var(--ct-text-muted)' }}>
                    Temporal Step: {data.time_step ?? '—'}
                  </span>
                </div>

                <div style={{ fontFamily: 'var(--ct-font-mono)', fontSize: '18px', fontWeight: 700, wordBreak: 'break-all' }}>
                  {activeId}
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <button type="button" className="primary" onClick={handleTrace}>
                  Open Full Link Investigation
                </button>
              </div>
            </div>

            {/* Risk Prioritization & Breakdown */}
            {risk && (
              <div
                style={{
                  padding: '14px',
                  backgroundColor: 'var(--ct-bg-elevated)',
                  border: '1px solid var(--ct-border)',
                  borderRadius: '2px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '10px',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '12px', textTransform: 'uppercase', fontWeight: 600, color: 'var(--ct-text-secondary)' }}>
                    Triage Priority Assessment
                  </span>
                  <RiskBadge tier={risk.priority_tier} score={risk.score} />
                </div>

                <RiskBreakdown riskScore={risk} />

                {risk.explanation && (
                  <div style={{ fontSize: '12px', color: 'var(--ct-text-secondary)', lineHeight: '1.4' }}>
                    {risk.explanation}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Aggregation Traceability for Wallets per AGENTS.md §3.7 */}
          {activeType === 'wallet' && data.aggregation && (
            <div className="ct-card">
              <div className="ct-card-header">
                <div>
                  <h2 style={{ fontSize: '15px' }}>Entity-Level Rollup Traceability</h2>
                  <div style={{ fontSize: '12px', color: 'var(--ct-text-secondary)', marginTop: '2px' }}>
                    Strict audit trail: Transaction scores never become blanket accusations against wallets.
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <label style={{ fontSize: '12px', color: 'var(--ct-text-secondary)' }}>Pooling Method:</label>
                  <select
                    value={aggregationMethod}
                    onChange={(e) => setAggregationMethod(e.target.value)}
                    style={{ fontSize: '12px', padding: '3px 8px' }}
                  >
                    <option value="max">Maximum (Peak)</option>
                    <option value="volume_weighted">Volume-Weighted</option>
                    <option value="mean">Arithmetic Mean</option>
                    <option value="frequency">Frequency Threshold</option>
                  </select>
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ fontSize: '13px', color: 'var(--ct-text-primary)' }}>
                  Aggregated Score: <strong style={{ fontFamily: 'var(--ct-font-mono)' }}>{data.aggregation.score.toFixed(1)}</strong> via <code>{data.aggregation.method}</code> across {data.aggregation.contributing_transaction_count} transactions.
                </div>

                {data.aggregation.contributing_scores && data.aggregation.contributing_scores.length > 0 ? (
                  <table>
                    <thead>
                      <tr>
                        <th>Contributing TxID</th>
                        <th>Risk Score</th>
                        <th>Priority Tier</th>
                        <th>Illicit Prob</th>
                        <th>Origin</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.aggregation.contributing_scores.map((cs) => (
                        <tr key={cs.transaction_id}>
                          <td style={{ fontFamily: 'var(--ct-font-mono)' }}>
                            <button
                              type="button"
                              onClick={() => handleSearch(String(cs.transaction_id), 'transaction')}
                              style={{ padding: '2px 6px', fontSize: '11px' }}
                            >
                              Tx #{cs.transaction_id}
                            </button>
                          </td>
                          <td style={{ fontFamily: 'var(--ct-font-mono)' }}>{cs.score.toFixed(1)}</td>
                          <td><RiskBadge tier={cs.priority_tier} /></td>
                          <td style={{ fontFamily: 'var(--ct-font-mono)' }}>
                            {cs.illicit_probability ? `${(cs.illicit_probability * 100).toFixed(0)}%` : '—'}
                          </td>
                          <td>{cs.is_synthetic ? <SyntheticTag /> : 'Real'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div style={{ color: 'var(--ct-text-muted)', fontSize: '12px' }}>
                    No individual transaction scores mapped.
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Behavioral Stats & Mini Ego-Graph Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
            {/* Behavioral Statistics */}
            <div className="ct-card">
              <h2 style={{ fontSize: '15px', marginBottom: '12px' }}>Interpretable Behavioral Profile</h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '10px' }}>
                {activeType === 'transaction' && data.interpretable_features ? (
                  Object.entries(data.interpretable_features).map(([k, v]) => (
                    <div key={k} style={{ padding: '8px 10px', backgroundColor: 'var(--ct-bg-elevated)', border: '1px solid var(--ct-border)' }}>
                      <div style={{ fontSize: '10px', color: 'var(--ct-text-secondary)', textTransform: 'uppercase' }}>
                        {k.replace(/_/g, ' ')}
                      </div>
                      <div style={{ fontFamily: 'var(--ct-font-mono)', fontWeight: 600, fontSize: '13px', marginTop: '2px' }}>
                        {v !== null && v !== undefined ? (typeof v === 'number' ? v.toLocaleString() : String(v)) : '—'}
                      </div>
                    </div>
                  ))
                ) : data.stats ? (
                  Object.entries(data.stats).map(([k, v]) => (
                    <div key={k} style={{ padding: '8px 10px', backgroundColor: 'var(--ct-bg-elevated)', border: '1px solid var(--ct-border)' }}>
                      <div style={{ fontSize: '10px', color: 'var(--ct-text-secondary)', textTransform: 'uppercase' }}>
                        {k.replace(/_/g, ' ')}
                      </div>
                      <div style={{ fontFamily: 'var(--ct-font-mono)', fontWeight: 600, fontSize: '13px', marginTop: '2px' }}>
                        {typeof v === 'object' ? JSON.stringify(v) : String(v ?? '—')}
                      </div>
                    </div>
                  ))
                ) : null}
              </div>
            </div>

            {/* Mini Ego-Graph */}
            <div className="ct-card" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2 style={{ fontSize: '15px' }}>Direct Link Neighborhood</h2>
                <span style={{ fontSize: '11px', color: 'var(--ct-text-muted)' }}>
                  {graphData.nodes.length} nodes
                </span>
              </div>

              <InvestigationGraph
                nodes={graphData.nodes}
                edges={graphData.edges}
                rootEntityId={activeId}
                height="320px"
              />
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
