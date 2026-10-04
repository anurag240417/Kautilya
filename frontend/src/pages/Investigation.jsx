import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import SearchInput from '../components/SearchInput';
import InvestigationGraph from '../components/InvestigationGraph';
import EvidenceTrail from '../components/EvidenceTrail';
import EntityTimeline from '../components/EntityTimeline';
import RiskBadge from '../components/RiskBadge';
import RiskBreakdown from '../components/RiskBreakdown';
import SyntheticTag from '../components/SyntheticTag';
import ForensicDisclaimer from '../components/ForensicDisclaimer';
import { getTransaction, getWallet, getGraph } from '../api/client';
import { formatBTC, formatScore } from '../utils/formatters';
import { getEntityColor } from '../utils/colors';

/**
 * Investigation Page per FRONTEND_BRIEF.md §3 & §4.
 * The core deliverable — trace one entity's evidence.
 * Interactive ego-graph + evidence-trail sequential reveal + forensic timeline.
 */
export default function Investigation() {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const initialEntity = searchParams.get('entity') || '1001';
  const initialType = searchParams.get('type') || 'transaction';

  const [activeEntityId, setActiveEntityId] = useState(initialEntity);
  const [activeEntityType, setActiveEntityType] = useState(initialType);

  const [entityData, setEntityData] = useState(null);
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('timeline');

  // Load entity details and graph subgraph
  const fetchEntityInvestigation = useCallback(async (id, type) => {
    if (!id) return;
    setLoading(true);
    setError(null);

    try {
      let details = null;
      if (type === 'transaction') {
        details = await getTransaction(id);
      } else {
        details = await getWallet(id);
      }

      setEntityData(details);

      // Fetch subgraph for visual link analysis
      try {
        const g = await getGraph(id, { depth: 1, maxNodes: 50 });
        let currentNodes = g?.nodes ? [...g.nodes] : [];
        let currentEdges = g?.edges ? [...g.edges] : [];

        // Enrich with correlated network IP observations if present
        if (details?.correlations && details.correlations.length > 0) {
          details.correlations.forEach((c) => {
            const ipId = String(c.candidate_ip);
            if (!currentNodes.some((n) => String(n.id) === ipId)) {
              currentNodes.push({
                id: ipId,
                type: 'network',
                label: ipId,
                is_synthetic: Boolean(c.is_synthetic),
                asn: c.asn,
                country: c.country,
                role: c.role || 'network_observation',
                confidence: c.correlation_confidence,
              });
            }
            const edgeExists = currentEdges.some(
              (e) => String(e.source) === String(id) && String(e.target) === ipId
            );
            if (!edgeExists) {
              currentEdges.push({
                source: String(id),
                target: ipId,
                relationship: c.role ? `${c.role}_ip` : 'network_ip',
                is_synthetic: Boolean(c.is_synthetic),
                confidence: c.correlation_confidence,
                provenance: 'synthetic_simulation',
              });
            }
          });
        }

        setGraphData({ nodes: currentNodes, edges: currentEdges });
      } catch (gErr) {
        console.warn('Subgraph fetch failed, fallback to direct node:', gErr);
        const fallbackNodes = [
          {
            id: String(id),
            type: type,
            priority_tier: details?.risk_score?.priority_tier || details?.aggregation?.priority_tier,
            risk_score: details?.risk_score?.score || details?.aggregation?.score,
            is_synthetic: details?.is_synthetic,
          },
        ];
        const fallbackEdges = [];
        if (details?.correlations && details.correlations.length > 0) {
          details.correlations.forEach((c) => {
            const ipId = String(c.candidate_ip);
            fallbackNodes.push({
              id: ipId,
              type: 'network',
              label: ipId,
              is_synthetic: Boolean(c.is_synthetic),
              asn: c.asn,
              country: c.country,
              role: c.role || 'network_observation',
              confidence: c.correlation_confidence,
            });
            fallbackEdges.push({
              source: String(id),
              target: ipId,
              relationship: c.role ? `${c.role}_ip` : 'network_ip',
              is_synthetic: Boolean(c.is_synthetic),
              confidence: c.correlation_confidence,
              provenance: 'synthetic_simulation',
            });
          });
        }
        setGraphData({
          nodes: fallbackNodes,
          edges: fallbackEdges,
        });
      }
    } catch (err) {
      console.error('Failed to load entity investigation:', err);
      setError(err.message || 'Entity not found in forensic database');
      setEntityData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchEntityInvestigation(activeEntityId, activeEntityType);

    const handleSimUpdate = () => {
      if (activeEntityId) {
        fetchEntityInvestigation(activeEntityId, activeEntityType);
      }
    };
    window.addEventListener('kautilya-simulation-update', handleSimUpdate);
    return () => window.removeEventListener('kautilya-simulation-update', handleSimUpdate);
  }, [activeEntityId, activeEntityType, fetchEntityInvestigation]);

  const handleSearch = (id, type) => {
    // Long hex transaction ids belong to the Forensics Lab (raw-transaction analysis);
    // this page only knows the numeric Elliptic++ ids.
    if (type === 'transaction' && /^[0-9a-f]{8,64}$/i.test(id) && /[a-f]/i.test(id)) {
      navigate(`/forensics?search=${encodeURIComponent(id)}`);
      return;
    }
    setActiveEntityId(id);
    setActiveEntityType(type);
    setSearchParams({ entity: id, type });
  };

  const handleSelectGraphNode = (nodeId, nodeType) => {
    if (nodeType === 'network' || nodeType === 'ip') {
      // IP nodes are network telemetry observations, not queryable transactions/wallets
      return;
    }
    if (String(nodeId) !== String(activeEntityId)) {
      handleSearch(nodeId, nodeType || 'transaction');
    }
  };

  // Compile evidence records from Evidence Ledger
  const evidenceRecords =
    entityData?.risk_score?.evidence_ledger?.records ||
    entityData?.aggregation?.contributing_scores?.map((cs, idx) => ({
      evidence_id: `ev-agg-${idx}`,
      category: 'ml_behavioral',
      headline: `Contributing Transaction #${cs.transaction_id}`,
      description: `Contributing transaction evaluated with priority score ${cs.score.toFixed(1)} (${cs.priority_tier}).`,
      confidence: cs.illicit_probability || 0.8,
      is_synthetic: Boolean(cs.is_synthetic),
    })) ||
    [];

  // Compile timeline events
  const timelineEvents = [];
  if (entityData) {
    if (activeEntityType === 'transaction') {
      timelineEvents.push({
        id: 'ev-init-tx',
        title: `Transaction Recorded at Time Step ${entityData.time_step}`,
        description: `Elliptic++ block appearance with total volume ${formatBTC(entityData.interpretable_features?.total_btc)}.`,
        time_step: entityData.time_step,
        is_synthetic: false,
      });

      if (entityData.correlations && entityData.correlations.length > 0) {
        entityData.correlations.forEach((c, idx) => {
          timelineEvents.push({
            id: `ev-corr-${idx}`,
            title: `Network Observation: IP ${c.candidate_ip}`,
            description: `Temporal correlation observed (confidence: ${(c.correlation_confidence * 100).toFixed(0)}%). Role: ${c.role || 'relay'}.`,
            candidate_ip: c.candidate_ip,
            asn: c.asn,
            country: c.country,
            timestamp: c.timestamp,
            is_synthetic: Boolean(c.is_synthetic),
          });
        });
      }
    } else {
      timelineEvents.push({
        id: 'ev-wallet-first',
        title: `First Observed at Block ${entityData.stats?.first_block_appeared_in ?? '—'}`,
        description: `Wallet lifetime spans ${entityData.stats?.lifetime_in_blocks ?? '—'} blocks with ${entityData.stats?.total_txs ?? '—'} total transactions.`,
        time_step: entityData.time_step,
        is_synthetic: false,
      });
    }

    // Add ledger records to timeline
    evidenceRecords.forEach((rec, idx) => {
      timelineEvents.push({
        id: `tl-rec-${idx}`,
        title: rec.headline,
        description: rec.description,
        is_synthetic: rec.is_synthetic,
      });
    });
  }

  const effectiveRisk = entityData?.risk_score || (entityData?.aggregation ? {
    score: entityData.aggregation.score,
    priority_tier: entityData.aggregation.priority_tier,
    explanation: entityData.aggregation.summary || 'Aggregated risk score across transactions.',
    active_signals: ['aggregation_max'],
  } : null);

  const isSynthetic = Boolean(entityData?.is_synthetic);

  return (
    <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px', maxWidth: '1440px', margin: '0 auto' }}>
      {/* Top Search & Triage Navigation */}
      <div className="ct-card">
        <h2 style={{ fontSize: '15px', marginBottom: '8px' }}>Forensic Entity Trace & Link Analysis</h2>
        <SearchInput onSearch={handleSearch} initialEntity={activeEntityId} initialType={activeEntityType} />
      </div>

      <ForensicDisclaimer text={entityData?.forensic_disclaimer} />

      {/* Entity Summary Bar */}
      {entityData && (
        <div
          className={`ct-card ${isSynthetic ? 'ct-synthetic-marker' : ''}`}
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '16px',
            borderLeft: isSynthetic ? '4px dashed #7B68AE' : `4px solid ${getEntityColor(activeEntityType)}`,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <span
              style={{
                fontSize: '12px',
                fontWeight: 700,
                textTransform: 'uppercase',
                color: getEntityColor(activeEntityType),
                backgroundColor: `${getEntityColor(activeEntityType)}1A`,
                padding: '4px 8px',
                borderRadius: '2px',
              }}
            >
              {activeEntityType}
            </span>

            <span style={{ fontFamily: 'var(--ct-font-mono)', fontSize: '15px', fontWeight: 600 }}>
              {activeEntityId}
            </span>

            {isSynthetic && <SyntheticTag />}

            <span style={{ fontSize: '12px', color: 'var(--ct-text-muted)' }}>
              Step: {entityData.time_step ?? '—'}
            </span>

            {entityData.label && (
              <span
                style={{
                  fontSize: '11px',
                  fontWeight: 600,
                  padding: '2px 6px',
                  borderRadius: '2px',
                  border: '1px solid var(--ct-border)',
                  color: entityData.label === 1 ? '#C9453E' : entityData.label === 2 ? '#4A7A5C' : '#5C6E7E',
                }}
              >
                Label: {entityData.label === 1 ? 'ILLICIT' : entityData.label === 2 ? 'LICIT' : 'UNKNOWN'}
              </span>
            )}
          </div>

          {effectiveRisk && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
              <RiskBreakdown riskScore={effectiveRisk} compact={true} />
              <RiskBadge tier={effectiveRisk.priority_tier} score={effectiveRisk.score} />
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="ct-card" style={{ color: '#C9453E', borderLeft: '4px solid #C9453E' }}>
          <strong>Query Notice:</strong> {error}
        </div>
      )}

      {/* Main Analysis Grid: Link Graph (Left 60%) + Evidence Trail Reveal (Right 40%) */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(400px, 1.2fr) minmax(350px, 0.8fr)',
          gap: '20px',
        }}
      >
        {/* Left Column: Link Analysis Graph */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2 style={{ fontSize: '15px' }}>Ego Subgraph & Link Analysis</h2>
            <span style={{ fontSize: '11px', color: 'var(--ct-text-muted)' }}>
              {graphData.nodes.length} nodes, {graphData.edges.length} edges
            </span>
          </div>

          <InvestigationGraph
            nodes={graphData.nodes}
            edges={graphData.edges}
            rootEntityId={activeEntityId}
            onSelectNode={handleSelectGraphNode}
            height="580px"
          />
        </div>

        {/* Right Column: Evidence Trail Reveal (FRONTEND_BRIEF.md §4) */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <EvidenceTrail
            seedEntity={{
              id: activeEntityId,
              type: activeEntityType,
              isSynthetic: isSynthetic,
            }}
            evidenceRecords={evidenceRecords}
            riskScore={effectiveRisk}
          />
        </div>
      </div>

      {/* Bottom Section: Tabs for Timeline, Features, and Connections */}
      <div className="ct-card">
        <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid var(--ct-border)', paddingBottom: '10px', marginBottom: '16px' }}>
          <button
            type="button"
            className={activeTab === 'timeline' ? 'primary' : ''}
            onClick={() => setActiveTab('timeline')}
          >
            Forensic Timeline
          </button>
          <button
            type="button"
            className={activeTab === 'features' ? 'primary' : ''}
            onClick={() => setActiveTab('features')}
          >
            Interpretable Features
          </button>
          <button
            type="button"
            className={activeTab === 'correlations' ? 'primary' : ''}
            onClick={() => setActiveTab('correlations')}
          >
            Correlated Observations ({entityData?.correlations?.length || 0})
          </button>
          <button
            type="button"
            className={activeTab === 'connections' ? 'primary' : ''}
            onClick={() => setActiveTab('connections')}
          >
            Connected Entities
          </button>
        </div>

        {/* Tab 1: Chronological Timeline */}
        {activeTab === 'timeline' && <EntityTimeline events={timelineEvents} />}

        {/* Tab 2: Interpretable Features (No anonymized features per §3.5) */}
        {activeTab === 'features' && (
          <div>
            <h3 style={{ fontSize: '14px', marginBottom: '10px' }}>Interpretable Behavioral Dimensions</h3>
            <div style={{ fontSize: '12px', color: 'var(--ct-text-muted)', marginBottom: '12px' }}>
              Forensic rule: Anonymized feature names are suppressed. Only verified behavioral dimensions are exposed.
            </div>

            {activeEntityType === 'transaction' && entityData?.interpretable_features ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '10px' }}>
                {Object.entries(entityData.interpretable_features).map(([k, v]) => (
                  <div key={k} style={{ padding: '8px 12px', backgroundColor: 'var(--ct-bg-elevated)', border: '1px solid var(--ct-border)' }}>
                    <div style={{ fontSize: '11px', color: 'var(--ct-text-secondary)', textTransform: 'uppercase' }}>
                      {k.replace(/_/g, ' ')}
                    </div>
                    <div style={{ fontFamily: 'var(--ct-font-mono)', fontWeight: 600, fontSize: '13px', marginTop: '2px' }}>
                      {v !== null && v !== undefined ? (typeof v === 'number' ? v.toLocaleString() : String(v)) : '—'}
                    </div>
                  </div>
                ))}
              </div>
            ) : entityData?.stats ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '10px' }}>
                {Object.entries(entityData.stats).map(([k, v]) => (
                  <div key={k} style={{ padding: '8px 12px', backgroundColor: 'var(--ct-bg-elevated)', border: '1px solid var(--ct-border)' }}>
                    <div style={{ fontSize: '11px', color: 'var(--ct-text-secondary)', textTransform: 'uppercase' }}>
                      {k.replace(/_/g, ' ')}
                    </div>
                    <div style={{ fontFamily: 'var(--ct-font-mono)', fontWeight: 600, fontSize: '13px', marginTop: '2px' }}>
                      {typeof v === 'object' ? JSON.stringify(v) : String(v ?? '—')}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ color: 'var(--ct-text-muted)', fontSize: '12px' }}>No interpretable features recorded.</div>
            )}
          </div>
        )}

        {/* Tab 3: Correlated Network Observations */}
        {activeTab === 'correlations' && (
          <div>
            <h3 style={{ fontSize: '14px', marginBottom: '10px' }}>Correlated Network Events</h3>
            <div style={{ fontSize: '12px', color: 'var(--ct-text-muted)', marginBottom: '12px' }}>
              Temporal correlation alone does NOT prove identity or wallet ownership. All IP observations derive from synthetic simulation.
            </div>

            {entityData?.correlations && entityData.correlations.length > 0 ? (
              <table>
                <thead>
                  <tr>
                    <th>Candidate IP</th>
                    <th>Role</th>
                    <th>Confidence</th>
                    <th>ASN</th>
                    <th>Country</th>
                    <th>Script Type</th>
                    <th>Origin</th>
                  </tr>
                </thead>
                <tbody>
                  {entityData.correlations.map((c, idx) => (
                    <tr key={idx}>
                      <td style={{ fontFamily: 'var(--ct-font-mono)' }}>{c.candidate_ip}</td>
                      <td>{c.role || 'relay'}</td>
                      <td>
                        <span style={{ fontWeight: 600, color: '#4A90D9' }}>
                          {((c.correlation_confidence || 0) * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td>{c.asn || '—'}</td>
                      <td>{c.country || '—'}</td>
                      <td><code>{c.script_type || 'P2PKH'}</code></td>
                      <td>{c.is_synthetic ? <SyntheticTag /> : 'Real'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div style={{ color: 'var(--ct-text-muted)', fontSize: '12px' }}>No network correlations recorded for this entity.</div>
            )}
          </div>
        )}

        {/* Tab 4: Connected Entities */}
        {activeTab === 'connections' && (
          <div>
            <h3 style={{ fontSize: '14px', marginBottom: '10px' }}>Direct Link Connections</h3>
            {activeEntityType === 'transaction' && entityData?.connected_wallets ? (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                <div>
                  <div style={{ fontWeight: 600, fontSize: '12px', marginBottom: '8px', color: 'var(--ct-text-secondary)' }}>
                    Input Wallets ({entityData.connected_wallets.inputs?.length || 0})
                  </div>
                  {entityData.connected_wallets.inputs?.length > 0 ? (
                    <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                      {entityData.connected_wallets.inputs.map((w) => (
                        <li key={w}>
                          <button
                            type="button"
                            onClick={() => handleSearch(w, 'wallet')}
                            style={{ fontFamily: 'var(--ct-font-mono)', fontSize: '12px', padding: '4px 8px', width: '100%', textAlign: 'left' }}
                          >
                            {w}
                          </button>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <div style={{ color: 'var(--ct-text-muted)', fontSize: '12px' }}>None listed</div>
                  )}
                </div>

                <div>
                  <div style={{ fontWeight: 600, fontSize: '12px', marginBottom: '8px', color: 'var(--ct-text-secondary)' }}>
                    Output Wallets ({entityData.connected_wallets.outputs?.length || 0})
                  </div>
                  {entityData.connected_wallets.outputs?.length > 0 ? (
                    <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                      {entityData.connected_wallets.outputs.map((w) => (
                        <li key={w}>
                          <button
                            type="button"
                            onClick={() => handleSearch(w, 'wallet')}
                            style={{ fontFamily: 'var(--ct-font-mono)', fontSize: '12px', padding: '4px 8px', width: '100%', textAlign: 'left' }}
                          >
                            {w}
                          </button>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <div style={{ color: 'var(--ct-text-muted)', fontSize: '12px' }}>None listed</div>
                  )}
                </div>
              </div>
            ) : entityData?.counterparties ? (
              <div>
                <div style={{ fontWeight: 600, fontSize: '12px', marginBottom: '8px', color: 'var(--ct-text-secondary)' }}>
                  Counterparties ({entityData.counterparties.length})
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {entityData.counterparties.map((cp) => (
                    <button
                      key={cp}
                      type="button"
                      onClick={() => handleSearch(cp, 'wallet')}
                      style={{ fontFamily: 'var(--ct-font-mono)', fontSize: '12px', padding: '4px 8px' }}
                    >
                      {cp}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ color: 'var(--ct-text-muted)', fontSize: '12px' }}>No direct connections found.</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
