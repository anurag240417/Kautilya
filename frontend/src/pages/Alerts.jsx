import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import RiskBadge from '../components/RiskBadge';
import SyntheticTag from '../components/SyntheticTag';
import ForensicDisclaimer from '../components/ForensicDisclaimer';
import { getAlerts, patchAlert } from '../api/client';
import { truncateIdentifier, formatTimestamp } from '../utils/formatters';
import { getEntityColor, getStatusColor } from '../utils/colors';

/**
 * Alerts Page per FRONTEND_BRIEF.md §3.
 * Full ranked triage list with sorting, filtering, and lifecycle status management.
 */
export default function Alerts() {
  const navigate = useNavigate();
  const [alerts, setAlerts] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filter state
  const [minTier, setMinTier] = useState('');
  const [entityType, setEntityType] = useState('');
  const [status, setStatus] = useState('');
  const [includeSynthetic, setIncludeSynthetic] = useState(true);
  const [minScore, setMinScore] = useState('');

  // Status updating state
  const [updatingAlertId, setUpdatingAlertId] = useState(null);

  async function loadAlerts() {
    try {
      setLoading(true);
      setError(null);
      const params = {};
      if (minTier) params.min_tier = minTier;
      if (entityType) params.entity_type = entityType;
      if (status) params.status = status;
      if (!includeSynthetic) params.include_synthetic = false;
      if (minScore) params.min_score = Number(minScore);

      const resp = await getAlerts(params);
      setAlerts(resp.alerts || []);
      setTotalCount(resp.total || 0);
    } catch (err) {
      console.error('Failed to load alerts:', err);
      setError(err.message || 'Error querying alert queue');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadAlerts();
  }, [minTier, entityType, status, includeSynthetic, minScore]);

  const handleStatusChange = async (alertId, newStatus) => {
    try {
      setUpdatingAlertId(alertId);
      await patchAlert(alertId, {
        status: newStatus,
        reviewer_notes: `Status updated to ${newStatus} via triage dashboard.`,
      });
      // Refresh alert in list
      setAlerts((prev) =>
        prev.map((a) => (a.alert_id === alertId ? { ...a, status: newStatus } : a))
      );
    } catch (err) {
      alert(`Failed to update alert: ${err.message}`);
    } finally {
      setUpdatingAlertId(null);
    }
  };

  const handleInvestigate = (entityId, type) => {
    navigate(`/investigation?entity=${encodeURIComponent(entityId)}&type=${encodeURIComponent(type || 'transaction')}`);
  };

  return (
    <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px', maxWidth: '1440px', margin: '0 auto' }}>
      {/* Page Header */}
      <div>
        <h1 style={{ fontSize: '22px', marginBottom: '4px' }}>Prioritized Alert Queue</h1>
        <p style={{ color: 'var(--ct-text-secondary)', fontSize: '13px' }}>
          Ranked investigative queue ordered by risk severity and multi-signal corroboration.
        </p>
      </div>

      <ForensicDisclaimer text="Alert ranks are for caseload triage prioritization only. Proximity or correlation does NOT establish guilt or wallet ownership." />

      {/* Filter Toolbar */}
      <div
        className="ct-card"
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          gap: '12px',
          padding: '14px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <label style={{ fontSize: '12px', color: 'var(--ct-text-secondary)' }}>Priority Tier:</label>
          <select value={minTier} onChange={(e) => setMinTier(e.target.value)}>
            <option value="">All Tiers</option>
            <option value="CRITICAL">Critical</option>
            <option value="HIGH">High</option>
            <option value="MEDIUM">Medium</option>
            <option value="LOW">Low</option>
          </select>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <label style={{ fontSize: '12px', color: 'var(--ct-text-secondary)' }}>Entity Type:</label>
          <select value={entityType} onChange={(e) => setEntityType(e.target.value)}>
            <option value="">All Types</option>
            <option value="transaction">Transaction</option>
            <option value="wallet">Wallet</option>
          </select>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <label style={{ fontSize: '12px', color: 'var(--ct-text-secondary)' }}>Status:</label>
          <select value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="">All Statuses</option>
            <option value="NEW">New</option>
            <option value="IN_REVIEW">In Review</option>
            <option value="ESCALATED">Escalated</option>
            <option value="CLOSED">Closed</option>
          </select>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <label style={{ fontSize: '12px', color: 'var(--ct-text-secondary)' }}>Min Score:</label>
          <input
            type="number"
            min="0"
            max="100"
            placeholder="0-100"
            value={minScore}
            onChange={(e) => setMinScore(e.target.value)}
            style={{ width: '80px' }}
          />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginLeft: 'auto' }}>
          <label style={{ fontSize: '12px', color: 'var(--ct-text-secondary)', display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={includeSynthetic}
              onChange={(e) => setIncludeSynthetic(e.target.checked)}
            />
            Include Synthetic Signals
          </label>
        </div>
      </div>

      {/* Alerts Table */}
      <div className="ct-card" style={{ padding: '0', overflowX: 'auto' }}>
        <div style={{ padding: '14px 16px', borderBottom: '1px solid var(--ct-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ fontSize: '15px' }}>
            Ranked Alert Results ({alerts.length} of {totalCount})
          </h2>
          <button type="button" onClick={loadAlerts} style={{ fontSize: '11px', padding: '3px 8px' }}>
            Refresh
          </button>
        </div>

        {loading ? (
          <div style={{ padding: '32px', textAlign: 'center', color: 'var(--ct-text-muted)' }}>
            Loading alert queue...
          </div>
        ) : error ? (
          <div style={{ padding: '24px', color: '#C9453E', textAlign: 'center' }}>
            {error}
          </div>
        ) : alerts.length === 0 ? (
          <div style={{ padding: '32px', textAlign: 'center', color: 'var(--ct-text-muted)' }}>
            No alerts match current filter criteria.
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Rank & Entity</th>
                <th>Priority Tier</th>
                <th>Score</th>
                <th>Headline & Summary</th>
                <th>Active Signals</th>
                <th>Lifecycle Status</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map((alert, idx) => {
                const entType = alert.entity_type || 'transaction';
                const entColor = getEntityColor(entType);
                const isSynth = Boolean(alert.contains_synthetic_input);

                return (
                  <tr key={alert.alert_id} className={isSynth ? 'ct-synthetic-marker' : ''}>
                    {/* Entity & Rank */}
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontFamily: 'var(--ct-font-mono)', color: 'var(--ct-text-muted)', fontSize: '11px' }}>
                          #{idx + 1}
                        </span>
                        <span
                          style={{
                            fontSize: '10px',
                            fontWeight: 700,
                            textTransform: 'uppercase',
                            color: entColor,
                            backgroundColor: `${entColor}1A`,
                            padding: '2px 4px',
                            borderRadius: '2px',
                          }}
                        >
                          {entType}
                        </span>
                        <span
                          style={{
                            fontFamily: 'var(--ct-font-mono)',
                            fontWeight: 600,
                            fontSize: '12px',
                            cursor: 'pointer',
                            color: 'var(--ct-text-primary)',
                          }}
                          onClick={() => handleInvestigate(alert.entity_id, entType)}
                          title={`Investigate ${alert.entity_id}`}
                        >
                          {truncateIdentifier(alert.entity_id, 10, 6)}
                        </span>
                        {isSynth && <SyntheticTag />}
                      </div>
                    </td>

                    {/* Priority Tier */}
                    <td>
                      <RiskBadge tier={alert.priority_tier} />
                    </td>

                    {/* Score */}
                    <td>
                      <span style={{ fontFamily: 'var(--ct-font-mono)', fontWeight: 700 }}>
                        {alert.risk_score ? alert.risk_score.toFixed(1) : '—'}
                      </span>
                    </td>

                    {/* Headline & Summary */}
                    <td style={{ maxWidth: '320px' }}>
                      <div style={{ fontWeight: 600, fontSize: '12px', marginBottom: '2px' }}>
                        {alert.headline}
                      </div>
                      <div
                        style={{
                          fontSize: '11px',
                          color: 'var(--ct-text-secondary)',
                          lineHeight: '1.3',
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                          overflow: 'hidden',
                        }}
                      >
                        {alert.summary}
                      </div>
                    </td>

                    {/* Active Signals */}
                    <td>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '3px', maxWidth: '160px' }}>
                        {alert.active_signals && alert.active_signals.length > 0 ? (
                          alert.active_signals.map((sig) => (
                            <span
                              key={sig}
                              style={{
                                fontSize: '10px',
                                fontFamily: 'var(--ct-font-mono)',
                                backgroundColor: 'var(--ct-bg-elevated)',
                                border: '1px solid var(--ct-border)',
                                padding: '1px 4px',
                                borderRadius: '2px',
                                color: 'var(--ct-text-secondary)',
                              }}
                            >
                              {sig}
                            </span>
                          ))
                        ) : (
                          <span style={{ color: 'var(--ct-text-muted)', fontSize: '11px' }}>composite</span>
                        )}
                      </div>
                    </td>

                    {/* Lifecycle Status dropdown */}
                    <td>
                      <select
                        value={alert.status}
                        disabled={updatingAlertId === alert.alert_id}
                        onChange={(e) => handleStatusChange(alert.alert_id, e.target.value)}
                        style={{
                          fontSize: '11px',
                          padding: '3px 6px',
                          color: getStatusColor(alert.status),
                          borderColor: `${getStatusColor(alert.status)}44`,
                        }}
                      >
                        <option value="NEW">NEW</option>
                        <option value="IN_REVIEW">IN REVIEW</option>
                        <option value="ESCALATED">ESCALATED</option>
                        <option value="CLOSED">CLOSED</option>
                      </select>
                    </td>

                    {/* Created */}
                    <td style={{ fontSize: '11px', color: 'var(--ct-text-muted)', whiteSpace: 'nowrap' }}>
                      {formatTimestamp(alert.created_at)}
                    </td>

                    {/* Action */}
                    <td>
                      <button
                        type="button"
                        className="primary"
                        onClick={() => handleInvestigate(alert.entity_id, entType)}
                        style={{ fontSize: '11px', padding: '3px 8px' }}
                      >
                        Trace
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
