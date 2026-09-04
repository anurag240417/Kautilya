import React from 'react';

/**
 * Renders an always-visible path to score components per FRONTEND_BRIEF.md §5.
 * Format: "35% model + 25% anomaly + 20% graph + 20% correlation"
 */
export default function RiskBreakdown({ riskScore, compact = false }) {
  if (!riskScore) return null;

  const {
    behavioral_signal,
    anomaly_signal,
    graph_signal,
    correlation_signal,
    known_indicator_signal,
    active_signals = [],
  } = riskScore;

  const components = [];
  if (behavioral_signal !== undefined && behavioral_signal !== null) {
    components.push({ label: 'Behavioral ML', value: behavioral_signal, key: 'behavioral' });
  }
  if (anomaly_signal !== undefined && anomaly_signal !== null) {
    components.push({ label: 'Anomaly', value: anomaly_signal, key: 'anomaly' });
  }
  if (graph_signal !== undefined && graph_signal !== null) {
    components.push({ label: 'Graph Topology', value: graph_signal, key: 'graph' });
  }
  if (correlation_signal !== undefined && correlation_signal !== null) {
    components.push({ label: 'Network Correlation', value: correlation_signal, key: 'correlation' });
  }
  if (known_indicator_signal !== undefined && known_indicator_signal !== null) {
    components.push({ label: 'Known Indicator', value: known_indicator_signal, key: 'known_indicator' });
  }

  // If no decomposed signals are provided, fall back to active_signals list
  if (components.length === 0 && active_signals.length > 0) {
    return (
      <div style={{ fontSize: '11px', color: 'var(--ct-text-secondary)' }}>
        <span style={{ fontWeight: 600 }}>Active signals:</span> {active_signals.join(' + ')}
      </div>
    );
  }

  if (components.length === 0) {
    return null;
  }

  const sumValues = components.reduce((acc, c) => acc + c.value, 0);
  const normalized = components.map(c => ({
    ...c,
    pct: sumValues > 0 ? Math.round((c.value / sumValues) * 100) : Math.round(100 / components.length),
  }));

  const oneLineFormula = normalized.map(c => `${c.pct}% ${c.label}`).join(' + ');

  if (compact) {
    return (
      <div
        style={{
          fontSize: '11px',
          color: 'var(--ct-text-secondary)',
          fontFamily: 'var(--ct-font-mono)',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
        title={`Component weighting: ${oneLineFormula}`}
      >
        {oneLineFormula}
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          fontSize: '11px',
          color: 'var(--ct-text-secondary)',
          fontFamily: 'var(--ct-font-mono)',
        }}
      >
        <span style={{ color: 'var(--ct-text-muted)', textTransform: 'uppercase', fontSize: '10px' }}>
          Decomposition:
        </span>
        <span>{oneLineFormula}</span>
      </div>

      {/* Segmented relative contribution bar */}
      <div
        style={{
          display: 'flex',
          height: '6px',
          borderRadius: '1px',
          overflow: 'hidden',
          backgroundColor: 'var(--ct-bg-elevated)',
          border: '1px solid var(--ct-border)',
        }}
      >
        {normalized.map((c, idx) => {
          const colors = ['#8FA3BF', '#A692C6', '#C7AA60', '#6BA18C', '#B87272'];
          return (
            <div
              key={c.key}
              style={{
                width: `${c.pct}%`,
                backgroundColor: colors[idx % colors.length],
                height: '100%',
              }}
              title={`${c.label}: ${c.pct}% (raw: ${c.value.toFixed(3)})`}
            />
          );
        })}
      </div>
    </div>
  );
}
