import React from 'react';
import { Link } from 'react-router-dom';
import RiskBadge from './RiskBadge';
import SyntheticTag from './SyntheticTag';
import { truncateIdentifier, formatTimestamp } from '../utils/formatters';
import { getEntityColor, getStatusColor } from '../utils/colors';

export default function AlertCard({ alert }) {
  if (!alert) return null;

  const isSynthetic = alert.contains_synthetic_input;
  const entityType = alert.entity_type || 'transaction';
  const entityColor = getEntityColor(entityType);
  const statusColor = getStatusColor(alert.status);

  return (
    <div
      className={`ct-card ${isSynthetic ? 'ct-synthetic-marker' : ''}`}
      style={{
        padding: '14px',
        display: 'flex',
        flexDirection: 'column',
        gap: '10px',
        borderLeft: isSynthetic ? '3px dashed #7B68AE' : `3px solid ${entityColor}`,
        backgroundColor: 'var(--ct-bg-surface)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <span
            style={{
              fontSize: '11px',
              fontWeight: 700,
              textTransform: 'uppercase',
              color: entityColor,
              backgroundColor: `${entityColor}1A`,
              padding: '2px 6px',
              borderRadius: '2px',
            }}
          >
            {entityType}
          </span>

          <Link
            to={`/investigation?entity=${encodeURIComponent(alert.entity_id)}&type=${encodeURIComponent(entityType)}`}
            style={{
              fontFamily: 'var(--ct-font-mono)',
              fontWeight: 600,
              color: 'var(--ct-text-primary)',
              textDecoration: 'none',
              fontSize: '13px',
            }}
            title={alert.entity_id}
          >
            {truncateIdentifier(alert.entity_id, 12, 8)}
          </Link>

          {isSynthetic && <SyntheticTag />}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span
            style={{
              fontSize: '10px',
              textTransform: 'uppercase',
              fontWeight: 600,
              color: statusColor,
              border: `1px solid ${statusColor}44`,
              padding: '1px 5px',
              borderRadius: '2px',
            }}
          >
            {alert.status || 'NEW'}
          </span>
          <RiskBadge tier={alert.priority_tier} score={alert.risk_score} />
        </div>
      </div>

      <div>
        <div style={{ fontWeight: 600, fontSize: '13px', color: 'var(--ct-text-primary)', marginBottom: '3px' }}>
          {alert.headline}
        </div>
        <div
          style={{
            fontSize: '12px',
            color: 'var(--ct-text-secondary)',
            lineHeight: '1.4',
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
          }}
        >
          {alert.summary}
        </div>
      </div>

      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '11px',
          color: 'var(--ct-text-muted)',
          paddingTop: '6px',
          borderTop: '1px solid var(--ct-border)',
        }}
      >
        <span>
          Signals: {alert.active_signals && alert.active_signals.length > 0 ? alert.active_signals.join(', ') : 'composite'}
        </span>
        <span>{formatTimestamp(alert.created_at)}</span>
      </div>
    </div>
  );
}
