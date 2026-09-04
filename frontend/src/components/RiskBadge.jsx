import React from 'react';
import { getSeverityColor } from '../utils/colors';

export default function RiskBadge({ tier, score }) {
  const tierStr = (tier || 'LOW').toUpperCase();
  const color = getSeverityColor(tier);

  return (
    <span
      className="ct-badge"
      style={{
        backgroundColor: `${color}22`,
        border: `1px solid ${color}66`,
        color: color,
        fontWeight: 700,
        fontSize: '11px',
        padding: '2px 6px',
        borderRadius: '2px',
        letterSpacing: '0.5px',
      }}
    >
      {tierStr}
      {score !== undefined && score !== null && (
        <span style={{ marginLeft: '4px', opacity: 0.9 }}>
          {Number(score).toFixed(0)}
        </span>
      )}
    </span>
  );
}
