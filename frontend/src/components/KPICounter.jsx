import React from 'react';

export default function KPICounter({ label, value, subtext, accentColor }) {
  return (
    <div
      className="ct-card"
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        padding: '16px',
        minWidth: '180px',
        position: 'relative',
      }}
    >
      <div
        style={{
          fontSize: '11px',
          textTransform: 'uppercase',
          letterSpacing: '0.6px',
          color: 'var(--ct-text-secondary)',
          fontWeight: 600,
          marginBottom: '8px',
        }}
      >
        {label}
      </div>

      <div
        style={{
          fontSize: '28px',
          fontWeight: 700,
          color: accentColor || 'var(--ct-text-primary)',
          lineHeight: '1.1',
          marginBottom: '4px',
          fontFamily: 'var(--ct-font-mono)',
        }}
      >
        {value}
      </div>

      {subtext && (
        <div
          style={{
            fontSize: '11px',
            color: 'var(--ct-text-muted)',
          }}
        >
          {subtext}
        </div>
      )}
    </div>
  );
}
