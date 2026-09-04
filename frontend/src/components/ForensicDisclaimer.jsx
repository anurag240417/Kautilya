import React from 'react';

export default function ForensicDisclaimer({ text, compact = false }) {
  const defaultText =
    text ||
    'Investigative decision-support prioritization only. Does not constitute proof of criminality or legal accusation.';

  if (compact) {
    return (
      <div
        style={{
          fontSize: '11px',
          color: 'var(--ct-text-muted)',
          lineHeight: '1.4',
          borderLeft: '2px solid var(--ct-border-light)',
          paddingLeft: '8px',
        }}
      >
        {defaultText}
      </div>
    );
  }

  return (
    <div
      style={{
        backgroundColor: 'rgba(26, 38, 52, 0.7)',
        border: '1px solid var(--ct-border)',
        borderLeft: '3px solid var(--ct-border-light)',
        padding: '10px 14px',
        borderRadius: '2px',
        fontSize: '12px',
        color: 'var(--ct-text-secondary)',
        lineHeight: '1.4',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '8px',
      }}
    >
      <div style={{ flex: 1 }}>
        <strong style={{ color: 'var(--ct-text-primary)', marginRight: '6px' }}>
          Forensic Notice:
        </strong>
        {defaultText}
      </div>
    </div>
  );
}
