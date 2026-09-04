import React from 'react';

export default function SyntheticTag({ title = 'Derived from synthetic network/temporal simulation' }) {
  return (
    <span
      className="ct-synthetic-badge"
      title={title}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '3px',
        padding: '1px 5px',
        border: '1px dashed #7B68AE',
        color: '#9D8BC9',
        fontSize: '10px',
        fontWeight: 700,
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
        borderRadius: '2px',
        backgroundColor: 'rgba(123, 104, 174, 0.08)',
      }}
    >
      SYNTH
    </span>
  );
}
