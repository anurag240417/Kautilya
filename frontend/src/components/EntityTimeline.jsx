import React from 'react';
import SyntheticTag from './SyntheticTag';
import { formatTimestamp } from '../utils/formatters';

export default function EntityTimeline({ events = [] }) {
  if (!events || events.length === 0) {
    return (
      <div
        className="ct-card"
        style={{ padding: '16px', color: 'var(--ct-text-muted)', fontSize: '12px' }}
      >
        No chronological timeline events recorded for this entity.
      </div>
    );
  }

  return (
    <div className="ct-card" style={{ padding: '16px' }}>
      <h3 style={{ fontSize: '14px', marginBottom: '14px' }}>
        Chronological Forensic Timeline
      </h3>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0', position: 'relative' }}>
        {events.map((ev, idx) => {
          const isLast = idx === events.length - 1;
          const isSynth = Boolean(ev.is_synthetic);

          return (
            <div
              key={ev.id || idx}
              style={{
                display: 'flex',
                gap: '14px',
                position: 'relative',
                paddingBottom: isLast ? '0' : '16px',
              }}
            >
              {/* Vertical timeline connector line */}
              {!isLast && (
                <div
                  style={{
                    position: 'absolute',
                    left: '7px',
                    top: '16px',
                    bottom: '0',
                    width: '2px',
                    backgroundColor: isSynth ? '#7B68AE' : 'var(--ct-border)',
                    borderStyle: isSynth ? 'dashed' : 'solid',
                  }}
                />
              )}

              {/* Timeline marker node */}
              <div
                style={{
                  width: '16px',
                  height: '16px',
                  borderRadius: '50%',
                  backgroundColor: isSynth ? '#7B68AE' : 'var(--ct-bg-elevated)',
                  border: isSynth ? '2px dashed #9D8BC9' : '2px solid var(--ct-border-light)',
                  flexShrink: 0,
                  marginTop: '2px',
                  zIndex: 1,
                }}
              />

              {/* Event Content */}
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '3px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ fontWeight: 600, fontSize: '13px', color: 'var(--ct-text-primary)' }}>
                      {ev.title || ev.headline || 'Observation Event'}
                    </span>
                    {isSynth && <SyntheticTag />}
                  </div>

                  <span
                    style={{
                      fontSize: '11px',
                      color: 'var(--ct-text-muted)',
                      fontFamily: 'var(--ct-font-mono)',
                    }}
                  >
                    {ev.timestamp ? formatTimestamp(ev.timestamp) : (ev.time_step ? `Step ${ev.time_step}` : '—')}
                  </span>
                </div>

                <div style={{ fontSize: '12px', color: 'var(--ct-text-secondary)', lineHeight: '1.4' }}>
                  {ev.description || ev.details || '—'}
                </div>

                {ev.candidate_ip && (
                  <div style={{ fontSize: '11px', color: 'var(--ct-text-secondary)', fontFamily: 'var(--ct-font-mono)', marginTop: '2px' }}>
                    IP: {ev.candidate_ip} | ASN: {ev.asn || '—'} | Country: {ev.country || '—'}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
