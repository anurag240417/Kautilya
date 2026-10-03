import React from 'react';
import { NavLink } from 'react-router-dom';

export default function Sidebar({ systemStatus = { offline: true, operational: true } }) {
  const navItems = [
    { to: '/', label: 'Overview', desc: 'Triage & KPIs' },
    { to: '/investigation', label: 'Investigation', desc: 'Trace & Link Graph' },
    { to: '/alerts', label: 'Alert Queue', desc: 'Ranked Triage List' },
    { to: '/forensics', label: 'Forensics Lab', desc: 'Raw Tx Analysis & Cases' },
    { to: '/entity-profile', label: 'Entity Profile', desc: 'Deep-Dive Forensic' },
    { to: '/data-sources', label: 'Data Sources', desc: 'Provenance Transparency' },
  ];

  return (
    <aside
      style={{
        width: 'var(--ct-sidebar-width)',
        backgroundColor: 'var(--ct-bg-surface)',
        borderRight: '1px solid var(--ct-border)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        minHeight: '100vh',
        flexShrink: 0,
      }}
    >
      {/* Brand Header */}
      <div>
        <div
          style={{
            padding: '20px 16px',
            borderBottom: '1px solid var(--ct-border)',
          }}
        >
          <div
            style={{
              fontSize: '16px',
              fontWeight: 700,
              letterSpacing: '0.5px',
              color: 'var(--ct-text-primary)',
            }}
          >
            ChainTrace
          </div>
          <div
            style={{
              fontSize: '11px',
              color: 'var(--ct-text-secondary)',
              marginTop: '2px',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
            }}
          >
            Forensic Decision Support
          </div>
        </div>

        {/* Navigation Menu */}
        <nav style={{ padding: '12px 8px' }}>
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              style={({ isActive }) => ({
                display: 'flex',
                flexDirection: 'column',
                gap: '2px',
                padding: '10px 12px',
                marginBottom: '4px',
                borderRadius: '2px',
                backgroundColor: isActive ? 'var(--ct-bg-elevated)' : 'transparent',
                borderLeft: isActive ? '3px solid #4A90D9' : '3px solid transparent',
                color: isActive ? 'var(--ct-text-primary)' : 'var(--ct-text-secondary)',
                textDecoration: 'none',
              })}
            >
              <span style={{ fontWeight: 600, fontSize: '13px' }}>{item.label}</span>
              <span style={{ fontSize: '11px', color: 'var(--ct-text-muted)' }}>{item.desc}</span>
            </NavLink>
          ))}
        </nav>
      </div>

      {/* System Status / Provenance Footer per §3 */}
      <div
        style={{
          padding: '16px',
          borderTop: '1px solid var(--ct-border)',
          fontSize: '11px',
          color: 'var(--ct-text-secondary)',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
        }}
      >
        <div style={{ textTransform: 'uppercase', fontSize: '10px', fontWeight: 600, color: 'var(--ct-text-muted)', letterSpacing: '0.5px' }}>
          System Operational State
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>Mode</span>
          <span style={{ fontFamily: 'var(--ct-font-mono)', fontWeight: 600, color: '#4A7A5C' }}>
            OFFLINE BATCH
          </span>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>Engine</span>
          <span style={{ fontFamily: 'var(--ct-font-mono)', fontWeight: 600, color: '#4A7A5C' }}>
            OPERATIONAL
          </span>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>Network Calls</span>
          <span style={{ fontFamily: 'var(--ct-font-mono)', color: 'var(--ct-text-muted)' }}>
            0 (LOCAL ONLY)
          </span>
        </div>
      </div>
    </aside>
  );
}
