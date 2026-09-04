import React from 'react';
import SyntheticTag from '../components/SyntheticTag';

/**
 * Data Sources Page per FRONTEND_BRIEF.md §3 & §6.
 * Full provenance transparency: distinguishes real Elliptic++ data from synthetic network layer.
 */
export default function DataSources() {
  return (
    <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px', maxWidth: '1280px', margin: '0 auto' }}>
      {/* Header */}
      <div>
        <h1 style={{ fontSize: '22px', marginBottom: '4px' }}>Data Sources & Provenance Transparency</h1>
        <p style={{ color: 'var(--ct-text-secondary)', fontSize: '13px' }}>
          Complete provenance audit trail distinguishing public benchmark blockchain data from local synthetic simulations.
        </p>
      </div>

      {/* Provenance Principle Banner */}
      <div
        className="ct-card"
        style={{
          backgroundColor: 'var(--ct-bg-elevated)',
          borderLeft: '4px solid #4A90D9',
        }}
      >
        <h2 style={{ fontSize: '15px', marginBottom: '6px' }}>Strict Provenance Boundary</h2>
        <p style={{ fontSize: '13px', color: 'var(--ct-text-secondary)', lineHeight: '1.5' }}>
          ChainTrace maintains an unbreakable boundary between real and synthetic data.
          Every synthetic record carries an explicit <code style={{ color: '#9D8BC9' }}>is_synthetic = True</code> flag
          at the domain data-model level. This flag survives ingestion, normalization, graph construction, ML inference,
          risk scoring, REST API responses, and is visibly marked across the entire user interface.
        </p>
      </div>

      {/* Two Pillars Grid: Real vs Synthetic */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        {/* Real Data Pillar */}
        <div className="ct-card" style={{ borderTop: '4px solid #4A90D9' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h2 style={{ fontSize: '16px' }}>Real Blockchain Layer</h2>
            <span
              style={{
                fontSize: '11px',
                fontWeight: 700,
                color: '#4A90D9',
                backgroundColor: 'rgba(74, 144, 217, 0.15)',
                padding: '2px 8px',
                borderRadius: '2px',
              }}
            >
              REAL DATA
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '13px', color: 'var(--ct-text-secondary)' }}>
            <div>
              <strong style={{ color: 'var(--ct-text-primary)' }}>Source:</strong> Elliptic++ Benchmark Dataset (MIT / Oxford).
            </div>

            <div>
              <strong style={{ color: 'var(--ct-text-primary)' }}>Scope:</strong> 203,769 Bitcoin transactions spanning 49 discrete, two-week temporal steps.
            </div>

            <div>
              <strong style={{ color: 'var(--ct-text-primary)' }}>Graph Topology:</strong> 4 native edgelists:
              <ul style={{ listStyle: 'disc', paddingLeft: '20px', marginTop: '6px', fontSize: '12px' }}>
                <li><code>tx_tx</code>: Transaction-to-transaction DAG flow</li>
                <li><code>addr_tx</code>: Input wallet addresses funding transactions</li>
                <li><code>tx_addr</code>: Output transactions paying to recipient addresses</li>
                <li><code>addr_addr</code>: Derived co-spending and counterparty address links</li>
              </ul>
            </div>

            <div>
              <strong style={{ color: 'var(--ct-text-primary)' }}>Features:</strong> 166 transaction features (interpretable volume, fees, sizes, in/out degrees, and anonymized aggregate dimensions) and 56 wallet statistics.
            </div>
          </div>
        </div>

        {/* Synthetic Network Pillar */}
        <div className="ct-card ct-synthetic-marker" style={{ borderTop: '4px solid #7B68AE' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h2 style={{ fontSize: '16px' }}>Synthetic Network Layer</h2>
              <SyntheticTag />
            </div>
            <span
              style={{
                fontSize: '11px',
                fontWeight: 700,
                color: '#7B68AE',
                backgroundColor: 'rgba(123, 104, 174, 0.15)',
                padding: '2px 8px',
                borderRadius: '2px',
              }}
            >
              SIMULATED
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '13px', color: 'var(--ct-text-secondary)' }}>
            <div>
              <strong style={{ color: 'var(--ct-text-primary)' }}>Source:</strong> Local ChainTrace Network Generator (Phase 4).
            </div>

            <div>
              <strong style={{ color: 'var(--ct-text-primary)' }}>Simulation Model:</strong> Generates P2P peer node pools, propagation delays, relay observations, and candidate originator IPs mapped deterministically from Elliptic++ time steps.
            </div>

            <div>
              <strong style={{ color: 'var(--ct-text-primary)' }}>GeoIP & ASN Resolution:</strong> 100% offline local database lookup. Resolves IP geolocation without external runtime DNS or API calls.
            </div>

            <div>
              <strong style={{ color: 'var(--ct-text-primary)' }}>Controlled Injections:</strong> Simulates rapid dispersion, burst originations, and geographic concentration anomalies with ground truth recorded for validation.
            </div>

            <div>
              <strong style={{ color: 'var(--ct-text-primary)' }}>Reproducibility:</strong> Controlled via explicit random seed configurations stored in model metadata.
            </div>
          </div>
        </div>
      </div>

      {/* Visual Marker Compliance Checklist */}
      <div className="ct-card">
        <h2 style={{ fontSize: '15px', marginBottom: '12px' }}>Visual Marker Compliance (§6)</h2>
        <div style={{ fontSize: '13px', color: 'var(--ct-text-secondary)', lineHeight: '1.5', marginBottom: '16px' }}>
          Every element in ChainTrace visually indicates whether it derives from the real blockchain layer or the synthetic network simulation:
        </div>

        <table>
          <thead>
            <tr>
              <th>Layer Element</th>
              <th>Data Provenance</th>
              <th>Visual Marker in UI</th>
              <th>Forensic Interpretation</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Transaction Nodes & Edges</strong></td>
              <td>Real (Elliptic++)</td>
              <td>Solid lines, Amber color (<code>#D4A843</code>)</td>
              <td>Direct blockchain observation</td>
            </tr>
            <tr>
              <td><strong>Wallet Address Nodes</strong></td>
              <td>Real (Elliptic++)</td>
              <td>Solid lines, Steel Blue color (<code>#4A90D9</code>)</td>
              <td>Direct blockchain address entity</td>
            </tr>
            <tr>
              <td><strong>IP & Network Observations</strong></td>
              <td>Synthetic Generator</td>
              <td><span className="ct-synthetic-badge">SYNTH</span> Dashed lines, Purple color (<code>#7B68AE</code>)</td>
              <td>Simulated peer propagation candidate</td>
            </tr>
            <tr>
              <td><strong>Temporal Correlation Edge</strong></td>
              <td>Inferred (Phase 6)</td>
              <td>Dashed border + confidence percentage</td>
              <td>Inferential hypothesis; NOT proof of ownership</td>
            </tr>
            <tr>
              <td><strong>Wallet Aggregation Score</strong></td>
              <td>Derived (Phase 7)</td>
              <td>Traceable contributing transaction list</td>
              <td>Triage prioritization; NOT blanket accusation</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Offline Sovereign Guarantee */}
      <div className="ct-card" style={{ backgroundColor: 'var(--ct-bg-elevated)' }}>
        <h2 style={{ fontSize: '15px', marginBottom: '6px' }}>Sovereign Offline Architecture</h2>
        <p style={{ fontSize: '13px', color: 'var(--ct-text-secondary)', lineHeight: '1.5' }}>
          ChainTrace is engineered for restricted public-sector forensic labs and air-gapped sovereign environments.
          All fonts, graph rendering scripts, machine learning weights, and IP databases are bundled locally.
          At runtime, the software initiates <strong>zero outbound network requests</strong>.
        </p>
      </div>
    </div>
  );
}
