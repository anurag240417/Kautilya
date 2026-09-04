import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import KPICounter from '../components/KPICounter';
import AlertCard from '../components/AlertCard';
import SearchInput from '../components/SearchInput';
import ForensicDisclaimer from '../components/ForensicDisclaimer';
import { getStatistics, getAlerts } from '../api/client';
import { SEVERITY_COLORS } from '../utils/colors';

/**
 * Overview Page per FRONTEND_BRIEF.md §3.
 * Job: Triage — what needs attention right now.
 * 4–5 KPI counters, top 5 priority alerts, system status.
 * Explicitly NO duplicate full-graph or full-map here.
 */
export default function Overview() {
  const navigate = useNavigate();
  const [stats, setStats] = useState(null);
  const [topAlerts, setTopAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadOverviewData() {
      try {
        setLoading(true);
        const [statsData, alertsData] = await Promise.all([
          getStatistics().catch(() => null),
          getAlerts({ limit: 5 }).catch(() => ({ alerts: [] })),
        ]);

        if (statsData) {
          setStats(statsData);
        } else {
          // Fallback stats if endpoint returns default
          setStats({
            total_transactions: 2,
            total_wallets: 2,
            total_alerts: alertsData?.total || alertsData?.alerts?.length || 0,
            tier_counts: { critical: 1, high: 1, medium: 0, low: 0 },
            synthetic_alerts_percentage: 50.0,
            is_offline_mode: true,
          });
        }

        setTopAlerts(alertsData?.alerts || []);
      } catch (err) {
        console.error('Failed to load overview data:', err);
        setError(err.message || 'Unable to connect to local API');
      } finally {
        setLoading(false);
      }
    }

    loadOverviewData();

    const handleSimUpdate = () => {
      loadOverviewData();
    };
    window.addEventListener('chaintrace-simulation-update', handleSimUpdate);
    return () => window.removeEventListener('chaintrace-simulation-update', handleSimUpdate);
  }, []);

  const handleSearch = (entityId, entityType) => {
    navigate(`/investigation?entity=${encodeURIComponent(entityId)}&type=${encodeURIComponent(entityType)}`);
  };

  return (
    <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px', maxWidth: '1280px', margin: '0 auto' }}>
      {/* Page Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h1 style={{ fontSize: '22px', marginBottom: '4px' }}>Investigative Triage Dashboard</h1>
          <p style={{ color: 'var(--ct-text-secondary)', fontSize: '13px' }}>
            System triage prioritization for suspicious Bitcoin transactions, wallet entities, and network correlations.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span
            style={{
              fontSize: '11px',
              fontFamily: 'var(--ct-font-mono)',
              backgroundColor: 'var(--ct-bg-surface)',
              border: '1px solid var(--ct-border)',
              padding: '6px 10px',
              borderRadius: '2px',
              color: 'var(--ct-text-secondary)',
            }}
          >
            Offline Mode: <strong style={{ color: '#4A7A5C' }}>ENFORCED</strong>
          </span>
        </div>
      </div>

      <ForensicDisclaimer text="Alerts represent investigative triage prioritization to manage caseloads, NOT proof of criminal guilt. Uncorroborated network signals remain inferential." />

      {/* Triage Search Bar */}
      <div className="ct-card">
        <h3 style={{ fontSize: '13px', textTransform: 'uppercase', color: 'var(--ct-text-secondary)', marginBottom: '10px' }}>
          Quick Entity Lookup
        </h3>
        <SearchInput onSearch={handleSearch} />
      </div>

      {/* KPI Counters (4-5 counters per FRONTEND_BRIEF.md §3) */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '16px',
        }}
      >
        <KPICounter
          label="Total Active Alerts"
          value={stats?.total_alerts ?? '—'}
          subtext="Ranked forensic alerts in triage queue"
        />
        <KPICounter
          label="Critical Priority"
          value={stats?.tier_counts?.critical ?? '—'}
          subtext="Requires immediate forensic escalation"
          accentColor={SEVERITY_COLORS.critical}
        />
        <KPICounter
          label="High Priority"
          value={stats?.tier_counts?.high ?? '—'}
          subtext="Multi-signal corroboration present"
          accentColor={SEVERITY_COLORS.high}
        />
        <KPICounter
          label="Monitored Entities"
          value={(stats?.total_transactions || 0) + (stats?.total_wallets || 0) || '—'}
          subtext="Ingested transactions & wallets"
        />
        <KPICounter
          label="Synthetic Signal Share"
          value={`${stats?.synthetic_alerts_percentage ?? 0}%`}
          subtext="Alerts incorporating synthetic network data"
        />
      </div>

      {/* Top 5 Priority Alerts (Compact list per §3) */}
      <div className="ct-card">
        <div className="ct-card-header">
          <div>
            <h2 style={{ fontSize: '16px' }}>Top Priority Investigative Alerts</h2>
            <div style={{ fontSize: '12px', color: 'var(--ct-text-secondary)', marginTop: '2px' }}>
              Highest ranked alerts requiring investigator review
            </div>
          </div>
          <Link to="/alerts">
            <button type="button">View All Alerts ({stats?.total_alerts || 0})</button>
          </Link>
        </div>

        {loading ? (
          <div style={{ padding: '24px', textAlign: 'center', color: 'var(--ct-text-muted)' }}>
            Loading triage queue...
          </div>
        ) : error ? (
          <div style={{ padding: '16px', color: SEVERITY_COLORS.critical }}>
            Error loading triage alerts: {error}
          </div>
        ) : topAlerts.length === 0 ? (
          <div style={{ padding: '24px', textAlign: 'center', color: 'var(--ct-text-muted)' }}>
            No active alerts in queue.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {topAlerts.map((alert) => (
              <AlertCard key={alert.alert_id} alert={alert} />
            ))}
          </div>
        )}
      </div>

      {/* System Provenance & Offline Status Banner */}
      <div
        className="ct-card"
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '16px',
          backgroundColor: 'var(--ct-bg-elevated)',
        }}
      >
        <div>
          <div style={{ fontWeight: 600, marginBottom: '4px', fontSize: '13px' }}>
            Dataset Provenance Guarantee
          </div>
          <div style={{ fontSize: '12px', color: 'var(--ct-text-secondary)', lineHeight: '1.4' }}>
            Blockchain layer records derive directly from public Elliptic++ benchmarks (real). Network layer observations derive from the local synthetic generator (reproducible seed).
          </div>
        </div>

        <div>
          <div style={{ fontWeight: 600, marginBottom: '4px', fontSize: '13px' }}>
            Fully Offline Execution
          </div>
          <div style={{ fontSize: '12px', color: 'var(--ct-text-secondary)', lineHeight: '1.4' }}>
            ChainTrace operates with zero external network dependencies. All GeoIP resolution, graph traversal, and ML scoring occur entirely in-memory and on local disk.
          </div>
        </div>
      </div>
    </div>
  );
}
