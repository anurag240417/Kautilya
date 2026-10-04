import React, { useState, useEffect, useRef } from 'react';
import { getSimulationStatus, injectSimulationStep, resetSimulation } from '../api/client';

export default function NetworkTelemetryBar() {
  const [telemetry, setTelemetry] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [liveEvent, setLiveEvent] = useState(null);
  const [streamSpeed, setStreamSpeed] = useState(1200); // 1.2s default
  const [loading, setLoading] = useState(false);
  const streamRef = useRef(null);

  const fetchTelemetry = async () => {
    try {
      const data = await getSimulationStatus();
      setTelemetry(data);
    } catch (err) {
      console.warn('Unable to reach telemetry service:', err);
    }
  };

  useEffect(() => {
    fetchTelemetry();
  }, []);

  // Automated streaming handler
  useEffect(() => {
    if (isStreaming) {
      streamRef.current = setInterval(async () => {
        try {
          const res = await injectSimulationStep();
          if (res && res.success) {
            handleEventDispatch(res);
            if (!res.status?.can_inject_next) {
              setIsStreaming(false);
            }
          } else {
            setIsStreaming(false);
          }
        } catch (e) {
          console.error('Streaming ingestion error:', e);
          setIsStreaming(false);
        }
      }, streamSpeed);
    } else {
      if (streamRef.current) {
        clearInterval(streamRef.current);
        streamRef.current = null;
      }
    }
    return () => {
      if (streamRef.current) clearInterval(streamRef.current);
    };
  }, [isStreaming, streamSpeed]);

  const handleEventDispatch = (res) => {
    setTelemetry(res.status);
    const inj = res.injected;
    if (inj) {
      setLiveEvent({
        txid: inj.txid,
        volume: inj.amount_btc,
        label: inj.label,
        title: res.batch_count > 1 ? `Batch Ingested (${res.batch_count} TXs)` : inj.title,
        headline: res.batch_count > 1 ? `Ingested batch of ${res.batch_count} transactions into active mempool.` : inj.headline,
        alerts: inj.added_alerts,
        edges: inj.edges_added_count,
        timestamp: new Date().toLocaleTimeString(),
      });
    }

    window.dispatchEvent(
      new CustomEvent('kautilya-simulation-update', {
        detail: res,
      })
    );
  };

  const handleManualStep = async () => {
    if (loading || !telemetry?.can_inject_next) return;
    setLoading(true);
    try {
      const res = await injectSimulationStep();
      if (res && res.success) {
        handleEventDispatch(res);
      }
    } catch (err) {
      console.error('Failed to ingest mempool transaction:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleBatchStep = async (batchSize = 10) => {
    if (loading || !telemetry?.can_inject_next) return;
    setLoading(true);
    try {
      const res = await injectSimulationStep(batchSize);
      if (res && res.success) {
        handleEventDispatch(res);
      }
    } catch (err) {
      console.error('Failed to ingest batch of transactions:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleResetStream = async (mode = 'baseline') => {
    setIsStreaming(false);
    setLoading(true);
    try {
      const res = await resetSimulation(mode);
      setTelemetry(res);
      setLiveEvent({
        title: mode === 'baseline' ? 'Node Telemetry Reset' : 'Full Scenario Loaded',
        headline: mode === 'baseline' ? 'Mempool flushed; baseline block loaded (TX 1005)' : '110 transactions across 5 clusters synchronized',
        timestamp: new Date().toLocaleTimeString(),
      });
      window.dispatchEvent(
        new CustomEvent('kautilya-simulation-update', {
          detail: { mode, status: res },
        })
      );
    } catch (err) {
      console.error('Failed to reset stream:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (liveEvent) {
      const timer = setTimeout(() => setLiveEvent(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [liveEvent]);

  if (!telemetry) return null;

  return (
    <header
      style={{
        backgroundColor: '#090D16',
        borderBottom: '1px solid #1E293B',
        padding: '8px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        fontSize: '11px',
        fontFamily: 'var(--ct-font-mono, "SF Mono", monospace)',
        color: '#94A3B8',
        position: 'sticky',
        top: 0,
        zIndex: 100,
        boxShadow: '0 2px 10px rgba(0, 0, 0, 0.5)',
      }}
    >
      {/* Left: Bitcoin Network & Node Telemetry */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
        {/* Node Status Indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span
            style={{
              display: 'inline-block',
              width: '7px',
              height: '7px',
              borderRadius: '50%',
              backgroundColor: '#10B981',
              boxShadow: '0 0 6px rgba(16, 185, 129, 0.6)',
            }}
          />
          <span style={{ fontWeight: 700, color: '#F1F5F9', letterSpacing: '0.5px' }}>
            BTC MAINNET
          </span>
          <span
            style={{
              backgroundColor: '#1E293B',
              color: '#38BDF8',
              fontSize: '10px',
              padding: '1px 5px',
              borderRadius: '3px',
              fontWeight: 600,
            }}
          >
            AIR-GAPPED REPLICA
          </span>
        </div>

        <span style={{ color: '#334155' }}>|</span>

        {/* Block Height */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
          <span style={{ color: '#64748B' }}>BLOCK:</span>
          <span style={{ fontWeight: 600, color: '#E2E8F0' }}>
            #{telemetry.block_height?.toLocaleString() || '854,230'}
          </span>
        </div>

        <span style={{ color: '#334155' }}>|</span>

        {/* Fee Rate */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
          <span style={{ color: '#64748B' }}>MEDIAN FEE:</span>
          <span style={{ fontWeight: 600, color: '#F59E0B' }}>
            {telemetry.median_fee_rate || 14.2} sat/vB
          </span>
        </div>

        <span style={{ color: '#334155' }}>|</span>

        {/* Mempool Backlog / Analyzed */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
          <span style={{ color: '#64748B' }}>INGESTED:</span>
          <span style={{ fontWeight: 600, color: '#38BDF8' }}>
            {telemetry.total_transactions} txs
          </span>
          <span style={{ color: '#475569' }}>/ {telemetry.total_steps || 110} stream queue</span>
        </div>

        <span style={{ color: '#334155' }}>|</span>

        {/* Active Alerts */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
          <span style={{ color: '#64748B' }}>TRIAGE ALERTS:</span>
          <span
            style={{
              fontWeight: 700,
              color: telemetry.total_alerts > 0 ? '#F87171' : '#10B981',
            }}
          >
            {telemetry.total_alerts}
          </span>
        </div>
      </div>

      {/* Right: Institutional Stream / Ingestion Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        {/* Stream Speed Selector */}
        <div style={{ display: 'flex', alignItems: 'center', backgroundColor: '#0F172A', borderRadius: '4px', border: '1px solid #1E293B', padding: '1px' }}>
          {[
            { label: '1x', val: 2000 },
            { label: '3x', val: 800 },
            { label: '10x', val: 250 },
          ].map((sp) => (
            <button
              key={sp.label}
              onClick={() => setStreamSpeed(sp.val)}
              style={{
                backgroundColor: streamSpeed === sp.val ? '#1E293B' : 'transparent',
                color: streamSpeed === sp.val ? '#38BDF8' : '#64748B',
                border: 'none',
                padding: '3px 6px',
                fontSize: '10px',
                fontWeight: 600,
                cursor: 'pointer',
                borderRadius: '3px',
                fontFamily: 'inherit',
              }}
              title={`Stream speed: ${sp.label} (${sp.val}ms per tx)`}
            >
              {sp.label}
            </button>
          ))}
        </div>

        {/* Live Stream Toggle */}
        <button
          onClick={() => setIsStreaming(!isStreaming)}
          disabled={!telemetry.can_inject_next && !isStreaming}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            backgroundColor: isStreaming ? '#7F1D1D' : '#1E293B',
            color: isStreaming ? '#FCA5A5' : '#E2E8F0',
            border: '1px solid ' + (isStreaming ? '#DC2626' : '#334155'),
            padding: '4px 10px',
            borderRadius: '4px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            fontFamily: 'inherit',
            transition: 'all 0.15s ease',
          }}
          title={isStreaming ? 'Pause live mempool stream' : `Start automated ingestion stream (${streamSpeed}ms per tx)`}
        >
          <span
            style={{
              display: 'inline-block',
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              backgroundColor: isStreaming ? '#EF4444' : '#10B981',
              boxShadow: isStreaming ? '0 0 6px #EF4444' : 'none',
            }}
          />
          <span>{isStreaming ? 'STREAM: PAUSE' : 'STREAM: LIVE'}</span>
        </button>

        {/* Single Step Button */}
        <button
          onClick={handleManualStep}
          disabled={loading || !telemetry.can_inject_next}
          style={{
            backgroundColor: telemetry.can_inject_next ? '#1E293B' : '#0F172A',
            color: telemetry.can_inject_next ? '#38BDF8' : '#475569',
            border: '1px solid ' + (telemetry.can_inject_next ? '#0284C7' : '#1E293B'),
            padding: '4px 9px',
            borderRadius: '4px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: telemetry.can_inject_next ? 'pointer' : 'not-allowed',
            fontFamily: 'inherit',
          }}
          title={telemetry.can_inject_next ? `Ingest next unconfirmed transaction (${telemetry.next_txid ? `TX ${telemetry.next_txid}` : ''})` : 'All transactions ingested'}
        >
          {telemetry.can_inject_next ? `+1 TX` : 'DRAINED'}
        </button>

        {/* Batch Step (+10) Button */}
        <button
          onClick={() => handleBatchStep(10)}
          disabled={loading || !telemetry.can_inject_next}
          style={{
            backgroundColor: telemetry.can_inject_next ? '#1E293B' : '#0F172A',
            color: telemetry.can_inject_next ? '#A7F3D0' : '#475569',
            border: '1px solid ' + (telemetry.can_inject_next ? '#059669' : '#1E293B'),
            padding: '4px 9px',
            borderRadius: '4px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: telemetry.can_inject_next ? 'pointer' : 'not-allowed',
            fontFamily: 'inherit',
          }}
          title="Ingest next batch of 10 transactions simultaneously"
        >
          +10 BATCH
        </button>

        {/* Reset / Reload Options */}
        <div style={{ display: 'flex', gap: '3px' }}>
          <button
            onClick={() => handleResetStream('baseline')}
            disabled={loading}
            style={{
              backgroundColor: '#0F172A',
              color: '#64748B',
              border: '1px solid #1E293B',
              padding: '4px 7px',
              borderRadius: '4px',
              fontSize: '10px',
              cursor: 'pointer',
              fontFamily: 'inherit',
            }}
            title="Reset to clean baseline state (1 control transaction, 0 alerts)"
          >
            RESET
          </button>
          <button
            onClick={() => handleResetStream('full')}
            disabled={loading}
            style={{
              backgroundColor: '#0F172A',
              color: '#38BDF8',
              border: '1px solid #0369A1',
              padding: '4px 7px',
              borderRadius: '4px',
              fontSize: '10px',
              cursor: 'pointer',
              fontFamily: 'inherit',
              fontWeight: 600,
            }}
            title="Synchronize all 110 transactions and 23 alerts immediately"
          >
            SYNC ALL (110)
          </button>
        </div>
      </div>

      {/* Institutional Telemetry HUD Dispatch (Discrete Top-Right Ticker) */}
      {liveEvent && (
        <div
          style={{
            position: 'fixed',
            top: '48px',
            right: '20px',
            backgroundColor: '#0F172A',
            border: '1px solid #2563EB',
            borderRadius: '4px',
            padding: '10px 14px',
            width: '360px',
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.7), 0 0 10px rgba(37, 99, 235, 0.25)',
            zIndex: 1000,
            fontFamily: 'var(--ct-font-sans, sans-serif)',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ color: '#38BDF8', fontSize: '12px' }}>📡</span>
              <span style={{ fontWeight: 700, fontSize: '11px', color: '#60A5FA', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                {liveEvent.txid ? `Mempool Event: TX ${liveEvent.txid}` : liveEvent.title}
              </span>
            </div>
            <span style={{ fontSize: '10px', color: '#64748B', fontFamily: 'var(--ct-font-mono, monospace)' }}>
              {liveEvent.timestamp}
            </span>
          </div>

          <div style={{ fontSize: '12px', color: '#CBD5E1', lineHeight: '1.35', marginTop: '4px' }}>
            {liveEvent.headline}
          </div>

          {liveEvent.volume !== undefined && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '8px', fontSize: '11px' }}>
              <span style={{ backgroundColor: '#1E293B', padding: '1px 6px', borderRadius: '3px', color: '#F59E0B', fontWeight: 600, fontFamily: 'monospace' }}>
                {liveEvent.volume} BTC
              </span>
              {liveEvent.alerts && liveEvent.alerts.length > 0 && (
                <span style={{ backgroundColor: '#7F1D1D', padding: '1px 6px', borderRadius: '3px', color: '#FCA5A5', fontWeight: 600 }}>
                  🚨 ALERT GENERATED
                </span>
              )}
              {liveEvent.edges > 0 && (
                <span style={{ backgroundColor: '#1E293B', padding: '1px 6px', borderRadius: '3px', color: '#38BDF8' }}>
                  +{liveEvent.edges} Edges
                </span>
              )}
            </div>
          )}
        </div>
      )}
    </header>
  );
}
