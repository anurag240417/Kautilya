import React, { useState, useEffect, useRef } from 'react';
import { getSimulationStatus, injectSimulationStep, resetSimulation } from '../api/client';

export default function SimulationControlBar() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [toast, setToast] = useState(null);
  const streamTimerRef = useRef(null);

  // Fetch initial status
  const fetchStatus = async () => {
    try {
      const st = await getSimulationStatus();
      setStatus(st);
    } catch (err) {
      console.warn('Simulation status check failed (backend starting?):', err);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  // Handle auto-streaming
  useEffect(() => {
    if (isStreaming) {
      streamTimerRef.current = setInterval(async () => {
        try {
          const res = await injectSimulationStep();
          if (res && res.success) {
            handleInjectionSuccess(res);
            if (!res.status?.can_inject_next) {
              setIsStreaming(false);
            }
          } else {
            setIsStreaming(false);
          }
        } catch (err) {
          console.error('Stream step error:', err);
          setIsStreaming(false);
        }
      }, 3000);
    } else {
      if (streamTimerRef.current) {
        clearInterval(streamTimerRef.current);
        streamTimerRef.current = null;
      }
    }
    return () => {
      if (streamTimerRef.current) {
        clearInterval(streamTimerRef.current);
      }
    };
  }, [isStreaming]);

  const handleInjectionSuccess = (res) => {
    setStatus(res.status);
    const inj = res.injected;
    setToast({
      step: inj.step,
      txid: inj.txid,
      title: inj.title,
      headline: inj.headline,
      volume: inj.amount_btc,
      alerts: inj.added_alerts,
      edges: inj.edges_added_count,
    });

    // Notify all active views to refresh their data
    window.dispatchEvent(
      new CustomEvent('kautilya-simulation-update', {
        detail: res,
      })
    );
  };

  const handleInject = async () => {
    if (loading || !status?.can_inject_next) return;
    setLoading(true);
    try {
      const res = await injectSimulationStep();
      if (res && res.success) {
        handleInjectionSuccess(res);
      }
    } catch (err) {
      console.error('Failed to inject step:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async (mode) => {
    setIsStreaming(false);
    setLoading(true);
    try {
      const res = await resetSimulation(mode);
      setStatus(res);
      setToast({
        title: mode === 'baseline' ? 'Scenario Reset to Clean Baseline' : 'Full Scenario Loaded',
        headline: mode === 'baseline' ? '1 Licit Control TX (TX 1005), 0 Alerts' : 'All 7 TXs, 5 Wallets, 6 Alerts Loaded',
      });
      window.dispatchEvent(
        new CustomEvent('kautilya-simulation-update', {
          detail: { mode, status: res },
        })
      );
    } catch (err) {
      console.error('Failed to reset simulation:', err);
    } finally {
      setLoading(false);
    }
  };

  // Close toast automatically after 4.5s
  useEffect(() => {
    if (toast) {
      const t = setTimeout(() => setToast(null), 4500);
      return () => clearTimeout(t);
    }
  }, [toast]);

  if (!status) return null;

  const progressPercent = Math.round((status.current_step / (status.total_steps || 6)) * 100);

  return (
    <>
      {/* Simulation Bar */}
      <div
        style={{
          backgroundColor: '#0F172A',
          borderBottom: '1px solid #1E293B',
          padding: '8px 20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '12px',
          color: '#E2E8F0',
          position: 'sticky',
          top: 0,
          zIndex: 100,
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.4)',
        }}
      >
        {/* Left: Status & Current Step */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span
              style={{
                display: 'inline-block',
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: isStreaming ? '#10B981' : '#38BDF8',
                boxShadow: isStreaming ? '0 0 8px #10B981' : 'none',
              }}
            />
            <span style={{ fontWeight: 700, letterSpacing: '0.5px', textTransform: 'uppercase', fontSize: '11px', color: '#94A3B8' }}>
              Live Mempool Simulator
            </span>
          </div>

          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              backgroundColor: '#1E293B',
              padding: '3px 10px',
              borderRadius: '4px',
              border: '1px solid #334155',
            }}
          >
            <span style={{ fontWeight: 600, color: '#F8FAFC' }}>
              Step {status.current_step} of {status.total_steps}
            </span>
            <div
              style={{
                width: '60px',
                height: '6px',
                backgroundColor: '#334155',
                borderRadius: '3px',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  width: `${progressPercent}%`,
                  height: '100%',
                  backgroundColor: '#38BDF8',
                  transition: 'width 0.3s ease',
                }}
              />
            </div>
            <span style={{ color: '#94A3B8', fontSize: '11px' }}>
              {status.step_title}
            </span>
          </div>
        </div>

        {/* Center / Right: Interactive Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {/* Step button */}
          <button
            onClick={handleInject}
            disabled={loading || !status.can_inject_next}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              backgroundColor: status.can_inject_next ? '#2563EB' : '#334155',
              color: status.can_inject_next ? '#FFFFFF' : '#94A3B8',
              border: 'none',
              padding: '5px 12px',
              borderRadius: '4px',
              fontWeight: 600,
              fontSize: '12px',
              cursor: status.can_inject_next ? 'pointer' : 'not-allowed',
              transition: 'background-color 0.2s',
            }}
            title={status.can_inject_next ? `Inject ${status.next_txid ? `TX ${status.next_txid}` : 'next transaction'}` : 'All scenario transactions injected'}
          >
            <span>⚡</span>
            <span>{status.can_inject_next ? `Inject Next (TX ${status.next_txid})` : 'Scenario Complete'}</span>
          </button>

          {/* Auto-stream toggle */}
          <button
            onClick={() => setIsStreaming(!isStreaming)}
            disabled={!status.can_inject_next && !isStreaming}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              backgroundColor: isStreaming ? '#DC2626' : '#1E293B',
              color: '#F8FAFC',
              border: '1px solid ' + (isStreaming ? '#DC2626' : '#475569'),
              padding: '5px 12px',
              borderRadius: '4px',
              fontWeight: 600,
              fontSize: '12px',
              cursor: 'pointer',
            }}
          >
            <span>{isStreaming ? '⏸' : '▶'}</span>
            <span>{isStreaming ? 'Pause Stream' : 'Auto-Stream (3s)'}</span>
          </button>

          {/* Reset dropdown/buttons */}
          <div style={{ display: 'flex', gap: '4px' }}>
            <button
              onClick={() => handleReset('baseline')}
              disabled={loading}
              style={{
                backgroundColor: '#1E293B',
                color: '#94A3B8',
                border: '1px solid #334155',
                padding: '5px 9px',
                borderRadius: '4px',
                fontSize: '11px',
                cursor: 'pointer',
              }}
              title="Reset database to Step 0 (1 licit baseline TX, 0 alerts)"
            >
              ↺ Reset Clean
            </button>
            <button
              onClick={() => handleReset('full')}
              disabled={loading}
              style={{
                backgroundColor: '#1E293B',
                color: '#94A3B8',
                border: '1px solid #334155',
                padding: '5px 9px',
                borderRadius: '4px',
                fontSize: '11px',
                cursor: 'pointer',
              }}
              title="Fast-forward to full scenario (all 7 transactions, 6 alerts)"
            >
              ⏩ Load All
            </button>
          </div>
        </div>
      </div>

      {/* Floating Ingestion Toast Notification */}
      {toast && (
        <div
          style={{
            position: 'fixed',
            top: '55px',
            right: '20px',
            backgroundColor: '#0F172A',
            border: '1px solid #3B82F6',
            borderRadius: '6px',
            padding: '12px 16px',
            width: '380px',
            boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.7), 0 0 10px rgba(59, 130, 246, 0.3)',
            zIndex: 1000,
            animation: 'fadeIn 0.2s ease-in',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '16px' }}>⚡</span>
              <span style={{ fontWeight: 700, fontSize: '13px', color: '#60A5FA' }}>
                {toast.txid ? `Ingested TX ${toast.txid}` : toast.title}
              </span>
            </div>
            <button
              onClick={() => setToast(null)}
              style={{
                background: 'none',
                border: 'none',
                color: '#64748B',
                cursor: 'pointer',
                fontSize: '14px',
                padding: 0,
              }}
            >
              ✕
            </button>
          </div>

          <div style={{ marginTop: '6px', fontSize: '12px', color: '#CBD5E1', lineHeight: '1.4' }}>
            {toast.headline}
          </div>

          {toast.volume !== undefined && (
            <div style={{ marginTop: '8px', display: 'flex', gap: '10px', fontSize: '11px' }}>
              <span style={{ backgroundColor: '#1E293B', padding: '2px 6px', borderRadius: '3px', color: '#F59E0B', fontWeight: 600 }}>
                {toast.volume} BTC
              </span>
              {toast.alerts && toast.alerts.length > 0 && (
                <span style={{ backgroundColor: '#7F1D1D', padding: '2px 6px', borderRadius: '3px', color: '#FCA5A5', fontWeight: 600 }}>
                  🚨 {toast.alerts.length} Alert Generated
                </span>
              )}
              {toast.edges > 0 && (
                <span style={{ backgroundColor: '#1E293B', padding: '2px 6px', borderRadius: '3px', color: '#38BDF8' }}>
                  +{toast.edges} Graph Edges
                </span>
              )}
            </div>
          )}
        </div>
      )}
    </>
  );
}
