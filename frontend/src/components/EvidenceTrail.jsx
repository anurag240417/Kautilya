import React, { useState, useEffect, useRef } from 'react';
import RiskBadge from './RiskBadge';
import RiskBreakdown from './RiskBreakdown';
import SyntheticTag from './SyntheticTag';
import ForensicDisclaimer from './ForensicDisclaimer';
import { getSeverityColor } from '../utils/colors';

/**
 * Explainability Evidence Trail Reveal per FRONTEND_BRIEF.md §4.
 * The ONLY required animation: assembles evidence in the order it accumulated.
 * Pure CSS transitions, step-by-step progress, play/pause/skip controls.
 */
export default function EvidenceTrail({
  seedEntity,
  evidenceRecords = [],
  riskScore,
  onStepChange,
}) {
  // Total steps = 1 (seed) + evidence count + 1 (final score)
  const totalSteps = 1 + evidenceRecords.length + (riskScore ? 1 : 0);
  const [currentStep, setCurrentStep] = useState(1);
  const [isPlaying, setIsPlaying] = useState(true);
  const timerRef = useRef(null);

  // Notify parent on step change (e.g. to synchronize graph node visibility)
  useEffect(() => {
    if (onStepChange) {
      onStepChange(currentStep, evidenceRecords.slice(0, Math.max(0, currentStep - 1)));
    }
  }, [currentStep, onStepChange, evidenceRecords]);

  // Autoplay progression
  useEffect(() => {
    if (isPlaying && currentStep < totalSteps) {
      timerRef.current = setTimeout(() => {
        setCurrentStep((prev) => Math.min(prev + 1, totalSteps));
      }, 900);
    } else if (currentStep >= totalSteps) {
      setIsPlaying(false);
    }

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [isPlaying, currentStep, totalSteps]);

  const handleSkip = () => {
    setIsPlaying(false);
    setCurrentStep(totalSteps);
  };

  const handleReplay = () => {
    setCurrentStep(1);
    setIsPlaying(true);
  };

  const handleTogglePlay = () => {
    if (currentStep >= totalSteps) {
      handleReplay();
    } else {
      setIsPlaying(!isPlaying);
    }
  };

  const finalStepIndex = totalSteps;
  const isFinalRevealed = currentStep === finalStepIndex;

  return (
    <div className="ct-card" style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      {/* Header with sequencing controls */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          paddingBottom: '10px',
          borderBottom: '1px solid var(--ct-border)',
          flexWrap: 'wrap',
          gap: '8px',
        }}
      >
        <div>
          <h2 style={{ fontSize: '15px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            Evidence Trail Reveal
            <span
              style={{
                fontSize: '11px',
                fontWeight: 600,
                color: 'var(--ct-text-secondary)',
                backgroundColor: 'var(--ct-bg-elevated)',
                padding: '2px 6px',
                borderRadius: '2px',
              }}
            >
              Step {currentStep} of {totalSteps}
            </span>
          </h2>
          <div style={{ fontSize: '11px', color: 'var(--ct-text-muted)', marginTop: '2px' }}>
            Accumulated analytical signals in forensic sequence
          </div>
        </div>

        {/* Sequencing action buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <button
            type="button"
            onClick={handleTogglePlay}
            style={{ fontSize: '12px', padding: '4px 10px' }}
            title={isPlaying ? 'Pause evidence sequence' : 'Play evidence sequence'}
          >
            {isPlaying ? 'Pause' : currentStep >= totalSteps ? 'Replay' : 'Play'}
          </button>
          <button
            type="button"
            onClick={() => setCurrentStep((prev) => Math.max(1, prev - 1))}
            disabled={currentStep <= 1}
            style={{ fontSize: '12px', padding: '4px 8px' }}
          >
            Prev
          </button>
          <button
            type="button"
            onClick={() => setCurrentStep((prev) => Math.min(totalSteps, prev + 1))}
            disabled={currentStep >= totalSteps}
            style={{ fontSize: '12px', padding: '4px 8px' }}
          >
            Next
          </button>
          <button
            type="button"
            onClick={handleSkip}
            disabled={currentStep >= totalSteps}
            style={{ fontSize: '12px', padding: '4px 10px', color: 'var(--ct-text-secondary)' }}
          >
            Skip to End
          </button>
        </div>
      </div>

      {/* Progress track bar */}
      <div
        style={{
          width: '100%',
          height: '4px',
          backgroundColor: 'var(--ct-bg-primary)',
          borderRadius: '1px',
          overflow: 'hidden',
          border: '1px solid var(--ct-border)',
        }}
      >
        <div
          style={{
            height: '100%',
            width: `${(currentStep / totalSteps) * 100}%`,
            backgroundColor: '#4A90D9',
            transition: 'width 0.25s linear',
          }}
        />
      </div>

      {/* Sequential Evidence Cards Container */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {/* Step 1: Seed Entity */}
        {currentStep >= 1 && (
          <div
            style={{
              padding: '12px',
              backgroundColor: 'var(--ct-bg-elevated)',
              border: seedEntity?.isSynthetic ? '1px dashed #7B68AE' : '1px solid var(--ct-border)',
              borderRadius: '2px',
              borderLeft: '4px solid #4A90D9',
              opacity: 1,
              transition: 'opacity 0.3s ease',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ fontSize: '10px', textTransform: 'uppercase', fontWeight: 700, color: '#4A90D9' }}>
                  Step 1: Seed Observation
                </span>
                {seedEntity?.isSynthetic && <SyntheticTag />}
              </div>
              <span style={{ fontSize: '11px', color: 'var(--ct-text-muted)' }}>Target Entity</span>
            </div>
            <div style={{ fontFamily: 'var(--ct-font-mono)', fontWeight: 600, fontSize: '13px' }}>
              {seedEntity?.type?.toUpperCase() || 'ENTITY'}: {seedEntity?.id || '—'}
            </div>
            <div style={{ fontSize: '12px', color: 'var(--ct-text-secondary)', marginTop: '4px' }}>
              Initial investigation target entered into forensic decision-support pipeline.
            </div>
          </div>
        )}

        {/* Steps 2 .. N: Evidence Records */}
        {evidenceRecords.map((rec, index) => {
          const stepNum = index + 2;
          const isVisible = currentStep >= stepNum;
          if (!isVisible) return null;

          const isSynth = rec.is_synthetic;
          const catColors = {
            ml_behavioral: '#8FA3BF',
            anomaly: '#C7AA60',
            graph_structural: '#7B68AE',
            temporal: '#A692C6',
            synthetic_network: '#7B68AE',
            known_indicator: '#D4763A',
          };
          const catColor = catColors[rec.category] || 'var(--ct-text-secondary)';

          return (
            <div
              key={rec.evidence_id || index}
              style={{
                padding: '12px',
                backgroundColor: 'var(--ct-bg-surface)',
                border: isSynth ? '1px dashed #7B68AE' : '1px solid var(--ct-border)',
                borderLeft: `4px solid ${catColor}`,
                borderRadius: '2px',
                opacity: 1,
                transition: 'opacity 0.35s ease',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span
                    style={{
                      fontSize: '10px',
                      textTransform: 'uppercase',
                      fontWeight: 700,
                      color: catColor,
                    }}
                  >
                    Step {stepNum}: {rec.category ? rec.category.replace('_', ' ') : 'Evidence'}
                  </span>
                  {isSynth && <SyntheticTag />}
                </div>

                {rec.confidence !== undefined && rec.confidence !== null && (
                  <span
                    style={{
                      fontSize: '11px',
                      color: 'var(--ct-text-secondary)',
                      fontFamily: 'var(--ct-font-mono)',
                    }}
                  >
                    conf: {(rec.confidence * 100).toFixed(0)}%
                  </span>
                )}
              </div>

              <div style={{ fontWeight: 600, fontSize: '13px', color: 'var(--ct-text-primary)' }}>
                {rec.headline}
              </div>

              <div style={{ fontSize: '12px', color: 'var(--ct-text-secondary)', marginTop: '4px', lineHeight: '1.4' }}>
                {rec.description}
              </div>

              {/* Natural language supporting metrics derived from known features */}
              {rec.supporting_metrics && Object.keys(rec.supporting_metrics).length > 0 && (
                <div
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '6px',
                    marginTop: '8px',
                    paddingTop: '6px',
                    borderTop: '1px solid rgba(255, 255, 255, 0.05)',
                  }}
                >
                  {Object.entries(rec.supporting_metrics).map(([k, v]) => (
                    <span
                      key={k}
                      style={{
                        fontSize: '10px',
                        fontFamily: 'var(--ct-font-mono)',
                        backgroundColor: 'var(--ct-bg-elevated)',
                        padding: '1px 5px',
                        borderRadius: '2px',
                        color: 'var(--ct-text-secondary)',
                      }}
                    >
                      {k}: {typeof v === 'number' ? v.toFixed(4) : String(v)}
                    </span>
                  ))}
                </div>
              )}
            </div>
          );
        })}

        {/* Final Step: Synthesized Risk Score Reveal */}
        {isFinalRevealed && riskScore && (
          <div
            style={{
              padding: '16px',
              backgroundColor: 'var(--ct-bg-elevated)',
              border: `1px solid ${getSeverityColor(riskScore.priority_tier)}`,
              borderRadius: '2px',
              display: 'flex',
              flexDirection: 'column',
              gap: '10px',
              opacity: 1,
              transition: 'opacity 0.4s ease',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ fontSize: '11px', textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.5px' }}>
                Final Synthesis & Triage Prioritization
              </div>
              <RiskBadge tier={riskScore.priority_tier} score={riskScore.score} />
            </div>

            <div style={{ fontSize: '13px', color: 'var(--ct-text-primary)' }}>
              {riskScore.explanation || 'Composite risk prioritization computed from contributing evidence records.'}
            </div>

            <RiskBreakdown riskScore={riskScore} />

            <ForensicDisclaimer compact={true} />
          </div>
        )}
      </div>
    </div>
  );
}
