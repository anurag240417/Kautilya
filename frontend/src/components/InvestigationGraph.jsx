import React, { useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import { ENTITY_COLORS } from '../utils/colors';

/**
 * Interactive Cytoscape.js graph component per FRONTEND_BRIEF.md §6 & §8.
 * Strict entity-type colors, dashed synthetic markers, neighbor hover highlight.
 */
export default function InvestigationGraph({
  nodes = [],
  edges = [],
  rootEntityId = null,
  onSelectNode = null,
  height = '520px',
}) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const [selectedNodeInfo, setSelectedNodeInfo] = useState(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Convert nodes to Cytoscape elements
    const cyNodes = nodes.map((n) => {
      const type = (n.type || 'unknown').toLowerCase();
      let color = ENTITY_COLORS[type] || ENTITY_COLORS.unknown;
      let shape = 'ellipse';

      if (type === 'transaction') shape = 'roundrectangle';
      if (type === 'network' || type === 'ip') shape = 'diamond';

      const isRoot = String(n.id) === String(rootEntityId);
      const isSynthetic = Boolean(n.is_synthetic);

      return {
        data: {
          id: String(n.id),
          label: n.id.length > 14 ? `${n.id.slice(0, 6)}...${n.id.slice(-4)}` : n.id,
          fullId: String(n.id),
          type: type,
          color: color,
          shape: shape,
          isRoot: isRoot,
          is_synthetic: isSynthetic,
          priority_tier: n.priority_tier,
          risk_score: n.risk_score,
        },
      };
    });

    // Convert edges to Cytoscape elements
    const cyEdges = edges.map((e, idx) => {
      const isSynthetic = Boolean(e.is_synthetic);
      return {
        data: {
          id: `e-${idx}-${e.source}-${e.target}`,
          source: String(e.source),
          target: String(e.target),
          relationship: e.relationship || 'connected',
          is_synthetic: isSynthetic,
          confidence: e.confidence,
        },
      };
    });

    const cy = cytoscape({
      container: containerRef.current,
      elements: [...cyNodes, ...cyEdges],
      style: [
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)',
            'label': 'data(label)',
            'color': '#E8ECF1',
            'font-family': 'Inter, sans-serif',
            'font-size': '10px',
            'text-valign': 'bottom',
            'text-margin-y': 4,
            'shape': 'data(shape)',
            'width': 28,
            'height': 28,
            'border-width': 2,
            'border-color': '#1A2634',
            'border-style': 'solid',
          },
        },
        // Synthetic node marker: dashed border
        {
          selector: 'node[?is_synthetic]',
          style: {
            'border-style': 'dashed',
            'border-width': 2,
            'border-color': '#7B68AE',
          },
        },
        // Root queried entity
        {
          selector: 'node[?isRoot]',
          style: {
            'border-width': 3,
            'border-color': '#FFFFFF',
            'width': 36,
            'height': 36,
            'font-size': '11px',
            'font-weight': 'bold',
          },
        },
        {
          selector: 'edge',
          style: {
            'width': 1.5,
            'line-color': '#3D4F63',
            'target-arrow-color': '#3D4F63',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'arrow-scale': 0.8,
          },
        },
        // Synthetic edge marker: dashed line
        {
          selector: 'edge[?is_synthetic]',
          style: {
            'line-style': 'dashed',
            'line-color': '#7B68AE',
            'target-arrow-color': '#7B68AE',
            'width': 1.5,
          },
        },
        // Interactive Hover: Direct neighbors highlighted, others dimmed (§4 secondary)
        {
          selector: '.highlighted',
          style: {
            'opacity': 1,
            'line-color': '#4A90D9',
            'target-arrow-color': '#4A90D9',
            'z-index': 999,
          },
        },
        {
          selector: '.dimmed',
          style: {
            'opacity': 0.18,
          },
        },
        {
          selector: 'node:selected',
          style: {
            'border-color': '#FFFFFF',
            'border-width': 3,
          },
        },
      ],
      layout: {
        name: 'cose',
        animate: false,
        padding: 30,
        randomize: false,
        componentSpacing: 50,
        nodeOverlap: 20,
        idealEdgeLength: 60,
      },
    });

    // Event: Hover to preview neighbors (FRONTEND_BRIEF.md §4 secondary)
    cy.on('mouseover', 'node', (evt) => {
      const node = evt.target;
      const neighborhood = node.neighborhood().add(node);
      cy.elements().addClass('dimmed');
      neighborhood.removeClass('dimmed').addClass('highlighted');
    });

    cy.on('mouseout', 'node', () => {
      cy.elements().removeClass('dimmed').removeClass('highlighted');
    });

    // Event: Node tap selection
    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      const data = node.data();
      setSelectedNodeInfo(data);
      if (onSelectNode) {
        onSelectNode(data.fullId, data.type);
      }
    });

    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        setSelectedNodeInfo(null);
      }
    });

    cyRef.current = cy;

    return () => {
      cy.destroy();
    };
  }, [nodes, edges, rootEntityId, onSelectNode]);

  const handleFit = () => {
    if (cyRef.current) cyRef.current.fit(undefined, 30);
  };

  const handleResetZoom = () => {
    if (cyRef.current) {
      cyRef.current.zoom(1);
      cyRef.current.center();
    }
  };

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        height: height,
        backgroundColor: 'var(--ct-bg-primary)',
        border: '1px solid var(--ct-border)',
        borderRadius: '2px',
        overflow: 'hidden',
      }}
    >
      {/* Graph Toolbar */}
      <div
        style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          zIndex: 10,
          display: 'flex',
          gap: '6px',
        }}
      >
        <button
          type="button"
          onClick={handleFit}
          style={{ fontSize: '11px', padding: '4px 8px', backgroundColor: 'rgba(26, 38, 52, 0.85)' }}
        >
          Fit View
        </button>
        <button
          type="button"
          onClick={handleResetZoom}
          style={{ fontSize: '11px', padding: '4px 8px', backgroundColor: 'rgba(26, 38, 52, 0.85)' }}
        >
          100% Zoom
        </button>
      </div>

      {/* Graph Legend per FRONTEND_BRIEF.md §6 */}
      <div
        style={{
          position: 'absolute',
          bottom: '10px',
          left: '10px',
          zIndex: 10,
          backgroundColor: 'rgba(26, 38, 52, 0.9)',
          border: '1px solid var(--ct-border)',
          padding: '8px 12px',
          borderRadius: '2px',
          fontSize: '11px',
          display: 'flex',
          flexDirection: 'column',
          gap: '4px',
        }}
      >
        <div style={{ fontWeight: 600, color: 'var(--ct-text-secondary)', marginBottom: '2px' }}>
          Graph Entity Legend
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: ENTITY_COLORS.wallet }} />
          <span>Wallet Node</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: ENTITY_COLORS.transaction }} />
          <span>Transaction Node</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ width: '10px', height: '10px', transform: 'rotate(45deg)', backgroundColor: ENTITY_COLORS.network }} />
          <span>Network / IP Node</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', borderTop: '1px solid var(--ct-border)', paddingTop: '4px' }}>
          <span style={{ width: '14px', borderTop: '2px dashed #7B68AE' }} />
          <span style={{ color: '#9D8BC9', fontWeight: 600 }}>Dashed: Synthetic Origin</span>
        </div>
      </div>

      {/* Selected Node Drawer / Tooltip */}
      {selectedNodeInfo && (
        <div
          style={{
            position: 'absolute',
            top: '10px',
            left: '10px',
            zIndex: 10,
            backgroundColor: 'rgba(26, 38, 52, 0.95)',
            border: '1px solid var(--ct-border)',
            padding: '10px 14px',
            borderRadius: '2px',
            maxWidth: '300px',
            fontSize: '12px',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <span style={{ textTransform: 'uppercase', fontWeight: 700, color: selectedNodeInfo.color }}>
              {selectedNodeInfo.type}
            </span>
            {selectedNodeInfo.is_synthetic && (
              <span className="ct-synthetic-badge">SYNTH</span>
            )}
          </div>
          <div style={{ fontFamily: 'var(--ct-font-mono)', wordBreak: 'break-all', fontWeight: 600, marginBottom: '6px' }}>
            {selectedNodeInfo.fullId}
          </div>
          {selectedNodeInfo.risk_score !== undefined && (
            <div style={{ color: 'var(--ct-text-secondary)', fontSize: '11px' }}>
              Priority Score: {Number(selectedNodeInfo.risk_score).toFixed(1)} ({selectedNodeInfo.priority_tier || 'LOW'})
            </div>
          )}
        </div>
      )}

      {/* Cytoscape DOM Canvas */}
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
}
