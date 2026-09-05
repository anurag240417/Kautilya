import React, { useEffect, useRef, useState, useMemo } from 'react';
import cytoscape from 'cytoscape';
import { ENTITY_COLORS } from '../utils/colors';

/**
 * Human-readable formatter for forensic edge relationships
 */
function formatRelationship(rel) {
  switch (rel) {
    case 'addr_tx':
      return 'Input';
    case 'tx_addr':
      return 'Output';
    case 'tx_tx':
      return 'Split / Flow';
    case 'addr_addr':
      return 'Transfers';
    case 'originator':
    case 'originator_ip':
      return 'Originator IP';
    case 'relay':
    case 'relay_ip':
      return 'Relay IP';
    case 'network_ip':
    case 'network_observation':
      return 'Observed IP';
    default:
      return rel ? rel.replace(/_/g, ' ') : 'Connected';
  }
}

/**
 * Bundles multiple parallel edges between the same source and target
 * to prevent visual entanglement while preserving all underlying data.
 */
function bundleEdges(rawEdges) {
  const edgeMap = new Map();

  rawEdges.forEach((e) => {
    const source = String(e.source);
    const target = String(e.target);
    const key = `${source}->${target}`;

    if (!edgeMap.has(key)) {
      edgeMap.set(key, {
        source,
        target,
        rawEdges: [],
        relationships: new Set(),
        is_synthetic: false,
        maxConfidence: null,
      });
    }

    const bundle = edgeMap.get(key);
    bundle.rawEdges.push(e);
    if (e.relationship) bundle.relationships.add(e.relationship);
    if (e.is_synthetic) bundle.is_synthetic = true;
    if (e.confidence !== undefined && e.confidence !== null) {
      if (bundle.maxConfidence === null || e.confidence > bundle.maxConfidence) {
        bundle.maxConfidence = e.confidence;
      }
    }
  });

  return Array.from(edgeMap.values()).map((bundle) => {
    const count = bundle.rawEdges.length;
    const relList = Array.from(bundle.relationships);
    const primaryRel = relList[0] || 'connected';
    const relDisplay = formatRelationship(primaryRel);

    const label = count > 1 ? `${count}× ${relDisplay}` : relDisplay;
    const width = count > 1 ? Math.min(2.0 + Math.log2(count) * 0.35, 4.0) : 1.8;

    return {
      data: {
        id: `bundle-${bundle.source}-${bundle.target}`,
        source: bundle.source,
        target: bundle.target,
        count,
        label,
        primaryRel,
        relationships: relList,
        is_synthetic: bundle.is_synthetic,
        confidence: bundle.maxConfidence,
        rawEdges: bundle.rawEdges,
        width,
      },
    };
  });
}

/**
 * Computes deterministic investigative topological ranks
 * Layer 0: Inbound Wallets (Inputs)
 * Layer 1: Investigated Entity (Root)
 * Layer 2: Outbound Wallets (Direct Outputs)
 * Layer 3: Downstream hops & Correlated Network/IP nodes
 */
function computeInvestigativePositions(nodes, edges, rootEntityId, orientation = 'vertical') {
  const rootId = String(rootEntityId);
  const nodeIds = new Set(nodes.map((n) => String(n.id)));

  const inEdges = new Map();
  const outEdges = new Map();
  nodes.forEach((n) => {
    inEdges.set(String(n.id), []);
    outEdges.set(String(n.id), []);
  });

  edges.forEach((e) => {
    const s = String(e.source);
    const t = String(e.target);
    if (outEdges.has(s)) outEdges.get(s).push(t);
    if (inEdges.has(t)) inEdges.get(t).push(s);
  });

  const levels = new Map();

  // Root entity is always Layer 1
  if (nodeIds.has(rootId)) {
    levels.set(rootId, 1);
  }

  // Nodes with edges going into root are Layer 0 (Inputs)
  const inputs = inEdges.get(rootId) || [];
  inputs.forEach((inNode) => {
    if (inNode !== rootId) {
      levels.set(inNode, 0);
    }
  });

  // Nodes with edges coming out of root
  const outputs = outEdges.get(rootId) || [];
  
  // First pass: Output wallets go to Layer 2, IPs go to Layer 3
  outputs.forEach((outNode) => {
    if (outNode !== rootId && !levels.has(outNode)) {
      const nodeObj = nodes.find((n) => String(n.id) === outNode);
      const isNetwork = nodeObj && (nodeObj.type === 'network' || nodeObj.type === 'ip');
      if (isNetwork) {
        levels.set(outNode, 3);
      } else if (nodeObj && nodeObj.type === 'wallet') {
        levels.set(outNode, 2);
      }
    }
  });

  // Second pass: Other outputs from root (e.g. child txs)
  outputs.forEach((outNode) => {
    if (outNode !== rootId && !levels.has(outNode)) {
      const ins = inEdges.get(outNode) || [];
      const fedByL2 = ins.some((src) => levels.get(src) === 2);
      levels.set(outNode, fedByL2 ? 3 : 2);
    }
  });

  // Remaining nodes (downstream hops or unconnected)
  nodes.forEach((n) => {
    const nid = String(n.id);
    if (!levels.has(nid)) {
      if (n.type === 'network' || n.type === 'ip') {
        levels.set(nid, 3);
      } else {
        const ins = inEdges.get(nid) || [];
        const fedByL2 = ins.some((src) => levels.get(src) === 2);
        levels.set(nid, fedByL2 ? 3 : 2);
      }
    }
  });

  // Group nodes by level
  const levelGroups = new Map();
  nodes.forEach((n) => {
    const nid = String(n.id);
    const lvl = levels.get(nid) ?? 1;
    if (!levelGroups.has(lvl)) levelGroups.set(lvl, []);
    levelGroups.get(lvl).push(nid);
  });

  const positions = {};
  const sortedLevels = Array.from(levelGroups.keys()).sort((a, b) => a - b);

  sortedLevels.forEach((lvl) => {
    const group = levelGroups.get(lvl);
    const count = group.length;

    group.forEach((nid, idx) => {
      const offset = (idx - (count - 1) / 2);
      if (orientation === 'horizontal') {
        positions[nid] = {
          x: lvl * 200 + 80,
          y: offset * 110 + 200,
        };
      } else {
        positions[nid] = {
          x: offset * 160 + 260,
          y: lvl * 130 + 70,
        };
      }
    });
  });

  return positions;
}

export default function InvestigationGraph({
  nodes = [],
  edges = [],
  rootEntityId = null,
  onSelectNode = null,
  height = '580px',
}) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const [selectedElement, setSelectedElement] = useState(null);
  const [layoutMode, setLayoutMode] = useState('flow-vertical'); // 'flow-vertical' | 'flow-horizontal' | 'concentric' | 'cose'
  const [copiedId, setCopiedId] = useState(false);

  // Convert and bundle elements for Cytoscape
  const { cyNodes, bundledEdges, positions } = useMemo(() => {
    const cyN = nodes.map((n) => {
      const type = (n.type || 'unknown').toLowerCase();
      const isRoot = String(n.id) === String(rootEntityId);
      const isSynthetic = Boolean(n.is_synthetic);

      let color = ENTITY_COLORS[type] || ENTITY_COLORS.unknown;
      let shape = 'ellipse';
      let prefix = '[W]';

      if (type === 'transaction') {
        shape = 'roundrectangle';
        prefix = '[TX]';
      } else if (type === 'network' || type === 'ip') {
        shape = 'diamond';
        prefix = '[IP]';
      }

      const shortId = n.id.length > 14 ? `${n.id.slice(0, 6)}...${n.id.slice(-4)}` : n.id;
      const displayLabel = isRoot ? `★ ${prefix} ${shortId}` : `${prefix} ${shortId}`;

      return {
        data: {
          id: String(n.id),
          label: displayLabel,
          fullId: String(n.id),
          type: type,
          color: color,
          shape: shape,
          isRoot: isRoot,
          is_synthetic: isSynthetic,
          priority_tier: n.priority_tier,
          risk_score: n.risk_score,
          asn: n.asn,
          country: n.country,
          role: n.role,
        },
      };
    });

    const bEdges = bundleEdges(edges);
    const pos = computeInvestigativePositions(
      nodes,
      edges,
      rootEntityId,
      layoutMode === 'flow-horizontal' ? 'horizontal' : 'vertical'
    );

    return { cyNodes: cyN, bundledEdges: bEdges, positions: pos };
  }, [nodes, edges, rootEntityId, layoutMode]);

  // Initialize and update Cytoscape instance
  useEffect(() => {
    if (!containerRef.current) return;

    const elements = [
      ...cyNodes.map((n) => ({
        ...n,
        position: positions[n.data.id] || { x: 0, y: 0 },
      })),
      ...bundledEdges,
    ];

    const cy = cytoscape({
      container: containerRef.current,
      elements: elements,
      boxSelectionEnabled: false,
      autounselectify: false,
      style: [
        // Base Node Style
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)',
            'label': 'data(label)',
            'color': '#E8ECF1',
            'font-family': 'Inter, sans-serif',
            'font-size': '10px',
            'font-weight': 600,
            'text-valign': 'bottom',
            'text-margin-y': 5,
            'shape': 'data(shape)',
            'width': 32,
            'height': 32,
            'border-width': 2,
            'border-color': '#1A2634',
            'border-style': 'solid',
            'transition-property': 'background-color, border-color, shadow-blur, opacity',
            'transition-duration': '0.15s',
          },
        },
        // Synthetic Node Marker: Dashed Purple Border
        {
          selector: 'node[?is_synthetic]',
          style: {
            'border-style': 'dashed',
            'border-width': 2.5,
            'border-color': '#9D8BC9',
          },
        },
        // Root Investigated Entity: Prominent with Glowing Halo
        {
          selector: 'node[?isRoot]',
          style: {
            'width': 42,
            'height': 42,
            'border-width': 3.5,
            'border-color': '#FFFFFF',
            'font-size': '11px',
            'font-weight': 'bold',
            'shadow-blur': 18,
            'shadow-color': 'rgba(74, 144, 217, 0.8)',
            'shadow-opacity': 0.9,
            'z-index': 100,
          },
        },
        // Node Selected State
        {
          selector: 'node:selected',
          style: {
            'border-color': '#F3BA2F',
            'border-width': 3.5,
            'shadow-blur': 22,
            'shadow-color': 'rgba(243, 186, 47, 0.85)',
            'shadow-opacity': 1,
            'z-index': 105,
          },
        },
        // Base Edge Style
        {
          selector: 'edge',
          style: {
            'width': 'data(width)',
            'line-color': '#4A90D9',
            'target-arrow-color': '#4A90D9',
            'target-arrow-shape': 'triangle',
            'arrow-scale': 1.1,
            'curve-style': 'bezier',
            'control-point-step-size': 35,
            'label': 'data(label)',
            'font-size': '9px',
            'font-family': 'Inter, sans-serif',
            'font-weight': 600,
            'color': '#CBD5E1',
            'text-rotation': 'autorotate',
            'text-background-opacity': 0.9,
            'text-background-color': '#101722',
            'text-background-padding': 3,
            'text-background-shape': 'roundrectangle',
            'text-border-color': '#2A3B4D',
            'text-border-width': 1,
            'text-border-opacity': 0.9,
          },
        },
        // Address Cluster Edges (addr_addr): Muted Slate with wider curvature
        {
          selector: 'edge[primaryRel = "addr_addr"]',
          style: {
            'line-color': '#6B7F96',
            'target-arrow-color': '#6B7F96',
            'control-point-step-size': 50,
          },
        },
        // Synthetic Edge Marker: Dashed Purple Line
        {
          selector: 'edge[?is_synthetic]',
          style: {
            'line-style': 'dashed',
            'line-color': '#9D8BC9',
            'target-arrow-color': '#9D8BC9',
          },
        },
        // Edge Selected State
        {
          selector: 'edge:selected',
          style: {
            'line-color': '#F3BA2F',
            'target-arrow-color': '#F3BA2F',
            'width': 3.5,
            'z-index': 105,
          },
        },
        // Interactive Hover: Direct neighborhood highlighted, background dimmed
        {
          selector: '.highlighted',
          style: {
            'opacity': 1,
            'z-index': 999,
          },
        },
        {
          selector: '.dimmed',
          style: {
            'opacity': 0.16,
          },
        },
      ],
      layout:
        layoutMode === 'flow-vertical' || layoutMode === 'flow-horizontal'
          ? {
              name: 'preset',
              positions: (node) => positions[node.id()] || { x: 0, y: 0 },
              padding: 40,
            }
          : layoutMode === 'concentric'
          ? {
              name: 'concentric',
              concentric: (node) => (node.data('isRoot') ? 3 : node.data('type') === 'transaction' ? 2 : 1),
              levelWidth: () => 1,
              padding: 40,
              minNodeSpacing: 60,
              animate: false,
            }
          : {
              name: 'cose',
              animate: false,
              padding: 40,
              componentSpacing: 80,
              nodeOverlap: 40,
              idealEdgeLength: 90,
              edgeElasticity: 100,
            },
    });

    // Fit view with clean margin
    cy.ready(() => {
      cy.fit(undefined, 40);
    });

    // Hover Highlight
    cy.on('mouseover', 'node', (evt) => {
      const node = evt.target;
      const neighborhood = node.neighborhood().add(node);
      cy.elements().addClass('dimmed');
      neighborhood.removeClass('dimmed').addClass('highlighted');
    });

    cy.on('mouseout', 'node', () => {
      cy.elements().removeClass('dimmed').removeClass('highlighted');
    });

    // Tap on Node
    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      const data = node.data();

      // Find connected in/out relationships
      const connectedEdges = node.connectedEdges();
      const inCount = connectedEdges.filter((e) => e.target().id() === node.id()).length;
      const outCount = connectedEdges.filter((e) => e.source().id() === node.id()).length;

      setSelectedElement({
        type: 'node',
        id: data.id,
        fullId: data.fullId,
        entityType: data.type,
        color: data.color,
        isRoot: data.isRoot,
        is_synthetic: data.is_synthetic,
        priority_tier: data.priority_tier,
        risk_score: data.risk_score,
        asn: data.asn,
        country: data.country,
        role: data.role,
        inCount,
        outCount,
      });
    });

    // Tap on Edge
    cy.on('tap', 'edge', (evt) => {
      const edge = evt.target;
      const data = edge.data();

      setSelectedElement({
        type: 'edge',
        id: data.id,
        source: data.source,
        target: data.target,
        primaryRel: data.primaryRel,
        count: data.count,
        label: data.label,
        is_synthetic: data.is_synthetic,
        confidence: data.confidence,
        rawEdges: data.rawEdges,
      });
    });

    // Tap on Canvas Background (Deselect)
    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        setSelectedElement(null);
      }
    });

    cyRef.current = cy;

    return () => {
      cy.destroy();
    };
  }, [cyNodes, bundledEdges, positions, layoutMode]);

  const handleFit = () => {
    if (cyRef.current) cyRef.current.fit(undefined, 40);
  };

  const handleZoomIn = () => {
    if (cyRef.current) cyRef.current.zoom(cyRef.current.zoom() * 1.25);
  };

  const handleZoomOut = () => {
    if (cyRef.current) cyRef.current.zoom(cyRef.current.zoom() * 0.8);
  };

  const handleResetZoom = () => {
    if (cyRef.current) {
      cyRef.current.zoom(1);
      cyRef.current.center();
    }
  };

  const handleCopyId = (text) => {
    navigator.clipboard.writeText(text);
    setCopiedId(true);
    setTimeout(() => setCopiedId(false), 2000);
  };

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        height: height,
        backgroundColor: 'var(--ct-bg-primary)',
        border: '1px solid var(--ct-border)',
        borderRadius: '4px',
        overflow: 'hidden',
      }}
    >
      {/* Top Toolbar: Controls & Layout Mode */}
      <div
        style={{
          position: 'absolute',
          top: '12px',
          right: '12px',
          zIndex: 10,
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          backgroundColor: 'rgba(16, 23, 34, 0.88)',
          backdropFilter: 'blur(6px)',
          border: '1px solid var(--ct-border)',
          borderRadius: '4px',
          padding: '4px 6px',
        }}
      >
        {/* Layout Switcher */}
        <select
          value={layoutMode}
          onChange={(e) => setLayoutMode(e.target.value)}
          style={{
            fontSize: '11px',
            padding: '4px 8px',
            backgroundColor: 'var(--ct-bg-elevated)',
            color: 'var(--ct-text-primary)',
            border: '1px solid var(--ct-border)',
            borderRadius: '2px',
            cursor: 'pointer',
          }}
          title="Change Graph Layout"
        >
          <option value="flow-vertical">Flow ↓ (Investigative DAG)</option>
          <option value="flow-horizontal">Flow → (Horizontal)</option>
          <option value="concentric">Radial ◉ (Concentric)</option>
          <option value="cose">Organic ☊ (Bundled Force)</option>
        </select>

        <div style={{ width: '1px', height: '16px', backgroundColor: 'var(--ct-border)', margin: '0 2px' }} />

        <button
          type="button"
          onClick={handleFit}
          title="Fit view to all nodes"
          style={{ fontSize: '11px', padding: '4px 8px', backgroundColor: 'var(--ct-bg-elevated)' }}
        >
          Fit View
        </button>
        <button
          type="button"
          onClick={handleZoomIn}
          title="Zoom In"
          style={{ fontSize: '12px', padding: '4px 8px', backgroundColor: 'var(--ct-bg-elevated)', fontWeight: 'bold' }}
        >
          +
        </button>
        <button
          type="button"
          onClick={handleZoomOut}
          title="Zoom Out"
          style={{ fontSize: '12px', padding: '4px 8px', backgroundColor: 'var(--ct-bg-elevated)', fontWeight: 'bold' }}
        >
          -
        </button>
        <button
          type="button"
          onClick={handleResetZoom}
          title="Reset to 100% Zoom"
          style={{ fontSize: '11px', padding: '4px 8px', backgroundColor: 'var(--ct-bg-elevated)' }}
        >
          100%
        </button>
      </div>

      {/* Forensic Legend */}
      <div
        style={{
          position: 'absolute',
          bottom: '12px',
          left: '12px',
          zIndex: 10,
          backgroundColor: 'rgba(16, 23, 34, 0.92)',
          backdropFilter: 'blur(6px)',
          border: '1px solid var(--ct-border)',
          padding: '8px 12px',
          borderRadius: '4px',
          fontSize: '11px',
          display: 'flex',
          flexDirection: 'column',
          gap: '5px',
        }}
      >
        <div style={{ fontWeight: 600, color: 'var(--ct-text-secondary)', marginBottom: '2px' }}>
          Investigative Forensics
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: ENTITY_COLORS.wallet }} />
          <span>[W] Wallet Address</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: ENTITY_COLORS.transaction }} />
          <span>[TX] Bitcoin Transaction</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ width: '10px', height: '10px', transform: 'rotate(45deg)', backgroundColor: ENTITY_COLORS.network }} />
          <span>[IP] Network Observation</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ width: '12px', height: '12px', border: '2px solid #FFFFFF', borderRadius: '2px', display: 'inline-block' }} />
          <span>★ Investigated Focus</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', borderTop: '1px solid var(--ct-border)', paddingTop: '4px' }}>
          <span style={{ width: '14px', borderTop: '2px dashed #9D8BC9' }} />
          <span style={{ color: '#9D8BC9', fontWeight: 600 }}>Dashed: Synthetic Origin</span>
        </div>
      </div>

      {/* Interactive Detail Inspector Drawer */}
      {selectedElement && (
        <div
          style={{
            position: 'absolute',
            top: '12px',
            left: '12px',
            zIndex: 10,
            backgroundColor: 'rgba(16, 23, 34, 0.95)',
            backdropFilter: 'blur(8px)',
            border: '1px solid var(--ct-border)',
            padding: '12px 14px',
            borderRadius: '4px',
            width: '320px',
            fontSize: '12px',
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5)',
          }}
        >
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span
                style={{
                  fontSize: '10px',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  padding: '2px 6px',
                  borderRadius: '2px',
                  backgroundColor: selectedElement.type === 'node' ? `${selectedElement.color}22` : '#3D506833',
                  color: selectedElement.type === 'node' ? selectedElement.color : '#CBD5E1',
                  border: `1px solid ${selectedElement.type === 'node' ? selectedElement.color : '#3D5068'}`,
                }}
              >
                {selectedElement.type === 'node' ? selectedElement.entityType : 'RELATIONSHIP'}
              </span>
              {selectedElement.is_synthetic && (
                <span className="ct-synthetic-badge" style={{ fontSize: '10px' }}>
                  SYNTH
                </span>
              )}
            </div>
            <button
              type="button"
              onClick={() => setSelectedElement(null)}
              style={{
                background: 'none',
                border: 'none',
                color: 'var(--ct-text-muted)',
                cursor: 'pointer',
                fontSize: '14px',
                padding: '0 4px',
              }}
              title="Close inspector"
            >
              ✕
            </button>
          </div>

          {/* Node Details */}
          {selectedElement.type === 'node' && (
            <div>
              <div
                style={{
                  fontFamily: 'var(--ct-font-mono)',
                  fontSize: '12px',
                  wordBreak: 'break-all',
                  fontWeight: 600,
                  marginBottom: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '6px',
                }}
              >
                <span>{selectedElement.fullId}</span>
                <button
                  type="button"
                  onClick={() => handleCopyId(selectedElement.fullId)}
                  style={{
                    fontSize: '10px',
                    padding: '2px 6px',
                    backgroundColor: 'var(--ct-bg-elevated)',
                    border: '1px solid var(--ct-border)',
                  }}
                >
                  {copiedId ? 'Copied' : 'Copy'}
                </button>
              </div>

              {selectedElement.risk_score !== undefined && selectedElement.risk_score !== null && (
                <div style={{ marginBottom: '6px', fontSize: '11px', color: 'var(--ct-text-secondary)' }}>
                  Priority Score:{' '}
                  <strong style={{ color: '#E2E8F0' }}>
                    {Number(selectedElement.risk_score).toFixed(1)}
                  </strong>{' '}
                  ({selectedElement.priority_tier || 'LOW'})
                </div>
              )}

              {selectedElement.entityType === 'network' && (
                <div style={{ fontSize: '11px', color: 'var(--ct-text-secondary)', marginBottom: '8px' }}>
                  <div>Role: <strong style={{ color: '#E2E8F0' }}>{selectedElement.role || 'relay'}</strong></div>
                  <div>ASN: <strong style={{ color: '#E2E8F0' }}>AS{selectedElement.asn || '—'}</strong></div>
                  <div>Country: <strong style={{ color: '#E2E8F0' }}>{selectedElement.country || '—'}</strong></div>
                </div>
              )}

              <div
                style={{
                  display: 'flex',
                  gap: '12px',
                  fontSize: '11px',
                  color: 'var(--ct-text-muted)',
                  borderTop: '1px solid var(--ct-border)',
                  paddingTop: '8px',
                  marginTop: '6px',
                  marginBottom: '10px',
                }}
              >
                <span>Inbound: <strong style={{ color: '#E2E8F0' }}>{selectedElement.inCount}</strong></span>
                <span>Outbound: <strong style={{ color: '#E2E8F0' }}>{selectedElement.outCount}</strong></span>
                {selectedElement.isRoot && <span style={{ color: '#F3BA2F' }}>Current Root Focus</span>}
              </div>

              {/* Action to Pivot Investigation */}
              {!selectedElement.isRoot && selectedElement.entityType !== 'network' && (
                <button
                  type="button"
                  className="primary"
                  onClick={() => {
                    if (onSelectNode) {
                      onSelectNode(selectedElement.fullId, selectedElement.entityType);
                    }
                  }}
                  style={{ width: '100%', padding: '6px', fontSize: '11px' }}
                >
                  🔍 Pivot Investigation to this {selectedElement.entityType}
                </button>
              )}
            </div>
          )}

          {/* Edge Details */}
          {selectedElement.type === 'edge' && (
            <div>
              <div style={{ fontWeight: 600, fontSize: '13px', marginBottom: '6px', color: '#E2E8F0' }}>
                {formatRelationship(selectedElement.primaryRel)}
                {selectedElement.count > 1 && ` (${selectedElement.count}× transfers)`}
              </div>

              <div style={{ fontSize: '11px', color: 'var(--ct-text-secondary)', marginBottom: '4px' }}>
                Source:{' '}
                <span style={{ fontFamily: 'var(--ct-font-mono)', color: '#CBD5E1' }}>
                  {selectedElement.source.length > 16
                    ? `${selectedElement.source.slice(0, 8)}...${selectedElement.source.slice(-6)}`
                    : selectedElement.source}
                </span>
              </div>
              <div style={{ fontSize: '11px', color: 'var(--ct-text-secondary)', marginBottom: '8px' }}>
                Target:{' '}
                <span style={{ fontFamily: 'var(--ct-font-mono)', color: '#CBD5E1' }}>
                  {selectedElement.target.length > 16
                    ? `${selectedElement.target.slice(0, 8)}...${selectedElement.target.slice(-6)}`
                    : selectedElement.target}
                </span>
              </div>

              {selectedElement.confidence !== null && selectedElement.confidence !== undefined && (
                <div style={{ fontSize: '11px', color: 'var(--ct-text-secondary)', marginBottom: '6px' }}>
                  Confidence:{' '}
                  <strong style={{ color: '#E2E8F0' }}>
                    {(selectedElement.confidence * 100).toFixed(0)}%
                  </strong>
                </div>
              )}

              <div
                style={{
                  fontSize: '11px',
                  color: 'var(--ct-text-muted)',
                  borderTop: '1px solid var(--ct-border)',
                  paddingTop: '8px',
                  marginTop: '6px',
                }}
              >
                {selectedElement.count > 1 ? (
                  <div>
                    This single conduit aggregates <strong>{selectedElement.count}</strong> underlying historical
                    transactions between these entities, eliminating visual clutter while preserving full evidence auditability.
                  </div>
                ) : (
                  <div>Direct link recorded in the forensic graph edgelist.</div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Cytoscape Canvas */}
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
}
