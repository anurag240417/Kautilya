import React, { useEffect, useMemo, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import { getSeverityColor } from '../utils/colors';

const TRAIL_FORWARD = '#D4763A';
const TRAIL_BACK = '#7B68AE';

function fmtTime(ns) {
  if (!ns) return '';
  return new Date(ns / 1e6).toISOString().replace('T', ' ').slice(0, 16) + 'Z';
}

/**
 * Link-analysis graph of value flow between entities, with a time slider.
 * Edges appear as their transactions occur; the money trail is highlighted.
 * Props: graph (API payload), onSelect(entityId).
 */
export default function FlowGraph({ graph, onSelect }) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const [t, setT] = useState(1);
  const [playing, setPlaying] = useState(false);

  const [t0, t1] = graph.time_range;
  const span = Math.max(t1 - t0, 1);
  const cutoff = t0 + span * t;

  const trailEdges = useMemo(() => {
    const keys = new Map();
    const add = (trail, color) => {
      for (let i = 1; i < (trail || []).length; i += 1) {
        const a = trail[i - 1].entity;
        const b = trail[i].entity;
        keys.set(`${a}>${b}`, color);
        keys.set(`${b}>${a}`, color);
      }
    };
    add(graph.trail_back, TRAIL_BACK);
    add(graph.trail_forward, TRAIL_FORWARD);
    return keys;
  }, [graph]);

  // Build the cytoscape instance once per graph.
  useEffect(() => {
    if (!containerRef.current) return undefined;
    const byEdge = new Map();
    graph.events.forEach((ev) => {
      const k = `${ev.src}>${ev.dst}`;
      if (!byEdge.has(k)) byEdge.set(k, []);
      byEdge.get(k).push(ev);
    });
    const elements = [
      ...graph.nodes.map((n) => ({
        data: {
          id: String(n.id),
          label: n.label,
          color: getSeverityColor(n.score > 0 ? n.tier : 'low'),
          size: 14 + Math.min(n.score, 100) * 0.22,
          center: n.id === graph.center,
          service: n.is_service,
          seed: n.is_seed,
        },
      })),
      ...graph.edges.map((e) => {
        const k = `${e.src}>${e.dst}`;
        return {
          data: {
            id: `e-${k}`,
            source: String(e.src),
            target: String(e.dst),
            events: byEdge.get(k) || [],
            trail: trailEdges.get(k) || '',
            label: '',
          },
        };
      }),
    ];
    const cy = cytoscape({
      container: containerRef.current,
      elements,
      minZoom: 0.3,
      maxZoom: 3,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)',
            width: 'data(size)',
            height: 'data(size)',
            label: 'data(label)',
            color: '#E8ECF1',
            'font-size': 9,
            'text-valign': 'bottom',
            'text-margin-y': 4,
            'border-width': 1,
            'border-color': '#0F1923',
          },
        },
        { selector: 'node[?center]', style: { 'border-width': 3, 'border-color': '#E8ECF1' } },
        { selector: 'node[?seed]', style: { 'border-width': 3, 'border-color': '#E8ECF1', 'border-style': 'dashed' } },
        { selector: 'node[?service]', style: { shape: 'round-rectangle', 'background-opacity': 0.55 } },
        {
          selector: 'edge',
          style: {
            width: 1.4,
            'line-color': '#3D4F63',
            'target-arrow-color': '#3D4F63',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            label: 'data(label)',
            'font-size': 8,
            color: '#8899A6',
            'text-background-color': '#0F1923',
            'text-background-opacity': 0.8,
          },
        },
        { selector: 'edge[trail != ""]', style: { width: 3, 'line-color': 'data(trail)', 'target-arrow-color': 'data(trail)' } },
        { selector: '.hidden', style: { display: 'none' } },
      ],
      layout: { name: 'cose', animate: false, nodeRepulsion: 9000, idealEdgeLength: 70, randomize: true },
    });
    cy.on('tap', 'node', (evt) => onSelect && onSelect(Number(evt.target.id())));
    cyRef.current = cy;
    setT(1);
    setPlaying(false);
    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, [graph, trailEdges, onSelect]);

  // Update which edges are visible as the slider moves.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.batch(() => {
      cy.edges().forEach((edge) => {
        const evs = edge.data('events');
        const seen = evs.filter((ev) => ev.ts <= cutoff);
        const total = seen.reduce((s, ev) => s + ev.amount, 0);
        if (evs.length && seen.length === 0) {
          edge.addClass('hidden');
        } else {
          edge.removeClass('hidden');
          edge.data('label', seen.length ? `${total.toFixed(3)} BTC` : '');
        }
      });
    });
  }, [cutoff, graph]);

  // Replay.
  useEffect(() => {
    if (!playing) return undefined;
    const id = setInterval(() => {
      setT((prev) => {
        if (prev >= 1) {
          setPlaying(false);
          return 1;
        }
        return Math.min(prev + 0.02, 1);
      });
    }, 100);
    return () => clearInterval(id);
  }, [playing]);

  return (
    <div>
      <div
        ref={containerRef}
        style={{ height: 380, border: '1px solid var(--ct-border)', background: '#0B131B', borderRadius: 2 }}
      />
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 8, fontSize: 12 }}>
        <button
          type="button"
          onClick={() => {
            if (t >= 1) setT(0);
            setPlaying((p) => !p);
          }}
          style={{ padding: '3px 10px' }}
        >
          {playing ? 'Pause' : 'Replay'}
        </button>
        <input
          type="range"
          min="0"
          max="1"
          step="0.005"
          value={t}
          onChange={(e) => {
            setPlaying(false);
            setT(Number(e.target.value));
          }}
          style={{ flex: 1 }}
          aria-label="Time slider"
        />
        <span style={{ fontFamily: 'var(--ct-font-mono)', color: 'var(--ct-text-secondary)', minWidth: 130 }}>
          {fmtTime(cutoff)}
        </span>
      </div>
      <div style={{ display: 'flex', gap: 16, marginTop: 6, fontSize: 11, color: 'var(--ct-text-muted)', flexWrap: 'wrap' }}>
        <span><span style={{ color: TRAIL_FORWARD }}>&#9632;</span> onward money trail</span>
        <span><span style={{ color: TRAIL_BACK }}>&#9632;</span> inbound trail (towards sources)</span>
        <span>Rounded square = service-like hub (exchange-style)</span>
        <span>Dashed white ring = known-illicit seed</span>
        <span>Node size = priority score. Click a node to open it.</span>
      </div>
    </div>
  );
}
