import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as d3 from 'd3';
import { ZoomIn, ZoomOut, X, RefreshCw, Tag, Maximize2 } from 'lucide-react';
import { VisualizerBundle, ThreatGraphNode, ThreatGraphLink } from '../types';
import { useSimulation } from '../context/SimulationContext';

interface NetworkGraphProps {
  bundle: VisualizerBundle | null;
  loading: boolean;
  onSelectEntity?: (entity: ThreatGraphNode) => void;
  onRefresh?: () => void;
}

export const NetworkGraph: React.FC<NetworkGraphProps> = ({
  bundle,
  loading,
  onSelectEntity,
}) => {
  const {
    latestTx,
    isStreaming,
    isGenerating,
    streamingStats,
    generationProgress,
    recentLiveTxs,
  } = useSimulation();

  const svgRef = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [selectedNode, setSelectedNode] = useState<ThreatGraphNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<ThreatGraphNode | null>(null);
  const [showAllLabels, setShowAllLabels] = useState<boolean>(false);
  const zoomBehaviorRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const particleLayerRef = useRef<d3.Selection<SVGGElement, unknown, null, undefined> | null>(null);
  const nodesRef = useRef<ThreatGraphNode[]>([]);
  const declutterRef = useRef<(() => void) | null>(null);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);

  const getNodeColor = (type: string) => {
    switch (type) {
      case 'syndicate':
        return '#E11D48'; // Rose
      case 'botnet':
        return '#EA580C'; // Orange
      case 'card':
        return '#7C3AED'; // Purple
      case 'breach_campaign':
        return '#64748B'; // Slate
      case 'bridge_card':
        return '#D97706'; // Amber
      case 'merchant':
        return '#0284C7'; // Sky
      case 'mule':
        return '#059669'; // Emerald
      default:
        return '#B85526'; // Copper
    }
  };

  const zoomToFit = useCallback((animate = true) => {
    if (!svgRef.current || !zoomBehaviorRef.current || nodesRef.current.length === 0) return;
    const container = containerRef.current;
    if (!container) return;
    const w = container.clientWidth || 1000;
    const h = container.clientHeight || 580;

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    nodesRef.current.forEach((n) => {
      if (n.x !== undefined && n.y !== undefined) {
        const r = n.radius || 15;
        minX = Math.min(minX, n.x - r - 35);
        maxX = Math.max(maxX, n.x + r + 35);
        minY = Math.min(minY, n.y - r - 35);
        maxY = Math.max(maxY, n.y + r + 35);
      }
    });

    if (!isFinite(minX) || !isFinite(maxX)) return;
    const graphW = maxX - minX;
    const graphH = maxY - minY;
    if (graphW <= 0 || graphH <= 0) return;

    const padX = 75;
    const padY = 70;
    const scale = Math.min(1.15, Math.max(0.25, Math.min((w - padX * 2) / graphW, (h - padY * 2) / graphH)));
    const midX = (minX + maxX) / 2;
    const midY = (minY + maxY) / 2;
    const tx = w / 2 - scale * midX;
    const ty = h / 2 - scale * midY;

    const transform = d3.zoomIdentity.translate(tx, ty).scale(scale);
    const svg = d3.select(svgRef.current);
    if (animate) {
      svg.transition().duration(400).ease(d3.easeCubicOut).call(zoomBehaviorRef.current.transform, transform);
    } else {
      svg.call(zoomBehaviorRef.current.transform, transform);
    }
  }, []);

  useEffect(() => {
    if (!bundle || !svgRef.current || !containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth || 1100;
    const height = Math.max(580, container.clientHeight || 580);

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Defs for arrowheads and glow filters
    const defs = svg.append('defs');

    // Arrow marker
    defs
      .append('marker')
      .attr('id', 'conduit-arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 18)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#A8A29E');

    // Amber glow filter for bridge pivot cards
    const filterAmber = defs
      .append('filter')
      .attr('id', 'glow-amber')
      .attr('x', '-30%')
      .attr('y', '-30%')
      .attr('width', '160%')
      .attr('height', '160%');
    filterAmber.append('feGaussianBlur').attr('stdDeviation', '2.5').attr('result', 'blur');
    const mergeAmber = filterAmber.append('feMerge');
    mergeAmber.append('feMergeNode').attr('in', 'blur');
    mergeAmber.append('feMergeNode').attr('in', 'SourceGraphic');

    const g = svg.append('g').attr('class', 'graph-root');

    // Zoom behavior
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    zoomBehaviorRef.current = zoom;
    svg.call(zoom);

    // Deep clone nodes and links to prevent d3 mutation collisions
    const rawNodes = bundle.threat_graph?.nodes || [];
    const rawLinks = bundle.threat_graph?.links || [];

    if (rawNodes.length === 0) return;

    // Scale target coordinates from 1400x900 base canvas into current container
    const scaleX = width / 1400;
    const scaleY = height / 900;
    const baseScale = Math.min(scaleX, scaleY) * 0.96;
    const offsetX = (width - 1400 * baseScale) / 2;
    const offsetY = (height - 900 * baseScale) / 2;

    const nodes: ThreatGraphNode[] = rawNodes.map((d) => {
      const existing = nodesRef.current.find((n) => n.id === d.id);
      const targetX = d.target_x !== undefined ? d.target_x * baseScale + offsetX : width / 2;
      const targetY = d.target_y !== undefined ? d.target_y * baseScale + offsetY : height / 2;
      return {
        ...d,
        x: existing?.x ?? targetX,
        y: existing?.y ?? targetY,
        vx: existing?.vx ?? 0,
        vy: existing?.vy ?? 0,
        target_x: targetX,
        target_y: targetY,
      };
    });

    const nodeById = new Map<string, ThreatGraphNode>();
    nodes.forEach((n) => nodeById.set(n.id, n));

    const links: ThreatGraphLink[] = rawLinks
      .map((l) => {
        const sourceId = typeof l.source === 'object' ? l.source.id : l.source;
        const targetId = typeof l.target === 'object' ? l.target.id : l.target;
        return {
          ...l,
          source: nodeById.get(sourceId) || sourceId,
          target: nodeById.get(targetId) || targetId,
        };
      })
      .filter((l) => typeof l.source === 'object' && typeof l.target === 'object');

    // Force simulation with sector anchoring and strong repulsion
    const simulation = d3
      .forceSimulation<ThreatGraphNode>(nodes)
      .alphaDecay(0.038)
      .velocityDecay(0.62)
      .force(
        'link',
        d3
          .forceLink<ThreatGraphNode, ThreatGraphLink>(links)
          .id((d) => d.id)
          .distance((d: any) => {
            if (d.type === 'OPERATES') return 85 * baseScale;
            if (d.type === 'CONTROLS') return 95 * baseScale;
            if (d.type === 'COMPROMISES') return 75 * baseScale;
            if (d.type === 'ATTACKS') return 110 * baseScale;
            if (d.type === 'TRANSACTS') return 130 * baseScale;
            if (d.type === 'CASH_OUT') return 120 * baseScale;
            return 95 * baseScale;
          })
          .strength(0.3)
      )
      .force(
        'charge',
        d3.forceManyBody().strength((d: any) => {
          if (d.type === 'syndicate') return -850;
          if (d.type === 'bridge_card') return -550;
          if (d.type === 'breach_campaign') return -480;
          if (d.type === 'botnet') return -380;
          if (d.type === 'merchant') return -300;
          return -200;
        })
      )
      .force('x', d3.forceX((d: any) => d.target_x ?? width / 2).strength(0.24))
      .force('y', d3.forceY((d: any) => d.target_y ?? height / 2).strength(0.24))
      .force('collision', d3.forceCollide().radius((d: any) => (d.radius || 12) + 16));

    // Warm-up simulation so steady state is achieved before painting
    for (let i = 0; i < 60; ++i) {
      simulation.tick();
    }

    // Render Links
    const linkGroup = g.append('g').attr('class', 'links');
    const linkLines = linkGroup
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('stroke', (d) => {
        if (d.declined_count && d.declined_count > 0) return '#F87171'; // Red for fraud/declined
        return '#CBD5E1'; // Neutral conduit
      })
      .attr('stroke-width', (d) => Math.max(1.2, Math.min(3.5, Math.log10((d.tx_count || 1) + 1) * 2)))
      .attr('stroke-opacity', 0.5)
      .attr('marker-end', 'url(#conduit-arrow)');

    // Real-Time Animated Particle Layer
    const particleLayer = g.append('g').attr('class', 'particles');
    particleLayerRef.current = particleLayer as any;
    nodesRef.current = nodes;

    // Render Nodes
    const nodeGroup = g.append('g').attr('class', 'nodes');
    const nodeElements = nodeGroup
      .selectAll('g.node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'node cursor-pointer')
      .call(
        d3
          .drag<SVGGElement, ThreatGraphNode>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    // Node glyphs based on type
    nodeElements.each(function (d) {
      const el = d3.select(this);
      const color = getNodeColor(d.type);
      const r = d.radius || 12;

      if (d.type === 'syndicate') {
        // Prominent Red Syndicate Threat Group Node
        el.append('circle')
          .attr('r', r + 7)
          .attr('fill', 'none')
          .attr('stroke', color)
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '4 3')
          .attr('opacity', 0.65);

        el.append('circle')
          .attr('r', r)
          .attr('fill', color)
          .attr('stroke', '#FFFFFF')
          .attr('stroke-width', 3)
          .attr('filter', 'drop-shadow(0 2px 5px rgba(225, 29, 72, 0.4))');

        const synLabel = d.id.replace('SYN_', '').replace('IN_', '').replace('US_', '').slice(0, 4);
        el.append('text')
          .attr('text-anchor', 'middle')
          .attr('dy', '0.35em')
          .attr('fill', '#FFFFFF')
          .attr('font-size', '10px')
          .attr('font-weight', '700')
          .attr('font-family', 'ui-monospace, monospace')
          .text(synLabel || 'SYN');

      } else if (d.type === 'bridge_card') {
        // Glowing Amber Diamond for Bridge Pivot Cards
        el.append('rect')
          .attr('width', r * 1.8)
          .attr('height', r * 1.8)
          .attr('x', -r * 0.9)
          .attr('y', -r * 0.9)
          .attr('transform', 'rotate(45)')
          .attr('fill', color)
          .attr('stroke', '#FEF08A')
          .attr('stroke-width', 2.5)
          .attr('filter', 'url(#glow-amber)');

        el.append('text')
          .attr('text-anchor', 'middle')
          .attr('dy', '0.35em')
          .attr('fill', '#FFFFFF')
          .attr('font-size', '8.5px')
          .attr('font-weight', '800')
          .attr('font-family', 'ui-monospace, monospace')
          .text('PIVOT');

      } else if (d.type === 'breach_campaign') {
        // Concentric Violet Circle for Contracted Breach Batches
        el.append('circle')
          .attr('r', r)
          .attr('fill', '#8B5CF6')
          .attr('stroke', '#DDD6FE')
          .attr('stroke-width', 2.5)
          .attr('filter', 'drop-shadow(0 2px 4px rgba(139, 92, 246, 0.35))');

        const cardCount = (d as any).card_count || 'Bat';
        el.append('text')
          .attr('text-anchor', 'middle')
          .attr('dy', '0.35em')
          .attr('fill', '#FFFFFF')
          .attr('font-size', '9px')
          .attr('font-weight', '700')
          .attr('font-family', 'ui-monospace, monospace')
          .text(`${cardCount}c`);

      } else {
        // Standard entity circle (Merchant, Mule, Card, Botnet)
        el.append('circle')
          .attr('r', r)
          .attr('fill', color)
          .attr('stroke', '#FFFFFF')
          .attr('stroke-width', 2)
          .attr('filter', 'drop-shadow(0 1px 3px rgba(0,0,0,0.15))');

        // Short inner label for cards and merchants
        if (d.type === 'merchant' && (d as any).details?.MCC) {
          el.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('fill', '#FFFFFF')
            .attr('font-size', '8px')
            .attr('font-weight', '700')
            .attr('font-family', 'ui-monospace, monospace')
            .text(String((d as any).details.MCC).slice(0, 4));
        } else if (d.type === 'card') {
          el.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('fill', '#FFFFFF')
            .attr('font-size', '7.5px')
            .attr('font-weight', '700')
            .attr('font-family', 'ui-monospace, monospace')
            .text(d.id.slice(-2));
        }
      }

      // Exterior text label
      const isHub = ['syndicate', 'botnet', 'bridge_card', 'breach_campaign'].includes(d.type);
      let labelText = d.label;
      if (d.type === 'syndicate') {
        labelText = d.label.length > 20 ? d.label.substring(0, 18) + '...' : d.label;
      } else if (d.type === 'bridge_card') {
        labelText = `Pivot: ..${d.id.slice(-4)}`;
      } else if (d.type === 'card') {
        labelText = `Card: ..${d.id.slice(-4)}`;
      } else if (d.type === 'merchant') {
        labelText = `MID: ${d.id.slice(-6)}`;
      } else if (d.type === 'mule') {
        labelText = `Mule: ${d.id.slice(-6)}`;
      }

      el.append('text')
        .attr('class', isHub ? 'node-label node-label-hub' : 'node-label node-label-minor')
        .attr('dy', r + 13)
        .attr('text-anchor', 'middle')
        .attr('fill', '#292524')
        .attr('font-size', isHub ? '11px' : '9.5px')
        .attr('font-family', 'ui-monospace, monospace')
        .attr('font-weight', isHub ? '700' : '500')
        .attr('paint-order', 'stroke')
        .attr('stroke', '#FFFFFF')
        .attr('stroke-width', '3px')
        .attr('stroke-linecap', 'round')
        .attr('stroke-linejoin', 'round')
        .style('display', isHub || showAllLabels ? 'block' : 'none')
        .text(labelText);
    });

    // Node interactions with neighborhood highlighting
    nodeElements
      .on('mouseenter', function (_, d) {
        setHoveredNode(d);

        // Highlight connected neighborhood
        const connectedIds = new Set<string>([d.id]);
        links.forEach((l: any) => {
          const sid = typeof l.source === 'object' ? l.source.id : l.source;
          const tid = typeof l.target === 'object' ? l.target.id : l.target;
          if (sid === d.id) connectedIds.add(tid);
          if (tid === d.id) connectedIds.add(sid);
        });

        nodeElements.attr('opacity', (n) => (connectedIds.has(n.id) ? 1.0 : 0.2));
        linkLines
          .attr('stroke-opacity', (l: any) => {
            const sid = typeof l.source === 'object' ? l.source.id : l.source;
            const tid = typeof l.target === 'object' ? l.target.id : l.target;
            return sid === d.id || tid === d.id ? 1.0 : 0.08;
          })
          .attr('stroke-width', (l: any) => {
            const sid = typeof l.source === 'object' ? l.source.id : l.source;
            const tid = typeof l.target === 'object' ? l.target.id : l.target;
            return sid === d.id || tid === d.id ? 2.8 : 1.0;
          });
      })
      .on('mouseleave', function () {
        setHoveredNode(null);
        nodeElements.attr('opacity', 1.0);
        linkLines
          .attr('stroke-opacity', 0.5)
          .attr('stroke-width', (d) => Math.max(1.2, Math.min(3.5, Math.log10((d.tx_count || 1) + 1) * 2)));
      })
      .on('click', (event, d) => {
        event.stopPropagation();
        setSelectedNode(d);
        if (onSelectEntity) onSelectEntity(d);
      });

    // Tick update
    simulation.on('tick', () => {
      linkLines
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      nodeElements.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
      nodesRef.current = nodes;
    });

    // Hub labels are always drawn, and they are far wider than the collision radius the force
    // layout uses, so in a dense cluster they print on top of each other and become unreadable.
    // Measure each label in screen space once the fit transform has been applied, keep the first
    // label to claim a region, and hide the ones that land on top of it. Hubs are considered
    // before minor labels so the labels that carry information win.
    const declutterLabels = () => {
      const placed: DOMRect[] = [];
      const labels = Array.from(g.selectAll<SVGTextElement, unknown>('.node-label').nodes());
      labels.sort(
        (a, b) =>
          Number(b.classList.contains('node-label-hub')) -
          Number(a.classList.contains('node-label-hub'))
      );
      labels.forEach((t) => {
        if (t.style.display === 'none') return;
        const box = t.getBoundingClientRect();
        if (box.width === 0 && box.height === 0) return;
        const collides = placed.some(
          (p) => !(box.right < p.left || box.left > p.right || box.bottom < p.top || box.top > p.bottom)
        );
        if (collides) {
          t.style.display = 'none';
        } else {
          placed.push(box);
        }
      });
    };
    declutterRef.current = declutterLabels;

    simulation.on('end', () => {
      zoomToFit(true);
      requestAnimationFrame(() => declutterLabels());
    });

    // Auto-fit to viewport without clipping
    zoomToFit(false);
    requestAnimationFrame(() => declutterLabels());

    // The canvas keeps its initial transform as the page layout changes, which is what clipped
    // nodes at the panel edge after a resize or a late-loading font.
    if (resizeObserverRef.current) resizeObserverRef.current.disconnect();
    const observer = new ResizeObserver(() => {
      zoomToFit(false);
      declutterLabels();
    });
    observer.observe(container);
    resizeObserverRef.current = observer;

    // Background click deselects
    svg.on('click', () => {
      setSelectedNode(null);
    });

    return () => {
      simulation.stop();
      observer.disconnect();
    };
  }, [bundle]);

  // Effect to toggle minor label visibility without restarting simulation
  useEffect(() => {
    if (!svgRef.current) return;
    d3.select(svgRef.current)
      .selectAll('.node-label-minor')
      .style('display', showAllLabels ? 'block' : 'none');
    // "All Labels" is an explicit request to see everything, so it is left alone. Collapsing
    // back to hub-only re-applies the declutter, which is where the overlap actually mattered.
    if (!showAllLabels) requestAnimationFrame(() => declutterRef.current?.());
  }, [showAllLabels]);

  // Authentic Real-Time Transaction Network Mutator
  useEffect(() => {
    if (!latestTx || !svgRef.current || nodesRef.current.length === 0) return;

    const layer = particleLayerRef.current;
    const nodes = nodesRef.current;
    const isFraud = latestTx.is_fraud === 1;

    const cardIdStr = (latestTx.card_id || '').toLowerCase();
    const merchantIdStr = (latestTx.merchant_id || '').toLowerCase();
    const syndicateIdStr = (latestTx.syndicate_id || '').toLowerCase();
    const botnetIdStr = (latestTx.botnet_cluster_id || '').toLowerCase();
    const muleIdStr = (latestTx.beneficiary_account_id || '').toLowerCase();

    // 1. Authentic Source Resolution
    // Priority A: Exact unrolled card or bridge card
    let sourceNode = nodes.find(
      (n) => (n.type === 'card' || n.type === 'bridge_card') &&
             (n.id.toLowerCase() === cardIdStr || (n.label && n.label.toLowerCase().includes(cardIdStr.slice(-6))))
    );

    // Priority B: If card is inside a breach campaign batch, map to that authentic campaign node
    if (!sourceNode && (botnetIdStr || syndicateIdStr)) {
      sourceNode = nodes.find(
        (n) => n.type === 'breach_campaign' &&
               ((botnetIdStr && (n.botnet_id?.toLowerCase() === botnetIdStr || n.id.toLowerCase().includes(botnetIdStr))) ||
                (syndicateIdStr && n.syndicate_id?.toLowerCase() === syndicateIdStr))
      );
    }

    // Priority C: If syndicate root node
    if (!sourceNode && syndicateIdStr) {
      sourceNode = nodes.find(
        (n) => n.type === 'syndicate' && n.id.toLowerCase() === syndicateIdStr
      );
    }

    // 2. Authentic Target Resolution
    let targetNode = nodes.find(
      (n) => n.type === 'merchant' &&
             (n.id.toLowerCase() === merchantIdStr || (n.label && n.label.toLowerCase().includes(merchantIdStr)))
    );

    if (!targetNode && muleIdStr) {
      targetNode = nodes.find(
        (n) => n.type === 'mule' &&
               (n.id.toLowerCase() === muleIdStr || (n.label && n.label.toLowerCase().includes(muleIdStr)))
      );
    }

    // ZERO ARBITRARY ASSUMPTIONS:
    // If authentic entities are not present in the active threat topology, DO NOT FAKE NODES!
    if (!sourceNode && !targetNode) {
      return;
    }

    // 3. In-Place Authentic State Mutation
    const mutateNode = (node: ThreatGraphNode) => {
      if (!node.details) node.details = {};
      if (node.type === 'card' || node.type === 'bridge_card') {
        const curTx = parseInt(String(node.details['Tx Attempts'] || '0'), 10) || 0;
        node.details['Tx Attempts'] = curTx + 1;
        node.details['Latest Response'] = latestTx.response_code;
        if (isFraud) {
          const prevVol = parseFloat(String(node.details['Fraud Volume'] || '$0').replace(/[^0-9.]/g, '')) || 0;
          node.details['Fraud Volume'] = `$${(prevVol + latestTx.amount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        }
      } else if (node.type === 'breach_campaign') {
        const curTx = parseInt(String(node.details['Tx Attempts'] || '0'), 10) || 0;
        node.details['Tx Attempts'] = curTx + 1;
        const prevVol = parseFloat(String(node.details['Total Extracted Volume'] || '$0').replace(/[^0-9.]/g, '')) || 0;
        node.details['Total Extracted Volume'] = `$${(prevVol + latestTx.amount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        node.total_volume_usd = (node.total_volume_usd || 0) + latestTx.amount;
      } else if (node.type === 'merchant') {
        const curTx = parseInt(String(node.details['Tx Ingress'] || '0'), 10) || 0;
        node.details['Tx Ingress'] = curTx + 1;
      }
    };

    if (sourceNode) mutateNode(sourceNode);
    if (targetNode) mutateNode(targetNode);

    // If currently inspecting one of these nodes, refresh the inspector drawer
    if (selectedNode && (selectedNode.id === sourceNode?.id || selectedNode.id === targetNode?.id)) {
      const activeNode = selectedNode.id === sourceNode?.id ? sourceNode : targetNode;
      if (activeNode) setSelectedNode({ ...activeNode });
    }

    // 4. Authentic Visual Feedback: Pulse the real nodes
    const pulseRing = (node: ThreatGraphNode, strokeColor: string) => {
      if (!layer || node.x === undefined || node.y === undefined) return;
      const r = node.radius || 12;
      const ring = layer
        .append('circle')
        .attr('cx', node.x)
        .attr('cy', node.y)
        .attr('r', r)
        .attr('fill', 'none')
        .attr('stroke', strokeColor)
        .attr('stroke-width', 2.5)
        .attr('opacity', 0.95);

      ring
        .transition()
        .duration(450)
        .ease(d3.easeCubicOut)
        .attr('r', r + 16)
        .attr('stroke-width', 0)
        .attr('opacity', 0)
        .on('end', () => ring.remove());
    };

    const pulseColor = isFraud ? '#E11D48' : '#0284C7';

    if (sourceNode) pulseRing(sourceNode, isFraud ? '#E11D48' : '#D97706');
    if (targetNode) pulseRing(targetNode, pulseColor);

    // 5. Authentic Conduit Lighting between Source and Target
    if (
      sourceNode &&
      targetNode &&
      sourceNode.x !== undefined &&
      sourceNode.y !== undefined &&
      targetNode.x !== undefined &&
      targetNode.y !== undefined
    ) {
      // Find and flash the authentic link line if present
      const svg = d3.select(svgRef.current);
      svg
        .selectAll<SVGLineElement, any>('.links line')
        .filter((l: any) => {
          const sid = typeof l.source === 'object' ? l.source.id : l.source;
          const tid = typeof l.target === 'object' ? l.target.id : l.target;
          return (
            (sid === sourceNode?.id && tid === targetNode?.id) ||
            (sid === targetNode?.id && tid === sourceNode?.id)
          );
        })
        .transition()
        .duration(80)
        .attr('stroke', isFraud ? '#E11D48' : '#0284C7')
        .attr('stroke-width', 4)
        .attr('stroke-opacity', 1.0)
        .transition()
        .duration(450)
        .attr('stroke', (d: any) => (d.declined_count && d.declined_count > 0 ? '#F87171' : '#CBD5E1'))
        .attr('stroke-width', (d: any) => Math.max(1.2, Math.min(3.5, Math.log10((d.tx_count || 1) + 1) * 2)))
        .attr('stroke-opacity', 0.5);

      // Render authentic directed authorization packet along the vector between source and target
      if (layer) {
        const startX = sourceNode.x;
        const startY = sourceNode.y;
        const endX = targetNode.x;
        const endY = targetNode.y;

        const packet = layer
          .append('circle')
          .attr('cx', startX)
          .attr('cy', startY)
          .attr('r', isFraud ? 4.5 : 3.5)
          .attr('fill', isFraud ? '#E11D48' : '#0284C7')
          .attr('stroke', '#FFFFFF')
          .attr('stroke-width', 1.5)
          .attr('opacity', 0.95);

        packet
          .transition()
          .duration(340)
          .ease(d3.easeQuadOut)
          .attr('cx', endX)
          .attr('cy', endY)
          .on('end', () => packet.remove());
      }
    }
  }, [latestTx]);

  const handleZoom = (factor: number) => {
    if (!svgRef.current || !zoomBehaviorRef.current) return;
    d3.select(svgRef.current)
      .transition()
      .duration(250)
      .call(zoomBehaviorRef.current.scaleBy, factor);
  };

  const handleResetZoom = () => {
    zoomToFit(true);
  };

  return (
    <div className="space-y-3">
      {/* Topology Canvas Card */}
      <div
        ref={containerRef}
        className="relative bg-white rounded-xl border border-border overflow-hidden shadow-sm h-[640px] flex flex-col"
      >
        {/* Top Overlay: Legend & Metrics Bar */}
        <div className="absolute top-3 left-3 right-3 z-10 flex flex-wrap items-center justify-between gap-2 pointer-events-none">
          {/* Legend Chips */}
          <div className="flex flex-wrap items-center gap-1.5 p-1.5 bg-white/95 backdrop-blur rounded-lg border border-border shadow-xs pointer-events-auto">
            <span className="text-[10px] font-semibold text-stone-500 uppercase tracking-wider px-1">
              Topology:
            </span>
            <span className="flex items-center space-x-1 text-[11px] font-medium text-stone-700 px-1.5 py-0.5 rounded bg-stone-50 border border-stone-200">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-600"></span>
              <span>Syndicate</span>
            </span>
            <span className="flex items-center space-x-1 text-[11px] font-medium text-stone-700 px-1.5 py-0.5 rounded bg-stone-50 border border-stone-200">
              <span className="w-2.5 h-2.5 rounded-full bg-orange-600"></span>
              <span>Proxy / Botnet</span>
            </span>
            <span className="flex items-center space-x-1 text-[11px] font-medium text-stone-700 px-1.5 py-0.5 rounded bg-stone-50 border border-stone-200">
              <span className="w-2.5 h-2.5 rounded-full bg-purple-600"></span>
              <span>Card</span>
            </span>
            <span className="flex items-center space-x-1 text-[11px] font-medium text-stone-700 px-1.5 py-0.5 rounded bg-stone-50 border border-stone-200">
              <span className="w-2.5 h-2.5 rotate-45 bg-amber-500"></span>
              <span>Bridge Pivot</span>
            </span>
            <span className="flex items-center space-x-1 text-[11px] font-medium text-stone-700 px-1.5 py-0.5 rounded bg-stone-50 border border-stone-200">
              <span className="w-2.5 h-2.5 rounded-full bg-sky-600"></span>
              <span>Merchant</span>
            </span>
            <span className="flex items-center space-x-1 text-[11px] font-medium text-stone-700 px-1.5 py-0.5 rounded bg-stone-50 border border-stone-200">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-600"></span>
              <span>Mule Ring</span>
            </span>
          </div>

          {/* Controls: Label Density & Zoom Toolbar */}
          <div className="flex items-center space-x-2 pointer-events-auto">
            {/* Live Streaming & Synthesis HUD Indicator */}
            {isStreaming ? (
              <div className="flex items-center space-x-1.5 px-3 py-1 rounded-full bg-emerald-50 border border-emerald-300 text-emerald-800 text-xs font-mono animate-pulse shadow-xs">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
                <span className="font-semibold">LIVE STREAM</span>
                <span className="text-emerald-700">({streamingStats?.current_tps || 25} TPS)</span>
              </div>
            ) : isGenerating ? (
              <div className="flex items-center space-x-1.5 px-3 py-1 rounded-full bg-copper-50 border border-copper-300 text-copper-800 text-xs font-mono shadow-xs">
                <RefreshCw className="w-3 h-3 animate-spin text-copper-600" />
                <span className="font-semibold">SYNTHESIZING {generationProgress?.pct || 0}%</span>
              </div>
            ) : (
              <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-stone-100 border border-stone-200 text-stone-600 text-xs font-mono">
                <span className="w-2 h-2 rounded-full bg-stone-400" />
                <span>TOPOLOGY ACTIVE</span>
              </div>
            )}

            {/* Label Density Toggle */}
            <button
              type="button"
              onClick={() => setShowAllLabels((v) => !v)}
              title={showAllLabels ? 'Click to show Hub labels only' : 'Click to show All node labels'}
              className={`px-2.5 py-1 rounded-lg text-xs font-mono font-medium border flex items-center space-x-1.5 cursor-pointer transition-colors shadow-2xs ${
                showAllLabels
                  ? 'bg-copper-600 border-copper-700 text-white'
                  : 'bg-white/95 border-border text-stone-700 hover:bg-stone-50'
              }`}
            >
              <Tag className="w-3.5 h-3.5" />
              <span>{showAllLabels ? 'All Labels' : 'Hub Labels'}</span>
            </button>

            {/* Zoom Toolbar */}
            <div className="flex items-center space-x-1 p-1 bg-white/95 backdrop-blur rounded-lg border border-border shadow-xs">
              <button
                type="button"
                onClick={() => handleZoom(1.3)}
                title="Zoom In"
                className="p-1.5 rounded hover:bg-stone-100 text-stone-700 cursor-pointer"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                type="button"
                onClick={() => handleZoom(0.7)}
                title="Zoom Out"
                className="p-1.5 rounded hover:bg-stone-100 text-stone-700 cursor-pointer"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <button
                type="button"
                onClick={handleResetZoom}
                title="Fit to Screen"
                className="p-1.5 rounded hover:bg-stone-100 text-stone-700 cursor-pointer"
              >
                <Maximize2 className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Real-Time Live Event Stream Ticker (Bottom of Graph) */}
        {recentLiveTxs && recentLiveTxs.length > 0 && !hoveredNode && (
          <div className="absolute bottom-3 left-3 right-3 z-10 bg-white/95 backdrop-blur-md rounded-xl border border-border shadow-md p-2 flex items-center justify-between gap-3 text-xs overflow-x-auto no-scrollbar pointer-events-auto">
            <div className="flex items-center space-x-2 shrink-0">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="font-bold text-stone-800 text-[11px] uppercase tracking-wide">Live Stream Feed</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-stone-100 text-stone-600 border border-stone-200">
                {recentLiveTxs.length} events
              </span>
            </div>

            <div className="flex items-center space-x-2 overflow-x-auto no-scrollbar py-0.5">
              {recentLiveTxs.slice(0, 4).map((tx) => (
                <div
                  key={tx.transaction_id}
                  className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-[11px] font-mono border shrink-0 transition-all ${
                    tx.is_fraud === 1
                      ? 'bg-rose-50 border-rose-300 text-rose-900 shadow-2xs'
                      : 'bg-stone-50 border-border text-stone-700'
                  }`}
                >
                  <span className="font-semibold">{tx.card_id}</span>
                  <span className="text-stone-400">→</span>
                  <span>{tx.merchant_id}</span>
                  <span className="font-bold text-stone-900">
                    {tx.currency === 'INR' ? '₹' : '$'}{tx.amount.toFixed(2)}
                  </span>
                  <span
                    className={`px-1 rounded text-[9px] font-bold uppercase ${
                      tx.is_fraud === 1
                        ? 'bg-rose-200 text-rose-800'
                        : 'bg-emerald-100 text-emerald-800'
                    }`}
                  >
                    {tx.is_fraud === 1 ? (tx.scenario_tag || 'FRAUD') : 'APPROV'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Hover / Tooltip HUD (Bottom Left) */}
        {hoveredNode && !selectedNode && (
          <div className="absolute bottom-3 left-3 z-10 p-3 bg-white/95 backdrop-blur rounded-lg border border-border shadow-md max-w-xs text-xs pointer-events-none">
            <div className="flex items-center space-x-1.5 mb-1">
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: getNodeColor(hoveredNode.type) }}
              />
              <span className="font-bold text-stone-900 font-mono">{hoveredNode.id}</span>
              <span className="text-[10px] uppercase font-mono px-1 rounded bg-stone-100 text-stone-600">
                {hoveredNode.type}
              </span>
            </div>
            <p className="text-stone-700 font-medium mb-1">{hoveredNode.label}</p>
            {hoveredNode.details && (
              <div className="space-y-0.5 text-[11px] text-stone-500 font-mono">
                {Object.entries(hoveredNode.details)
                  .slice(0, 3)
                  .map(([k, v]) => (
                    <div key={k} className="flex justify-between">
                      <span>{k}:</span>
                      <span className="text-stone-800 font-semibold">{String(v)}</span>
                    </div>
                  ))}
              </div>
            )}
          </div>
        )}

        {/* Entity Inspector Side Card (When clicked) */}
        {selectedNode && (
          <div className="absolute top-16 right-3 z-20 w-80 p-4 bg-white/95 backdrop-blur rounded-xl border border-border shadow-lg text-xs space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-border">
              <div className="flex items-center space-x-2">
                <span
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: getNodeColor(selectedNode.type) }}
                />
                <span className="font-bold font-mono text-stone-900">{selectedNode.id}</span>
              </div>
              <button
                type="button"
                onClick={() => setSelectedNode(null)}
                className="p-1 rounded hover:bg-stone-100 text-stone-400 hover:text-stone-800 cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>

            <div>
              <span className="text-[10px] text-stone-400 uppercase font-semibold">Entity Type</span>
              <p className="font-semibold text-stone-800 capitalize">{selectedNode.type.replace('_', ' ')}</p>
            </div>

            {selectedNode.type === 'breach_campaign' && (
              <div className="p-2.5 rounded-lg bg-amber-50/80 border border-amber-200 text-xs text-amber-900 space-y-1">
                <div className="flex items-center space-x-1.5 font-semibold text-amber-950">
                  <Tag className="w-3.5 h-3.5 text-amber-700" />
                  <span>Forensic Breach Rollup</span>
                </div>
                <p className="text-[11px] leading-relaxed text-amber-800">
                  Aggregates {selectedNode.card_count || 'remainder'} cards sharing vector and botnet {selectedNode.botnet_id || ''} to enforce topological readability and prevent star-graph dandelion collapse (degree ratio &le; 0.35).
                </p>
              </div>
            )}

            {selectedNode.details && (
              <div className="space-y-1.5 bg-stone-50 p-2.5 rounded-lg border border-border font-mono text-[11px]">
                {Object.entries(selectedNode.details).map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span className="text-stone-500">{k}:</span>
                    <span className="text-stone-900 font-medium">{String(v)}</span>
                  </div>
                ))}
              </div>
            )}

            {selectedNode.type === 'breach_campaign' && selectedNode.constituent_cards && selectedNode.constituent_cards.length > 0 && (
              <div className="space-y-1.5 pt-1">
                <div className="flex items-center justify-between text-[11px] font-semibold text-stone-700">
                  <span>Constituent Cards ({selectedNode.constituent_cards.length})</span>
                  <span className="text-[10px] text-stone-400 font-mono">Click card to inspect</span>
                </div>
                <div className="max-h-44 overflow-y-auto space-y-1.5 pr-1 border border-border rounded-lg p-1.5 bg-stone-50/60 no-scrollbar">
                  {selectedNode.constituent_cards.map((c) => (
                    <div
                      key={c.id}
                      onClick={() => {
                        setSelectedNode({
                          id: c.id,
                          label: c.label,
                          type: 'card',
                          details: {
                            'Cardholder ID': c.id,
                            'Batch Rollup Parent': selectedNode.id,
                            'Breach Vector': selectedNode.details?.['Primary Vector'] || 'Botnet Compromise',
                            'Controlling Botnet': selectedNode.botnet_id || 'N/A',
                            'Affiliated Syndicate': selectedNode.syndicate_id || 'N/A',
                            'Tx Attempts': c.tx_count,
                            'Fraud Volume': `$${c.volume.toFixed(2)}`,
                            'Approval Rate': `${c.approval_rate}%`,
                            'Targeted Merchants': c.merchants.join(', ') || 'N/A',
                          },
                        });
                      }}
                      className="p-1.5 bg-white rounded border border-stone-200 text-[11px] font-mono hover:border-amber-400 hover:bg-amber-50/50 cursor-pointer transition-colors"
                    >
                      <div className="flex justify-between items-center font-bold text-stone-800">
                        <span>{c.label}</span>
                        <span className="text-rose-700 font-semibold">${c.volume.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between text-[10px] text-stone-500">
                        <span>{c.tx_count} txs</span>
                        <span>{c.approval_rate}% approved</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Active Real-Time Synthesis Ingestion State */}
        {isGenerating && (!bundle || !bundle.threat_graph || bundle.threat_graph.nodes.length === 0) ? (
          <div className="flex-1 flex flex-col items-center justify-center p-8 text-center space-y-4">
            <div className="relative flex items-center justify-center w-24 h-24">
              <div className="absolute inset-0 rounded-full border-2 border-copper-300 border-t-copper-600 animate-spin" />
              <div className="w-16 h-16 rounded-full bg-copper-50 flex items-center justify-center text-copper-700 shadow-inner">
                <RefreshCw className="w-7 h-7 animate-spin text-copper-600" />
              </div>
            </div>
            <div className="space-y-1.5 max-w-md">
              <div className="flex items-center justify-center space-x-2 text-sm font-semibold text-stone-900">
                <span>Synthesizing Payment Telemetry</span>
                <span className="font-mono text-copper-600 font-bold">({generationProgress?.pct || 0}%)</span>
              </div>
              <p className="text-xs text-stone-500 leading-relaxed">
                Previous graph cleared. Sampling continuous Hawkes MTPP arrivals, generating adversary decisions, and discovering criminal threat topology in real time.
              </p>
            </div>
            <div className="w-72 bg-stone-100 rounded-full h-2 overflow-hidden border border-border">
              <div
                className="bg-copper-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${generationProgress?.pct || 0}%` }}
              />
            </div>
            <div className="flex items-center space-x-3 text-[11px] font-mono text-stone-500 bg-stone-50 px-3 py-1 rounded-full border border-stone-200">
              <span className="font-semibold text-stone-800">{generationProgress?.current || 0} / {generationProgress?.total || 0} txs</span>
              <span>•</span>
              <span className="text-copper-700 font-bold">{generationProgress?.tps || 0} tx/s</span>
            </div>
          </div>
        ) : loading ? (
          <div className="flex-1 flex items-center justify-center text-xs text-stone-400">
            <span>Computing forensic entity topology...</span>
          </div>
        ) : !bundle || !bundle.threat_graph || bundle.threat_graph.nodes.length === 0 ? (
          <div className="flex-1 flex items-center justify-center text-xs text-stone-400">
            <span>No transaction network available. Generate a simulation batch to explore.</span>
          </div>
        ) : (
          <svg ref={svgRef} className="w-full h-full cursor-grab active:cursor-grabbing" />
        )}
      </div>
    </div>
  );
};
