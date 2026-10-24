import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react';
import {
  SimulationMetadata,
  VisualizerBundle,
  LiveTransactionEvent,
  StreamingStats,
  GenerationProgress,
} from '../types';

interface SimulationContextType {
  connected: boolean;
  isGenerating: boolean;
  generationProgress: GenerationProgress | null;
  isStreaming: boolean;
  streamingStats: StreamingStats | null;
  latestTx: LiveTransactionEvent | null;
  recentLiveTxs: LiveTransactionEvent[];
  metadata: SimulationMetadata | undefined;
  bundle: VisualizerBundle | null;
  bundleLoading: boolean;
  startGeneration: (params: {
    region: string;
    n_transactions?: number;
    fraud_prevalence: number;
    adversary_mode: string;
    seed: number;
    simulation_mode?: 'transactions' | 'days';
    time_span_days?: number;
  }) => Promise<void>;
  startLiveStream: (params: {
    region: string;
    target_tps: number;
    fraud_prevalence: number;
    adversary_mode: string;
    seed: number;
  }) => void;
  stopLiveStream: () => void;
  refreshBundle: () => void;
}

const SimulationContext = createContext<SimulationContextType | undefined>(undefined);

export const SimulationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [connected, setConnected] = useState<boolean>(false);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [generationProgress, setGenerationProgress] = useState<GenerationProgress | null>(null);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [streamingStats, setStreamingStats] = useState<StreamingStats | null>(null);
  const [latestTx, setLatestTx] = useState<LiveTransactionEvent | null>(null);
  const [recentLiveTxs, setRecentLiveTxs] = useState<LiveTransactionEvent[]>([]);
  const [metadata, setMetadata] = useState<SimulationMetadata | undefined>(undefined);
  const [bundle, setBundle] = useState<VisualizerBundle | null>(null);
  const [bundleLoading, setBundleLoading] = useState<boolean>(false);

  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const pingIntervalRef = useRef<number | null>(null);

  const fetchBundle = useCallback(() => {
    setBundleLoading(true);
    fetch('/api/visualizer/bundle')
      .then((res) => res.json())
      .then((data) => {
        setBundle(data);
        setBundleLoading(false);
      })
      .catch(() => setBundleLoading(false));
  }, []);

  const connectWebSocket = useCallback(() => {
    if (socketRef.current && (socketRef.current.readyState === WebSocket.OPEN || socketRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/simulation`;
    const ws = new WebSocket(wsUrl);
    socketRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      // Heartbeat ping
      pingIntervalRef.current = window.setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ action: 'ping' }));
        }
      }, 15000);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        switch (data.type) {
          case 'init':
            if (data.metadata && data.metadata.n_transactions > 0) {
              setMetadata(data.metadata);
            }
            setIsGenerating(Boolean(data.is_generating));
            setIsStreaming(Boolean(data.is_streaming));
            if (data.progress) setGenerationProgress(data.progress);
            break;

          case 'progress':
            setIsGenerating(true);
            if (data.progress) setGenerationProgress(data.progress);
            if (data.sample_tx) {
              setLatestTx(data.sample_tx);
              setRecentLiveTxs((prev) => [data.sample_tx, ...prev.slice(0, 24)]);
            }
            if (data.threat_graph) {
              setBundle((prev) => ({
                metadata: prev?.metadata || {
                  region: 'IN',
                  seed: 42,
                  total_transactions: data.progress?.current || 0,
                  fraud_count: data.threat_graph.nodes.filter((n: any) => n.type === 'card' || n.type === 'bridge_card').length,
                  fraud_rate_pct: 5.0,
                  legitimate_count: data.progress?.current || 0,
                  hard_negative_count: 0,
                },
                threat_graph: data.threat_graph,
                switch_funnel: prev?.switch_funnel || {
                  total_ingress: data.progress?.current || 0,
                  approved: data.progress?.current || 0,
                  declined: 0,
                  approval_rate: 100,
                  hops: [],
                  iso_distribution: [],
                },
                temporal_series: prev?.temporal_series || [],
              }));
            }
            break;

          case 'live_tx':
            if (data.tx) {
              setLatestTx(data.tx);
              setRecentLiveTxs((prev) => [data.tx, ...prev.slice(0, 24)]);
            }
            if (data.stats) setStreamingStats(data.stats);
            break;

          case 'stream_started':
            setIsStreaming(true);
            break;

          case 'stream_stopped':
            setIsStreaming(false);
            break;

          case 'complete':
            setIsGenerating(false);
            setGenerationProgress(null);
            if (data.metadata) setMetadata(data.metadata);
            if (data.bundle) {
              setBundle(data.bundle);
            } else {
              fetchBundle();
            }
            break;

          case 'error':
            setIsGenerating(false);
            setGenerationProgress(null);
            break;

          default:
            break;
        }
      } catch (err) {
        console.error('WebSocket parse error:', err);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      socketRef.current = null;
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
        pingIntervalRef.current = null;
      }
      // Reconnect after 2 seconds
      reconnectTimeoutRef.current = window.setTimeout(() => {
        connectWebSocket();
      }, 2000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [fetchBundle]);

  useEffect(() => {
    connectWebSocket();
    fetchBundle();

    // Initial metadata fetch
    fetch('/api/metadata')
      .then((res) => res.json())
      .then((data) => {
        if (data && data.n_transactions > 0) {
          setMetadata(data);
        }
      })
      .catch(() => {});

    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
      if (socketRef.current) socketRef.current.close();
    };
  }, [connectWebSocket, fetchBundle]);

  const startGeneration = async (params: {
    region: string;
    n_transactions?: number;
    fraud_prevalence: number;
    adversary_mode: string;
    seed: number;
    simulation_mode?: 'transactions' | 'days';
    time_span_days?: number;
  }) => {
    // 1. Immediately clear stale graph from previous batch!
    setBundle(null);
    setLatestTx(null);
    setRecentLiveTxs([]);
    setIsGenerating(true);
    const mode = params.simulation_mode || 'transactions';
    const estTotal = mode === 'days'
      ? Math.round((params.time_span_days || 14) * Math.max(100, Math.min(2000, (params.time_span_days || 14) * 35)) * 2.13)
      : (params.n_transactions || 1000);

    setGenerationProgress({
      current: 0,
      total: estTotal,
      pct: 0,
      tps: 0,
      status: 'simulating',
      simulation_mode: mode,
      time_span_days: params.time_span_days,
    });

    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Failed to synthesize simulation batch');
      }
      setMetadata(data.metadata);
      fetchBundle();
    } catch (err: any) {
      setIsGenerating(false);
      setGenerationProgress(null);
      throw err;
    }
  };

  const startLiveStream = (params: {
    region: string;
    target_tps: number;
    fraud_prevalence: number;
    adversary_mode: string;
    seed: number;
  }) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(
        JSON.stringify({
          action: 'start_stream',
          ...params,
        })
      );
    } else {
      fetch('/api/stream/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      }).catch(console.error);
    }
    setIsStreaming(true);
  };

  const stopLiveStream = () => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ action: 'stop_stream' }));
    } else {
      fetch('/api/stream/stop', { method: 'POST' }).catch(console.error);
    }
    setIsStreaming(false);
  };

  return (
    <SimulationContext.Provider
      value={{
        connected,
        isGenerating,
        generationProgress,
        isStreaming,
        streamingStats,
        latestTx,
        recentLiveTxs,
        metadata,
        bundle,
        bundleLoading,
        startGeneration,
        startLiveStream,
        stopLiveStream,
        refreshBundle: fetchBundle,
      }}
    >
      {children}
    </SimulationContext.Provider>
  );
};

export const useSimulation = (): SimulationContextType => {
  const context = useContext(SimulationContext);
  if (!context) {
    throw new Error('useSimulation must be used within a SimulationProvider');
  }
  return context;
};
