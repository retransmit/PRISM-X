import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { StudioView } from './components/StudioView';
import { BenchmarkTab } from './components/BenchmarkTab';
import { SimulationMetadata } from './types';
import { SimulationProvider } from './context/SimulationContext';

export const App: React.FC = () => {
  const getTabFromHash = (): 'studio' | 'benchmark' => {
    const hash = window.location.hash.replace('#', '').toLowerCase();
    if (hash.startsWith('benchmark')) return 'benchmark';
    return 'studio';
  };

  const [activeTab, setActiveTabState] = useState<'studio' | 'benchmark'>(getTabFromHash);
  const [metadata, setMetadata] = useState<SimulationMetadata | undefined>(undefined);

  const setActiveTab = (tab: 'studio' | 'benchmark') => {
    setActiveTabState(tab);
    window.location.hash = tab;
  };

  useEffect(() => {
    const handleHashChange = () => {
      setActiveTabState(getTabFromHash());
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const handleMetadataUpdated = (newMeta: SimulationMetadata) => {
    setMetadata(newMeta);
  };

  return (
    <SimulationProvider>
      <div className="min-h-screen bg-background text-stone-900 flex flex-col font-sans">
        <Navbar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          metadata={metadata}
        />

        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className={activeTab === 'studio' ? 'block' : 'hidden'}>
            <StudioView
              metadata={metadata}
              onMetadataUpdated={handleMetadataUpdated}
            />
          </div>
          <div className={activeTab === 'benchmark' ? 'block' : 'hidden'}>
            <BenchmarkTab />
          </div>
        </main>

        <footer className="border-t border-border bg-surface py-4 text-center text-xs text-stone-500">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2">
            <span className="text-stone-700 font-medium">
              Prism Generator Studio: Synthetic Payment Rail Telemetry & Causal Benchmark
            </span>
            <span className="font-mono text-stone-500 text-[11px]">
              Atharva College of Engineering, Dept of IT
            </span>
          </div>
        </footer>
      </div>
    </SimulationProvider>
  );
};

export default App;
