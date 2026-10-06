import React, { useState, useEffect } from 'react';

export default function App() {
  const [health, setHealth] = useState({ loading: true, data: null, error: null });
  const [dbHealth, setDbHealth] = useState({ loading: true, data: null, error: null });

  const fetchHealth = async () => {
    try {
      const res = await fetch('/health');
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      setHealth({ loading: false, data, error: null });
    } catch (err) {
      setHealth({ loading: false, data: null, error: err.message });
    }

    try {
      const res = await fetch('/health/db');
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      setDbHealth({ loading: false, data, error: null });
    } catch (err) {
      setDbHealth({ loading: false, data: null, error: err.message });
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-6 font-sans">
      <div className="max-w-xl w-full bg-slate-900 border border-slate-800 rounded-xl p-8 shadow-2xl space-y-6">
        <div className="flex items-center space-x-3 border-b border-slate-800 pb-4">
          <div className="w-10 h-10 rounded-lg bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center text-emerald-400 font-bold text-xl">
            🌱
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-100">Multi-Crop Monitoring System</h1>
            <p className="text-xs text-slate-400">Step 0: Initial Monorepo & Service Architecture</p>
          </div>
        </div>

        <div className="space-y-4">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Backend Connectivity Check</h2>
          
          {/* Main API Health Status */}
          <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-between">
            <div>
              <span className="text-xs text-slate-400 block font-mono">GET /health</span>
              <span className="text-sm font-medium text-slate-200">API Service</span>
            </div>
            {health.loading ? (
              <span className="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-400 animate-pulse">Checking...</span>
            ) : health.error ? (
              <span className="text-xs px-2.5 py-1 rounded bg-rose-500/20 text-rose-400 border border-rose-500/30">Offline</span>
            ) : (
              <span className="text-xs px-2.5 py-1 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                {health.data?.status === 'ok' ? 'Online' : 'Degraded'}
              </span>
            )}
          </div>

          {/* DB & PostGIS Health Status */}
          <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-between">
            <div>
              <span className="text-xs text-slate-400 block font-mono">GET /health/db</span>
              <span className="text-sm font-medium text-slate-200">PostgreSQL + PostGIS</span>
            </div>
            {dbHealth.loading ? (
              <span className="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-400 animate-pulse">Checking...</span>
            ) : dbHealth.error ? (
              <span className="text-xs px-2.5 py-1 rounded bg-rose-500/20 text-rose-400 border border-rose-500/30">Offline</span>
            ) : (
              <span className="text-xs px-2.5 py-1 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                {dbHealth.data?.status === 'ok' ? 'Connected' : 'Error'}
              </span>
            )}
          </div>
        </div>

        <div className="pt-2 flex justify-between items-center text-xs text-slate-500 border-t border-slate-800">
          <span>Phase: Step 0 (Project Setup)</span>
          <button 
            onClick={fetchHealth} 
            className="px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
          >
            Recheck API
          </button>
        </div>
      </div>
    </div>
  );
}
