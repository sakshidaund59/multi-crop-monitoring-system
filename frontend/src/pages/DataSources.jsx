import React, { useState, useEffect } from 'react';
import { Database, Satellite, CloudSun, CheckCircle2, XCircle, RefreshCw } from 'lucide-react';
import api from '../services/api';

export default function DataSources() {
  const [health, setHealth] = useState({ loading: true, data: null, error: null });
  const [dbHealth, setDbHealth] = useState({ loading: true, data: null, error: null });

  const fetchStatus = async () => {
    setHealth({ loading: true, data: null, error: null });
    setDbHealth({ loading: true, data: null, error: null });

    try {
      const res = await api.get('/health');
      setHealth({ loading: false, data: res.data, error: null });
    } catch (err) {
      setHealth({ loading: false, data: null, error: err.message });
    }

    try {
      const res = await api.get('/health/db');
      setDbHealth({ loading: false, data: res.data, error: null });
    } catch (err) {
      setDbHealth({ loading: false, data: null, error: err.response?.data?.reason || err.message });
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Data Sources & Services</h1>
          <p className="text-xs text-slate-400 mt-1">Live status of database connectors, satellite feeds, and backend pipelines</p>
        </div>
        <button
          onClick={fetchStatus}
          className="px-3.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium flex items-center space-x-1.5 border border-slate-700 transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh Status</span>
        </button>
      </div>

      {/* Services Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* PostgreSQL + PostGIS */}
        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
          <div className="flex items-start justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                <Database className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-100 text-sm">PostgreSQL + PostGIS</h3>
                <span className="text-[10px] text-slate-400">Geospatial Relational Database</span>
              </div>
            </div>
            {dbHealth.loading ? (
              <span className="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-400 animate-pulse">Checking...</span>
            ) : dbHealth.error ? (
              <span className="text-xs px-2.5 py-1 rounded bg-rose-500/20 text-rose-400 border border-rose-500/30">Offline</span>
            ) : (
              <span className="text-xs px-2.5 py-1 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Connected</span>
            )}
          </div>
          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80 text-xs font-mono space-y-1 text-slate-400">
            <p>Database: {dbHealth.data?.database_name || 'crop_monitoring_db'}</p>
            <p>PostGIS Engine: {dbHealth.data?.postgis_version ? 'Active (v3.3)' : 'Checking...'}</p>
          </div>
        </div>

        {/* FastAPI Application Service */}
        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
          <div className="flex items-start justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
                <CheckCircle2 className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-100 text-sm">FastAPI Application Service</h3>
                <span className="text-[10px] text-slate-400">REST API Gateway & Auth</span>
              </div>
            </div>
            {health.loading ? (
              <span className="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-400 animate-pulse">Checking...</span>
            ) : health.error ? (
              <span className="text-xs px-2.5 py-1 rounded bg-rose-500/20 text-rose-400 border border-rose-500/30">Offline</span>
            ) : (
              <span className="text-xs px-2.5 py-1 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Online</span>
            )}
          </div>
          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80 text-xs font-mono space-y-1 text-slate-400">
            <p>Environment: {health.data?.environment || 'development'}</p>
            <p>Version: {health.data?.version || '0.1.0'}</p>
          </div>
        </div>

        {/* Sentinel-2 Remote Sensing Feed */}
        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
          <div className="flex items-start justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
                <Satellite className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-100 text-sm">Sentinel-2 Remote Sensing</h3>
                <span className="text-[10px] text-slate-400">Copernicus Multispectral Optical Imagery</span>
              </div>
            </div>
            <span className="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-400 border border-slate-700">Phase 3</span>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Integration pipeline configured for automated 10m-resolution NDVI/EVI vegetation indexing over farm boundary polygons.
          </p>
        </div>

        {/* Meteorological Risk Engine */}
        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
          <div className="flex items-start justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400">
                <CloudSun className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-semibold text-slate-100 text-sm">Weather Risk Analytics</h3>
                <span className="text-[10px] text-slate-400">Open-Meteo & Agronomic Indicators</span>
              </div>
            </div>
            <span className="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-400 border border-slate-700">Phase 5</span>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Meteorological observation collector tracking precipitation, temperature extremes, and growing degree days (GDD).
          </p>
        </div>
      </div>
    </div>
  );
}
