import React, { useState, useEffect } from 'react';
import { Upload, Scan, AlertTriangle, Cpu } from 'lucide-react';
import api from '../services/api';

export default function Analyze() {
  const [activeModel, setActiveModel] = useState(null);
  const [loadingModel, setLoadingModel] = useState(true);

  useEffect(() => {
    const checkModelStatus = async () => {
      try {
        const res = await api.get('/api/models/active');
        setActiveModel(res.data);
      } catch (err) {
        setActiveModel(null);
      } finally {
        setLoadingModel(false);
      }
    };

    checkModelStatus();
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-slate-800 pb-6">
        <h1 className="text-2xl font-bold text-slate-100">Crop Health Analysis</h1>
        <p className="text-xs text-slate-400 mt-1">
          Perform computer vision diagnostics on leaf symptoms and pest infestations
        </p>
      </div>

      {/* Model Availability Check Banner */}
      {loadingModel ? (
        <div className="p-8 text-center text-slate-500 text-sm">Checking machine learning model registry...</div>
      ) : !activeModel ? (
        <div className="p-8 rounded-2xl bg-amber-500/10 border border-amber-500/20 text-center space-y-4">
          <div className="w-14 h-14 rounded-full bg-amber-500/20 border border-amber-500/30 flex items-center justify-center text-amber-400 mx-auto">
            <Cpu className="w-7 h-7" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-100">Model not available</h2>
            <p className="text-xs text-slate-400 max-w-md mx-auto mt-1.5 leading-relaxed">
              No machine learning model is currently active in the Model Registry.
              Deep learning models (PyTorch / ONNX) and trained weights will be linked and activated in Phase 4.
            </p>
          </div>
          <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-full bg-amber-500/10 border border-amber-500/30 text-[11px] text-amber-400 font-mono">
            <span>model_registry: 0 models active</span>
          </div>
        </div>
      ) : (
        <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
          <div className="flex items-center space-x-3 text-emerald-400">
            <Scan className="w-5 h-5" />
            <span className="text-sm font-semibold text-slate-200">Active Model: {activeModel.name} ({activeModel.version})</span>
          </div>
          <p className="text-xs text-slate-400">Provenance: {activeModel.weights_hash.substring(0, 16)}...</p>
        </div>
      )}

      {/* Upload Zone (Disabled when model is not available) */}
      <div className="p-10 rounded-2xl bg-slate-900/40 border-2 border-dashed border-slate-800 text-center space-y-3 opacity-60 cursor-not-allowed">
        <div className="w-12 h-12 rounded-full bg-slate-800 flex items-center justify-center text-slate-500 mx-auto">
          <Upload className="w-6 h-6" />
        </div>
        <div>
          <p className="text-sm font-medium text-slate-300">Image upload disabled</p>
          <p className="text-xs text-slate-500 mt-1">Inference gateway requires an active registered model</p>
        </div>
      </div>
    </div>
  );
}
