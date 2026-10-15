import React, { useState, useEffect } from 'react';
import { Bot, Send, Cpu } from 'lucide-react';
import api from '../services/api';

export default function Chatbot() {
  const [activeModel, setActiveModel] = useState(null);
  const [loadingModel, setLoadingModel] = useState(true);
  const [messages] = useState([]); // Real state, no sample data

  useEffect(() => {
    const checkModel = async () => {
      try {
        const res = await api.get('/api/models/active');
        setActiveModel(res.data);
      } catch (err) {
        setActiveModel(null);
      } finally {
        setLoadingModel(false);
      }
    };

    checkModel();
  }, []);

  return (
    <div className="space-y-6 flex flex-col h-[calc(100vh-8rem)]">
      {/* Header */}
      <div className="border-b border-slate-800 pb-4 flex-shrink-0">
        <h1 className="text-2xl font-bold text-slate-100">Agronomic Chatbot Assistant</h1>
        <p className="text-xs text-slate-400 mt-1">
          Consult agricultural advisory models for crop protection, fertilizer, and disease guidance
        </p>
      </div>

      {/* Model availability check */}
      {loadingModel ? (
        <div className="p-4 text-center text-xs text-slate-500">Checking model registry...</div>
      ) : !activeModel ? (
        <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center space-x-3 text-amber-400 text-xs flex-shrink-0">
          <Cpu className="w-5 h-5 flex-shrink-0" />
          <div>
            <span className="font-bold">Model not available: </span>
            <span>No conversational agronomy model is currently registered in the backend model registry.</span>
          </div>
        </div>
      ) : null}

      {/* Chat messages container */}
      <div className="flex-1 overflow-y-auto bg-slate-900/40 border border-slate-800 rounded-2xl p-6 flex flex-col justify-center items-center text-center">
        <div className="max-w-sm space-y-2">
          <div className="w-12 h-12 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-400 mx-auto">
            <Bot className="w-6 h-6 text-emerald-400" />
          </div>
          <h3 className="font-semibold text-slate-200 text-sm">No conversations yet</h3>
          <p className="text-xs text-slate-400">
            {activeModel
              ? 'Ask questions regarding crop pathology, pest remedies, or sowing schedules.'
              : 'Chat is disabled until an active advisory model is registered in Phase 4.'}
          </p>
        </div>
      </div>

      {/* Input box */}
      <div className="flex items-center space-x-2 flex-shrink-0">
        <input
          type="text"
          disabled={!activeModel}
          placeholder={activeModel ? 'Ask agronomy questions...' : 'Model not available — input disabled'}
          className="flex-1 px-4 py-3 rounded-xl bg-slate-900 border border-slate-800 text-slate-100 placeholder-slate-500 text-xs focus:outline-none focus:border-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed"
        />
        <button
          disabled={!activeModel}
          className="p-3 rounded-xl bg-emerald-600 text-white disabled:opacity-40 disabled:cursor-not-allowed hover:bg-emerald-500 transition-colors"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
