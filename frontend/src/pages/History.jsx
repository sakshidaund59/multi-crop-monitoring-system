import React, { useState } from 'react';
import { Clock } from 'lucide-react';
import EmptyState from '../components/EmptyState';

export default function History() {
  const [analyses] = useState([]); // Real state, no sample data

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-slate-800 pb-6">
        <h1 className="text-2xl font-bold text-slate-100">Analysis History</h1>
        <p className="text-xs text-slate-400 mt-1">
          Historical record of leaf diagnostic scans and pest detection results
        </p>
      </div>

      {/* Empty State */}
      {analyses.length === 0 ? (
        <EmptyState
          title="No analyses yet"
          description="You haven't conducted any crop scans yet. Scanned analyses and disease diagnostic reports will appear here."
          icon={Clock}
        />
      ) : (
        <div className="space-y-3">
          {analyses.map((item) => (
            <div key={item.id} className="p-4 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-sm font-medium">{item.analysis_type}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
