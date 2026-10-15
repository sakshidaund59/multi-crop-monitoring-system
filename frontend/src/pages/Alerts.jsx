import React, { useState } from 'react';
import { Bell } from 'lucide-react';
import EmptyState from '../components/EmptyState';

export default function Alerts() {
  const [alerts] = useState([]); // Real state, no sample data

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-slate-800 pb-6">
        <h1 className="text-2xl font-bold text-slate-100">Agricultural Risk Alerts</h1>
        <p className="text-xs text-slate-400 mt-1">
          Real-time notices on crop disease outbreaks, pest threshold alerts, and frost/heat risks
        </p>
      </div>

      {/* Empty State */}
      {alerts.length === 0 ? (
        <EmptyState
          title="No alerts found"
          description="There are currently no active warnings or critical risk thresholds triggered across your farm plots."
          icon={Bell}
        />
      ) : (
        <div className="space-y-3">
          {alerts.map((alert) => (
            <div key={alert.id} className="p-4 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-sm font-semibold">{alert.title}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
