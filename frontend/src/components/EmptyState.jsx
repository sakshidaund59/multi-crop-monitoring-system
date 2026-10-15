import React from 'react';
import { AlertCircle } from 'lucide-react';

export default function EmptyState({
  title = "No data found",
  description = "There are no records to display yet.",
  icon: Icon = AlertCircle,
  action,
}) {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center bg-slate-900/50 border border-slate-800 rounded-xl my-6">
      <div className="w-14 h-14 rounded-full bg-slate-800 flex items-center justify-center text-slate-400 mb-4 border border-slate-700">
        <Icon className="w-7 h-7 text-emerald-400" />
      </div>
      <h3 className="text-lg font-semibold text-slate-100 mb-1">{title}</h3>
      <p className="text-sm text-slate-400 max-w-sm mb-6">{description}</p>
      {action && <div>{action}</div>}
    </div>
  );
}
