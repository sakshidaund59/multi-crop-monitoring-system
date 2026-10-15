import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { MapPin, Scan, Bell, CheckCircle2, ArrowRight } from 'lucide-react';
import api from '../services/api';
import { useAuth } from '../context/AuthContext';
import EmptyState from '../components/EmptyState';

export default function Dashboard() {
  const { user } = useAuth();
  const [farms, setFarms] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        const res = await api.get('/api/farms');
        setFarms(res.data);
      } catch (err) {
        console.error("Failed to load dashboard farms:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  return (
    <div className="space-y-8">
      {/* Welcome Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Farm Overview</h1>
          <p className="text-xs text-slate-400 mt-1">
            Welcome back, <span className="text-emerald-400 font-medium">{user?.full_name}</span> ({user?.role})
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <Link
            to="/farms"
            className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold flex items-center space-x-1.5 transition-colors shadow-lg shadow-emerald-950"
          >
            <MapPin className="w-3.5 h-3.5" />
            <span>Manage Farms</span>
          </Link>
          <Link
            to="/analyze"
            className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium flex items-center space-x-1.5 border border-slate-700 transition-colors"
          >
            <Scan className="w-3.5 h-3.5" />
            <span>Scan Crop</span>
          </Link>
        </div>
      </div>

      {/* KPI Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 flex items-center space-x-4">
          <div className="w-12 h-12 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
            <MapPin className="w-6 h-6" />
          </div>
          <div>
            <span className="text-xs text-slate-400 block font-medium">Registered Farms</span>
            <span className="text-2xl font-bold text-slate-100">{loading ? '...' : farms.length}</span>
          </div>
        </div>

        <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 flex items-center space-x-4">
          <div className="w-12 h-12 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
            <Scan className="w-6 h-6" />
          </div>
          <div>
            <span className="text-xs text-slate-400 block font-medium">Completed Scans</span>
            <span className="text-2xl font-bold text-slate-100">0</span>
          </div>
        </div>

        <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 flex items-center space-x-4">
          <div className="w-12 h-12 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400">
            <Bell className="w-6 h-6" />
          </div>
          <div>
            <span className="text-xs text-slate-400 block font-medium">Active Alerts</span>
            <span className="text-2xl font-bold text-slate-100">0</span>
          </div>
        </div>

        <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 flex items-center space-x-4">
          <div className="w-12 h-12 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
            <CheckCircle2 className="w-6 h-6" />
          </div>
          <div>
            <span className="text-xs text-slate-400 block font-medium">Spatial Engine</span>
            <span className="text-sm font-semibold text-emerald-400">PostGIS Active</span>
          </div>
        </div>
      </div>

      {/* Farm Activity / Plots Section */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-slate-100">Your Agricultural Plots</h2>
          {farms.length > 0 && (
            <Link to="/farms" className="text-xs text-emerald-400 hover:underline flex items-center space-x-1">
              <span>View all</span>
              <ArrowRight className="w-3 h-3" />
            </Link>
          )}
        </div>

        {loading ? (
          <div className="p-8 text-center text-slate-500 text-sm">Loading farm plots...</div>
        ) : farms.length === 0 ? (
          <EmptyState
            title="No farms registered yet"
            description="Register your agricultural plots with GPS coordinates and polygon boundaries to begin monitoring crop health and disease risks."
            icon={MapPin}
            action={
              <Link
                to="/farms"
                className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold inline-flex items-center space-x-1.5 transition-colors"
              >
                <span>Add Your First Farm</span>
              </Link>
            }
          />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {farms.map((farm) => (
              <div key={farm.id} className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
                <div className="flex items-start justify-between">
                  <h3 className="font-semibold text-slate-100 text-sm">{farm.name}</h3>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    {farm.area_ha ? `${farm.area_ha} ha` : 'Active'}
                  </span>
                </div>
                <div className="text-xs text-slate-400 space-y-1 font-mono">
                  <p>GPS: {farm.latitude?.toFixed(4)}, {farm.longitude?.toFixed(4)}</p>
                  <p>Boundary: {farm.boundary ? 'GeoJSON Polygon Defined' : 'Point Location'}</p>
                </div>
                <div className="text-[10px] text-slate-500 pt-2 border-t border-slate-800">
                  Added on {new Date(farm.created_at).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
