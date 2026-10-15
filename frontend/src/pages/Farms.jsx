import React, { useState, useEffect } from 'react';
import { Plus, Trash2, MapPin, AlertCircle, X } from 'lucide-react';
import api from '../services/api';
import EmptyState from '../components/EmptyState';

export default function Farms() {
  const [farms, setFarms] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [error, setError] = useState(null);

  // Form state
  const [name, setName] = useState('');
  const [latitude, setLatitude] = useState('');
  const [longitude, setLongitude] = useState('');
  const [areaHa, setAreaHa] = useState('');
  const [boundaryJson, setBoundaryJson] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const fetchFarms = async () => {
    try {
      const res = await api.get('/api/farms');
      setFarms(res.data);
    } catch (err) {
      setError(err.response?.data?.reason || err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFarms();
  }, []);

  const handleCreateFarm = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);

    let parsedBoundary = null;
    if (boundaryJson.trim()) {
      try {
        parsedBoundary = JSON.parse(boundaryJson);
      } catch (err) {
        setError('Invalid GeoJSON Polygon format in boundary field.');
        setSubmitting(false);
        return;
      }
    }

    try {
      await api.post('/api/farms', {
        name,
        latitude: parseFloat(latitude),
        longitude: parseFloat(longitude),
        area_ha: areaHa ? parseFloat(areaHa) : null,
        boundary: parsedBoundary,
      });

      setName('');
      setLatitude('');
      setLongitude('');
      setAreaHa('');
      setBoundaryJson('');
      setShowModal(false);
      fetchFarms();
    } catch (err) {
      setError(err.response?.data?.reason || err.response?.data?.detail || err.message);
    } finally {
      setSubmitting(false);
    }
  };

  const handleDeleteFarm = async (id) => {
    if (!window.confirm('Are you sure you want to delete this farm plot?')) return;
    try {
      await api.delete(`/api/farms/${id}`);
      setFarms((prev) => prev.filter((f) => f.id !== id));
    } catch (err) {
      alert(err.response?.data?.reason || err.message);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Agricultural Farms</h1>
          <p className="text-xs text-slate-400 mt-1">Manage field boundaries and GPS center locations</p>
        </div>
        <button
          onClick={() => {
            setError(null);
            setShowModal(true);
          }}
          className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold flex items-center space-x-1.5 transition-colors shadow-lg shadow-emerald-950"
        >
          <Plus className="w-4 h-4" />
          <span>Add New Farm</span>
        </button>
      </div>

      {error && (
        <div className="p-3.5 rounded-lg bg-rose-500/10 border border-rose-500/20 flex items-center space-x-3 text-rose-400 text-xs">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Farms List */}
      {loading ? (
        <div className="p-12 text-center text-slate-500 text-sm">Loading farms...</div>
      ) : farms.length === 0 ? (
        <EmptyState
          title="No farms registered yet"
          description="You have not created any farm plots yet. Click 'Add New Farm' to register a field with GPS coordinates and polygon boundaries."
          icon={MapPin}
          action={
            <button
              onClick={() => setShowModal(true)}
              className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold inline-flex items-center space-x-1.5 transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span>Register Farm</span>
            </button>
          }
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {farms.map((farm) => (
            <div key={farm.id} className="p-5 rounded-xl bg-slate-900 border border-slate-800 flex flex-col justify-between space-y-4 hover:border-slate-700 transition-colors">
              <div className="space-y-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                      <MapPin className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-slate-100 text-sm">{farm.name}</h3>
                      <span className="text-[10px] text-slate-400 font-mono">
                        {farm.latitude?.toFixed(4)}, {farm.longitude?.toFixed(4)}
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={() => handleDeleteFarm(farm.id)}
                    className="p-1.5 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
                    title="Delete farm"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs pt-2 border-t border-slate-800/80">
                  <div className="p-2 rounded bg-slate-950/60 border border-slate-800/40">
                    <span className="text-[10px] text-slate-500 block uppercase">Area</span>
                    <span className="font-medium text-slate-200">{farm.area_ha ? `${farm.area_ha} ha` : 'Unspecified'}</span>
                  </div>
                  <div className="p-2 rounded bg-slate-950/60 border border-slate-800/40">
                    <span className="text-[10px] text-slate-500 block uppercase">Boundary</span>
                    <span className="font-medium text-slate-200">{farm.boundary ? 'Polygon' : 'Point'}</span>
                  </div>
                </div>
              </div>

              <div className="text-[10px] text-slate-500 border-t border-slate-800 pt-3 flex justify-between items-center">
                <span>SRID 4326 (WGS84)</span>
                <span>Created {new Date(farm.created_at).toLocaleDateString()}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create Farm Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="max-w-md w-full bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-4">
              <h3 className="font-bold text-slate-100 text-base">Register Farm Plot</h3>
              <button
                onClick={() => setShowModal(false)}
                className="text-slate-400 hover:text-slate-200 p-1 rounded-lg"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleCreateFarm} className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Farm / Plot Name</label>
                <input
                  type="text"
                  required
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. North Orchard"
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 text-xs focus:outline-none focus:border-emerald-500"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">Latitude (GPS)</label>
                  <input
                    type="number"
                    step="any"
                    required
                    value={latitude}
                    onChange={(e) => setLatitude(e.target.value)}
                    placeholder="e.g. 19.8762"
                    className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 text-xs focus:outline-none focus:border-emerald-500 font-mono"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">Longitude (GPS)</label>
                  <input
                    type="number"
                    step="any"
                    required
                    value={longitude}
                    onChange={(e) => setLongitude(e.target.value)}
                    placeholder="e.g. 75.3433"
                    className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 text-xs focus:outline-none focus:border-emerald-500 font-mono"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Area in Hectares (Optional)</label>
                <input
                  type="number"
                  step="any"
                  value={areaHa}
                  onChange={(e) => setAreaHa(e.target.value)}
                  placeholder="e.g. 15.5"
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 text-xs focus:outline-none focus:border-emerald-500 font-mono"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Boundary GeoJSON Polygon (Optional)
                </label>
                <textarea
                  rows={3}
                  value={boundaryJson}
                  onChange={(e) => setBoundaryJson(e.target.value)}
                  placeholder='{"type": "Polygon", "coordinates": [[[75.34, 19.87], ...]]}'
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-100 text-xs focus:outline-none focus:border-emerald-500 font-mono"
                />
                <span className="text-[10px] text-slate-500">Leave blank if using point coordinates only.</span>
              </div>

              <div className="pt-2 flex items-center justify-end space-x-2 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setShowModal(false)}
                  className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={submitting}
                  className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold transition-colors disabled:opacity-50"
                >
                  {submitting ? 'Saving...' : 'Save Farm'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
