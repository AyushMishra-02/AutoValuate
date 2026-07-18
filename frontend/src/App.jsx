import React, { useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceDot } from 'recharts';
import { Activity, ShieldCheck, ChevronRight } from 'lucide-react';

export default function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  const [formData, setFormData] = useState({
    name: 'Maruti Swift Dzire VDI',
    year: 2017,
    km_driven: 45000,
    fuel: 'Diesel',
    seller_type: 'Individual',
    transmission: 'Manual',
    owner: 'First Owner'
  });

  const handleChange = (e) => {
    setFormData({...formData, [e.target.name]: e.target.value});
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const response = await axios.post('http://localhost:8000/predict', {
        ...formData,
        year: parseInt(formData.year),
        km_driven: parseInt(formData.km_driven)
      });
      
      // Artificial delay for cinematic effect
      setTimeout(() => {
        setResult(response.data);
        setLoading(false);
      }, 800);
      
    } catch (err) {
      setError(err.message || 'Failed to fetch valuation');
      setLoading(false);
    }
  };

  const formatPrice = (val) => new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(val);

  // Generate mockup depreciation curve
  const generateCurve = () => {
    if (!result) return [];
    const age = 2026 - parseInt(formData.year);
    const decay = 0.12;
    const base = result.predicted_price / Math.pow(1 - decay, age);
    
    return Array.from({length: 15}, (_, i) => ({
      age: i + 1,
      value: base * Math.pow(1 - decay, i + 1),
      isCurrent: (i + 1) === age ? result.predicted_price : null
    }));
  };

  return (
    <div className="app-container fade-in">
      
      {/* Left Column: Form */}
      <div className="fade-in delay-1">
        <h1>AutoValuate</h1>
        
        <div className="glass-panel">
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Vehicle Configuration</label>
              <input type="text" name="name" value={formData.name} onChange={handleChange} required />
            </div>
            
            <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem'}}>
              <div className="form-group">
                <label>Year</label>
                <input type="number" name="year" value={formData.year} onChange={handleChange} required />
              </div>
              <div className="form-group">
                <label>Kilometers</label>
                <input type="number" name="km_driven" value={formData.km_driven} onChange={handleChange} required />
              </div>
            </div>
            
            <div className="form-group">
              <label>Fuel Type</label>
              <select name="fuel" value={formData.fuel} onChange={handleChange}>
                <option>Petrol</option>
                <option>Diesel</option>
                <option>CNG</option>
              </select>
            </div>
            
            <div className="form-group">
              <label>Transmission</label>
              <select name="transmission" value={formData.transmission} onChange={handleChange}>
                <option>Manual</option>
                <option>Automatic</option>
              </select>
            </div>

            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? 'Analyzing...' : 'Generate Valuation'} <ChevronRight size={16} style={{position: 'absolute', right: '1rem', top: '1rem'}} />
            </button>
          </form>
        </div>
      </div>
      
      {/* Right Column: Results */}
      <div style={{display: 'flex', flexDirection: 'column', justifyContent: 'center'}}>
        
        {loading && (
          <div className="fade-in">
            <div className="loading-spinner"></div>
            <p style={{textAlign: 'center', color: 'var(--subtle-gray)', letterSpacing: '0.1em'}}>COMPUTING CONFORMAL BOUNDS</p>
          </div>
        )}
        
        {error && (
          <div className="glass-panel fade-in" style={{borderColor: 'var(--ferrari-red)'}}>
            <p style={{color: 'var(--ferrari-red)'}}>{error}</p>
          </div>
        )}
        
        {result && !loading && (
          <div className="glass-panel fade-in delay-2">
            
            <div className="metric-row">
              <div className="metric-box">
                <div className="metric-label">Lower Bound (80%)</div>
                <div className="metric-value">{formatPrice(result.confidence_lower_80)}</div>
              </div>
              <div className="metric-box">
                <div className="metric-label">Estimated Value</div>
                <div className="metric-value primary">{formatPrice(result.predicted_price)}</div>
              </div>
              <div className="metric-box">
                <div className="metric-label">Upper Bound (80%)</div>
                <div className="metric-value">{formatPrice(result.confidence_upper_80)}</div>
              </div>
            </div>
            
            <div style={{marginBottom: '2rem'}}>
              <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem'}}>
                <Activity size={16} color="var(--ferrari-red)" />
                <h3 style={{fontSize: '0.9rem'}}>Depreciation Curve</h3>
              </div>
              <div style={{height: '250px', width: '100%'}}>
                <ResponsiveContainer>
                  <LineChart data={generateCurve()}>
                    <XAxis dataKey="age" stroke="var(--subtle-gray)" fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis stroke="var(--subtle-gray)" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(v) => `₹${(v/100000).toFixed(1)}L`} />
                    <Tooltip 
                      contentStyle={{backgroundColor: 'var(--carbon-black)', border: '1px solid rgba(255,40,0,0.3)', borderRadius: 0}}
                      itemStyle={{color: 'white'}}
                      formatter={(val) => formatPrice(val)}
                      labelFormatter={(label) => `Age: ${label} Years`}
                    />
                    <Line type="monotone" dataKey="value" stroke="rgba(255,255,255,0.2)" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="isCurrent" stroke="var(--ferrari-red)" strokeWidth={0} dot={{r: 6, fill: 'var(--ferrari-red)'}} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            <div style={{marginBottom: '2rem'}}>
              <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem'}}>
                <ShieldCheck size={16} color="var(--ferrari-red)" />
                <h3 style={{fontSize: '0.9rem'}}>Key Drivers</h3>
              </div>
              {result.top_3_shap_features.map((f, i) => (
                <div key={i} style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', paddingBottom: '0.5rem', borderBottom: '1px solid rgba(255,255,255,0.05)'}}>
                  <span style={{color: 'var(--subtle-gray)', fontSize: '0.85rem'}}>{f.feature.replace('_', ' ').toUpperCase()}</span>
                  <span style={{color: f.contribution > 0 ? '#10b981' : '#ef4444', fontSize: '0.85rem'}}>
                    {f.contribution > 0 ? '+' : ''}{formatPrice(f.contribution)}
                  </span>
                </div>
              ))}
            </div>

            <div style={{background: 'rgba(255, 40, 0, 0.05)', padding: '1.5rem', borderRadius: '8px', border: '1px solid rgba(255, 40, 0, 0.2)'}}>
              <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem'}}>
                <div style={{width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--ferrari-red)', boxShadow: '0 0 10px var(--ferrari-red)'}}></div>
                <h3 style={{fontSize: '0.9rem', color: 'var(--ferrari-red)'}}>AI Negotiation Intelligence</h3>
              </div>
              <ul style={{listStyleType: 'none', padding: 0}}>
                {result.negotiation_insights.map((insight, i) => (
                  <li key={i} style={{fontSize: '0.85rem', color: 'var(--stark-white)', marginBottom: '1rem', lineHeight: '1.5', paddingLeft: '1rem', borderLeft: '2px solid rgba(255,40,0,0.5)'}}>
                    {insight}
                  </li>
                ))}
              </ul>
            </div>

          </div>
        )}
        
        {!result && !loading && !error && (
          <div className="fade-in delay-3" style={{textAlign: 'center', color: 'var(--subtle-gray)', paddingTop: '4rem'}}>
            <p style={{letterSpacing: '0.1em', fontSize: '0.8rem'}}>AWAITING VEHICLE SPECIFICATIONS</p>
          </div>
        )}
        
      </div>
    </div>
  );
}
