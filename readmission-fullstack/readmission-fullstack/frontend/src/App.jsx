import { useEffect, useMemo, useState } from 'react';
import { BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const defaultApi = 'http://127.0.0.1:8000';
const baseForm = {
  race: 'Caucasian', gender: 'Female', age: '[70-80)', admission_type_id: 1, discharge_disposition_id: 1,
  admission_source_id: 7, time_in_hospital: 4, num_lab_procedures: 43, num_procedures: 1, num_medications: 13,
  number_outpatient: 0, number_emergency: 0, number_inpatient: 1, diag_1: '428', diag_2: '276', diag_3: '250',
  number_diagnoses: 8, max_glu_serum: 'None', A1Cresult: 'None', metformin: 'No', insulin: 'Up', change: 'Ch', diabetesMed: 'Yes'
};
const metricRisk = (prob) => prob >= 0.5 ? 'high' : prob >= 0.2 ? 'medium' : 'low';

export default function App() {
  const [apiBase, setApiBase] = useState(defaultApi);
  const [summary, setSummary] = useState(null);
  const [examples, setExamples] = useState([]);
  const [topFeatures, setTopFeatures] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [form, setForm] = useState(baseForm);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState('');

  const fetchJson = async (path, options = {}) => {
    const res = await fetch(`${apiBase}${path}`, options);
    if (!res.ok) throw new Error(`${path} failed with ${res.status}`);
    return await res.json();
  };
  const loadDashboard = async () => {
    try {
      setError(''); setLoading(true);
      const data = await fetchJson('/dashboard/summary');
      setSummary(data.metrics); setTopFeatures(data.top_features || []); setExamples(data.patient_examples || []);
    } catch {
      setError(`Could not reach backend at ${apiBase}. Start FastAPI first, because the frontend can’t read your mind.`);
    } finally { setLoading(false); }
  };
  useEffect(() => { loadDashboard(); }, [apiBase]);

  const chartData = useMemo(() => (topFeatures || []).map(item => ({
    feature: String(item.feature || '').replaceAll('_', ' '),
    value: Number((item.abs_coefficient ?? 0).toFixed ? item.abs_coefficient.toFixed(3) : item.abs_coefficient ?? 0)
  })), [topFeatures]);

  const handleChange = (key, value) => setForm(prev => ({ ...prev, [key]: value }));
  const handlePredict = async () => {
    try {
      setPredicting(true); setError('');
      const payload = {
        ...form,
        admission_type_id: Number(form.admission_type_id), discharge_disposition_id: Number(form.discharge_disposition_id),
        admission_source_id: Number(form.admission_source_id), time_in_hospital: Number(form.time_in_hospital),
        num_lab_procedures: Number(form.num_lab_procedures), num_procedures: Number(form.num_procedures),
        num_medications: Number(form.num_medications), number_outpatient: Number(form.number_outpatient),
        number_emergency: Number(form.number_emergency), number_inpatient: Number(form.number_inpatient), number_diagnoses: Number(form.number_diagnoses)
      };
      setPrediction(await fetchJson('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }));
    } catch {
      setError('Prediction failed. Check that the backend is running and the dataset zip is in the backend folder.');
    } finally { setPredicting(false); }
  };

  const m = summary || { classification_report: { '1': {} }, threshold: 0.5 };

  return <div className="container">
    <div className="header"><div><h1>Hospital Readmission XAI Dashboard</h1><p>Live React dashboard connected to FastAPI for diabetic patient readmission risk and top-3 driver explanations.</p></div><div className="badge">Backend: {apiBase}</div></div>
    <div className="card" style={{ marginBottom: 16 }}><div className="form-grid" style={{ gridTemplateColumns: '2fr 1fr' }}><div className="input-group"><label>API Base URL</label><input value={apiBase} onChange={(e) => setApiBase(e.target.value)} /></div><div className="button-row" style={{ alignItems: 'end' }}><button className="secondary" onClick={loadDashboard}>Reload data</button></div></div></div>
    {error && <div className="error" style={{ marginBottom: 16 }}>{error}</div>}
    <div className="grid grid-4" style={{ marginBottom: 16 }}>
      <MetricCard label="Accuracy" value={loading ? '...' : (m.accuracy ?? 0).toFixed(4)} foot="Overall test-set accuracy" />
      <MetricCard label="ROC-AUC" value={loading ? '...' : (m.roc_auc ?? 0).toFixed(4)} foot="Ranking quality for readmission risk" />
      <MetricCard label="Recall class 1" value={loading ? '...' : `${((m.classification_report?.['1']?.recall ?? 0) * 100).toFixed(2)}%`} foot="Caught readmissions" />
      <MetricCard label="Threshold" value={loading ? '...' : m.threshold} foot="Decision cutoff" />
    </div>
    <div className="grid grid-2" style={{ marginBottom: 16 }}>
      <div className="card"><h3 className="section-title">Global feature drivers</h3><div style={{ width: '100%', height: 320 }}><ResponsiveContainer><BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 15, left: 25, bottom: 5 }}><CartesianGrid strokeDasharray="3 3" stroke="#263453" /><XAxis type="number" stroke="#9eb0cc" /><YAxis dataKey="feature" type="category" width={130} stroke="#9eb0cc" /><Tooltip /><Bar dataKey="value" fill="#5eead4" radius={[0, 8, 8, 0]} /></BarChart></ResponsiveContainer></div><div className="footer-note">These values come straight from the backend coefficient output.</div></div>
      <div className="card"><h3 className="section-title">Predict a patient</h3><div className="form-grid">
        <Field label="Age" value={form.age} onChange={(v) => handleChange('age', v)} options={['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)','[50-60)','[60-70)','[70-80)','[80-90)','[90-100)']} />
        <Field label="Gender" value={form.gender} onChange={(v) => handleChange('gender', v)} options={['Male','Female']} />
        <Field label="Race" value={form.race} onChange={(v) => handleChange('race', v)} options={['Caucasian','AfricanAmerican','Hispanic','Asian','Other']} />
        <NumberField label="Hospital days" value={form.time_in_hospital} onChange={(v) => handleChange('time_in_hospital', v)} />
        <NumberField label="Medications" value={form.num_medications} onChange={(v) => handleChange('num_medications', v)} />
        <NumberField label="Prior inpatient" value={form.number_inpatient} onChange={(v) => handleChange('number_inpatient', v)} />
        <NumberField label="Lab procedures" value={form.num_lab_procedures} onChange={(v) => handleChange('num_lab_procedures', v)} />
        <NumberField label="Diagnosis count" value={form.number_diagnoses} onChange={(v) => handleChange('number_diagnoses', v)} />
        <Field label="Insulin" value={form.insulin} onChange={(v) => handleChange('insulin', v)} options={['No','Up','Down','Steady']} />
        <TextField label="Primary diagnosis" value={form.diag_1} onChange={(v) => handleChange('diag_1', v)} />
        <TextField label="Secondary diagnosis" value={form.diag_2} onChange={(v) => handleChange('diag_2', v)} />
        <TextField label="Tertiary diagnosis" value={form.diag_3} onChange={(v) => handleChange('diag_3', v)} />
      </div><div className="button-row"><button className="primary" onClick={handlePredict} disabled={predicting}>{predicting ? 'Predicting...' : 'Predict risk'}</button><button className="secondary" onClick={() => { setForm(baseForm); setPrediction(null); }}>Reset</button></div>{prediction ? <PredictionCard prediction={prediction} /> : <div className="callout" style={{ marginTop: 16 }}>No prediction yet. Run one and the backend will return probability, class, and top 3 drivers.</div>}</div>
    </div>
    <div className="card"><h3 className="section-title">Example test patients</h3><div className="table-wrap"><table className="table"><thead><tr><th>ID</th><th>Age</th><th>Gender</th><th>Hospital days</th><th>Medications</th><th>Prior inpatient</th><th>Probability</th><th>Actual</th></tr></thead><tbody>{examples.map((row) => <tr key={row.id}><td>{row.id}</td><td>{row.age}</td><td>{row.gender}</td><td>{row.time_in_hospital}</td><td>{row.num_medications}</td><td>{row.number_inpatient}</td><td><span className={`pill ${metricRisk(row.predicted_probability)}`}>{row.predicted_probability.toFixed(3)}</span></td><td>{row.actual}</td></tr>)}</tbody></table></div></div>
  </div>;
}

function MetricCard({ label, value, foot }) { return <div className="card"><div className="metric-label">{label}</div><div className="metric-value">{value}</div><div className="metric-foot">{foot}</div></div>; }
function PredictionCard({ prediction }) { return <div style={{ marginTop: 16 }}><div className={`pill ${metricRisk(prediction.predicted_probability)}`}>Probability {prediction.predicted_probability.toFixed(4)} • Class {prediction.predicted_class}</div><div className="callout" style={{ marginTop: 12 }}><strong>Top 3 risk drivers</strong><div style={{ marginTop: 8 }}>{prediction.top_3_risk_drivers?.length ? prediction.top_3_risk_drivers.map((item) => <div key={item.feature} className="driver"><div><div>{item.label}</div><div className="small">Encoded value: {item.value.toFixed(3)}</div></div><div className="small">Contribution: {item.contribution.toFixed(4)}</div></div>) : <div className="small">No positive drivers were found for this profile.</div>}</div></div></div>; }
function Field({ label, value, onChange, options }) { return <div className="input-group"><label>{label}</label><select value={value} onChange={(e) => onChange(e.target.value)}>{options.map(option => <option key={option} value={option}>{option}</option>)}</select></div>; }
function NumberField({ label, value, onChange }) { return <div className="input-group"><label>{label}</label><input type="number" value={value} onChange={(e) => onChange(e.target.value)} /></div>; }
function TextField({ label, value, onChange }) { return <div className="input-group"><label>{label}</label><input value={value} onChange={(e) => onChange(e.target.value)} /></div>; }
