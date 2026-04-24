import { useState } from 'react'
import './App.css' // Assuming standard Vite CSS file

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

function App() {
  const [emailText, setEmailText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const checkSpam = async () => {
    if (!emailText.trim()) return;
    
    setLoading(true)
    try {
      // Send the text to your FastAPI backend
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: emailText }),
      })
      
      const data = await response.json()
      setResult(data)
    } catch (error) {
      console.error("Error connecting to API:", error)
      setResult({ error: "Failed to connect to the server." })
    }
    setLoading(false)
  }

  return (
    <div className="container" style={{ maxWidth: '600px', margin: '0 auto', padding: '2rem', fontFamily: 'sans-serif' }}>
      <h1 style={{ textAlign: 'center', color: '#4ea9c4' }}>Email Classifier</h1>
      <p style={{ textAlign: 'center', color: '#666' }}>Powered by an Ensemble Voting Model (SVC, RF, NB)</p>
      
      <textarea 
        rows="8" 
        placeholder="Paste the email content here to check..."
        value={emailText}
        onChange={(e) => setEmailText(e.target.value)}
        style={{ width: '100%', padding: '10px', fontSize: '16px', borderRadius: '8px', border: '1px solid #ccc', marginBottom: '1rem' }}
      />
      
      <button 
        onClick={checkSpam} 
        disabled={loading}
        style={{ width: '100%', padding: '12px', fontSize: '16px', backgroundColor: '#007BFF', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer' }}
      >
        {loading ? 'Analyzing...' : 'Analyze Email'}
      </button>

      {result && !result.error && (
        <div style={{ 
          marginTop: '2rem', 
          padding: '1.5rem', 
          borderRadius: '8px', 
          textAlign: 'center',
          backgroundColor: result.is_spam ? '#ffebee' : '#e8f5e9',
          border: `2px solid ${result.is_spam ? '#f44336' : '#4caf50'}`
        }}>
          <h2 style={{ color: result.is_spam ? '#d32f2f' : '#2e7d32', margin: 0 }}>
            {result.is_spam ? '🚨 WARNING: SPAM DETECTED' : '✅ SAFE: NOT SPAM'}
          </h2>
          <p style={{ margin: '10px 0 0 0', color: '#555' }}>
            Model Confidence: <strong>{result.confidence}%</strong>
          </p>
        </div>
      )}

      {result && result.error && (
        <div style={{ marginTop: '2rem', color: 'red', textAlign: 'center' }}>
          {result.error} Make sure your FastAPI backend is running!
        </div>
      )}
    </div>
  )
}

export default App