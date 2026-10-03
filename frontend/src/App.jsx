import React, { useState, useEffect } from 'react';
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import NetworkTelemetryBar from './components/NetworkTelemetryBar';
import Overview from './pages/Overview';
import Investigation from './pages/Investigation';
import Alerts from './pages/Alerts';
import EntityProfile from './pages/EntityProfile';
import DataSources from './pages/DataSources';
import ForensicsLab from './pages/ForensicsLab';
import { getHealth } from './api/client';

export default function App() {
  const [systemStatus, setSystemStatus] = useState({ offline: true, operational: true });

  useEffect(() => {
    async function checkHealth() {
      try {
        const h = await getHealth();
        if (h && h.status === 'healthy') {
          setSystemStatus({ offline: true, operational: true, version: h.version });
        }
      } catch {
        // Backend not yet running or demo offline mode
        setSystemStatus({ offline: true, operational: true, version: '0.1.0' });
      }
    }
    checkHealth();
  }, []);

  return (
    <HashRouter>
      <div style={{ display: 'flex', minHeight: '100vh', width: '100%' }}>
        {/* Left Fixed Navigation Sidebar */}
        <Sidebar systemStatus={systemStatus} />

        {/* Main Content Area */}
        <main
          style={{
            flex: 1,
            backgroundColor: 'var(--ct-bg-primary)',
            minHeight: '100vh',
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {/* Institutional Bitcoin Network & Forensic Telemetry Header */}
          <NetworkTelemetryBar />

          <div style={{ flex: 1 }}>
            <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/investigation" element={<Investigation />} />
            <Route path="/alerts" element={<Alerts />} />
            <Route path="/forensics" element={<ForensicsLab />} />
            <Route path="/entity-profile" element={<EntityProfile />} />
            <Route path="/data-sources" element={<DataSources />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </main>
    </div>
    </HashRouter>
  );
}
