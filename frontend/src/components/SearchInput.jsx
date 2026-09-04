import React, { useState } from 'react';

export default function SearchInput({ onSearch, initialEntity = '', initialType = 'transaction' }) {
  const [entityId, setEntityId] = useState(initialEntity);
  const [entityType, setEntityType] = useState(initialType);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (entityId.trim()) {
      onSearch(entityId.trim(), entityType);
    }
  };

  const handleQuickLoad = (id, type) => {
    setEntityId(id);
    setEntityType(type);
    onSearch(id, type);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', width: '100%' }}>
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '8px', width: '100%' }}>
        <select
          value={entityType}
          onChange={(e) => setEntityType(e.target.value)}
          style={{ width: '130px', flexShrink: 0 }}
        >
          <option value="transaction">Transaction</option>
          <option value="wallet">Wallet Address</option>
        </select>

        <input
          type="text"
          placeholder={
            entityType === 'transaction'
              ? 'Enter Elliptic++ Transaction ID (e.g. 1001, 230425446)'
              : 'Enter Wallet Address (e.g. 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa, 1dice8EM2Ws...)'
          }
          value={entityId}
          onChange={(e) => setEntityId(e.target.value)}
          style={{ flex: 1 }}
        />

        <button type="submit" className="primary" style={{ flexShrink: 0, padding: '0 18px', fontWeight: 600 }}>
          Trace Entity
        </button>
      </form>

      {/* Quick query presets for offline demo convenience */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', color: 'var(--ct-text-muted)' }}>
        <span>Quick Samples:</span>
        <button
          type="button"
          onClick={() => handleQuickLoad('1001', 'transaction')}
          style={{ fontSize: '11px', padding: '2px 6px', background: 'transparent' }}
        >
          Tx #1001 (Illicit)
        </button>
        <button
          type="button"
          onClick={() => handleQuickLoad('1002', 'transaction')}
          style={{ fontSize: '11px', padding: '2px 6px', background: 'transparent' }}
        >
          Tx #1002 (Correlated)
        </button>
        <button
          type="button"
          onClick={() => handleQuickLoad('1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa', 'wallet')}
          style={{ fontSize: '11px', padding: '2px 6px', background: 'transparent' }}
        >
          Genesis Wallet
        </button>
      </div>
    </div>
  );
}
