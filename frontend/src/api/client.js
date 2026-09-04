/**
 * ChainTrace API client module.
 * Fully offline-compliant; wraps browser fetch() against local WSGI API.
 */

const BASE_URL = '';

async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`;
  const defaultHeaders = {
    'Accept': 'application/json',
  };

  if (options.body && typeof options.body === 'object' && !(options.body instanceof FormData)) {
    defaultHeaders['Content-Type'] = 'application/json';
    options.body = JSON.stringify(options.body);
  }

  const response = await fetch(url, {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  });

  if (!response.ok) {
    let errorDetail = { code: 'HTTP_ERROR', message: `Request failed with status ${response.status}` };
    try {
      const errJson = await response.json();
      if (errJson && errJson.error) {
        errorDetail = errJson.error;
      }
    } catch {
      // Body not JSON
    }
    const err = new Error(errorDetail.message || 'API request failed');
    err.code = errorDetail.code;
    err.details = errorDetail.details;
    err.status = response.status;
    throw err;
  }

  if (response.status === 204) {
    return null;
  }

  return response.json();
}

export async function getHealth() {
  return request('/health');
}

export async function getStatistics() {
  return request('/statistics');
}

export async function getTransaction(txid) {
  return request(`/transactions/${encodeURIComponent(txid)}`);
}

export async function getWallet(address, aggregationMethod = 'max') {
  return request(`/wallets/${encodeURIComponent(address)}?aggregation_method=${encodeURIComponent(aggregationMethod)}`);
}

export async function getAlerts(params = {}) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, val]) => {
    if (val !== undefined && val !== null && val !== '') {
      query.append(key, String(val));
    }
  });
  const qs = query.toString();
  return request(`/alerts${qs ? `?${qs}` : ''}`);
}

export async function getAlertById(alertId) {
  return request(`/alerts/${encodeURIComponent(alertId)}`);
}

export async function patchAlert(alertId, updateData) {
  return request(`/alerts/${encodeURIComponent(alertId)}`, {
    method: 'PATCH',
    body: updateData,
  });
}

export async function getGraph(entityId, options = {}) {
  const query = new URLSearchParams();
  if (options.depth) query.append('depth', String(options.depth));
  if (options.relationship) query.append('relationship', String(options.relationship));
  if (options.maxNodes) query.append('max_nodes', String(options.maxNodes));
  const qs = query.toString();
  return request(`/graph/${encodeURIComponent(entityId)}${qs ? `?${qs}` : ''}`);
}

export async function getGraphPath(source, target) {
  const query = new URLSearchParams({ source, target });
  return request(`/graph/path?${query.toString()}`);
}
