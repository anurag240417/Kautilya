/**
 * Forensic data formatters for Kautilya.
 */

export function truncateIdentifier(id, startLen = 8, endLen = 6) {
  if (!id) return '';
  const str = String(id);
  if (str.length <= startLen + endLen + 3) return str;
  return `${str.slice(0, startLen)}...${str.slice(-endLen)}`;
}

export function formatBTC(value) {
  if (value === null || value === undefined || isNaN(value)) return '0.0000 BTC';
  return `${Number(value).toLocaleString('en-US', {
    minimumFractionDigits: 4,
    maximumFractionDigits: 6,
  })} BTC`;
}

export function formatScore(score) {
  if (score === null || score === undefined || isNaN(score)) return '0.0';
  return Number(score).toFixed(1);
}

export function formatTimestamp(isoString) {
  if (!isoString) return '—';
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return d.toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
  } catch {
    return String(isoString);
  }
}

export function formatPercent(value) {
  if (value === null || value === undefined || isNaN(value)) return '0%';
  return `${Math.round(Number(value))}%`;
}
