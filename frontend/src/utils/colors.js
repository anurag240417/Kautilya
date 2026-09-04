/**
 * Strict Color System Mapping per FRONTEND_BRIEF.md §6
 * 
 * RULE: Entity-type colors and severity colors NEVER share hues.
 * - Entity-types: Wallet (steel blue), Transaction (amber), Network/IP (purple), Unknown (gray)
 * - Severity: Critical (muted red), High (burnt orange), Medium (warm yellow), Low (muted green)
 */

export const ENTITY_COLORS = {
  wallet: '#4A90D9',
  transaction: '#D4A843',
  network: '#7B68AE',
  ip: '#7B68AE',
  unknown: '#5C6E7E',
};

export const SEVERITY_COLORS = {
  critical: '#C9453E',
  high: '#D4763A',
  medium: '#C9A93E',
  low: '#4A7A5C',
};

export const STATUS_COLORS = {
  new: '#4A90D9',
  in_review: '#C9A93E',
  escalated: '#C9453E',
  closed: '#5C6E7E',
};

export function getEntityColor(type) {
  if (!type) return ENTITY_COLORS.unknown;
  const key = String(type).toLowerCase();
  return ENTITY_COLORS[key] || ENTITY_COLORS.unknown;
}

export function getSeverityColor(tier) {
  if (!tier) return SEVERITY_COLORS.low;
  const key = String(tier).toLowerCase();
  return SEVERITY_COLORS[key] || SEVERITY_COLORS.low;
}

export function getStatusColor(status) {
  if (!status) return STATUS_COLORS.new;
  const key = String(status).toLowerCase();
  return STATUS_COLORS[key] || STATUS_COLORS.new;
}
