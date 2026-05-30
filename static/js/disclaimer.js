// disclaimer.js — Task 48: persistent footer disclaimer (in markup) +
// one-time acknowledgment modal stored in localStorage.

import { openModal, closeModal } from './modal.js';

const ACK_KEY = 'dash_disclaimer_ack_v1';

function isAcknowledged() {
    try { return localStorage.getItem(ACK_KEY) === '1'; } catch (_e) { return false; }
}

function acknowledge() {
    try { localStorage.setItem(ACK_KEY, '1'); } catch (_e) { /* ignore */ }
}

export function initDisclaimer() {
    const overlay = document.getElementById('disclaimerModal');
    const ackBtn = document.getElementById('disclaimerAck');
    if (!overlay) return;

    if (ackBtn) {
        ackBtn.addEventListener('click', () => {
            acknowledge();
            closeModal(overlay);
        });
    }

    // Footer "view disclaimer" re-opens it on demand.
    const footerLink = document.getElementById('footerDisclaimerLink');
    if (footerLink) {
        footerLink.addEventListener('click', (e) => {
            e.preventDefault();
            openModal(overlay, { trigger: footerLink });
        });
    }

    if (!isAcknowledged()) {
        // Defer slightly so it appears after the dashboard renders.
        setTimeout(() => openModal(overlay), 400);
    }
}
