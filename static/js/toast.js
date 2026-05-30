// toast.js — non-blocking notifications (replaces alert()).

let container = null;

function ensureContainer() {
    if (container) return container;
    container = document.createElement('div');
    container.className = 'toast-container';
    container.setAttribute('role', 'region');
    container.setAttribute('aria-label', 'Notifications');
    document.body.appendChild(container);
    return container;
}

const ICONS = { success: '\u2713', error: '\u2715', info: '\u2139', warning: '\u26A0' };

/**
 * Show a toast.
 * @param {string} message
 * @param {('success'|'error'|'info'|'warning')} type
 * @param {number} duration ms (0 = sticky)
 */
export function toast(message, type = 'info', duration = 4000) {
    const root = ensureContainer();
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    // Errors are assertive; everything else polite.
    el.setAttribute('role', type === 'error' ? 'alert' : 'status');
    el.setAttribute('aria-live', type === 'error' ? 'assertive' : 'polite');

    const icon = document.createElement('span');
    icon.className = 'toast-icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = ICONS[type] || ICONS.info;

    const msg = document.createElement('span');
    msg.className = 'toast-msg';
    msg.textContent = message;

    const close = document.createElement('button');
    close.className = 'toast-close';
    close.setAttribute('aria-label', 'Dismiss notification');
    close.innerHTML = '&times;';

    const dismiss = () => {
        el.classList.add('leaving');
        el.addEventListener('animationend', () => el.remove(), { once: true });
        // Fallback removal
        setTimeout(() => el.remove(), 400);
    };
    close.addEventListener('click', dismiss);

    el.appendChild(icon);
    el.appendChild(msg);
    el.appendChild(close);
    root.appendChild(el);

    if (duration > 0) setTimeout(dismiss, duration);
    return el;
}

export const notify = {
    success: (m, d) => toast(m, 'success', d),
    error: (m, d) => toast(m, 'error', d ?? 6000),
    info: (m, d) => toast(m, 'info', d),
    warning: (m, d) => toast(m, 'warning', d),
};
