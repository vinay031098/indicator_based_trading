// modal.js — accessible modal helper: focus trap, Escape to close,
// restores focus to the trigger, overlay-click to dismiss.

const FOCUSABLE = [
    'a[href]', 'button:not([disabled])', 'textarea:not([disabled])',
    'input:not([disabled])', 'select:not([disabled])', '[tabindex]:not([tabindex="-1"])',
].join(',');

const openStack = [];

function getFocusable(container) {
    return Array.from(container.querySelectorAll(FOCUSABLE))
        .filter((el) => el.offsetParent !== null || el === document.activeElement);
}

function onKeydown(e) {
    const top = openStack[openStack.length - 1];
    if (!top) return;
    if (e.key === 'Escape') {
        e.preventDefault();
        closeModal(top.overlay);
        return;
    }
    if (e.key === 'Tab') {
        const focusables = getFocusable(top.overlay);
        if (focusables.length === 0) { e.preventDefault(); return; }
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        if (e.shiftKey && document.activeElement === first) {
            e.preventDefault();
            last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
            e.preventDefault();
            first.focus();
        }
    }
}

export function openModal(overlay, opts = {}) {
    if (!overlay || overlay.classList.contains('show')) return;
    const trigger = opts.trigger || document.activeElement;
    overlay.classList.add('show');
    overlay.setAttribute('aria-hidden', 'false');
    openStack.push({ overlay, trigger });

    if (openStack.length === 1) {
        document.addEventListener('keydown', onKeydown, true);
        document.body.style.overflow = 'hidden';
    }

    // Focus first sensible element.
    const initial = opts.initialFocus
        || overlay.querySelector('[data-autofocus]')
        || getFocusable(overlay)[0]
        || overlay.querySelector('.modal-box');
    if (initial) {
        if (!initial.hasAttribute('tabindex') && initial.classList.contains('modal-box')) {
            initial.setAttribute('tabindex', '-1');
        }
        setTimeout(() => initial.focus(), 30);
    }
}

export function closeModal(overlay) {
    if (!overlay || !overlay.classList.contains('show')) return;
    overlay.classList.remove('show');
    overlay.setAttribute('aria-hidden', 'true');
    const idx = openStack.findIndex((s) => s.overlay === overlay);
    let entry = null;
    if (idx !== -1) entry = openStack.splice(idx, 1)[0];

    if (openStack.length === 0) {
        document.removeEventListener('keydown', onKeydown, true);
        document.body.style.overflow = '';
    }
    if (entry && entry.trigger && typeof entry.trigger.focus === 'function') {
        entry.trigger.focus();
    }
}

/** Wire close-on-overlay-click and any [data-close-modal] buttons. */
export function bindModalDismiss(overlay) {
    if (!overlay) return;
    overlay.addEventListener('mousedown', (e) => { if (e.target === overlay) closeModal(overlay); });
    overlay.querySelectorAll('[data-close-modal]').forEach((btn) => {
        btn.addEventListener('click', () => closeModal(overlay));
    });
}
