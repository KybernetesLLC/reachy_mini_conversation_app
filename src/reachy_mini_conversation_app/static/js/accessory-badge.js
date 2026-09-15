/**
 * Header accessory badge: what is on the NFC reader right now.
 *
 * Lives in the app shell (index.html) alongside the personality badge, so it
 * persists across view changes. Hidden entirely on robots with no reader — a
 * badge that could never say anything but "no reader" is just noise.
 */

import { getRfidStatus, subscribe, untilReady } from "./api.js";
import { prettifyProfileName } from "./ui.js";

const LABEL_BY_STATE = Object.freeze({
  none: "None",
  blank: "Blank",
  unknown: "Unrecognized",
});

let rootEl = null;
let groupEl = null;
let nameEl = null;
let supported = false;
// The row only exists on Talk; elsewhere the whole row is hidden, so the badge
// must not re-show itself when a tag notification lands on another view.
let onTalkView = false;
let lastAccessory = { state: "none", personality: null };
const listeners = new Set();
// The badge lives as long as the page, unlike a view that unmounts.
const neverAborts = new AbortController().signal;

/** Bind to the static markup and start tracking the reader. Safe to call twice. */
export function mountAccessoryBadge(headerRoot = document) {
  const next = headerRoot.querySelector('[data-component="accessory-badge"]');
  if (!next || rootEl === next) return;
  rootEl = next;
  groupEl = next.closest('[data-component="accessory-group"]') || next;
  nameEl = next.querySelector('[data-role="accessory-name"]');

  // The reader only ever broadcasts while it is being polled, which needs a
  // driver, so a notification is proof enough on its own.
  subscribe("rfid.tag", (payload) => {
    supported = true;
    render(payload?.accessory);
  });

  void (async () => {
    try {
      // The desktop panel loads this page as the app starts, well before /rpc
      // answers; without the retry the badge would stay hidden for the session.
      const status = await untilReady(getRfidStatus, neverAborts);
      supported = Boolean(status?.driver_available);
      render(status?.accessory);
    } catch (error) {
      console.warn("Accessory reader unavailable; hiding the badge:", error);
      supported = false;
      refreshVisibility();
    }
  })();
}

function render(accessory) {
  lastAccessory = accessory || { state: "none", personality: null };
  if (nameEl) {
    nameEl.textContent = lastAccessory.personality
      ? prettifyProfileName(lastAccessory.personality)
      : LABEL_BY_STATE[lastAccessory.state] || LABEL_BY_STATE.none;
  }
  if (rootEl) rootEl.dataset.state = lastAccessory.state;
  refreshVisibility();
  for (const listener of listeners) {
    try {
      listener(lastAccessory);
    } catch (error) {
      console.error("accessory badge listener threw:", error);
    }
  }
}

function refreshVisibility() {
  if (groupEl) groupEl.hidden = !(supported && onTalkView);
}

/** Observe accessory changes. Returns an unsubscribe function. */
export function onAccessoryChange(listener) {
  listeners.add(listener);
  listener(lastAccessory);
  return () => listeners.delete(listener);
}

export function showAccessoryBadge() {
  onTalkView = true;
  refreshVisibility();
}

export function hideAccessoryBadge() {
  onTalkView = false;
  refreshVisibility();
}
