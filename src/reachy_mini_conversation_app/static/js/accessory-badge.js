/**
 * Header accessory badge: what is on the NFC reader right now.
 *
 * Lives in the app shell (index.html) alongside the personality badge, so it
 * persists across view changes. Hidden entirely on robots with no reader — a
 * badge that could never say anything but "no reader" is just noise.
 */

import { getRfidStatus, subscribe } from "./api.js";
import { prettifyProfileName } from "./ui.js";

const LABEL_BY_STATE = Object.freeze({
  none: "None",
  blank: "Blank",
  unknown: "Unrecognized",
});

let rootEl = null;
let nameEl = null;
let supported = false;
let lastAccessory = { state: "none", personality: null };
const listeners = new Set();

/** Bind to the static markup and start tracking the reader. Safe to call twice. */
export function mountAccessoryBadge(headerRoot = document) {
  const next = headerRoot.querySelector('[data-component="accessory-badge"]');
  if (!next || rootEl === next) return;
  rootEl = next;
  nameEl = next.querySelector('[data-role="accessory-name"]');

  subscribe("rfid.tag", (payload) => render(payload?.accessory));
  void (async () => {
    try {
      const status = await getRfidStatus();
      supported = Boolean(status?.driver_available);
      render(status?.accessory);
    } catch {
      // No reader on this robot, or rfid.* never registered: stay hidden.
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
  if (rootEl) rootEl.hidden = !supported;
}

/** Whether this robot has an accessory reader at all. */
export function hasAccessoryReader() {
  return supported;
}

/** The accessory currently on the reader, as the badge last saw it. */
export function currentAccessory() {
  return lastAccessory;
}

/** Observe accessory changes. Returns an unsubscribe function. */
export function onAccessoryChange(listener) {
  listeners.add(listener);
  listener(lastAccessory);
  return () => listeners.delete(listener);
}

export function showAccessoryBadge() {
  refreshVisibility();
}

export function hideAccessoryBadge() {
  if (rootEl) rootEl.hidden = true;
}
