/**
 * Header accessory badge: what is on the NFC reader right now.
 *
 * Lives in the app shell (index.html) alongside the personality badge, so it
 * persists across view changes. It stays visible without a reader, reading
 * "Unavailable", so the feature is discoverable rather than silently absent.
 */

import { getRfidStatus, subscribe, untilReady } from "./api.js";
import { prettifyProfileName } from "./ui.js";

const LABEL_BY_STATE = Object.freeze({
  none: "None",
  blank: "Blank",
  known: "Linked",
  unknown: "Unrecognized",
});
const NO_READER_STATE = "unavailable";

let rootEl = null;
let groupEl = null;
let nameEl = null;
let supported = false;
// The row only exists on Talk; elsewhere the whole row is hidden, so the badge
// must not re-show itself when a tag notification lands on another view.
let onTalkView = false;
let lastAccessory = { state: "none", personality: null };
let activeProfile = null;
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
      // answers; without the retry the badge would read "Unavailable" all session.
      const status = await untilReady(getRfidStatus, neverAborts);
      supported = Boolean(status?.driver_available);
      render(status?.accessory);
    } catch (error) {
      console.warn("Accessory reader unavailable:", error);
      supported = false;
      render(lastAccessory);
    }
  })();
}

/**
 * Naming the personality here would only repeat the badge next to it, which
 * already says it. The name earns its place solely when the two disagree.
 */
function labelFor(accessory) {
  if (!supported) return "Unavailable";
  if (accessory.state === "known" && accessory.personality !== activeProfile) {
    return prettifyProfileName(accessory.personality);
  }
  return LABEL_BY_STATE[accessory.state] || LABEL_BY_STATE.none;
}

function render(accessory) {
  lastAccessory = accessory || { state: "none", personality: null };
  if (nameEl) nameEl.textContent = labelFor(lastAccessory);
  if (rootEl) {
    rootEl.dataset.state = supported ? lastAccessory.state : NO_READER_STATE;
    // "Linked" and another personality's name both read as a known tag; only the
    // second one is something the user may want to act on.
    rootEl.toggleAttribute(
      "data-mismatch",
      supported && lastAccessory.state === "known" && lastAccessory.personality !== activeProfile
    );
  }
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
  if (groupEl) groupEl.hidden = !onTalkView;
}

/** Tell the badge which personality is running, so it can stay out of its way. */
export function setActivePersonality(profile) {
  if (profile === activeProfile) return;
  activeProfile = profile;
  render(lastAccessory);
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
