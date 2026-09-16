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
  unavailable: "Unavailable",
  none: "None",
  blank: "Blank",
  known: "Linked",
  unknown: "Unrecognized",
});
// What to show before the first answer, and whenever the reader cannot be read.
const NO_READER = Object.freeze({ state: "unavailable", personality: null });

let rootEl = null;
let groupEl = null;
let nameEl = null;
// The row only exists on Talk; elsewhere the whole row is hidden, so the badge
// must not re-show itself when a tag notification lands on another view.
let onTalkView = false;
let lastAccessory = NO_READER;
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

  // The poller broadcasts whether or not there is a reader to read, so the
  // state in the payload is what says which of the two this is.
  subscribe("rfid.tag", (payload) => render(payload?.accessory));

  void (async () => {
    try {
      // The desktop panel loads this page as the app starts, well before /rpc
      // answers; without the retry the badge would read "Unavailable" all session.
      const status = await untilReady(getRfidStatus, neverAborts);
      render(status?.accessory);
    } catch (error) {
      console.warn("Accessory reader unavailable:", error);
      render(NO_READER);
    }
  })();
}

/**
 * Naming the personality here would only repeat the badge next to it, which
 * already says it. The name earns its place solely when the two disagree.
 */
function labelFor(accessory) {
  if (accessory.state === "known" && accessory.personality !== activeProfile) {
    return prettifyProfileName(accessory.personality);
  }
  return LABEL_BY_STATE[accessory.state] || LABEL_BY_STATE.unavailable;
}

function render(accessory) {
  lastAccessory = accessory || NO_READER;
  if (nameEl) nameEl.textContent = labelFor(lastAccessory);
  if (rootEl) {
    rootEl.dataset.state = lastAccessory.state;
    // "Linked" and another personality's name both read as a known tag; only the
    // second one is something the user may want to act on.
    rootEl.toggleAttribute(
      "data-mismatch",
      lastAccessory.state === "known" && lastAccessory.personality !== activeProfile
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
