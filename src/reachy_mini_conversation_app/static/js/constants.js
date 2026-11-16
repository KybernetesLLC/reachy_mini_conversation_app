/** Shared constants for the modern web UI. */

export const ORB_STATES = Object.freeze({
  MUTED: "muted",
  IDLE: "idle",
  CONNECTING: "connecting",
  LISTENING: "listening",
  THINKING: "thinking",
  SPEAKING: "speaking",
  ERROR: "error",
});

export const GLOW_BY_STATE = Object.freeze({
  [ORB_STATES.MUTED]: "#94a3b8",      // paused, waiting for the user
  [ORB_STATES.IDLE]: "#34d399",       // ready / breathing
  [ORB_STATES.CONNECTING]: "#facc15", // negotiating
  [ORB_STATES.LISTENING]: "#22d3ee",  // user speaks
  [ORB_STATES.THINKING]: "#f59e0b",   // model composes
  [ORB_STATES.SPEAKING]: "#8b7dff",   // model speaks
  [ORB_STATES.ERROR]: "#ff6a75",
});

export const ROUTES = Object.freeze({
  TALK: "#/",
  SETTINGS: "#/settings",
});
