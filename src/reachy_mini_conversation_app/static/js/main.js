/** Bootstrap: wire the shell header to the hash router and mount views on demand. */

import { ROUTES } from "./constants.js";
import { createRouter } from "./router.js";
import { $ } from "./ui.js";
import { mountTalkView } from "./views/talk.js";
import { mountSettingsView } from "./views/settings.js";

const SETTINGS_RETURN_KEY = "settings-return-route";

/** Survives reloads while on /settings (sessionStorage may be unavailable when embedded). */
function readSettingsReturn() {
  try {
    const route = sessionStorage.getItem(SETTINGS_RETURN_KEY) || "";
    return route.split("?")[0] === ROUTES.TALK ? route : ROUTES.TALK;
  } catch {
    return ROUTES.TALK;
  }
}

function storeSettingsReturn(route) {
  try {
    sessionStorage.setItem(SETTINGS_RETURN_KEY, route);
  } catch {
    // in-memory fallback is enough
  }
}

/** Honor the desktop app's ?theme=dark|light, overriding prefers-color-scheme. */
function applyEmbeddedTheme() {
  const theme = new URLSearchParams(window.location.search).get("theme");
  if (theme !== "dark" && theme !== "light") return;
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
}

function boot() {
  applyEmbeddedTheme();

  const outlet = $("#view-outlet");
  if (!outlet) {
    console.error("#view-outlet missing from index.html");
    return;
  }

  const router = createRouter(
    {
      [ROUTES.TALK]: (ctx) => mountTalkView(ctx),
      [ROUTES.SETTINGS]: (ctx) => mountSettingsView(ctx),
    },
    { fallback: ROUTES.TALK, outlet, onRouteChange: syncHeaderForRoute }
  );

  // Settings is an overlay: closing it returns to where it was opened
  let settingsReturn = readSettingsReturn();
  const gear = $('[data-action="open-settings"]');
  if (gear) {
    gear.addEventListener("click", () => {
      const route = router.currentRoute() || ROUTES.TALK;
      if (route.split("?")[0] === ROUTES.SETTINGS) {
        router.navigate(settingsReturn);
      } else {
        settingsReturn = route;
        storeSettingsReturn(settingsReturn);
        router.navigate(ROUTES.SETTINGS);
      }
    });
  }

  const brand = $('[data-action="go-home"]');
  if (brand) {
    brand.addEventListener("click", (event) => {
      event.preventDefault();
      router.navigate(ROUTES.TALK);
    });
  }

  const back = $('[data-action="go-back"]');
  if (back) {
    back.addEventListener("click", () => {
      const route = router.currentRoute() || ROUTES.TALK;
      const routeName = route.split("?")[0];
      if (routeName === ROUTES.SETTINGS) {
        router.navigate(settingsReturn);
      } else {
        router.navigate(ROUTES.TALK);
      }
    });
  }

  function syncHeaderForRoute(route = router.currentRoute() || ROUTES.TALK) {
    const routeName = route.split("?")[0];
    if (gear) {
      const onSettings = routeName === ROUTES.SETTINGS;
      gear.classList.toggle("is-active", onSettings);
      gear.setAttribute("aria-label", onSettings ? "Close settings" : "Open settings");
    }
    if (back) {
      back.hidden = routeName === ROUTES.TALK;
      back.setAttribute("aria-label", "Back to conversation");
    }
  }
  router.start();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot, { once: true });
} else {
  boot();
}
