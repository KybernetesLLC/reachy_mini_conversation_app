/** Accessory view: the NFC reader, and what the accessory on it carries. */

import {
  describeError,
  eraseRfidTag,
  getRfidStatus,
  linkRfidTag,
  listPersonalities,
  subscribe,
  untilReady,
} from "../api.js";
import { h, prettifyProfileName } from "../ui.js";
import { confirmDialog } from "../components/confirm-dialog.js";

const ACCESSORY_COPY = Object.freeze({
  none: {
    title: "No accessory",
    hint: "Place an accessory on Reachy Mini's head to see what it carries.",
  },
  blank: {
    title: "Blank accessory",
    hint: "This accessory carries nothing yet. Link it to a personality below, or ask Reachy Mini to invent one for it.",
  },
  known: {
    title: "Linked accessory",
    hint: "Reachy Mini takes on this personality while the accessory is on its head.",
  },
  unknown: {
    title: "Unrecognized accessory",
    hint: "This accessory carries something this robot cannot use — a personality it does not have, or a tag written elsewhere.",
  },
});

export async function mountAccessoryView({ outlet, signal }) {
  const readerStatus = h("p", { class: "settings-hint", "data-role": "reader" }, "Checking the reader…");
  const accessoryCard = h("div", { class: "accessory-card", "aria-live": "polite" });
  const personalitySelect = h("select", {
    class: "settings-select",
    "aria-label": "Personality to link",
    disabled: "disabled",
  });
  const linkButton = h("button", { type: "button", class: "btn btn--primary", disabled: "disabled" }, "Link accessory");
  const eraseButton = h("button", { type: "button", class: "btn btn--ghost", disabled: "disabled" }, "Unlink accessory");
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });

  const view = h(
    "section",
    { class: "view view--accessory" },
    h(
      "header",
      { class: "view-header" },
      h("h1", { class: "view-title" }, "Accessory"),
      h(
        "p",
        { class: "view-subtitle" },
        "See what the accessory on Reachy Mini's head carries, and choose the personality it should apply."
      )
    ),
    h(
      "section",
      { class: "settings-section" },
      h("h2", { class: "settings-section-title" }, "On the reader"),
      readerStatus,
      accessoryCard
    ),
    h(
      "section",
      { class: "settings-section" },
      h("h2", { class: "settings-section-title" }, "Link a personality"),
      h(
        "p",
        { class: "settings-hint settings-section-intro" },
        "The accessory carries the personality itself, so it means the same thing on any Reachy Mini. "
          + "Unlinking leaves it blank, and Reachy Mini stops changing personality for it."
      ),
      h(
        "label",
        { class: "settings-field" },
        h("span", { class: "settings-label" }, "Personality"),
        personalitySelect
      ),
      h("div", { class: "settings-actions" }, eraseButton, linkButton),
      status
    )
  );
  outlet.replaceChildren(view);

  let latest = null;
  let busy = false;

  function setBusy(nextBusy, label) {
    busy = nextBusy;
    view.toggleAttribute("aria-busy", nextBusy);
    linkButton.disabled = nextBusy || !canLink();
    eraseButton.disabled = nextBusy || !canErase();
    personalitySelect.disabled = nextBusy || !personalitySelect.options.length;
    linkButton.textContent = nextBusy && label ? label : "Link accessory";
  }

  function canLink() {
    return Boolean(latest?.connected) && latest?.accessory?.state !== "none" && personalitySelect.value !== "";
  }

  function canErase() {
    const state = latest?.accessory?.state;
    return Boolean(latest?.connected) && (state === "known" || state === "unknown");
  }

  function renderReader(payload) {
    if (!payload?.driver_available) {
      readerStatus.textContent = "No NFC reader on this Reachy Mini.";
      return;
    }
    if (!payload.connected) {
      readerStatus.textContent = "The NFC reader is not responding.";
      return;
    }
    readerStatus.textContent = payload.port
      ? `Reader connected on ${payload.port}.`
      : "Reader connected.";
  }

  function renderAccessory(payload) {
    const accessory = payload?.accessory || { state: "none" };
    const copy = ACCESSORY_COPY[accessory.state] || ACCESSORY_COPY.none;
    accessoryCard.dataset.state = accessory.state;
    accessoryCard.replaceChildren(
      h(
        "div",
        { class: "accessory-card__copy" },
        h("strong", null, copy.title),
        accessory.personality
          ? h("span", { class: "accessory-card__name" }, prettifyProfileName(accessory.personality))
          : null,
        h("span", { class: "settings-hint" }, copy.hint),
        tagDetails(accessory)
      )
    );
  }

  /** What the tag literally holds. Useful when a tag is not recognized. */
  function tagDetails(accessory) {
    if (accessory.state === "none" || accessory.state === "blank") return null;
    return h(
      "dl",
      { class: "accessory-card__details" },
      detailRow("Personality", accessory.personality ? prettifyProfileName(accessory.personality) : "—"),
      detailRow("Profile", accessory.personality || "—"),
      detailRow("Written on the tag", accessory.content || "—")
    );
  }

  function detailRow(label, value) {
    return h(
      "div",
      { class: "accessory-card__detail" },
      h("dt", null, label),
      h("dd", null, value)
    );
  }

  function render(payload) {
    latest = payload;
    renderReader(payload);
    renderAccessory(payload);
    // A tag arriving or leaving changes what the buttons can do.
    setBusy(busy);
  }

  let choices;
  try {
    choices = await untilReady(listPersonalities, signal, () => {
      readerStatus.textContent = "Waiting for Reachy to finish starting…";
    });
  } catch (error) {
    if (signal.aborted) return;
    status.textContent = describeError(error);
    status.classList.add("is-error");
    return;
  }
  if (signal.aborted) return;

  for (const name of choices?.choices || []) {
    personalitySelect.appendChild(h("option", { value: name }, prettifyProfileName(name)));
  }
  if (!personalitySelect.options.length) {
    personalitySelect.appendChild(h("option", { value: "" }, "No personality to link"));
  }

  try {
    render(await getRfidStatus());
  } catch (error) {
    if (!signal.aborted) {
      status.textContent = describeError(error);
      status.classList.add("is-error");
    }
  }

  const unsubscribe = subscribe("rfid.tag", (payload) => {
    if (!signal.aborted) render({ ...payload, accessory: payload.accessory || { state: "none" } });
  });
  signal.addEventListener("abort", unsubscribe, { once: true });

  personalitySelect.addEventListener("change", () => setBusy(busy));

  linkButton.addEventListener("click", async () => {
    if (busy || !canLink()) return;
    const personality = personalitySelect.value;
    status.classList.remove("is-error");
    status.textContent = "";
    setBusy(true, "Linking…");
    try {
      const result = await linkRfidTag(personality);
      status.textContent = result?.written
        ? `Accessory linked to ${prettifyProfileName(personality)}.`
        : `This accessory already carries ${prettifyProfileName(personality)}.`;
      render(await getRfidStatus());
    } catch (error) {
      status.textContent = describeError(error);
      status.classList.add("is-error");
    } finally {
      setBusy(false);
    }
  });

  eraseButton.addEventListener("click", async () => {
    if (busy || !canErase()) return;
    const confirmed = await confirmDialog({
      title: "Unlink this accessory?",
      message: "It will carry nothing afterwards, and Reachy Mini will stop changing personality for it.",
      confirmLabel: "Unlink",
      danger: true,
      signal,
    });
    if (!confirmed) return;
    status.classList.remove("is-error");
    status.textContent = "";
    setBusy(true, "Erasing…");
    try {
      await eraseRfidTag(false);
      status.textContent = "Accessory unlinked.";
      render(await getRfidStatus());
    } catch (error) {
      status.textContent = describeError(error);
      status.classList.add("is-error");
    } finally {
      setBusy(false);
    }
  });
}
