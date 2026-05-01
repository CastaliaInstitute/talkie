import { test, expect } from "@playwright/test";

/** Fixed prompt for smoke / Playwright validation (upstream Talkie + proxy must be live). */
export const VALIDATION_PROMPT = "What year is it?";

test.describe("Talkie chat validation", () => {
  test("assistant answers the validation question with a plausible year", async ({
    page,
  }) => {
    await page.goto("/");

    await page.getByLabel("Your dispatch").fill(VALIDATION_PROMPT);
    await page.getByRole("button", { name: /Seal & send/i }).click();

    const lastAssistant = page.locator(".turn.assistant").last();
    await expect(lastAssistant).not.toHaveClass(/pending/);
    await expect(page.locator("#err")).not.toHaveClass(/visible/);

    const body = lastAssistant.locator(".body");
    const text = await body.innerText();
    expect(text.trim().length).toBeGreaterThan(4);
    // Expect a spoken year (e.g. 1931, MCMXXXI, or similar digit span)
    expect(text).toMatch(/\b(?:1[0-9]{3}|20[0-9]{2})\b/);
  });
});
