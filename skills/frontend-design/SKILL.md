name: frontend-design
description: Use when the user asks to build, restyle, or review a frontend UI, page, component, or layout. Enforces an intentional visual direction, accessible/responsive implementation, and browser verification instead of generic AI-looking output.
instructions: |
  # Frontend Design

  Applies when the user asks to create or change anything a user sees.

  1. **Decide the visual direction before writing markup.** Name the audience, the one
     feeling the screen should convey, and the reference aesthetic (editorial, brutalist,
     soft minimal, technical, playful, …). If the repo already has a design language,
     extend it instead of inventing a second one.
  2. **Build a deliberate type scale.** Use at most two typefaces, one for display and one
     for body. Set a clear hierarchy (e.g. 12/14/16/20/28/40 px), line-height 1.4–1.6 for
     body copy, and measure 60–75 characters per line. Never use default browser sizes for
     headings.
  3. **Define color as tokens.** Put the palette in CSS custom properties (`--bg`,
     `--surface`, `--text`, `--text-muted`, `--accent`, `--border`) and use only those.
     Pick a restrained palette with one accent; verify text contrast is at least 4.5:1
     (3:1 for large text and UI borders).
  4. **Use a spacing system, not ad-hoc values.** Restrict margins and padding to a scale
     (`4/8/12/16/24/32/48/64`). Prefer generous whitespace and one focal point per view
     over dense decoration.
  5. **Lay out with intent.** Use CSS grid/flex with `gap`; define breakpoints at content
     boundaries, not device names. Start mobile-first and verify at ~360 px, ~768 px, and
     ~1280 px. Nothing may overflow horizontally, overlap, or truncate critical text.
  6. **Give motion a purpose.** Animate only feedback, state changes, and entrances, with
     150–300 ms durations and `ease-out`/`ease-in-out`. Disable or reduce motion under
     `@media (prefers-reduced-motion: reduce)`.
  7. **Make it accessible by default.** Use semantic elements (`button`, `nav`, `main`,
     `label`) rather than clickable `div`s. Every input needs a label, every image alt text,
     every icon-only control an `aria-label`. Provide visible `:focus-visible` styles and
     full keyboard operation.
  8. **Design all states.** Implement hover, active, focus, disabled, loading, empty, and
     error states for every interactive element; never ship a happy path only.
  9. **Avoid generic AI aesthetics.** No purple/blue gradient hero, no emoji as icons, no
     card-inside-card stacked shadows, no centered everything. Use real content, one
     coherent icon set (SVG), subtle borders or a single shadow level, and typography that
     carries the design.
  10. **Verify before declaring done.** Run the app, open it in a browser, check the three
      breakpoints, tab through the page, and confirm console is free of errors. Screenshot
      or describe the result; do not rely on reading the code alone.
