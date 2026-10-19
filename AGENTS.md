## GitHub Workflow

Applies when the user asks to push code to GitHub.

1. **Never push directly to `main`**, even when explicitly asked.
2. Create a descriptive `feature/` or `fix/` branch for the finalized changes.
3. Fetch the latest `main` and confirm the branch is conflict-free before opening a PR.
   ```bash
   git fetch origin
   git rebase origin/main   # resolve conflicts locally, never push force without asking
   ```
4. Commit using **Conventional Commits**: `<type>(<scope>): <description>`, e.g.
   `feat(backend): add user authentication`, `fix(frontend): resolve login validation issue`,
   `docs(redis): document cache TTLs`. Allowed types: `feat`, `fix`, `docs`, `refactor`, `test`,
   `chore`, `build`, `ci`, `perf`, `style`.
5. Push the branch, then open a PR into `main`.
   ```bash
   git push -u origin <branch>
   gh pr create --base main --head <branch> --title "release(<scope>): <description>" --body "$(cat <<'EOF'
   ## Summary
   - ...

   ## Changes
   - ...

   ## Verification
   - ...
   EOF
   )"
   ```
6. **PR title must be `release(<scope>): <description>`** with a free-form scope
   (`backend`, `sdk`, `ci`, `visual`, …).
7. **PR body must not be empty.** Both are enforced by the PR Hygiene workflow and will fail CI
   otherwise.