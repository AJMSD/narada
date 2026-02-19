# Contributing

## Code Quality Bar
- Keep code readable and professional.
- Add comments only when logic is non-obvious.
- Avoid behavior changes to flags or transcript format without matching tests.

## Required Local Checks
Run before opening a PR:

```powershell
./scripts/verify.ps1
```

The change is not ready unless syntax, format, lint, type checks, and tests all pass.
