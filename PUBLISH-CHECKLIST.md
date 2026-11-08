# Publish checklist (for Dan) - the upload commands below have NOT been run

**Release version: 0.4.0.** The registry has only 0.2.0 (0.3.0 was never uploaded), so 0.4.0 is the
next number. It is not a breaking API change, but scores moved (and round 5 changed how very long
unpunctuated sentences are split into chunks), so people who tuned a threshold on 0.2 should
re-run `rerankgate conformal` and re-ingest their documents. The version is already 0.4.0 in `pyproject.toml`,
`src/rerankgate/__init__.py` and `CHANGELOG.md`; do not bump it.

## Check first (about 10 minutes)
0. **Is the GitHub repository public?** The package page will link to
   `github.com/calasdan-sketch/rerankgate` (Repository, Issues, Changelog) and the README links
   to files there. If the repository is private, those links will 404 for everyone: either make
   it public first, or delete the three `Project-URL` lines for it in `pyproject.toml` and the
   links in the README. I could not check from here.
1. Push the branch to GitHub and open the **Actions** tab. The `ci` workflow must be green on all
   Windows, macOS and Linux jobs. Windows has never been run; if a Windows job fails, do not
   publish - send me the log.
2. `README.md` has a note under "Install" that says "Until 0.4.0 is on PyPI, install from the
   repository". **Delete that sentence** (and keep the `>=0.4` in the install lines) right
   before you upload, because PyPI freezes the README forever.
3. Merge to `master` (the repository's default branch) so the GitHub links in the README work.
4. Check the PyPI project page for `rerankgate` is yours (it already holds 0.2.0). Use a project-scoped
   API token, not your account password.

## Commands (in a clean checkout, from the repository folder)
```bash
python -m venv .release && . .release/bin/activate      # Windows: .release\Scripts\activate
pip install build twine
python -m build                                          # makes dist/*.whl and dist/*.tar.gz
twine check dist/*                                       # both must say PASSED
python -m venv /tmp/try && /tmp/try/bin/pip install dist/*.whl && /tmp/try/bin/rerankgate --version
twine upload --repository testpypi dist/*                # optional rehearsal (needs a TestPyPI token)
twine upload dist/*                                      # the real upload; cannot be undone
git tag v0.4.0 && git push origin v0.4.0
```
A PyPI version number can never be re-used, even if you delete the release.

## After
- `pip install "rerankgate>=0.4"` in a fresh environment and run the README "60-second demo".
- Cost: PyPI and TestPyPI are free; GitHub Actions minutes for a public repo are free (private repos
  have a monthly allowance - check yours).

## What changed in round 5 (so you know what you are shipping)
- Author and maintainer are exactly `Calas Automations Inc.` (no email anywhere).
- Two chunking bugs fixed (words cut in half; repeated text), a `ScoreGate` split bug, tamper
  check on gate files, and `eval` now also reports nDCG, precision@k, MAP and the false-answer
  rate with a confidence interval. Details: `CHANGELOG.md` and `docs/ROUND5.md`.
- After building, `unzip -p dist/*.whl '*/METADATA' | head` must show `Author: Calas Automations Inc.`
