# Contributing

This project is an experimental native OBS effect filter. The supported build target
is native Linux x86-64; macOS, Windows, ARM64, and Flatpak packaging need additional
implementation and validation. Start with the [setup and tuning guide](README.md).

## Reporting a problem

Include the OBS version and installation type, OS and architecture, plugin commit or
version, selected model filename and SHA256, inference device, and exact error.
For build failures, include the failing command and compiler/CMake/Python versions.
For runtime failures, include the refreshed filter status and relevant OBS log lines.
See [troubleshooting](README.md#troubleshooting) before opening an issue.

Share only the information needed to reproduce the problem. Remove credentials,
private paths, and unrelated scene/account details from logs. Do not upload private
camera recordings, downloaded model weights, `.env` files, or complete local build and
runtime directories. Use a minimal synthetic fixture when possible; otherwise provide
media you have permission to publish.

## Changing the plugin

Keep a change focused and explain the problem, resulting behavior, and validation in
the pull request. Preserve model signature checks, bounded buffering, and original-source
passthrough on loading/errors/stale results. Document changes to the filter controls or
defaults, especially any effects on video delay and audio synchronization.

Build with the chosen backend from the [Linux instructions](README.md#build-linux), then run:

```sh
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

For changes to inference or rendering, also run the relevant
[image CLI](README.md#test-an-image-without-obs) and
[OBS integration checks](README.md#obs-integration-test) with appropriate fixtures.
Report the backend, resolution, frame rate, and measurement conditions. Separate
synthetic alignment checks from real-camera matte quality, and isolated model timings
from complete OBS scene performance. Documentation-only changes need link and command
checks; they do not require rerunning camera benchmarks.

## Licenses and submitted files

Submit code you have the right to contribute under the project's
[GPL-2.0-or-later license](LICENSE), and preserve third-party attribution and license
notices. Model weights and runtime dependencies have separate terms; see the
[release guide](docs/PUBLIC_RELEASE.md).

Keep generated models, SDKs, CUDA libraries, recordings, logs, and private Git backups
out of commits. `.gitignore` helps prevent accidental additions, but does not remove a
file already committed or included in a release asset. Review the staged diff and file
list before committing.

Git commits contain an author name and email. If you want to keep your email private,
copy the GitHub-provided noreply address from your account's email settings and configure
it for this checkout before committing. This affects future commits; it does not change
the identity already stored in older commits.
