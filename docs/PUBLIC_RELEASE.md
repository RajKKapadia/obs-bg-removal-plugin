# Preparing a public release

This guide covers publishing the source repository and preparing downloadable assets.
A personal `cmake --install` directory may contain models and NVIDIA libraries; review
it separately before distributing it. Changing GitHub visibility does not change any
license terms.

## What belongs in the source repository

Keep the plugin, command-line tools, tests, download/build scripts, documentation, and
license notices. Downloaded models, ONNX Runtime SDKs, CUDA libraries, local recordings,
benchmark output, OBS settings, and Git history backups stay outside the tracked tree.
The existing `.gitignore` excludes the main generated directories and model formats.

Before publication, check every branch and tag that will become visible, including old
versions of files and commit author/committer identities. A clean working tree does not
prove that old commits are free of credentials or personal information. `.gitignore`
does not remove previously committed files.

If history is rewritten for privacy, first retain a private backup. Verify that each
commit's implementation files and merge relationships are preserved. Rewritten commits
have new IDs. A `.mailmap` or a new Git email setting alone does not remove the original
email from old commit objects. Review the prepared history before replacing remote refs.
On an existing GitHub repository, old commits can also be referenced by pull requests,
forks, or cached views; updating branch tips alone is not proof of complete removal.
See [GitHub's history-removal guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

## Licenses and dependencies

| Component | Source of terms | Publication guidance |
| --- | --- | --- |
| This plugin and its tools | [LICENSE](../LICENSE), [COPYING](../COPYING): GPL-2.0-or-later | Preserve the license and attribution. For a binary distribution, meet the applicable GPL source-code requirements. |
| RVM | [Upstream project](https://github.com/PeterL1n/RobustVideoMatting), [retained GPL-3.0 text](../data/licenses/rvm/LICENSE), [provenance](../data/licenses/rvm/NOTICE.txt) | The source checkout links to upstream model downloads. Preserve notices and review the applicable terms before bundling upstream code or model files. |
| BRIA RMBG-1.4 | [Official model card](https://huggingface.co/briaai/RMBG-1.4) and the terms attached to model access | Keep the weights out of the public repository and release assets unless redistribution permission is confirmed. Commercial use requires a separate agreement according to the model card. The plugin's GPL license grants no additional model rights. |
| ONNX Runtime | `LICENSE` and `ThirdPartyNotices.txt` in the downloaded SDK | Preserve the exact SDK's required notices when packaging its libraries. CMake installs these under the plugin's `data/licenses/`. |
| NVIDIA CUDA/cuBLAS/cuDNN | Notices and agreements supplied with the exact downloaded packages | The private-library helper stages libraries for local use. Confirm which files may be redistributed and fulfill their terms before publishing them. |
| Demo media and benchmark inputs | Rights attached to the supplied input files | Publish only deliberately reviewed media and metadata. Local test access does not establish redistribution rights. |

The setup examples use RVM and skip the BRIA download. Existing RMBG support remains
available through the README's optional setup. As checked on 2026-09-10, BRIA's model
card describes noncommercial use and a commercial-license option, but its linked detailed
agreement returned HTTP 404. Treat RMBG redistribution as unconfirmed; do not infer a
permission from the broken link or from the availability of a download.

## Source archive

Commit the reviewed changes first. The following exports only files tracked in `HEAD`;
it does not include Git history or untracked local models, libraries, and artifacts.
Inspect the file list before using the archive as a release asset:

```sh
git status --short
git ls-tree -r --name-only HEAD
mkdir -p artifacts/public-release
git archive --format=tar.gz --prefix=obs-bg-removal-plugin/ \
  --output=artifacts/public-release/obs-bg-removal-plugin-source.tar.gz HEAD
tar -tzf artifacts/public-release/obs-bg-removal-plugin-source.tar.gz
sha256sum artifacts/public-release/obs-bg-removal-plugin-source.tar.gz
```

An archive of reviewed source is different from an archive of the entire checkout or
`dist/`. Keep private history backups and local validation packages out of release assets.

## Binary packages

Before offering binaries, validate them on the exact OS/architecture/OBS combination
being advertised. The current implementation supports native Linux x86-64 only; no
Windows, macOS, ARM64, or Flatpak compatibility claim has been established.

For a package intended to omit all model weights, configure a separate build directory
with the appropriate runtime arguments and all three model-install options explicitly off:

```text
-DRMBG_INSTALL_MODEL=OFF
-DRVM_INSTALL_MODELS=OFF
-DRVM_INSTALL_RESNET_MODELS=OFF
```

Stage into a new, empty output directory. Disabling an install option does not remove
files already left in an older staging directory. Inspect the resulting tree, preserve
required licenses and corresponding source, and review any bundled runtime libraries.
Omitting weights does not by itself settle the licenses of everything else in the package.

## Validation evidence

The [v0.4.0 report](v0.4.0-validation.md) retains the measured results and their
limitations. Its artifact names refer to local development files, which are intentionally
excluded from the public repository. Use the [README's reproduction commands](../README.md#reproduce-motion-and-model-comparisons)
to generate new evidence from suitable inputs. If evidence is published later, review
paths, metadata, camera footage, and redistribution rights before attaching it.
