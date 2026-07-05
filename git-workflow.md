# Git Workflow (local — fork-specific)

`origin` (`monkut/music-react-video-splitter`) is upstream and the PR target. Contributors do **not** have push access to it. Push branches to the `fork` remote (`weyucou/music-react-video-splitter`) and open cross-repo PRs against `origin`:

```bash
git push -u fork <branch-name>
gh pr create --repo monkut/music-react-video-splitter --head "weyucou:<branch-name>" --base main
```

Verify your remotes match this layout before pushing:

```
origin  https://github.com/monkut/music-react-video-splitter.git   # upstream, PR target
fork    https://github.com/weyucou/music-react-video-splitter.git  # push branches here
```
