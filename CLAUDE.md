You are an expert Python developer with experience in AWS serverless architectures, Zappa, Flask, and audio/video processing pipelines.

This project (`sanji`) provides a SaaS service for splitting music reaction streams — it processes YouTube videos to separate music segments from reaction commentary, making music portions individually accessible.

## Key Architecture

- **Zappa + Flask**: serverless API layer (`sanji/app.py`)
- **SQS + S3**: async job queue and artifact storage
- **`is_music` classifier**: heuristic that detects music vs. speech segments
- **`run_pipeline`**: core processing pipeline (to be ported to async worker)

## Git Workflow

If `git-workflow.md` exists at the repo root, read it for fork/origin remote conventions and the PR-creation recipe. The file is gitignored because it is specific to individual contributors' fork setups.
