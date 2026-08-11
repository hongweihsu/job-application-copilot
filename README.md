# Job Application Copilot

An evidence-grounded resume and job-description matcher. It identifies which requirements are
supported, partially supported, or missing—and refuses to invent experience that is not present in
the resume.

> **Milestone 1:** This repository currently contains a transparent deterministic baseline. It is
> intentionally useful without an API key and creates a measurable benchmark before RAG and LLM
> components are introduced.

## Why this project exists

Many resume assistants generate polished text without checking whether each claim is supported.
That creates a real risk for applicants. This project treats resume content as evidence: every
positive assessment must cite resume text, while missing experience must remain explicitly missing.

## Current features

- Paste resume and job-description text
- Import text-based PDF resumes up to 5 MB
- Extract job requirements using a transparent baseline
- Classify each requirement as `supported`, `partial`, or `missing`
- Cite matching resume evidence
- Validate structured API responses with Pydantic
- Run a golden-dataset evaluation in CI
- Run locally or with Docker without an API key

## Demo

![Job Application Copilot interface](docs/screenshot-placeholder.svg)

## Architecture

```text
PDF / pasted resume ──> text extraction ──> evidence units ─┐
                                                            ├─> matcher ─> validated result ─> UI
Job description ──────> requirement extraction ─────────────┘
```

The next milestone will replace the baseline matcher with an evaluated retrieval pipeline while
preserving the same schemas and golden dataset.

## Run locally

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync --dev
uv run uvicorn app.main:app --reload
```

Open <http://localhost:8000>. API documentation is available at
<http://localhost:8000/docs>.

Alternatively, use Docker:

```bash
docker compose up --build
```

## Verify the project

```bash
uv run ruff check .
uv run pytest --cov=app
uv run python -m evaluation.run_evaluation
```

The evaluation command returns a non-zero exit code when baseline status accuracy falls below 90%,
so regressions fail the GitHub Actions workflow.

## API example

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "resume_text": "Software Engineer. Built Python APIs used by three internal teams.",
    "job_description": "Required: professional experience building Python API services."
  }'
```

## Roadmap

- [x] Evidence-grounded deterministic baseline
- [x] Structured output validation
- [x] PDF text extraction
- [x] Golden dataset and CI quality gate
- [ ] Expand and manually label the evaluation dataset
- [ ] Add full-text and vector retrieval baselines
- [ ] Compare vector, hybrid, and reranked retrieval
- [ ] Add citation correctness and unsupported-claim metrics
- [ ] Add request tracing, latency, token, and cost metrics
- [ ] Deploy a public demo

## Responsible-use note

Do not upload resumes containing information you are not authorised to process. Review all output
before using it in an application. The copilot should improve how genuine experience is communicated,
not manufacture qualifications or employment history.

## License

MIT

