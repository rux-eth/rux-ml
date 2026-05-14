# vibe-rails

Structured vibe coding with Claude. Clone it, start Claude, and build.

> Vibe coding, but on rails — so you don't derail.

## What is this?

A project template that turns Claude Code into a disciplined development partner. No boilerplate code, no language lock-in — just the methodology that makes AI-assisted development actually work.

You clone this repo, start Claude Code, and it asks: **"What do you want to build?"**

From there, Claude guides you through structured design planning (with integrated research rounds), sets up your architecture docs, defines PRs with verification criteria, and enforces a research-backed implementation process as the project evolves. You bring the ideas, Claude handles the structure.

## What's included

- **Onboarding flow** — Claude bootstraps your project from a single conversation
- **Design planning procedure** — structured idea → decisions (with research rounds) → convergence → docs → implementation
- **PR research procedure** — mandatory state assessment + research before every PR implementation. Catches drift between design and code.
- **PR workflow** — one file per PR with mandatory "Before Implementation" gate, scope, dependencies, verification criteria, and research findings
- **Research backlog** — per-PR research status tracker with drift watch
- **Code audit procedure** — post-design alignment checks to catch doc/code drift
- **Architecture docs** — living documents that Claude maintains as the project evolves
- **Constraints system** — define your non-negotiables and Claude enforces them

## Quick start

```bash
# Clone the template
git clone https://github.com/YOUR_USERNAME/vibe-rails.git my-project
cd my-project

# Remove template git history, start fresh
rm -rf .git
git init

# Start Claude Code
claude

# Kick it off
> init

# Claude will ask "What do you want to build?" — go from there.
```

## How it works

1. **Clone & init** — you get a skeleton with procedures and doc templates
2. **First conversation** — Claude detects the project is uninitialized, runs the design planning procedure, and asks what you want to build
3. **Design session** — Claude iterates research rounds with you until every significant decision is research-backed. Then converges. Then populates docs.
4. **CLAUDE.md rewrites itself** — the bootstrap instructions are replaced with your project-specific guidance
5. **Build** — each PR starts with `PROCEDURE-pr-research.md` (mandatory state assessment, research where needed) before any code is written. PRs are defined, roadmap is set.
6. **Iterate** — new design sessions update docs (with research), code audits catch drift, PRs track progress

## Philosophy

- **Enforce structure, not opinions** — the template defines *how* to plan and build, not *what* to build
- **Research before decisions** — "I think" is not sufficient. Every significant decision cites reputable sources.
- **State drifts** — research done at design time can go stale by implementation time. PR research catches it.
- **One PR, one thing** — no scope creep, every change is reviewable
- **Docs are living** — architecture docs update with the code, in the same commit
- **No phantom implementations** — if it's not tested and verified, it's not done
- **Design before code** — think first, research, build second, audit third

## File structure

```
├── CLAUDE.md                        # the brain — drives onboarding + ongoing behavior
├── docs/
│   ├── ARCHITECTURE.md              # project architecture (populated during onboarding)
│   ├── CONSTRAINTS.md               # non-negotiable rules (inc. research-backed decisions)
│   ├── ROADMAP.md                   # PR index with phases and dependencies
│   ├── RESEARCH-BACKLOG.md          # per-PR research status + drift watch
│   └── DESIGN-log.md                # design conversation log (created during sessions)
├── prs/
│   ├── PR-TEMPLATE.md               # template for new PR files (includes research gate)
│   └── PR-XXX-*.md                  # one file per PR (copied from template)
├── PROCEDURE-design-planning.md     # how to run a design session (with research rounds)
├── PROCEDURE-pr-research.md         # mandatory research before every PR implementation
├── PROCEDURE-code-audit.md          # how to audit code after design changes
└── .gitignore
```

## License

MIT
