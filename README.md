# ml-daily

Daily practice repo focused on **ML pipelines** and **AI agents** — aligned with an Associate ML Engineer role.

A scheduled script picks a task from `curriculum.yaml` each day, scaffolds a dated folder with a starter template, commits the scaffold, and pushes to GitHub. Then I fill in the actual work during the day and commit on top.

## Layout

```
ml-daily/
├── pipelines/        # ML pipeline exercises (YYYY-MM-DD-topic/)
├── agents/           # AI agent exercises (YYYY-MM-DD-topic/)
├── notes/            # paper / concept summaries
├── curriculum.yaml   # rotating task pool — edit freely
├── progress.md       # log of scaffolded days
└── scripts/
    ├── setup.sh      # one-time setup (run once)
    ├── daily.sh      # entrypoint launchd calls
    └── pick_task.py  # picks task + writes starter template
```

## Setup (run once)

```bash
cd ~/Desktop/Github/ml-daily
bash scripts/setup.sh
```

The setup script will:
1. `git init` and set repo-local author to `Chelappan Kumar <chelappankumar23@gmail.com>`
2. Make the initial commit
3. Prompt you to create the GitHub remote with `gh repo create` (or print manual instructions if `gh` is missing)
4. Install the launchd plist at `~/Library/LaunchAgents/com.chelappan.mldaily.plist`

## How the daily run works

1. launchd fires `scripts/daily.sh` at 09:00 local time
2. Script sleeps a random 0–540 minutes (so the commit lands somewhere in 09:00–18:00)
3. `pick_task.py` chooses today's topic from `curriculum.yaml` (rotates pipeline → agent → notes)
4. A new folder `pipelines/2026-05-20-<topic-slug>/` is created with `starter.py` (or `starter.md` for notes), populated with TODOs, suggested approach, and references
5. The scaffold is committed and pushed
6. **You then do the actual exercise** during the day and commit your real work on top

The scaffold commit is a real artifact (your curated curriculum + dated exercise definition), not filler. Your actual learning commits build on it.

## Manual run

```bash
bash scripts/daily.sh --now    # skip the random sleep
```

## Editing the curriculum

Open `curriculum.yaml` and add/remove tasks. Categories: `pipelines`, `agents`, `notes`. The picker avoids repeating a topic within 30 days.

## Logs

- `progress.md` — one line per scaffolded day
- `scripts/daily.log` — stdout/stderr from cron runs (gitignored)

## Disabling

```bash
launchctl unload ~/Library/LaunchAgents/com.chelappan.mldaily.plist
```
