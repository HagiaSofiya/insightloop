.PHONY: web up down db api demo test replay compare lock-prompts personas respond embed neighbors index-bench chunk-compare ask retrieval-eval themes agent eval-seed eval eval-agreement research agent-compare worker submit request flow

agent:
	cd services/api && .venv/bin/python -m app.cli agent $(ARGS)

themes:
	cd services/api && .venv/bin/python -m app.cli themes $(ARGS)

retrieval-eval:
	cd services/api && .venv/bin/python -m app.cli retrieval-eval $(ARGS)

ask:
	cd services/api && .venv/bin/python -m app.cli ask $(ARGS)

index-bench:
	cd services/api && .venv/bin/python -m app.cli index-bench $(ARGS)

chunk-compare:
	cd services/api && .venv/bin/python -m app.cli chunk-compare $(ARGS)

neighbors:
	cd services/api && .venv/bin/python -m app.cli neighbors $(ARGS)

embed:
	cd services/api && .venv/bin/python -m app.cli embed $(ARGS)

respond:
	cd services/api && .venv/bin/python -m app.cli respond --brief "$(BRIEF)" $(ARGS)

BRIEF ?= A meal-kit startup wants to know why customers pause or cancel after the first month.

personas:
	cd services/api && .venv/bin/python -m app.cli personas --brief "$(BRIEF)" $(ARGS)

lock-prompts:
	cd services/api && .venv/bin/python -m app.prompts.manifest

compare:
	cd services/api && .venv/bin/python -m app.cli compare discussion_guide v1 v2 --cases ../../evals/briefs.json $(ARGS)

replay:
	cd services/api && .venv/bin/python -m app.cli replay $(ID)

db:
	docker compose exec -T db psql -U insightloop -d insightloop < services/api/schema.sql

test:
	cd services/api && .venv/bin/python -m pytest -v

demo:
	cd services/api && .venv/bin/python -m app.demo

api:
	cd services/api && .venv/bin/uvicorn app.main:app --reload

up:
	docker compose up -d --wait

down:
	docker compose down

eval-seed:
	cd services/api && .venv/bin/python -m app.cli eval-seed

eval:
	cd services/api && .venv/bin/python -m app.cli eval $(ARGS)

eval-agreement:
	cd services/api && .venv/bin/python -m app.cli eval-agreement

research:
	cd services/api && .venv/bin/python -m app.cli research "$(Q)"

agent-compare:
	cd services/api && .venv/bin/python -m app.cli agent-compare

worker:
	cd services/api && .venv/bin/python -m app.cli worker $(ARGS)

submit:
	cd services/api && .venv/bin/python -m app.cli submit "$(Q)"

request:
	cd services/api && .venv/bin/python -m app.cli request $(ID)

web:
	cd services/web && npm run dev -- -p 3100

flow:
	cd services/api && .venv/bin/python -m app.cli workflow $(ACTION) "$(TARGET)" $(ARGS)
