SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

# A dirty working tree is gated against HEAD; a clean tree against origin/main.
BASE_REF ?= $(shell git diff --quiet HEAD -- && echo origin/main || echo HEAD)
DIFF_COVERAGE ?= 80
MUTATION_WORKERS ?= 8

GATE_DIR := .data/gate

.PHONY: gate clean

gate:
	@git rev-parse --verify "$(BASE_REF)^{commit}" >/dev/null 2>&1 || { echo "Unknown BASE_REF: $(BASE_REF)" >&2; exit 2; }
	@echo "gate base: $(BASE_REF)"
	@rm -f .coverage.* .coveragerc.gremlins
	@mkdir -p $(GATE_DIR)
	uv run ruff format --check .
	uv run ruff check .
	@base="$$(git merge-base "$(BASE_REF)" HEAD)"; \
	changed="$$( { git diff --name-only --diff-filter=ACMR "$$base" -- src/priorart; git ls-files --others --exclude-standard -- src/priorart; } | sort -u | grep '\.py$$' || true )"; \
	report="coverage/gremlins/gremlins.json"; \
	log="$(GATE_DIR)/mutations.log"; \
	args=(); \
	rm -f "$$report"; \
	if [ -n "$$changed" ]; then \
		targets="$$(printf '%s\n' "$$changed" | paste -sd, -)"; \
		args=(--gremlins --gremlin-targets="$$targets" \
			--gremlin-workers=$(MUTATION_WORKERS) --gremlin-executor=subprocess \
			--gremlin-cache --gremlin-report=json); \
	fi; \
	env COVERAGE_FILE=$(GATE_DIR)/.coverage uv run python -m pytest \
		--cov=priorart --cov-report=xml:$(GATE_DIR)/coverage.xml "$${args[@]}" \
		2>&1 | tee "$$log"; \
	status="$$?"; \
	if [ "$$status" -ne 0 ]; then exit "$$status"; fi; \
	uv run diff-cover $(GATE_DIR)/coverage.xml --compare-branch=$(BASE_REF) --fail-under=$(DIFF_COVERAGE) --show-uncovered --include-untracked || exit "$$?"; \
	if [ -z "$$changed" ]; then \
		echo "No changed Python source files; mutation testing skipped."; \
	elif [ -f "$$report" ]; then \
		uv run python -c 'import json, sys; s = json.load(open(sys.argv[1]))["summary"]; sys.exit(any(s[key] for key in ("survived", "timeout", "error")))' "$$report"; \
	elif ! grep -q "No gremlins found" "$$log"; then \
		echo "Mutation report is missing." >&2; exit 1; \
	fi

clean:
	@rm -rf $(GATE_DIR) coverage .gremlins_cache .pytest_cache
	@rm -f .coverage .coverage.* .coveragerc.gremlins
