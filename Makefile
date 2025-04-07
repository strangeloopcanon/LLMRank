PYPROJECT = pyproject.toml
VERSION = $(shell grep '^version' $(PYPROJECT) | sed -E 's/.*"([0-9]+\.[0-9]+\.[0-9]+)"/\1/')

.PHONY: clean build check upload bump-patch bump-minor bump-major git-release publish

clean:
	rm -rf build dist *.egg-info

build: clean
	python -m build

check:
	twine check dist/*

upload:
	twine upload dist/*

bump-patch:
	@python scripts/bump_version.py patch

bump-minor:
	@python scripts/bump_version.py minor

bump-major:
	@python scripts/bump_version.py major

git-release:
	git add -A
	git commit -m "Release v$(VERSION)" || echo "Nothing to commit"
	@if git rev-parse "v$(VERSION)" >/dev/null 2>&1; then \
		echo "⚠️ Tag v$(VERSION) already exists. Skipping tag creation."; \
	else \
		git tag v$(VERSION); \
	fi
	git push
	git push --tags
	@python scripts/create_github_release.py v$(VERSION)

BUMP ?= patch

publish:
	@make bump-$(BUMP)
	@make build
	@make check
	@make upload
	@make git-release
