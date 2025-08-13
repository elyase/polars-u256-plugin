SHELL=/bin/bash

.PHONY: build lib copy-lib wheel test clean

build: lib

lib:
	cargo build --release

copy-lib: lib
	@mkdir -p polars_u256_plugin/bin
	@if [[ "$$(uname)" == "Darwin" ]]; then \
		cp target/release/libpolars_u256_plugin.dylib polars_u256_plugin/bin/; \
	elif [[ "$$(uname)" == "Linux" ]]; then \
		cp target/release/libpolars_u256_plugin.so polars_u256_plugin/bin/; \
	else \
		cp target/release/polars_u256_plugin.dll polars_u256_plugin/bin/; \
	fi

wheel: copy-lib
	python -m pip install --upgrade pip build
	python -m build

test:
	pytest -q

clean:
	rm -rf dist build polars_u256_plugin/bin/*
