---
name: Plugin Protocol Tests
description: "[semantic] Plugin Protocol Tests"
depends:
  - plugin_interface
  - ir_module
  - skill_module
  - assumptions_module
  - runs_module
parent: test_suite
kind: file
source_paths:
  - tests/test_connector_plugin.py
---
