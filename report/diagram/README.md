# Diagram sources

- `source.mmd` is the editable Mermaid source.
- `diagram.svg` is the exported/standalone version used by the report.

The diagram is an original illustration for this project. It shows the LoRA
training branches, the merged inference form, and the simulated QLoRA
composition (simple absmax 4-bit base, **not NF4**).

Optional re-export with Mermaid CLI:

```bash
npx -y @mermaid-js/mermaid-cli -i report/diagram/source.mmd -o report/diagram/diagram.svg
```
