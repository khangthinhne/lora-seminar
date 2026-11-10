# Open manual items

## Decisions applied

- exp07 (PEFT DistilBERT/SST-2) is **omitted** from the results and comparison
  text, per the author's decision. A one-line limitation note states that it was
  not run because it requires a GPU and the optional Hugging Face dependencies.
- exp05 is included briefly via Figure 3 and one paragraph.
- exp06 is included via the comparison table and one paragraph; it is explicitly
  labelled as simple absmax 4-bit quantization, not NF4.
- No result placeholders remain in the fragment.

## Items to check before submission

1. Merge `references_additions.bib` into the main bibliography and confirm the
   main document uses the keys `hu2022lora` and `dettmers2023qlora`.
2. Confirm that `booktabs`, `graphicx`, `amsmath`, and `amssymb` are in the main
   preamble (see `required_packages.md`).
3. Check the page budget after pasting into the main document. The fragment has
   three tables and three figures; if the section exceeds 3-4 pages, reduce the
   figure width or move one figure to an appendix.
4. Verify the hardware sentence against the machine actually used for the final
   run (Windows 11, Python 3.10.19, PyTorch 2.2.2+cu121, i7-13700H, 32 GB RAM).
   The results were produced on CPU only; no GPU results are claimed.
5. If exp07 is run later, replace the limitation sentence with a short results
   paragraph and add the corresponding table/figure; also update
   `number_sources.md`.
6. Independently verify all claims and numbers against
   `report/results_summary.md` and `results/tables/*.csv` before submission.
