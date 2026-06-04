# psee_adt

This directory contains the bundled dataset-evaluation toolbox used by the
reproduction scripts. It is packaged locally so that the supplementary code can
be installed without relying on a separate repository checkout.

## Installation

```bash
cd psee_adt-master
pip install .
```

## Notes

Only packaging-level changes are included here:
- Python package layout
- `setup.py` for editable/local installation
- evaluator return values wired for programmatic use

The third-party copyright and license notices required for redistribution are
kept in the source files and in `LICENSE`.
