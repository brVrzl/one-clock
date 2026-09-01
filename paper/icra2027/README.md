# ICRA 2027 manuscript notes

The canonical manuscript source is `main.tex`. It uses the PaperCept
`ieeeconf.cls` package and `IEEEtran.bst` downloaded from the manuscript support
links referenced by the official ICRA 2027 author instructions. The class
options are `letterpaper, 10 pt, conference`, and the source uses
`\overrideIEEEmargins` as required by the PaperCept sample.

The initial submission is double-anonymous and limited to eight pages for the
complete paper, including text, figures, tables, acknowledgments, and
references.

## Frozen anonymous-artifact policy

The reviewer artifact will be constructed using an explicit file allowlist. It
must not be produced by copying the repository, by copying `paper/icra2027/`, or
from a git clone wholesale. No generalized export infrastructure is authorized
at this stage.

The eventual anonymous artifact must exclude:

- `.git/`;
- `paper/icra2027/cdta_draft.md`;
- `paper/icra2027/dynamic_horizon_draft.md`;
- `paper/icra2027/README.md` (this internal note);
- author names and email addresses;
- absolute local paths;
- repository owner/name identifiers;
- project branding; and
- identity-revealing URLs.
