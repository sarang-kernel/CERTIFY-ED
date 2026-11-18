# docs/project/STYLE.md: writing rules for the documents of this repository

These rules hold for every Markdown document under docs/, blog/, lean/ and for docs/project/CHANGELOG_PROJECT.md.
The script tools/lint_docs.py rejects a document that breaks rules 1 to 5; rules 6 to 9 are checked by reading.

Rule 1. No bullet points and no lists of any kind. No line may start with a dash, an asterisk, a plus sign, a bullet
character, or an enumeration such as "1." or "(a)". Write connected paragraphs; use "First, ... Second, ..." inside
sentences when order matters. Markdown pipe tables are allowed for numbers, files and statuses.

Rule 2. No em dash and no en dash anywhere, and no double hyphen used as a dash. Write ranges as "4 to 12 October"
or "dimensions 924 to 112720". Hyphenated words (rank-one, S=1/2, two-stack) are fine.

Rule 3. No promotional or filler vocabulary; the list is in tools/lint_docs.py. Plain, short sentences. Define every
technical term the first time it appears in a document, or point to the document that defines it. A reader with
undergraduate linear algebra and some programming must be able to follow.

Rule 4. Say exactly how strongly something has been checked. A statement is "kernel-checked in Lean" (a Lean theorem
with no sorry, accepted by the Lean kernel), "re-accepted by a second checker" (the independent kernel nanoda, level K2; or a replay by leanchecker, level K+), "recomputed
in Lean" (a compiled Lean program reproduced the number), "machine-checked" (a script tested it on random and
adversarial cases), or "checked by" a named person. The words `verified` and `proven` are not used, because under the
project's decision D-12 new mathematics counts as established only after a person has re-derived it, and that check
by Sarang Vehale is still pending for the derivation documents. Never write that a check was made by a person when it
was not.

Rule 5. The documents describe the work, not the writing process. They do not mention software assistants, agents,
sessions, prompts or chats. The record of how the material was produced is kept in the separate agent-material
package (see README.md). Where a check was produced automatically rather than by a person, the text says so in those
words ("a second derivation written independently from the statement alone, not a human check").

Rule 6. Every experiment and every failure gets a reason. For each one write, as prose and in this order: why it was a
reasonable next step (which earlier result, gap, theorem or cost argument made it worth doing, and what outcome would
have changed the plan), what exactly was done (object, model, sizes, settings), how the instrument was checked
(controls, negative controls, comparisons), what happened (numbers), and what the result does and does not show.
"Because it was requested" or "for completeness" alone is not a reason.

Rule 7. Copy every number exactly from its source file and cite that file once, by its path relative to
archive_v5/ (for example code/certify-ed/RESULT_evidence_run_3.2.1.md). A number that is in no source is not
written. Keep the difference between a certified bound, a floating-point estimate and a published value.

Rule 8. Code, file names, record fields and commands go in backticks or fenced blocks. Everything needed to recreate
a result is written down as commands that run without any interactive help.

Rule 9. Dates are written as "21 October 2026". Versions as "3.2.1", tags as `v3.2.1`, commits by their short hash.
