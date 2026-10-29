# Planted-error corpus: certify-ed

Checker command: `{python} -m certify_ed.cli certify --from {case} -o {out}`  
Run started 2026-10-29T03:33:46+00:00; 60 cases.

**detected 49 / missed 4 (unsound 0, silent 4, error 0) / false alarms 0 (weak 0) of 53 planted cases, 7 controls (clean 7); conditional 0**

Verdicts: *detected* (a statement the error falsifies is FAILED or INCONCLUSIVE), *unsound* (such a statement is CERTIFIED), *silent* (all of them NOT_CHECKED), *refused* (input rejected; accepted as a detection where expected.json allows it), *conditional* (CERTIFIED on a case that allows it under a stronger model), *error* (no readable result). False alarm: FAILED on a statement a correct checker certifies; weak alarm: INCONCLUSIVE on one.

| case | class | k | verdict | S1 | S2 | S3 | S4 | S8 | notes |
|---|---|---|---|---|---|---|---|---|---|
| heis10_correct | correct | 5 | **control** | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| heis10_duplicated_vector | duplicated_vector | 5 | **detected** | CERTIFIED | INCONCLUSIVE | INCONCLUSIVE | NOT_CHECKED | INCONCLUSIVE |  |
| heis10_mislabeled_sector | mislabeled_sector | 3 | **detected** | CERTIFIED | CERTIFIED | FAILED | NOT_CHECKED | NOT_CHECKED |  |
| heis10_missing_multiplet_member | missing_multiplet_member | 2 | **detected** | CERTIFIED | CERTIFIED | INCONCLUSIVE | NOT_CHECKED | NOT_CHECKED |  |
| heis10_nan_eigenvalue | nan_eigenvalue | 5 | **refused** | - | - | - | - | - | exit 1 |
| heis10_nan_eigenvector_entry | nan_eigenvector_entry | 5 | **refused** | - | - | - | - | - | exit 1 |
| heis10_non_hermitian | non_hermitian | 5 | **detected** | FAILED | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| heis10_subnormal_entries | subnormal_entries | 5 | **detected** | CERTIFIED | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |  |
| heis10_swapped_pairs | swapped_pairs | 5 | **detected** | CERTIFIED | FAILED | INCONCLUSIVE | NOT_CHECKED | INCONCLUSIVE |  |
| heis10_unnormalised_cluster_member | unnormalised_cluster_member | 5 | **silent** | CERTIFIED | INCONCLUSIVE | INCONCLUSIVE | NOT_CHECKED | INCONCLUSIVE |  |
| heis10_unnormalised_isolated | unnormalised_isolated | 5 | **control** | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| heis10_wrong_coupling | wrong_coupling | 7 | **detected** | FAILED | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |  |
| heis10_wrong_eigenvalue_1e-05 | wrong_eigenvalue | 5 | **detected** | CERTIFIED | FAILED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| heis10_wrong_eigenvalue_large | wrong_eigenvalue | 5 | **detected** | CERTIFIED | FAILED | INCONCLUSIVE | NOT_CHECKED | CERTIFIED |  |
| heis10_wrong_matrix_entry | wrong_matrix_entry | 5 | **detected** | FAILED | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| heis10_wrong_window_count | wrong_window_count |  | **detected** | CERTIFIED | CERTIFIED | FAILED | NOT_CHECKED | CERTIFIED |  |
| heis8_correct | correct | 5 | **control** | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| heis8_duplicated_vector | duplicated_vector | 5 | **detected** | CERTIFIED | INCONCLUSIVE | INCONCLUSIVE | NOT_CHECKED | INCONCLUSIVE |  |
| heis8_mislabeled_sector | mislabeled_sector | 3 | **detected** | CERTIFIED | CERTIFIED | FAILED | NOT_CHECKED | NOT_CHECKED |  |
| heis8_missing_multiplet_member | missing_multiplet_member | 2 | **detected** | CERTIFIED | CERTIFIED | INCONCLUSIVE | NOT_CHECKED | NOT_CHECKED |  |
| heis8_nan_eigenvalue | nan_eigenvalue | 5 | **refused** | - | - | - | - | - | exit 1 |
| heis8_nan_eigenvector_entry | nan_eigenvector_entry | 5 | **refused** | - | - | - | - | - | exit 1 |
| heis8_non_hermitian | non_hermitian | 5 | **detected** | FAILED | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| heis8_subnormal_entries | subnormal_entries | 5 | **detected** | CERTIFIED | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |  |
| heis8_swapped_pairs | swapped_pairs | 5 | **detected** | CERTIFIED | FAILED | INCONCLUSIVE | NOT_CHECKED | INCONCLUSIVE |  |
| heis8_unnormalised_cluster_member | unnormalised_cluster_member | 5 | **silent** | CERTIFIED | INCONCLUSIVE | INCONCLUSIVE | NOT_CHECKED | INCONCLUSIVE |  |
| heis8_unnormalised_isolated | unnormalised_isolated | 5 | **control** | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| heis8_wrong_coupling | wrong_coupling | 7 | **detected** | FAILED | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| heis8_wrong_eigenvalue_1e-05 | wrong_eigenvalue | 5 | **detected** | CERTIFIED | FAILED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| heis8_wrong_eigenvalue_large | wrong_eigenvalue | 5 | **detected** | CERTIFIED | FAILED | INCONCLUSIVE | NOT_CHECKED | CERTIFIED |  |
| heis8_wrong_matrix_entry | wrong_matrix_entry | 5 | **detected** | FAILED | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| heis8_wrong_window_count | wrong_window_count |  | **detected** | CERTIFIED | CERTIFIED | FAILED | NOT_CHECKED | CERTIFIED |  |
| j1j2_8_correct | correct | 5 | **control** | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| j1j2_8_duplicated_vector | duplicated_vector | 5 | **detected** | CERTIFIED | INCONCLUSIVE | INCONCLUSIVE | NOT_CHECKED | INCONCLUSIVE |  |
| j1j2_8_mislabeled_sector | mislabeled_sector | 3 | **detected** | CERTIFIED | CERTIFIED | INCONCLUSIVE | NOT_CHECKED | CERTIFIED |  |
| j1j2_8_missing_multiplet_member | missing_multiplet_member | 3 | **detected** | CERTIFIED | CERTIFIED | INCONCLUSIVE | NOT_CHECKED | CERTIFIED |  |
| j1j2_8_nan_eigenvalue | nan_eigenvalue | 5 | **refused** | - | - | - | - | - | exit 1 |
| j1j2_8_nan_eigenvector_entry | nan_eigenvector_entry | 5 | **refused** | - | - | - | - | - | exit 1 |
| j1j2_8_non_hermitian | non_hermitian | 5 | **detected** | FAILED | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| j1j2_8_subnormal_entries | subnormal_entries | 5 | **detected** | CERTIFIED | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |  |
| j1j2_8_unnormalised_cluster_member | unnormalised_cluster_member | 5 | **silent** | CERTIFIED | INCONCLUSIVE | INCONCLUSIVE | NOT_CHECKED | INCONCLUSIVE |  |
| j1j2_8_wrong_coupling | wrong_coupling | 7 | **detected** | FAILED | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| j1j2_8_wrong_eigenvalue_1e-05 | wrong_eigenvalue | 5 | **detected** | CERTIFIED | FAILED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| j1j2_8_wrong_eigenvalue_large | wrong_eigenvalue | 5 | **detected** | CERTIFIED | FAILED | INCONCLUSIVE | NOT_CHECKED | INCONCLUSIVE |  |
| j1j2_8_wrong_matrix_entry | wrong_matrix_entry | 5 | **detected** | FAILED | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| tfim8_correct | correct | 5 | **control** | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| tfim8_duplicated_vector | duplicated_vector | 5 | **detected** | CERTIFIED | INCONCLUSIVE | INCONCLUSIVE | NOT_CHECKED | INCONCLUSIVE |  |
| tfim8_mislabeled_sector | mislabeled_sector | 6 | **detected** | CERTIFIED | CERTIFIED | FAILED | NOT_CHECKED | CERTIFIED |  |
| tfim8_missing_multiplet_member | missing_multiplet_member | 4 | **detected** | CERTIFIED | CERTIFIED | INCONCLUSIVE | NOT_CHECKED | NOT_CHECKED |  |
| tfim8_nan_eigenvalue | nan_eigenvalue | 5 | **refused** | - | - | - | - | - | exit 1 |
| tfim8_nan_eigenvector_entry | nan_eigenvector_entry | 5 | **refused** | - | - | - | - | - | exit 1 |
| tfim8_non_hermitian | non_hermitian | 5 | **detected** | FAILED | FAILED | INCONCLUSIVE | NOT_CHECKED | INCONCLUSIVE |  |
| tfim8_swapped_pairs | swapped_pairs | 5 | **detected** | CERTIFIED | FAILED | INCONCLUSIVE | NOT_CHECKED | INCONCLUSIVE |  |
| tfim8_unnormalised_cluster_member | unnormalised_cluster_member | 5 | **silent** | CERTIFIED | INCONCLUSIVE | INCONCLUSIVE | NOT_CHECKED | INCONCLUSIVE |  |
| tfim8_unnormalised_isolated | unnormalised_isolated | 5 | **control** | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| tfim8_wrong_coupling | wrong_coupling | 5 | **detected** | FAILED | CERTIFIED | CERTIFIED | NOT_CHECKED | NOT_CHECKED |  |
| tfim8_wrong_eigenvalue_1e-05 | wrong_eigenvalue | 5 | **detected** | CERTIFIED | FAILED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| tfim8_wrong_eigenvalue_large | wrong_eigenvalue | 5 | **detected** | CERTIFIED | FAILED | INCONCLUSIVE | NOT_CHECKED | INCONCLUSIVE |  |
| tfim8_wrong_matrix_entry | wrong_matrix_entry | 5 | **detected** | FAILED | CERTIFIED | CERTIFIED | CERTIFIED | CERTIFIED |  |
| tfim8_wrong_window_count | wrong_window_count |  | **detected** | CERTIFIED | CERTIFIED | FAILED | NOT_CHECKED | CERTIFIED |  |

## By error class

| class | cases | verdicts |
|---|---|---|
| correct | 4 | control 4 |
| duplicated_vector | 4 | detected 4 |
| mislabeled_sector | 4 | detected 4 |
| missing_multiplet_member | 4 | detected 4 |
| nan_eigenvalue | 4 | refused 4 |
| nan_eigenvector_entry | 4 | refused 4 |
| non_hermitian | 4 | detected 4 |
| subnormal_entries | 3 | detected 3 |
| swapped_pairs | 3 | detected 3 |
| unnormalised_cluster_member | 4 | silent 4 |
| unnormalised_isolated | 3 | control 3 |
| wrong_coupling | 4 | detected 4 |
| wrong_eigenvalue | 8 | detected 8 |
| wrong_matrix_entry | 4 | detected 4 |
| wrong_window_count | 3 | detected 3 |

Total checker time 885 s.
