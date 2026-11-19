# Verifier validation results

Certificates produced by certify-ed: 63 (8 refused by certify-ed, no certificate written; 55 verified).

- verifier exit 0 (every check passed): 55
- verifier exit 1 (a check failed): 0
- verifier error (unreadable input): 0
- S2/S3 statuses certify-ed FAILED: 18; reproduced by the verifier from the stored data: 18; not reproduced: 0

## Mutation tests

| mutation | mutants | caught (exit 1 with a FAIL check) |
|---|---|---|
| radius | 20 | 20 |
| array_hash | 20 | 20 |
| body_hash | 20 | 20 |
| status | 20 | 20 |

| case | mutation | what | exit | caught by |
|---|---|---|---|---|
| heis10_duplicated_vector | radius | pair 4: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| heis10_duplicated_vector | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis10_duplicated_vector | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis10_duplicated_vector | status | S2.eigenpairs INCONCLUSIVE -> CERTIFIED (statement + scope block, body resealed) | 1 | S2.status |
| heis10_mislabeled_sector | radius | pair 0: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| heis10_mislabeled_sector | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis10_mislabeled_sector | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis10_mislabeled_sector | status | S3.completeness FAILED -> CERTIFIED (statement + scope block, body resealed) | 1 | S3.inertia, S3.bound, S3.ordering, S3.gap, S3.window_record |
| heis10_missing_multiplet_member | radius | pair 0: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| heis10_missing_multiplet_member | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis10_missing_multiplet_member | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis10_missing_multiplet_member | status | S3.completeness INCONCLUSIVE -> CERTIFIED (statement + scope block, body resealed) | 1 | S3.inertia, S3.bound, S3.ordering, S3.gap, S3.window_record |
| heis10_subnormal_entries | radius | pair 3: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| heis10_subnormal_entries | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis10_subnormal_entries | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis10_subnormal_entries | status | S2.eigenpairs INCONCLUSIVE -> CERTIFIED (statement + scope block, body resealed) | 1 | S2.status |
| heis10_swapped_pairs | radius | pair 4: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| heis10_swapped_pairs | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis10_swapped_pairs | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis10_swapped_pairs | status | S2.eigenpairs FAILED -> CERTIFIED (statement + scope block, body resealed) | 1 | S2.status |
| heis10_unnormalised_cluster_member | radius | pair 1: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| heis10_unnormalised_cluster_member | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis10_unnormalised_cluster_member | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis10_unnormalised_cluster_member | status | S2.eigenpairs INCONCLUSIVE -> CERTIFIED (statement + scope block, body resealed) | 1 | S2.status |
| heis10_wrong_coupling | radius | pair 6: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| heis10_wrong_coupling | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis10_wrong_coupling | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis10_wrong_coupling | status | S2.eigenpairs INCONCLUSIVE -> CERTIFIED (statement + scope block, body resealed) | 1 | S2.status |
| heis10_wrong_eigenvalue_1e-05 | radius | pair 1: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| heis10_wrong_eigenvalue_1e-05 | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis10_wrong_eigenvalue_1e-05 | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis10_wrong_eigenvalue_1e-05 | status | S2.eigenpairs FAILED -> CERTIFIED (statement + scope block, body resealed) | 1 | S2.status |
| heis10_wrong_eigenvalue_large | radius | pair 4: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| heis10_wrong_eigenvalue_large | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis10_wrong_eigenvalue_large | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis10_wrong_eigenvalue_large | status | S2.eigenpairs FAILED -> CERTIFIED (statement + scope block, body resealed) | 1 | S2.status |
| heis10_wrong_window_count | radius | pair 2: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| heis10_wrong_window_count | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis10_wrong_window_count | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis10_wrong_window_count | status | S3.completeness FAILED -> CERTIFIED (statement + scope block, body resealed) | 1 | S3.ordering, S3.window_record |
| heis8_duplicated_vector | radius | pair 0: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| heis8_duplicated_vector | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis8_duplicated_vector | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis8_duplicated_vector | status | S2.eigenpairs INCONCLUSIVE -> CERTIFIED (statement + scope block, body resealed) | 1 | S2.status |
| heis8_mislabeled_sector | radius | pair 0: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| heis8_mislabeled_sector | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis8_mislabeled_sector | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis8_mislabeled_sector | status | S3.completeness FAILED -> CERTIFIED (statement + scope block, body resealed) | 1 | S3.inertia, S3.bound, S3.ordering, S3.gap, S3.window_record |
| heis8_missing_multiplet_member | radius | pair 0: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| heis8_missing_multiplet_member | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis8_missing_multiplet_member | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis8_missing_multiplet_member | status | S3.completeness INCONCLUSIVE -> CERTIFIED (statement + scope block, body resealed) | 1 | S3.inertia, S3.bound, S3.ordering, S3.gap, S3.window_record |
| heis8_subnormal_entries | radius | pair 0: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| heis8_subnormal_entries | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis8_subnormal_entries | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis8_subnormal_entries | status | S2.eigenpairs INCONCLUSIVE -> CERTIFIED (statement + scope block, body resealed) | 1 | S2.status |
| heis8_swapped_pairs | radius | pair 0: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| heis8_swapped_pairs | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis8_swapped_pairs | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis8_swapped_pairs | status | S2.eigenpairs FAILED -> CERTIFIED (statement + scope block, body resealed) | 1 | S2.status |
| heis8_unnormalised_cluster_member | radius | pair 1: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | C4.arrays_vs_records, R2.residual_bound, R3.existence_radius |
| heis8_unnormalised_cluster_member | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis8_unnormalised_cluster_member | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis8_unnormalised_cluster_member | status | S2.eigenpairs INCONCLUSIVE -> CERTIFIED (statement + scope block, body resealed) | 1 | S2.status |
| heis8_wrong_eigenvalue_1e-05 | radius | pair 1: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| heis8_wrong_eigenvalue_1e-05 | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis8_wrong_eigenvalue_1e-05 | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis8_wrong_eigenvalue_1e-05 | status | S2.eigenpairs FAILED -> CERTIFIED (statement + scope block, body resealed) | 1 | S2.status |
| heis8_wrong_eigenvalue_large | radius | pair 4: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| heis8_wrong_eigenvalue_large | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis8_wrong_eigenvalue_large | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis8_wrong_eigenvalue_large | status | S2.eigenpairs FAILED -> CERTIFIED (statement + scope block, body resealed) | 1 | S2.status |
| heis8_wrong_window_count | radius | pair 2: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| heis8_wrong_window_count | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| heis8_wrong_window_count | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| heis8_wrong_window_count | status | S3.completeness FAILED -> CERTIFIED (statement + scope block, body resealed) | 1 | S3.ordering, S3.window_record |
| j1j2_8_duplicated_vector | radius | pair 0: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed) | 1 | R2.residual_bound, R3.existence_radius |
| j1j2_8_duplicated_vector | array_hash | arrays.sha256[radius] first hex digit changed, body resealed | 1 | H2.radius |
| j1j2_8_duplicated_vector | body_hash | integrity.body_sha256 last hex digit changed | 1 | H1.body_sha256 |
| j1j2_8_duplicated_vector | status | S2.eigenpairs INCONCLUSIVE -> CERTIFIED (statement + scope block, body resealed) | 1 | S2.status |

## Per certificate

| case | certify-ed S1 S2 S3 S4 S8 | verifier exit | pass/fail/skip | reproduced | skipped |
|---|---|---|---|---|---|
| heis10_correct | C C C C C | 0 | 44/0/0 | - | - |
| heis10_duplicated_vector | C I I N I | 0 | 37/0/0 | - | - |
| heis10_mislabeled_sector | C C F N N | 0 | 27/0/0 | S3.failed=REPRODUCED | - |
| heis10_missing_multiplet_member | C C I N N | 0 | 27/0/0 | - | - |
| heis10_nan_eigenvalue | refused | - | - | - | - |
| heis10_nan_eigenvector_entry | refused | - | - | - | - |
| heis10_non_hermitian | F C C C C | 0 | 44/0/0 | - | - |
| heis10_subnormal_entries | C I I I I | 0 | 34/0/0 | - | - |
| heis10_swapped_pairs | C F I N I | 0 | 35/0/0 | S2.failed=REPRODUCED | - |
| heis10_unnormalised_cluster_member | C I I N I | 0 | 27/0/0 | - | - |
| heis10_unnormalised_isolated | C C C C C | 0 | 44/0/0 | - | - |
| heis10_wrong_coupling | F I I I I | 0 | 44/0/0 | - | - |
| heis10_wrong_eigenvalue_1e-05 | C F C C C | 0 | 42/0/0 | S2.failed=REPRODUCED | - |
| heis10_wrong_eigenvalue_large | C F I N C | 0 | 35/0/0 | S2.failed=REPRODUCED | - |
| heis10_wrong_matrix_entry | F C C C C | 0 | 44/0/0 | - | - |
| heis10_wrong_window_count | C C F N C | 0 | 37/0/0 | S3.failed=REPRODUCED | - |
| heis8_correct | C C C C C | 0 | 44/0/0 | - | - |
| heis8_duplicated_vector | C I I N I | 0 | 37/0/0 | - | - |
| heis8_mislabeled_sector | C C F N N | 0 | 27/0/0 | S3.failed=REPRODUCED | - |
| heis8_missing_multiplet_member | C C I N N | 0 | 27/0/0 | - | - |
| heis8_nan_eigenvalue | refused | - | - | - | - |
| heis8_nan_eigenvector_entry | refused | - | - | - | - |
| heis8_non_hermitian | F C C C C | 0 | 44/0/0 | - | - |
| heis8_subnormal_entries | C I I I I | 0 | 34/0/0 | - | - |
| heis8_swapped_pairs | C F I N I | 0 | 35/0/0 | S2.failed=REPRODUCED | - |
| heis8_unnormalised_cluster_member | C I I N I | 0 | 27/0/0 | - | - |
| heis8_unnormalised_isolated | C C C C C | 0 | 44/0/0 | - | - |
| heis8_wrong_coupling | F C C C C | 0 | 54/0/0 | - | - |
| heis8_wrong_eigenvalue_1e-05 | C F C C C | 0 | 42/0/0 | S2.failed=REPRODUCED | - |
| heis8_wrong_eigenvalue_large | C F I N C | 0 | 35/0/0 | S2.failed=REPRODUCED | - |
| heis8_wrong_matrix_entry | F C C C C | 0 | 44/0/0 | - | - |
| heis8_wrong_window_count | C C F N C | 0 | 37/0/0 | S3.failed=REPRODUCED | - |
| j1j2_8_correct | C C C C C | 0 | 54/0/0 | - | - |
| j1j2_8_duplicated_vector | C I I N I | 0 | 37/0/0 | - | - |
| j1j2_8_mislabeled_sector | C C I N C | 0 | 37/0/0 | - | - |
| j1j2_8_missing_multiplet_member | C C I N C | 0 | 37/0/0 | - | - |
| j1j2_8_nan_eigenvalue | refused | - | - | - | - |
| j1j2_8_nan_eigenvector_entry | refused | - | - | - | - |
| j1j2_8_non_hermitian | F C C C C | 0 | 54/0/0 | - | - |
| j1j2_8_subnormal_entries | C I I I I | 0 | 44/0/0 | - | - |
| j1j2_8_unnormalised_cluster_member | C I I N I | 0 | 37/0/0 | - | - |
| j1j2_8_wrong_coupling | F C C C C | 0 | 54/0/0 | - | - |
| j1j2_8_wrong_eigenvalue_1e-05 | C F C C C | 0 | 52/0/0 | S2.failed=REPRODUCED | - |
| j1j2_8_wrong_eigenvalue_large | C F I N I | 0 | 45/0/0 | S2.failed=REPRODUCED | - |
| j1j2_8_wrong_matrix_entry | F C C C C | 0 | 54/0/0 | - | - |
| tfim8_correct | C C C C C | 0 | 44/0/0 | - | - |
| tfim8_duplicated_vector | C I I N I | 0 | 27/0/0 | - | - |
| tfim8_mislabeled_sector | C C F N C | 0 | 37/0/0 | S3.failed=REPRODUCED | - |
| tfim8_missing_multiplet_member | C C I N N | 0 | 27/0/0 | - | - |
| tfim8_nan_eigenvalue | refused | - | - | - | - |
| tfim8_nan_eigenvector_entry | refused | - | - | - | - |
| tfim8_non_hermitian | F F I N I | 0 | 45/0/0 | S2.failed=REPRODUCED | - |
| tfim8_swapped_pairs | C F I N I | 0 | 45/0/0 | S2.failed=REPRODUCED | - |
| tfim8_unnormalised_cluster_member | C I I N I | 0 | 27/0/0 | - | - |
| tfim8_unnormalised_isolated | C C C C C | 0 | 44/0/0 | - | - |
| tfim8_wrong_coupling | F C C N N | 0 | 34/0/0 | - | - |
| tfim8_wrong_eigenvalue_1e-05 | C F C C C | 0 | 42/0/0 | S2.failed=REPRODUCED | - |
| tfim8_wrong_eigenvalue_large | C F I N I | 0 | 35/0/0 | S2.failed=REPRODUCED | - |
| tfim8_wrong_matrix_entry | F C C C C | 0 | 44/0/0 | - | - |
| tfim8_wrong_window_count | C C F N C | 0 | 37/0/0 | S3.failed=REPRODUCED | - |
| dense-arrays | C C C C C | 0 | 38/0/1 | - | R2.residual_bound |
| matrix-free | C C C N N | 0 | 30/0/1 | - | R2.residual_bound |
| npz | C C C C C | 0 | 44/0/0 | - | - |
