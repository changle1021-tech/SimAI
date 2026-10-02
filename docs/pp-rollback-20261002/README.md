# PP rollback archive

Two snapshots are published on `codex/pp-rollback-20261002`.

The before snapshot retains the PP implementation, native profiles, experiment records, original TP backups and external CPU entrypoints/tables. The after snapshot restores the original TP source and CPU table. No main branch or local working index is changed by publication. The three existing child forks are published first, and this snapshot points at their verified commits.

DeepGEMM, cutlass, fmt and sarathi-serve forks were unavailable. Their exact original versions and source URLs are recorded in `nested-repositories-before.json`. DeepGEMM local changes are preserved in the verified Git bundle, whose prerequisite is the recorded original DeepGEMM commit. The other three have no local changes. Upstream URLs remain for those clean dependencies.

To recover DeepGEMM changes, clone the recorded upstream repository, ensure the prerequisite commit is present, and fetch `DeepGEMM-local-changes.bundle` with `refs/heads/codex/pp-rollback-20261002` as the source ref. No model weights or transient caches are included.

Historical reports inside the ZIP describe intermediate PP experiments; see the original-path regression report for the later TP regression findings.
