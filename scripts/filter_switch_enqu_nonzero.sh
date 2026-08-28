#!/usr/bin/env bash
set -euo pipefail

workspace_root="${1:-/home/changle/SA/SimAI}"
output_file="${2:-${workspace_root}/switch_enqu_nonzero_matches.tsv}"
summary_file="${output_file%.tsv}_summary.tsv"

printf 'scenario\ttrace_file\tline\ttime_ns\tswitch_id\tport_queue\tqlen_before_bytes\trecord\n' > "${output_file}"
printf 'scenario\ttrace_file\tmatch_count\tfirst_line\tfirst_record\n' > "${summary_file}"

while IFS= read -r trace_file; do
    scenario="$(basename "$(dirname "${trace_file}")")"
    LC_ALL=C awk \
        -v scenario="${scenario}" \
        -v output_file="${output_file}" \
        -v summary_file="${summary_file}" '
        BEGIN { OFS = "\t" }
        $5 == "Enqu" && ($4 + 0) > 0 {
            node = $2
            sub(/^n:/, "", node)
            if (node != 48 && node != 49 && node != 50) {
                next
            }

            count++
            if (count == 1) {
                first_line = FNR
                first_record = $0
            }
            print scenario, FILENAME, FNR, $1, node, $3, $4, $0 >> output_file
        }
        END {
            print scenario, FILENAME, count + 0, first_line, first_record >> summary_file
        }
        ' "${trace_file}"
done < <(find "${workspace_root}/results" -mindepth 2 -maxdepth 2 -type f -name trace.txt | sort)

printf 'Matches: %s\n' "${output_file}"
printf 'Summary: %s\n' "${summary_file}"
