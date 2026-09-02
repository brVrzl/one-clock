# Gate M remote exposure audit

The audit inspected 26 fetched `origin/*` refs, 149 reachable commits, 5311 reachable objects, and parsed 1122 unique outcome JSON/JSONL blobs.

The previously proposed 134 blocks contained four additional raw outcome exposures, all in `experiments/component_temporal_reuse/rapid_component_smoke/results_libero_object_task6.json` at commit `38046a961cd796b30b554c9de407d64aa82518cf` (blob `018c26fbb8ada57dc459bfaa3b91403f87a99ca5`): task 6 states 25, 26, 28, and 29.

Those four blocks and only those blocks were removed. No replacement states were added. The frozen Gate M cohort contains **130 paired blocks**, above the minimum of 120.

No outcome record was found for any remaining cell before preregistration. Protocol-only exposure was not used as an automatic exclusion.

Exact final states by task:

- Object task 1: 30,32,33,36,37,40,41,42,43,46,49
- Object task 2: 24,25,26,28,29,30,32,33,36,37,40,41,42,43,46,49
- Object task 3: 24,25,26,28,29,30,32,33,36,37,40,41,42,43,46,49
- Object task 4: 30,32,33,36,37,40,41,42,43,46,49
- Object task 5: 24,25,26,28,29,30,32,33,36,37,40,41,42,43,46,49
- Object task 6: 24,30,32,33,36,37,40,41,42,43,46,49
- Object task 7: 24,25,26,28,29,30,32,33,36,37,40,41,42,43,46,49
- Object task 8: 24,25,26,28,29,30,32,33,36,37,40,41,42,43,46,49
- Object task 9: 24,25,26,28,29,30,32,33,36,37,40,41,42,43,46,49
