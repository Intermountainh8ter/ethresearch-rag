# Discord (short)

ethrag: Q&A over 901 EIPs/ERCs/consensus specs/Flashbots docs where a deterministic checker (no
model) can withhold the answer. Every sentence must be cited, and every number/hex/identifier
must be in the cited chunk next to its name. `ethrag ask --explain "What is
BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?"` -> `8` + the source line. Wrong draft -> withheld.

Honest numbers (504 synthetic questions, test n=151): reranker R@5 0.934, hybrid 0.815 vs BM25
0.788 (not significant). Checker accepts 97% of correct answers, 18% of swapped-value ones. Not
correct != verified; blind spots documented. numpy only, offline tests, CLI / HTTP / MCP, MIT.
