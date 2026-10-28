# LinkedIn post (draft)

I built a Q&A system over Ethereum specs that is allowed to say "no".

Ask "What is BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?" and you get `8`, plus the chunk and the
source line it came from. A draft that says `16` is withheld before anyone sees it. The model is
untrusted; a deterministic checker, with no model in it, decides what is released: every sentence
cited, every cited chunk retrieved, every number, hex value and identifier present in the cited
text next to the name it is attached to.

What is in the box: 901 EIPs, ERCs, consensus specs and Flashbots docs (17k chunks), hybrid
BM25 + dense retrieval, a small learned reranker, a citation graph, a CLI, an HTTP API and an MCP
server. numpy is the only required dependency; the tests (312 of them) run offline.

What I measured, on 504 synthetic questions with a train/test split I fixed before tuning:

- the reranker lifts held-out Recall@5 from 0.815 to 0.934;
- the "hybrid" does NOT clearly beat plain BM25 (+2.6 points, confidence interval spans zero);
- the checker rejected 98% of answers with a negation added and 82% of answers whose value was
  swapped for another number from the same chunk, and accepted 97% of correct ones;
- but "verified" is not "correct": the no-LLM baseline verifies 138 of 151 test questions and
  only 75 of those answers contain the reference value.

The write-up says what the checker cannot catch (for example a chunk that mentions both 0xEF and
0xFE will happily support a sentence that swaps them) and how the evaluation could be fooling me.

Repository and the full evaluation write-up are linked in the first comment: https://github.com/Intermountainh8ter/ethresearch-rag

#ethereum #rag #opensource #python
