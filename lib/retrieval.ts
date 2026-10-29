import { getContextDocs, type ContextDocRow } from "./db";

/**
 * Deliberately simple retrieval: keyword overlap + region/category boosting,
 * not real embeddings. The corpus here is a handful of short event blurbs,
 * so TF-IDF/BM25 or a vector index would be pure overhead. At real scale
 * (hundreds+ of docs) swap this for Postgres + pgvector with proper
 * embeddings — the call site (lib/agent.ts) doesn't need to change, only
 * this function's implementation.
 */
export function retrieveContext(
  region: string,
  category: string,
  limit = 3
): ContextDocRow[] {
  const docs = getContextDocs();
  const scored = docs.map((doc) => {
    let score = 0;
    if (doc.region === "all" || doc.region === region) score += 2;
    if (doc.category === "all" || doc.category === category) score += 2;
    return { doc, score };
  });
  return scored
    .filter((s) => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((s) => s.doc);
}
