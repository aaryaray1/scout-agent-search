# LinkedIn post draft: Scout production-readiness pass

I spent this week getting Scout, my agent-facing search project, closer to something I'd actually host. I went in planning a cleanup pass and came out having fixed things I didn't know were broken.

The plan was simple: run a cyclomatic complexity review before adding more features. I used a Claude Code skill that measures complexity per function, ranks the hotspots, and refactors worst-first.

The interesting part is that the complexity was basically fine. Highest function scored 9 against a refactor-now threshold of 11. Under the old workflow I'd have glanced at that, called it clean, and moved on.

But measuring every function means reading every function. And that's where the actual problems were:

My 5MB response limit didn't limit anything. It checked the size after the entire body was already in memory, and otherwise trusted the Content-Length header, which a malicious server just omits. It read like a working safety limit. It was decorative.

My retriever wasn't thread-safe. FastAPI runs sync endpoints in a threadpool, so a search can land in the middle of an ingest. I was swapping three parallel data structures one at a time. I proved it by reverting the fix and running the new test: searches scoring documents against the wrong embeddings.

And the one that stung: Scout couldn't be deployed as the ingest-only service that is, on paper, the entire point of the project. It refused to boot without a folder of local markdown. I'd never tested that path because I always had the demo corpus sitting there.

Also: index writes through file handles that were never closed, a config typo that would hang the process forever, top_k=0 silently becoming 3.

What I appreciated about the skill was that it pushed back on the metric itself. It told me not to game the number, and left five functions alone with a note that they read honestly as-is. I also deleted one of my own new tests after checking it against a deliberately broken version and finding it passed anyway. A test that can't fail is worse than no test.

Two things I got wrong along the way, for the record. I was sure exceeding the redirect limit would return the redirect body as content. httpx already handles that. And my logging change buried the CLI output under HuggingFace download chatter, which I only caught because I actually ran it.

Tests went 52 to 106 and the suite got 3x faster. Complexity average dropped from 2.98 to 2.40.

Bugs were never really about tangled code. They were about code that read fine and did something else.
