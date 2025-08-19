---
"kilo-code": patch
---

OpenRouter inference providers whose context window is smaller than that of the top provider for a particular model are now automatically ignored by default. They can still be used by selecting them specifically in the Provider Routing settings.
